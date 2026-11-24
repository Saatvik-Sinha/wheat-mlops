"""Evaluate detector robustness and run a bounded SPSA degradation search.

Usage:
    python -m wheat_mlops.robustness --dataset-dir data/perturbed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from wheat_mlops.perturbations import (
    CORRUPTIONS,
    apply_spsa_perturbation,
    load_config,
)


def select_search_ids(split_csv: str | Path, count: int, seed: int) -> list[str]:
    """Select a deterministic, approximately stratified SPSA subset of validation IDs."""
    split = pd.read_csv(split_csv)
    validation = split.loc[split["split"] == "val"].copy()
    if len(validation) <= count:
        raise ValueError(
            f"Need more than {count} validation images to keep SPSA search and evaluation disjoint"
        )
    if "source" in validation:
        source = validation["source"].fillna("unknown").astype(str)
    else:
        source = pd.Series("unknown", index=validation.index)
    if "n_boxes" in validation:
        density = pd.to_numeric(validation["n_boxes"], errors="coerce").fillna(0) // 15
        density = density.clip(upper=6).astype(int).astype(str)
    else:
        density = pd.Series("all", index=validation.index)
    validation["_stratum"] = source + "_" + density
    rng = random.Random(seed)
    strata: dict[str, list[str]] = {}
    for key, group in validation.groupby("_stratum", sort=True):
        ids = group["image_id"].astype(str).tolist()
        rng.shuffle(ids)
        strata[str(key)] = ids

    selected: list[str] = []
    keys = list(strata)
    while len(selected) < count:
        made_progress = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
    return selected


def spsa_optimize(
    objective,
    parameter_bounds: dict[str, list[float]],
    *,
    iterations: int,
    seed: int,
    a: float,
    c: float,
    alpha: float,
    gamma: float,
    stability_constant: float,
) -> tuple[dict[str, float], list[dict]]:
    """Maximize a scalar objective using projected SPSA in normalized [0, 1] space."""
    if iterations < 1 or not parameter_bounds:
        raise ValueError("SPSA requires at least one iteration and one parameter")
    names = list(parameter_bounds)
    bounds = [
        (float(parameter_bounds[name][0]), float(parameter_bounds[name][1])) for name in names
    ]
    if any(low >= high for low, high in bounds):
        raise ValueError("Each SPSA lower bound must be less than its upper bound")

    def to_values(point: list[float]) -> dict[str, float]:
        return {
            name: low + min(1.0, max(0.0, point[i])) * (high - low)
            for i, (name, (low, high)) in enumerate(zip(names, bounds, strict=True))
        }

    rng = random.Random(seed)
    point = [0.5] * len(names)
    best_point = point.copy()
    best_value = -math.inf
    history: list[dict] = []
    for iteration in range(iterations):
        ak = a / (stability_constant + iteration + 1) ** alpha
        ck = c / (iteration + 1) ** gamma
        delta = [1.0 if rng.random() >= 0.5 else -1.0 for _ in names]
        plus = [min(1.0, max(0.0, point[i] + ck * delta[i])) for i in range(len(names))]
        minus = [min(1.0, max(0.0, point[i] - ck * delta[i])) for i in range(len(names))]
        plus_value = float(objective(to_values(plus)))
        minus_value = float(objective(to_values(minus)))
        if not math.isfinite(plus_value) or not math.isfinite(minus_value):
            raise ValueError("SPSA objective returned a non-finite value")
        if plus_value > best_value:
            best_value, best_point = plus_value, plus.copy()
        if minus_value > best_value:
            best_value, best_point = minus_value, minus.copy()
        gradient = [(plus_value - minus_value) / (2.0 * ck * delta[i]) for i in range(len(names))]
        point = [min(1.0, max(0.0, point[i] + ak * gradient[i])) for i in range(len(names))]
        history.append(
            {
                "iteration": iteration + 1,
                "objective_plus": plus_value,
                "objective_minus": minus_value,
                "best_objective": best_value,
                "plus_parameters": to_values(plus),
                "minus_parameters": to_values(minus),
                "updated_parameters": to_values(point),
            }
        )
    return to_values(best_point), history


def _stable_seed(seed: int, image_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _write_dataset_yaml(root: Path, val: str, destination: Path) -> Path:
    spec = {
        "path": str(root.resolve()),
        "train": val,
        "val": val,
        "nc": 1,
        "names": ["wheat"],
    }
    destination.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return destination


def _metric_values(result) -> dict[str, float]:
    box = result.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def _evaluate(
    model,
    dataset_yaml: Path,
    *,
    output_dir: Path,
    name: str,
    evaluation: dict,
) -> dict[str, float]:
    result = model.val(
        data=str(dataset_yaml),
        imgsz=int(evaluation.get("imgsz", 640)),
        batch=int(evaluation.get("batch", 8)),
        device=evaluation.get("device") or None,
        conf=float(evaluation.get("confidence", 0.001)),
        iou=float(evaluation.get("iou", 0.7)),
        plots=False,
        save_json=False,
        verbose=False,
        workers=0,
        project=str(output_dir / "ultralytics"),
        name=name,
        exist_ok=True,
    )
    return _metric_values(result)


def _condition_metrics(
    model,
    dataset_dir: Path,
    condition: str,
    image_ids: list[str],
    *,
    temp_dir: Path,
    output_dir: Path,
    evaluation: dict,
) -> dict[str, float]:
    condition_images = dataset_dir / "images" / condition
    available = {path.stem: path for path in condition_images.iterdir() if path.is_file()}
    missing = set(image_ids) - set(available)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing from condition {condition}")
    list_file = temp_dir / f"{condition}.txt"
    list_file.write_text(
        "\n".join(str(available[item].resolve()) for item in image_ids) + "\n", encoding="utf-8"
    )
    spec = {
        "path": str(dataset_dir.resolve()),
        "train": str(list_file.resolve()),
        "val": str(list_file.resolve()),
        "nc": 1,
        "names": ["wheat"],
    }
    data_yaml = temp_dir / f"{condition}.yaml"
    data_yaml.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return _evaluate(model, data_yaml, output_dir=output_dir, name=condition, evaluation=evaluation)


def _candidate_metrics(
    model,
    dataset_dir: Path,
    image_ids: list[str],
    parameters: dict[str, float],
    *,
    seed: int,
    temp_root: Path,
    output_dir: Path,
    evaluation: dict,
    example_dir: Path | None = None,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="spsa_candidate_", dir=temp_root) as temporary:
        candidate = Path(temporary)
        images_out, labels_out = candidate / "images" / "val", candidate / "labels" / "val"
        images_out.mkdir(parents=True)
        labels_out.mkdir(parents=True)
        source_root = dataset_dir / "images" / "clean"
        source_labels = dataset_dir / "labels" / "clean"
        for image_id in image_ids:
            with Image.open(source_root / f"{image_id}.jpg") as original:
                transformed = apply_spsa_perturbation(
                    original,
                    parameters,
                    seed=_stable_seed(seed, image_id),
                )
                transformed.save(images_out / f"{image_id}.jpg", format="JPEG", quality=95)
            shutil.copy2(source_labels / f"{image_id}.txt", labels_out / f"{image_id}.txt")
            if example_dir is not None and image_id in image_ids[:8]:
                example_dir.mkdir(parents=True, exist_ok=True)
                original_path = source_root / f"{image_id}.jpg"
                with Image.open(original_path) as original:
                    clean = original.convert("RGB")
                    changed = Image.open(images_out / f"{image_id}.jpg").convert("RGB")
                    preview = Image.new("RGB", (clean.width * 2, clean.height))
                    preview.paste(clean, (0, 0))
                    preview.paste(changed, (clean.width, 0))
                    preview.save(example_dir / f"{image_id}_clean_vs_spsa.jpg", quality=90)
        data_yaml = _write_dataset_yaml(candidate, "images/val", candidate / "dataset.yaml")
        return _evaluate(
            model,
            data_yaml,
            output_dir=output_dir,
            name=f"spsa_{len(image_ids)}_{parameters['blur_sigma']:.2f}_"
            f"{parameters['noise_sigma']:.2f}_{parameters['brightness_delta']:.2f}",
            evaluation=evaluation,
        )


def _metric_rows(standard: dict[str, dict[str, float]], spsa_metrics: dict) -> list[dict]:
    return [{"condition": name, **metrics} for name, metrics in standard.items()] + [
        {"condition": "spsa_best_heldout", **spsa_metrics}
    ]


def run_experiment(
    dataset_dir: str | Path,
    *,
    config: dict,
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    results_dir: str | Path = "data/robustness/results",
    log_mlflow: bool = True,
) -> dict:
    """Run the standard benchmark and SPSA search against a disjoint validation subset."""
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest.csv"
    split_path = dataset_dir / "split.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generated manifest not found: {manifest_path}")
    evaluation = config["evaluation"]
    spsa = config["spsa"]
    selected_checkpoint = Path(checkpoint or evaluation["checkpoint"]).resolve()
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {selected_checkpoint}")
    manifest = pd.read_csv(manifest_path)
    clean_ids = manifest.loc[manifest["corruption"] == "clean", "image_id"].astype(str).tolist()
    search_ids = select_search_ids(
        split_path, int(evaluation.get("search_images", 64)), int(spsa.get("seed", 42))
    )
    if not set(search_ids).issubset(clean_ids):
        raise ValueError("Generated dataset does not contain every ID in the SPSA subset")
    eval_ids = [image_id for image_id in clean_ids if image_id not in set(search_ids)]
    if not eval_ids:
        raise ValueError("No held-out validation images remain after selecting the SPSA subset")

    results_dir = Path(results_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(selected_checkpoint))
    standard: dict[str, dict[str, float]] = {}
    condition_names = ["clean"] + [
        f"{name}_s{level}" for name in CORRUPTIONS for level in range(1, 6)
    ]

    with tempfile.TemporaryDirectory(prefix="wheat_eval_", dir=run_dir) as temporary:
        temp_dir = Path(temporary)
        for condition in condition_names:
            standard[condition] = _condition_metrics(
                model,
                dataset_dir,
                condition,
                eval_ids,
                temp_dir=temp_dir,
                output_dir=run_dir,
                evaluation=evaluation,
            )

        bounds = spsa["parameters"]
        history: list[dict] = []

        def objective(parameters: dict[str, float]) -> float:
            metrics = _candidate_metrics(
                model,
                dataset_dir,
                search_ids,
                parameters,
                seed=int(spsa.get("seed", 42)),
                temp_root=run_dir,
                output_dir=run_dir,
                evaluation=evaluation,
            )
            return 1.0 - metrics["map50_95"]

        best_parameters, history = spsa_optimize(
            objective,
            bounds,
            iterations=int(spsa["iterations"]),
            seed=int(spsa.get("seed", 42)),
            a=float(spsa["a"]),
            c=float(spsa["c"]),
            alpha=float(spsa["alpha"]),
            gamma=float(spsa["gamma"]),
            stability_constant=float(spsa["stability_constant"]),
        )

        spsa_metrics = _candidate_metrics(
            model,
            dataset_dir,
            eval_ids,
            best_parameters,
            seed=int(spsa.get("seed", 42)),
            temp_root=run_dir,
            output_dir=run_dir,
            evaluation=evaluation,
            example_dir=run_dir / "examples",
        )

    report = {
        "checkpoint": str(selected_checkpoint),
        "dataset": str(dataset_dir),
        "objective": "1 - mAP@0.5:0.95 (maximized on the SPSA search subset)",
        "evaluation_settings": evaluation,
        "spsa_settings": spsa,
        "search_image_count": len(search_ids),
        "search_image_ids": search_ids,
        "heldout_image_count": len(eval_ids),
        "heldout_image_ids": eval_ids,
        "standard_metrics": standard,
        "spsa": {
            "best_parameters": best_parameters,
            "best_search_objective": history[-1]["best_objective"],
            "heldout_metrics": spsa_metrics,
            "history": history,
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(_metric_rows(standard, spsa_metrics)).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "spsa_best_parameters.json").write_text(
        json.dumps(best_parameters, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "iteration": row["iteration"],
                "objective_plus": row["objective_plus"],
                "objective_minus": row["objective_minus"],
                "best_objective": row["best_objective"],
                **{f"plus_{key}": value for key, value in row["plus_parameters"].items()},
                **{f"minus_{key}": value for key, value in row["minus_parameters"].items()},
                **{f"updated_{key}": value for key, value in row["updated_parameters"].items()},
            }
            for row in history
        ]
    ).to_csv(run_dir / "spsa_history.csv", index=False)

    if log_mlflow:
        import mlflow

        mlflow_cfg = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8")).get(
            "mlflow", {}
        )
        if mlflow_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "wheat-mlops-pipeline-a"))
        with mlflow.start_run(run_name=f"robustness-{run_id}"):
            mlflow.log_param("robustness.checkpoint", str(selected_checkpoint))
            mlflow.log_param("robustness.dataset_images", len(clean_ids))
            mlflow.log_param("robustness.spsa_search_images", len(search_ids))
            mlflow.log_param("robustness.spsa_iterations", int(spsa["iterations"]))
            mlflow.log_param("robustness.spsa_objective", "1-map50_95")
            mlflow.log_param("robustness.imgsz", int(evaluation.get("imgsz", 640)))
            mlflow.log_param("robustness.confidence", float(evaluation.get("confidence", 0.001)))
            mlflow.log_param("robustness.iou", float(evaluation.get("iou", 0.7)))
            mlflow.log_params({f"spsa.best.{key}": value for key, value in best_parameters.items()})
            for condition, metrics in standard.items():
                safe = condition.replace("-", "_")
                mlflow.log_metrics({f"{safe}_{key}": value for key, value in metrics.items()})
            mlflow.log_metrics(
                {f"spsa_best_heldout_{key}": value for key, value in spsa_metrics.items()}
            )
            for artifact in (
                "report.json",
                "metrics.csv",
                "spsa_history.csv",
                "spsa_best_parameters.json",
            ):
                mlflow.log_artifact(str(run_dir / artifact), artifact_path="robustness")
            mlflow.log_artifact(str(Path(config_path)), artifact_path="robustness")
            mlflow.log_artifact(str(manifest_path), artifact_path="robustness")
            examples = run_dir / "examples"
            if examples.is_dir():
                for path in examples.glob("*.jpg"):
                    mlflow.log_artifact(str(path), artifact_path="robustness/examples")
    return {"run_dir": str(run_dir), "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/perturbed")
    parser.add_argument("--config", default="perturbations.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--results-dir", default="data/robustness/results")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_experiment(
        args.dataset_dir,
        config=config,
        config_path=args.config,
        checkpoint=args.checkpoint,
        results_dir=args.results_dir,
        log_mlflow=not args.no_mlflow,
    )
    print(json.dumps({"run_dir": result["run_dir"], "report": result["report"]}, indent=2))


if __name__ == "__main__":
    main()
