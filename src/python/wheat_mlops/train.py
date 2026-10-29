"""Train a YOLO model on the prepared wheat dataset.

Meant to run on a Kaggle/Colab GPU (see notebooks/kaggle_train.ipynb). It is deliberately
standalone: it imports nothing from the rest of the package, so it runs on Kaggle without
installing the project (whose Python requirement is newer than Kaggle's).

Each invocation writes to --out-dir, replacing that run's local output files so best.pt stays
at a stable path for the next training loop. Every invocation is tracked as a separate MLflow
run (per-epoch metrics, loss curves, val plots, and the best checkpoint are logged). The best.pt
is also registered in the MLflow Model Registry under the name "wheat-yolo".

Usage:
    python -m wheat_mlops.train --data-yaml data/processed/dataset.yaml --out-dir trained [--smoke]
    python -m wheat_mlops.train --data-yaml data/processed/dataset.yaml --out-dir trained --run-name "exp-lr0.01"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

# --smoke: a 2-epoch run on 5% of the data at low resolution, to prove the setup works.
SMOKE_ARGS = {"epochs": 2, "fraction": 0.05, "imgsz": 320, "cache": False}

# MLflow experiment / registry name used by all train runs.
MLFLOW_EXPERIMENT = "wheat-train"
MLFLOW_REGISTERED_MODEL = "wheat-yolo"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def load_train_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train"]
    cfg.setdefault("args", {})
    return cfg


def validate_config(cfg: dict) -> None:
    if not cfg.get("model"):
        raise ValueError("train.model is missing")
    args = cfg["args"]
    if args.get("epochs", 1) < 1:
        raise ValueError("train.args.epochs must be >= 1")
    if args.get("imgsz", 640) % 32 != 0:
        raise ValueError("train.args.imgsz must be a multiple of 32")


def with_smoke_overrides(cfg: dict) -> dict:
    return {**cfg, "args": {**cfg["args"], **SMOKE_ARGS}}


def resolve_dataset_yaml(data_yaml: str | Path, out_dir: str | Path) -> Path:
    """Write a copy of dataset.yaml with an absolute `path:` so Ultralytics cannot misresolve."""
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(data_yaml.resolve().parent)
    out = Path(out_dir) / "dataset.resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _pick(results: dict, needle: str) -> float | None:
    for key, value in results.items():
        if needle in key:
            return float(value)
    return None


def extract_metrics(results_dict: dict) -> dict:
    """Map Ultralytics' result keys (e.g. 'metrics/mAP50(B)') to stable names."""
    return {
        "precision": _pick(results_dict, "precision"),
        "recall": _pick(results_dict, "recall"),
        "map50": _pick(results_dict, "mAP50("),
        "map50_95": _pick(results_dict, "mAP50-95"),
    }


def _log_epoch_metrics_from_csv(csv_path: Path) -> None:
    """Read Ultralytics results.csv and log each row as a step in the active MLflow run."""
    import mlflow

    if not csv_path.is_file():
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row.get("epoch", 0)))
            metrics: dict[str, float] = {}
            for raw_key, raw_val in row.items():
                key = raw_key.strip()
                if key == "epoch":
                    continue
                # Ultralytics names detection metrics with a (B) suffix, while
                # MLflow rejects parentheses in metric names. Match the naming
                # used by Ultralytics' own MLflow callback.
                key = key.replace("(", "").replace(")", "")
                try:
                    metrics[key] = float(raw_val)
                except (ValueError, TypeError):
                    pass
            if metrics:
                mlflow.log_metrics(metrics, step=step)


def train(
    data_yaml: str | Path,
    out_dir: str | Path,
    cfg: dict,
    *,
    run_name: str | None = None,
    register_model: bool = True,
) -> dict:
    """Train the YOLO model, log everything to MLflow, and optionally register the best.pt.

    Args:
        data_yaml: Path to the YOLO dataset.yaml produced by prepare_data.
        out_dir: Stable output directory. Local output files are overwritten on each run.
        cfg: Parsed train config dict (from load_train_config).
        run_name: Human-readable name surfaced in the MLflow UI.  Defaults to a timestamp.
        register_model: If True, register best.pt in the MLflow Model Registry.
    """
    import mlflow
    import mlflow.pytorch
    import torch
    from ultralytics import YOLO, settings
    from ultralytics import __version__ as ultralytics_version

    os.environ.setdefault("WANDB_MODE", "disabled")

    # ------------------------------------------------------------------ MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Enable Ultralytics' built-in MLflow callback so per-epoch metrics and plots
    # are streamed automatically during model.train().
    try:
        settings.update({"mlflow": True, "wandb": False})
    except Exception as exc:  # noqa: BLE001
        print(f"note: could not enable Ultralytics MLflow integration: {exc}")

    args = dict(cfg["args"])
    on_gpu = torch.cuda.is_available()
    if not on_gpu:
        print("WARNING: no GPU found - training on CPU will be extremely slow.")

    # Keep local checkpoints at a stable path so the next short training run can
    # reuse the same model path from improve_params.yaml.
    run_out_dir = Path(out_dir).resolve()
    run_out_dir.mkdir(parents=True, exist_ok=True)
    run_dir_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    data = resolve_dataset_yaml(data_yaml, run_out_dir)

    with mlflow.start_run(run_name=run_dir_name) as active_run:
        run_id = active_run.info.run_id
        print(f"\n=== MLflow Run: {run_dir_name} [ID: {run_id}] ===")

        # --- log hyper-parameters ------------------------------------------------
        mlflow.log_param("model_source", cfg["model"])
        mlflow.log_param("device", "gpu" if on_gpu else "cpu")
        mlflow.log_param("python", platform.python_version())
        mlflow.log_param("ultralytics", ultralytics_version)
        mlflow.log_param("torch", torch.__version__)
        for k, v in args.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(k, v)

        # --- train ---------------------------------------------------------------
        started = time.time()
        # Tell Ultralytics to use our active run's experiment so its autologger
        # writes into the same run. Keep it active because the callback normally
        # ends MLflow runs at train end, while we still log validation/artifacts.
        os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT
        os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
        YOLO(cfg["model"]).train(
            data=str(data),
            project=str(run_out_dir),
            name="run",
            exist_ok=True,
            **args,
        )
        train_minutes = (time.time() - started) / 60

        ultralytics_run_dir = run_out_dir / "run"
        best = ultralytics_run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"training finished but {best} does not exist")

        # Copy outputs to stable locations so later runs can reuse best.pt.
        shutil.copy2(best, run_out_dir / "best.pt")
        if (ultralytics_run_dir / "results.csv").is_file():
            shutil.copy2(ultralytics_run_dir / "results.csv", run_out_dir / "results.csv")
        stats_file = Path(data_yaml).parent / "prepare_stats.json"
        if stats_file.is_file():
            shutil.copy2(stats_file, run_out_dir / "prepare_stats.json")

        # --- per-epoch metrics from results.csv ----------------------------------
        # Ultralytics autologger streams these live; we also parse the CSV afterwards
        # to make sure every epoch row is captured (handles Kaggle / offline runs).
        _log_epoch_metrics_from_csv(run_out_dir / "results.csv")

        # --- final validation run ------------------------------------------------
        batch = args.get("batch", 16)
        val = YOLO(str(run_out_dir / "best.pt")).val(
            data=str(data),
            imgsz=args.get("imgsz", 640),
            batch=batch if isinstance(batch, int) and batch > 0 else 16,
            plots=True,
            project=str(run_out_dir),
            name="val",
            exist_ok=True,
        )
        final_metrics = extract_metrics(val.results_dict)

        # --- log summary metrics -------------------------------------------------
        mlflow.log_metric("train_minutes", round(train_minutes, 1))
        for k, v in final_metrics.items():
            if v is not None:
                mlflow.log_metric(f"final_{k}", v)
        mlflow.log_metric(
            "best_checkpoint_mb",
            round((run_out_dir / "best.pt").stat().st_size / 1e6, 2),
        )

        # --- log artifacts -------------------------------------------------------
        mlflow.log_artifact(str(run_out_dir / "best.pt"), artifact_path="weights")
        if (run_out_dir / "results.csv").is_file():
            mlflow.log_artifact(str(run_out_dir / "results.csv"), artifact_path="training")
        if (run_out_dir / "prepare_stats.json").is_file():
            mlflow.log_artifact(str(run_out_dir / "prepare_stats.json"), artifact_path="training")

        # Ultralytics plot PNGs (loss curves, confusion matrix, PR curve …)
        for png in sorted(ultralytics_run_dir.glob("*.png")):
            mlflow.log_artifact(str(png), artifact_path="plots")
        # Val batch previews
        val_dir = run_out_dir / "val"
        for img in sorted(val_dir.glob("*.jpg")) if val_dir.is_dir() else []:
            mlflow.log_artifact(str(img), artifact_path="val_previews")

        # --- write metrics.json (same as before, for backward compat) ------------
        report = {
            "metrics": final_metrics,
            "train_minutes": round(train_minutes, 1),
            "model": cfg["model"],
            "args": args,
            "run_dir": str(run_out_dir),
            "best_checkpoint": "best.pt",
            "best_checkpoint_mb": round((run_out_dir / "best.pt").stat().st_size / 1e6, 2),
            "device": torch.cuda.get_device_name(0) if on_gpu else "cpu",
            "ultralytics": ultralytics_version,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "mlflow_run_id": run_id,
        }
        (run_out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(run_out_dir / "metrics.json"), artifact_path="training")

        # --- register the model --------------------------------------------------
        if register_model:
            # MLflow 3 registers logged MLflow Models, not arbitrary run artifacts.
            checkpoint_model = YOLO(str(run_out_dir / "best.pt")).model
            model_info = mlflow.pytorch.log_model(
                pytorch_model=checkpoint_model,
                name="wheat-yolo-checkpoint",
            )
            mv = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=MLFLOW_REGISTERED_MODEL,
            )
            mlflow.set_tag("registered_model_version", mv.version)
            print(f"Registered '{MLFLOW_REGISTERED_MODEL}' version {mv.version}")

        print(f"=== Run finished. Outputs in: {run_out_dir} ===\n")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-yaml", required=True, help="dataset.yaml from prepare_data")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for best.pt, metrics.json, results.csv, and plots.",
    )
    parser.add_argument("--params", default="train_params.yaml")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable name for this run (shown in MLflow UI).  Defaults to a timestamp.",
    )
    parser.add_argument("--smoke", action="store_true", help="quick 2-epoch sanity run")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip registering best.pt in the MLflow Model Registry.",
    )
    a = parser.parse_args()

    cfg = load_train_config(a.params)
    if a.smoke:
        cfg = with_smoke_overrides(cfg)
    validate_config(cfg)
    report = train(
        a.data_yaml,
        a.out_dir,
        cfg,
        run_name=a.run_name,
        register_model=not a.no_register,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
