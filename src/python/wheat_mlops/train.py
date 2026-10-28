"""Train a YOLO model on the prepared wheat dataset.

Meant to run on a Kaggle/Colab GPU (see notebooks/kaggle_train.ipynb). It is deliberately
standalone: it imports nothing from the rest of the package, so it runs on Kaggle without
installing the project (whose Python requirement is newer than Kaggle's).

No MLflow / DVC here. Registering the results is a separate step, done from the files this
script writes into --out-dir: best.pt, metrics.json, results.csv, prepare_stats.json.

Usage:
    python train.py --data-yaml data/processed/dataset.yaml --out-dir trained [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

import yaml

# --smoke: a 2-epoch run on 5% of the data at low resolution, to prove the setup works.
SMOKE_ARGS = {"epochs": 2, "fraction": 0.05, "imgsz": 320, "cache": False}


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


def train(data_yaml: str | Path, out_dir: str | Path, cfg: dict) -> dict:
    # Heavy imports live here so the helpers above stay importable (and testable) anywhere.
    import torch
    from ultralytics import YOLO, settings
    from ultralytics import __version__ as ultralytics_version

    # Part 7 only produces files; experiment tracking is added later by the pipeline owners.
    os.environ.setdefault("WANDB_MODE", "disabled")
    try:
        settings.update({"mlflow": False, "wandb": False})
    except Exception as exc:  # noqa: BLE001
        print(f"note: could not switch off Ultralytics logger integrations: {exc}")

    out_dir = Path(out_dir).resolve()  # absolute, or Ultralytics nests it under runs/detect
    out_dir.mkdir(parents=True, exist_ok=True)
    args = dict(cfg["args"])
    on_gpu = torch.cuda.is_available()
    if not on_gpu:
        print("WARNING: no GPU found - training on CPU will be extremely slow.")

    data = resolve_dataset_yaml(data_yaml, out_dir)
    started = time.time()
    YOLO(cfg["model"]).train(
        data=str(data), project=str(out_dir), name="run", exist_ok=True, **args
    )
    train_minutes = (time.time() - started) / 60

    run_dir = out_dir / "run"
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} does not exist")
    shutil.copy2(best, out_dir / "best.pt")
    if (run_dir / "results.csv").is_file():
        shutil.copy2(run_dir / "results.csv", out_dir / "results.csv")
    stats_file = Path(data_yaml).parent / "prepare_stats.json"
    if stats_file.is_file():
        shutil.copy2(stats_file, out_dir / "prepare_stats.json")

    batch = args.get("batch", 16)
    val = YOLO(str(best)).val(
        data=str(data),
        imgsz=args.get("imgsz", 640),
        batch=batch if isinstance(batch, int) and batch > 0 else 16,
        plots=False,
        project=str(out_dir),
        name="val",
        exist_ok=True,
    )

    report = {
        "metrics": extract_metrics(val.results_dict),
        "train_minutes": round(train_minutes, 1),
        "model": cfg["model"],
        "args": args,
        "best_checkpoint": "best.pt",
        "best_checkpoint_mb": round((out_dir / "best.pt").stat().st_size / 1e6, 2),
        "device": torch.cuda.get_device_name(0) if on_gpu else "cpu",
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-yaml", required=True, help="dataset.yaml from prepare_data")
    parser.add_argument("--out-dir", required=True, help="where to write best.pt, metrics.json ...")
    parser.add_argument("--params", default="train_params.yaml")
    parser.add_argument("--smoke", action="store_true", help="quick 2-epoch sanity run")
    a = parser.parse_args()

    cfg = load_train_config(a.params)
    if a.smoke:
        cfg = with_smoke_overrides(cfg)
    validate_config(cfg)
    print(json.dumps(train(a.data_yaml, a.out_dir, cfg), indent=2))


if __name__ == "__main__":
    import sys
    main()
