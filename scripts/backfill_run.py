"""Backfill the existing training run into MLflow.

Run this once from the project root to import the data from
  data/working/trained/metrics.json
  data/working/trained/results.csv
  data/working/trained/run/   (plots, weights)

into a new MLflow run so it appears in the UI.

Usage:
    .venv/Scripts/python.exe scripts/backfill_run.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mlflow

TRAINED_DIR = Path("data/working/trained")
METRICS_JSON = TRAINED_DIR / "metrics.json"
RESULTS_CSV = TRAINED_DIR / "results.csv"
ULTRALYTICS_RUN_DIR = TRAINED_DIR / "run"
BEST_PT = TRAINED_DIR / "best.pt"
PREPARE_STATS = TRAINED_DIR / "prepare_stats.json"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "wheat-train"
REGISTERED_MODEL = "wheat-yolo"
RUN_NAME = "backfill-run-1"


def register_checkpoint(run_id: str) -> None:
    """Register the checkpoint in an already logged run without duplicating it."""
    if not BEST_PT.is_file():
        raise FileNotFoundError(f"Cannot find checkpoint: {BEST_PT}")

    import mlflow.pytorch
    from ultralytics import YOLO

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pytorch.log_model(
            pytorch_model=YOLO(str(BEST_PT)).model,
            name="wheat-yolo-checkpoint",
            serialization_format="pickle",
        )
        mv = mlflow.register_model(model_uri=model_info.model_uri, name=REGISTERED_MODEL)
        mlflow.set_tag("registered_model_version", mv.version)
    print(f"Registered '{REGISTERED_MODEL}' as version {mv.version} on run {run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--register-only-run-id",
        help="Register best.pt to a run that already has its metrics and artifacts.",
    )
    options = parser.parse_args()

    if not METRICS_JSON.is_file():
        raise FileNotFoundError(f"Cannot find {METRICS_JSON}. Run training first.")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    if options.register_only_run_id:
        register_checkpoint(options.register_only_run_id)
        return

    report = json.loads(METRICS_JSON.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"MLflow run: {RUN_NAME} [{run_id}]")

        # ---- params -------------------------------------------------------
        args = report.get("args", {})
        mlflow.log_param("model_source", report.get("model", "unknown"))
        mlflow.log_param("device", report.get("device", "unknown"))
        mlflow.log_param("python", report.get("python", "unknown"))
        mlflow.log_param("ultralytics", report.get("ultralytics", "unknown"))
        mlflow.log_param("torch", report.get("torch", "unknown"))
        for k, v in args.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(k, v)

        # ---- per-epoch metrics from results.csv ---------------------------
        latest_epoch_metrics: dict[str, float] = {}
        if RESULTS_CSV.is_file():
            with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step = int(float(row.get("epoch", 0)))
                    metrics: dict[str, float] = {}
                    for raw_key, raw_val in row.items():
                        key = raw_key.strip()
                        if key == "epoch":
                            continue
                        # Ultralytics adds a (B) suffix to detection metrics;
                        # MLflow rejects parentheses in metric names.
                        key = key.replace("(", "").replace(")", "")
                        try:
                            metrics[key] = float(raw_val)
                        except (ValueError, TypeError):
                            pass
                    if metrics:
                        mlflow.log_metrics(metrics, step=step)
                        latest_epoch_metrics = metrics
            print(f"  Logged per-epoch metrics from {RESULTS_CSV}")

        # ---- final / summary metrics --------------------------------------
        # The training process can stop after writing results.csv but before
        # refreshing metrics.json. Prefer the latest CSV epoch in that case.
        metrics_json_is_stale = RESULTS_CSV.is_file() and BEST_PT.is_file() and (
            RESULTS_CSV.stat().st_mtime > METRICS_JSON.stat().st_mtime
            or BEST_PT.stat().st_mtime > METRICS_JSON.stat().st_mtime
        )
        if metrics_json_is_stale:
            summary_metrics = {}
            for name, suffix in {
                "precision": "precisionB",
                "recall": "recallB",
                "map50": "mAP50B",
                "map50_95": "mAP50-95B",
            }.items():
                value = next(
                    (v for key, v in latest_epoch_metrics.items() if key.endswith(suffix)), None
                )
                if value is not None:
                    summary_metrics[name] = value
        else:
            summary_metrics = report.get("metrics", {})

        if not metrics_json_is_stale and report.get("train_minutes") is not None:
            mlflow.log_metric("train_minutes", report["train_minutes"])
        checkpoint_mb = (
            BEST_PT.stat().st_size / 1e6
            if BEST_PT.is_file()
            else report.get("best_checkpoint_mb")
        )
        if checkpoint_mb is not None:
            mlflow.log_metric("best_checkpoint_mb", checkpoint_mb)
        for k, v in summary_metrics.items():
            if v is not None:
                mlflow.log_metric(f"final_{k}", float(v))

        # ---- artifacts ----------------------------------------------------
        if BEST_PT.is_file():
            mlflow.log_artifact(str(BEST_PT), artifact_path="weights")
            print(f"  Logged artifact: {BEST_PT}")
        if RESULTS_CSV.is_file():
            mlflow.log_artifact(str(RESULTS_CSV), artifact_path="training")
        if PREPARE_STATS.is_file():
            mlflow.log_artifact(str(PREPARE_STATS), artifact_path="training")
        if METRICS_JSON.is_file() and not metrics_json_is_stale:
            mlflow.log_artifact(str(METRICS_JSON), artifact_path="training")

        # Ultralytics-generated plots
        if ULTRALYTICS_RUN_DIR.is_dir():
            for png in sorted(ULTRALYTICS_RUN_DIR.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="plots")
                print(f"  Logged plot: {png.name}")

        val_dir = TRAINED_DIR / "val"
        if val_dir.is_dir():
            for img in sorted(val_dir.glob("*.jpg")):
                mlflow.log_artifact(str(img), artifact_path="val_previews")

        # ---- register model -----------------------------------------------
        # MLflow 3 registers logged MLflow Models, not arbitrary run artifacts.
        from ultralytics import YOLO

        model_info = mlflow.pytorch.log_model(
            pytorch_model=YOLO(str(BEST_PT)).model,
            name="wheat-yolo-checkpoint",
            serialization_format="pickle",
        )
        mv = mlflow.register_model(model_uri=model_info.model_uri, name=REGISTERED_MODEL)
        mlflow.set_tag("registered_model_version", mv.version)
        print(f"  Registered '{REGISTERED_MODEL}' as version {mv.version}")

        print(f"\nDone!  View at: mlflow ui --backend-store-uri {TRACKING_URI}")
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
