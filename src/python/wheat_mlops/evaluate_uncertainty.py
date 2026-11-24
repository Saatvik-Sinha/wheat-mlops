"""CLI and MLflow tracking script for Detection Uncertainty Estimation (M2).

Runs YOLO detection on clean, mildly degraded, moderately degraded, heavily degraded,
and domain-shifted images, matches predictions to ground truth via one-to-one matching,
and evaluates confidence calibration (Reliability diagrams, ECE, MCE, Brier score).
All parameters, metrics, plots, and summary artifacts are logged to MLflow.

Usage:
    python -m wheat_mlops.evaluate_uncertainty
    python -m wheat_mlops.evaluate_uncertainty --smoke
    python -m wheat_mlops.evaluate_uncertainty --model-path best.pt --num-samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

from wheat_mlops.config import load_params
from wheat_mlops.uncertainty.evaluator import UncertaintyEvaluator, load_yolo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2: Detection Uncertainty Estimation & Confidence Calibration"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt). Defaults to model.pretrained or trained/best.pt.",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="data/processed/dataset.yaml",
        help="Path to dataset.yaml containing image and label locations.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/working/uncertainty",
        help="Output directory for reliability diagrams, metrics CSV, and plots.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for one-to-one matching of predictions to ground-truth boxes.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.10,
        help="Minimum confidence threshold for predicted boxes.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of confidence bins for calibration curves and ECE.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional limit on the number of validation images to evaluate.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on a small subset (15 images) at 320px for fast smoke testing / CI.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (run standalone).",
    )
    return parser.parse_args()


def load_validation_records(
    data_yaml_path: str | Path,
    max_samples: int | None = None,
) -> list[dict]:
    """Load validation image paths and their ground-truth bounding boxes."""
    yaml_path = Path(data_yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset yaml not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_dir = yaml_path.resolve().parent
    val_images_rel = data_cfg.get("val", "images/val")
    val_images_dir = base_dir / val_images_rel

    if not val_images_dir.is_dir():
        # Fallback to direct path
        val_images_dir = Path("data/processed/images/val")

    image_paths = sorted(val_images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"no validation images found in {val_images_dir}")

    if max_samples and max_samples < len(image_paths):
        image_paths = image_paths[:max_samples]

    records = []
    labels_dir = val_images_dir.parent.parent / "labels" / "val"

    for img_p in image_paths:
        label_file = labels_dir / f"{img_p.stem}.txt"
        gts = load_yolo_labels(label_file) if label_file.is_file() else []
        records.append(
            {
                "image_path": str(img_p),
                "image_id": img_p.stem,
                "ground_truths": gts,
            }
        )

    return records


def main() -> None:
    args = parse_args()
    params = load_params()

    # Determine model path
    model_path = args.model_path
    if not model_path:
        trained_best = Path("data/working/trained/best.pt")
        pretrained = Path(params["model"]["pretrained"])
        base_ckpt = Path(params["model"]["base_checkpoint"])
        if trained_best.is_file():
            model_path = str(trained_best)
        elif pretrained.is_file():
            model_path = str(pretrained)
        elif base_ckpt.is_file():
            model_path = str(base_ckpt)
        else:
            model_path = "yolov8n.pt"

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    imgsz = 320 if args.smoke else int(params["model"].get("imgsz", 640))
    num_samples = 15 if args.smoke else args.num_samples

    print(f"Loading validation images (smoke={args.smoke}, max_samples={num_samples})...")
    records = load_validation_records(args.data_yaml, max_samples=num_samples)
    print(f"Loaded {len(records)} validation records.")

    evaluator = UncertaintyEvaluator(
        model=model,
        imgsz=imgsz,
        iou_threshold=args.iou_thresh,
        conf_threshold=args.conf_thresh,
        device="cpu",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        "clean",
        "mild_degraded",
        "moderate_degraded",
        "heavy_degraded",
        "domain_shifted_fog",
        "domain_shifted_illumination",
    ]

    print("Beginning uncertainty evaluation across conditions...")
    results = evaluator.evaluate_suite(
        image_records=records,
        output_dir=out_dir,
        conditions=conditions,
        n_bins=args.n_bins,
    )

    # Print summary table
    summary_df = pd.read_csv(out_dir / "uncertainty_summary.csv")
    print("\n=== Uncertainty & Confidence Calibration Summary ===")
    print(summary_df.to_string(index=False))

    # MLflow tracking
    if not args.no_mlflow:
        try:
            import mlflow

            tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("wheat-uncertainty")

            run_name = args.run_name or f"uncertainty-{Path(model_path).stem}"
            if args.smoke:
                run_name += "-smoke"

            with mlflow.start_run(run_name=run_name):
                # Log evaluation parameters
                mlflow.log_params(
                    {
                        "model_path": model_path,
                        "iou_threshold": args.iou_thresh,
                        "conf_threshold": args.conf_thresh,
                        "n_bins": args.n_bins,
                        "imgsz": imgsz,
                        "num_eval_images": len(records),
                        "smoke": args.smoke,
                    }
                )

                # Log metrics for each condition
                all_metrics: dict[str, float] = {}
                for cond_name, cond_res in results.items():
                    cond_metrics = cond_res.calibration.to_mlflow_metrics(prefix=cond_name)
                    all_metrics.update(cond_metrics)

                # Log delta metrics showing calibration degradation
                if "clean" in results and "heavy_degraded" in results:
                    clean_ece = results["clean"].calibration.ece
                    heavy_ece = results["heavy_degraded"].calibration.ece
                    all_metrics["ece_degradation_delta"] = heavy_ece - clean_ece

                    clean_brier = results["clean"].calibration.brier_score
                    heavy_brier = results["heavy_degraded"].calibration.brier_score
                    all_metrics["brier_degradation_delta"] = heavy_brier - clean_brier

                mlflow.log_metrics(all_metrics)

                # Log artifacts: reliability diagrams, comparative plot, summary CSV/JSON
                mlflow.log_artifacts(str(out_dir), artifact_path="uncertainty_plots")
                print("\nMLflow run logged successfully under experiment 'wheat-uncertainty'.")
        except Exception as e:
            print(f"Note: MLflow logging skipped or encountered an issue: {e}")

    print(f"\nEvaluation completed. Results saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
