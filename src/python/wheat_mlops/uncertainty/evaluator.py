"""Evaluation engine for detection uncertainty estimation across degradations and domain shifts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wheat_mlops.uncertainty.calibration import (
    CalibrationResult,
    compute_calibration_metrics,
    plot_reliability_diagram,
)
from wheat_mlops.uncertainty.degradation import (
    apply_degradation,
    apply_domain_shift,
)
from wheat_mlops.uncertainty.matching import MatchedPrediction, match_predictions


@dataclass
class ConditionEvaluationResult:
    """Evaluation result for a specific degradation or domain condition."""

    condition_name: str
    calibration: CalibrationResult
    matched_predictions: list[MatchedPrediction]
    reliability_plot_path: str | None = None

    def to_summary_dict(self) -> dict:
        return {
            "condition": self.condition_name,
            "total_predictions": self.calibration.total_samples,
            "tp_count": self.calibration.tp_count,
            "fp_count": self.calibration.fp_count,
            "accuracy_precision": self.calibration.accuracy,
            "mean_confidence": self.calibration.mean_confidence,
            "brier_score": self.calibration.brier_score,
            "ece": self.calibration.ece,
            "mce": self.calibration.mce,
        }


def load_yolo_labels(
    label_path: str | Path,
    img_width: int = 1024,
    img_height: int = 1024,
) -> list[tuple[float, float, float, float]]:
    """Parse YOLO txt format (class cx cy bw bh in [0, 1]) to [x1, y1, x2, y2] coordinates."""
    path = Path(label_path)
    if not path.is_file():
        return []

    boxes: list[tuple[float, float, float, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Format: class_id, cx, cy, bw, bh
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * img_width
                y1 = (cy - bh / 2.0) * img_height
                x2 = (cx + bw / 2.0) * img_width
                y2 = (cy + bh / 2.0) * img_height
                boxes.append((x1, y1, x2, y2))
    return boxes


def plot_degradation_comparison(
    results: dict[str, ConditionEvaluationResult],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot summary comparison of calibration metrics across degradations.

    Compares ECE, Brier score, and accuracy across severity levels.
    """
    conditions = list(results.keys())
    ece_vals = [results[c].calibration.ece for c in conditions]
    brier_vals = [results[c].calibration.brier_score for c in conditions]
    acc_vals = [results[c].calibration.accuracy for c in conditions]
    mean_confs = [results[c].calibration.mean_confidence for c in conditions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. ECE across conditions
    axes[0].bar(conditions, ece_vals, color="#e74c3c", alpha=0.75, edgecolor="#c0392b")
    axes[0].set_title("Expected Calibration Error (ECE)", fontweight="bold")
    axes[0].set_ylabel("ECE (lower is better)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Brier Score across conditions
    axes[1].bar(conditions, brier_vals, color="#f39c12", alpha=0.75, edgecolor="#d68910")
    axes[1].set_title("Brier Score", fontweight="bold")
    axes[1].set_ylabel("Brier Score (lower is better)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    # 3. Accuracy vs Mean Confidence
    x = np.arange(len(conditions))
    width = 0.35
    axes[2].bar(
        x - width / 2,
        acc_vals,
        width,
        label="Accuracy",
        color="#2980b9",
        alpha=0.75,
        edgecolor="#1f618d",
    )
    axes[2].bar(
        x + width / 2,
        mean_confs,
        width,
        label="Mean Confidence",
        color="#8e44ad",
        alpha=0.75,
        edgecolor="#6c3483",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(conditions, rotation=30)
    axes[2].set_title("Accuracy vs Mean Confidence", fontweight="bold")
    axes[2].set_ylabel("Score")
    axes[2].legend()
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


class UncertaintyEvaluator:
    """Main evaluator for detection confidence calibration and uncertainty estimation."""

    def __init__(
        self,
        model,
        imgsz: int = 640,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.10,
        nms_iou: float = 0.7,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.device = device

    def predict_image(
        self,
        image: Image.Image,
    ) -> list[tuple[float, float, float, float, float]]:
        """Run YOLO inference on a single image and return list of (x1, y1, x2, y2, score)."""
        result = self.model.predict(
            image,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.nms_iou,
        )[0]

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            (float(x1), float(y1), float(x2), float(y2), float(s))
            for (x1, y1, x2, y2), s in zip(xyxy, scores, strict=True)
        ]

    def evaluate_condition(
        self,
        image_records: Sequence[dict],
        condition_name: str,
        degradation_severity: str | None = None,
        domain_shift: str | None = None,
        n_bins: int = 10,
        output_dir: str | Path | None = None,
    ) -> ConditionEvaluationResult:
        """Evaluate calibration on a set of image records under a specific condition.

        Each record in `image_records` should be a dict with:
        - 'image_path': Path or str to image
        - 'ground_truths': list of (x1, y1, x2, y2)
        - 'image_id': str optional
        """
        all_matches: list[MatchedPrediction] = []

        for idx, record in enumerate(image_records):
            img_path = Path(record["image_path"])
            gts = record["ground_truths"]
            image_id = record.get("image_id", img_path.stem)

            img = Image.open(img_path).convert("RGB")

            # Apply degradation if specified
            if degradation_severity and degradation_severity != "clean":
                img = apply_degradation(img, severity=degradation_severity, seed=42 + idx)

            # Apply domain shift if specified
            if domain_shift:
                sev = degradation_severity or "moderate"
                img = apply_domain_shift(img, shift_type=domain_shift, severity=sev)

            # Run YOLO prediction
            predictions = self.predict_image(img)

            # Match predictions against ground truth using one-to-one matching
            matches = match_predictions(
                predictions=predictions,
                ground_truths=gts,
                iou_threshold=self.iou_threshold,
                image_id=image_id,
            )
            all_matches.extend(matches)

        # Extract confidences and binary correctness targets
        confidences = [m.confidence for m in all_matches]
        targets = [1 if m.is_true_positive else 0 for m in all_matches]

        # Compute calibration metrics
        calib_res = compute_calibration_metrics(confidences, targets, n_bins=n_bins)

        # Generate reliability diagram
        plot_path = None
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            plot_path = str(out_p / f"reliability_{condition_name}.png")
            plot_reliability_diagram(
                confidences=confidences,
                targets=targets,
                n_bins=n_bins,
                title=f"Reliability Diagram: {condition_name}",
                save_path=plot_path,
            )
            plt.close("all")

        return ConditionEvaluationResult(
            condition_name=condition_name,
            calibration=calib_res,
            matched_predictions=all_matches,
            reliability_plot_path=plot_path,
        )

    def evaluate_suite(
        self,
        image_records: Sequence[dict],
        output_dir: str | Path,
        conditions: list[str] | None = None,
        n_bins: int = 10,
    ) -> dict[str, ConditionEvaluationResult]:
        """Run uncertainty evaluation across clean, degraded, and domain-shifted images."""
        if conditions is None:
            conditions = [
                "clean",
                "mild_degraded",
                "moderate_degraded",
                "heavy_degraded",
                "domain_shifted_fog",
                "domain_shifted_illumination",
            ]

        results: dict[str, ConditionEvaluationResult] = {}
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        for cond in conditions:
            print(f"--> Evaluating condition: {cond}...")
            deg_sev = None
            dom_shift = None

            if cond == "clean":
                deg_sev = "clean"
            elif cond == "mild_degraded":
                deg_sev = "mild"
            elif cond == "moderate_degraded":
                deg_sev = "moderate"
            elif cond == "heavy_degraded":
                deg_sev = "heavy"
            elif cond == "domain_shifted_fog":
                deg_sev = "moderate"
                dom_shift = "fog"
            elif cond == "domain_shifted_illumination":
                deg_sev = "moderate"
                dom_shift = "color_temperature"
            else:
                deg_sev = "moderate"

            res = self.evaluate_condition(
                image_records=image_records,
                condition_name=cond,
                degradation_severity=deg_sev,
                domain_shift=dom_shift,
                n_bins=n_bins,
                output_dir=out_p,
            )
            results[cond] = res

        # Generate comparative summary plot
        comparison_plot = out_p / "degradation_comparison.png"
        plot_degradation_comparison(results, save_path=comparison_plot)
        plt.close("all")

        # Save summary CSV and JSON
        summary_rows = [res.to_summary_dict() for res in results.values()]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_p / "uncertainty_summary.csv", index=False)
        with open(out_p / "uncertainty_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        return results
