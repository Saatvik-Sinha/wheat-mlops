"""Confidence calibration metrics and reliability diagrams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for headless / CI environments
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class CalibrationResult:
    """Structured results from evaluating confidence calibration."""

    total_samples: int
    tp_count: int
    fp_count: int
    accuracy: float  # Overall precision / TP rate
    mean_confidence: float
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    bin_edges: list[float]
    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]
    bin_proportions: list[float]

    def to_mlflow_metrics(self, prefix: str = "") -> dict[str, float]:
        """Convert scalar summary metrics to a dictionary for MLflow logging."""
        p = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix
        return {
            f"{p}total_samples": float(self.total_samples),
            f"{p}tp_count": float(self.tp_count),
            f"{p}fp_count": float(self.fp_count),
            f"{p}accuracy": float(self.accuracy),
            f"{p}mean_confidence": float(self.mean_confidence),
            f"{p}brier_score": float(self.brier_score),
            f"{p}ece": float(self.ece),
            f"{p}mce": float(self.mce),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def compute_brier_score(
    confidences: Sequence[float] | np.ndarray,
    targets: Sequence[int | float | bool] | np.ndarray,
) -> float:
    """Calculate the Brier score: mean squared difference between confidence and binary outcome.

    BS = (1 / N) * sum((confidence_i - target_i)^2)
    Lower is better (0 is perfect calibration and sharp predictions).
    """
    confs = np.asarray(confidences, dtype=float)
    targs = np.asarray(targets, dtype=float)

    if len(confs) == 0:
        return 0.0
    if len(confs) != len(targs):
        raise ValueError(
            f"confidences ({len(confs)}) and targets ({len(targs)}) must have the same length"
        )

    return float(np.mean((confs - targs) ** 2))


def compute_calibration_curve(
    confidences: Sequence[float] | np.ndarray,
    targets: Sequence[int | float | bool] | np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve bins: confidences, accuracies, counts, proportions.

    Args:
        confidences: Predicted confidence scores in [0, 1].
        targets: Binary correctness targets in {0, 1}.
        n_bins: Number of confidence bins between 0 and 1.

    Returns:
        (bin_edges, bin_confidences, bin_accuracies, bin_counts, bin_proportions)
    """
    confs = np.asarray(confidences, dtype=float)
    targs = np.asarray(targets, dtype=float)

    if len(confs) != len(targs):
        raise ValueError("confidences and targets must have matching length")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins, dtype=float)
    bin_accuracies = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    n_samples = len(confs)
    if n_samples == 0:
        bin_proportions = np.zeros(n_bins, dtype=float)
        return bin_edges, bin_confidences, bin_accuracies, bin_counts, bin_proportions

    # Assign samples to bins: (0, b1], (b1, b2], ..., (b_{n-1}, 1]
    # For index 0, include 0.0 as well
    bin_indices = np.digitize(confs, bin_edges) - 1
    # Clip to handle potential boundary values (1.0 goes into last bin)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = float(np.mean(confs[mask]))
            bin_accuracies[i] = float(np.mean(targs[mask]))
        else:
            # Empty bin center
            bin_confidences[i] = float((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            bin_accuracies[i] = 0.0

    bin_proportions = bin_counts / n_samples
    return bin_edges, bin_confidences, bin_accuracies, bin_counts, bin_proportions


def compute_ece(
    confidences: Sequence[float] | np.ndarray,
    targets: Sequence[int | float | bool] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    _, bin_confs, bin_accs, bin_counts, bin_props = compute_calibration_curve(
        confidences, targets, n_bins=n_bins
    )
    if len(confidences) == 0:
        return 0.0

    non_empty = bin_counts > 0
    ece = np.sum(bin_props[non_empty] * np.abs(bin_accs[non_empty] - bin_confs[non_empty]))
    return float(ece)


def compute_mce(
    confidences: Sequence[float] | np.ndarray,
    targets: Sequence[int | float | bool] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Maximum Calibration Error (MCE).

    MCE = max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|
    """
    _, bin_confs, bin_accs, bin_counts, _ = compute_calibration_curve(
        confidences, targets, n_bins=n_bins
    )
    if len(confidences) == 0:
        return 0.0

    non_empty = bin_counts > 0
    if not np.any(non_empty):
        return 0.0

    mce = np.max(np.abs(bin_accs[non_empty] - bin_confs[non_empty]))
    return float(mce)


def compute_calibration_metrics(
    confidences: Sequence[float] | np.ndarray,
    targets: Sequence[int | float | bool] | np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute comprehensive calibration metrics and binning statistics."""
    confs = np.asarray(confidences, dtype=float)
    targs = np.asarray(targets, dtype=float)

    n_samples = len(confs)
    tp_count = int(np.sum(targs == 1)) if n_samples > 0 else 0
    fp_count = int(np.sum(targs == 0)) if n_samples > 0 else 0
    accuracy = float(tp_count / n_samples) if n_samples > 0 else 0.0
    mean_conf = float(np.mean(confs)) if n_samples > 0 else 0.0

    brier = compute_brier_score(confs, targs)
    bin_edges, bin_confs, bin_accs, bin_counts, bin_props = compute_calibration_curve(
        confs, targs, n_bins=n_bins
    )

    non_empty = bin_counts > 0
    if np.any(non_empty):
        ece = float(
            np.sum(bin_props[non_empty] * np.abs(bin_accs[non_empty] - bin_confs[non_empty]))
        )
        mce = float(np.max(np.abs(bin_accs[non_empty] - bin_confs[non_empty])))
    else:
        ece = 0.0
        mce = 0.0

    return CalibrationResult(
        total_samples=n_samples,
        tp_count=tp_count,
        fp_count=fp_count,
        accuracy=accuracy,
        mean_confidence=mean_conf,
        brier_score=brier,
        ece=ece,
        mce=mce,
        bin_edges=bin_edges.tolist(),
        bin_confidences=bin_confs.tolist(),
        bin_accuracies=bin_accs.tolist(),
        bin_counts=bin_counts.tolist(),
        bin_proportions=bin_props.tolist(),
    )


def plot_reliability_diagram(
    confidences: Sequence[float] | np.ndarray,
    targets: Sequence[int | float | bool] | np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a publication-quality reliability diagram with confidence histogram.

    Top panel: Accuracy vs Confidence per bin, diagonal ideal calibration line,
               and shaded calibration gaps.
    Bottom panel: Sample count distribution across confidence bins.
    """
    res = compute_calibration_metrics(confidences, targets, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    bin_edges = np.array(res.bin_edges)
    bin_accs = np.array(res.bin_accuracies)
    bin_confs = np.array(res.bin_confidences)
    bin_counts = np.array(res.bin_counts)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = 1.0 / n_bins

    # Top plot: Reliability Diagram
    # Plot ideal calibration line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)

    # Plot accuracy bars
    ax1.bar(
        bin_centers,
        bin_accs,
        width=bin_width * 0.9,
        alpha=0.6,
        color="#2b5c8f",
        edgecolor="#163252",
        label="Accuracy",
    )

    # Plot error gaps
    non_empty = bin_counts > 0
    for i in range(n_bins):
        if non_empty[i]:
            acc = bin_accs[i]
            conf = bin_confs[i]
            x = bin_centers[i]
            # Red outline / error gap indicating miscalibration
            ax1.plot([x, x], [acc, conf], color="#d9534f", linewidth=2.0)
            ax1.scatter(
                [x],
                [conf],
                color="#d9534f",
                marker="o",
                s=25,
                zorder=4,
                label="Mean Confidence" if i == 0 else "",
            )

    ax1.set_ylabel("Accuracy (Precision)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Metrics textbox
    text_str = (
        f"Samples: {res.total_samples}\n"
        f"ECE: {res.ece:.4f}\n"
        f"MCE: {res.mce:.4f}\n"
        f"Brier: {res.brier_score:.4f}\n"
        f"Avg Conf: {res.mean_confidence:.3f}\n"
        f"Overall Acc: {res.accuracy:.3f}"
    )
    props = {"boxstyle": "round,pad=0.5", "facecolor": "#f8f9fa", "edgecolor": "#cccccc"}
    ax1.text(
        0.05,
        0.95,
        text_str,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )
    ax1.legend(loc="lower right", fontsize=9)

    # Bottom plot: Frequency Histogram
    ax2.bar(
        bin_centers,
        bin_counts,
        width=bin_width * 0.9,
        color="#6c757d",
        edgecolor="#343a40",
        alpha=0.7,
    )
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
