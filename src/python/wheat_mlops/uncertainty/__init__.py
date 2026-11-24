"""Detection Uncertainty Estimation package for wheat head detection."""

from wheat_mlops.uncertainty.calibration import (
    CalibrationResult,
    compute_brier_score,
    compute_calibration_curve,
    compute_calibration_metrics,
    compute_ece,
    compute_mce,
    plot_reliability_diagram,
)
from wheat_mlops.uncertainty.degradation import (
    DegradationSeverity,
    apply_degradation,
    create_degraded_dataset,
)
from wheat_mlops.uncertainty.evaluator import UncertaintyEvaluator
from wheat_mlops.uncertainty.matching import (
    MatchedPrediction,
    box_iou,
    compute_iou_matrix,
    match_predictions,
)

__all__ = [
    "CalibrationResult",
    "DegradationSeverity",
    "MatchedPrediction",
    "UncertaintyEvaluator",
    "apply_degradation",
    "box_iou",
    "compute_brier_score",
    "compute_calibration_curve",
    "compute_calibration_metrics",
    "compute_ece",
    "compute_iou_matrix",
    "compute_mce",
    "create_degraded_dataset",
    "match_predictions",
    "plot_reliability_diagram",
]
