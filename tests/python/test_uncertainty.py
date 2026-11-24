"""Tests for M2: Detection Uncertainty Estimation & Confidence Calibration."""

import numpy as np
import pytest
from PIL import Image
from wheat_mlops.uncertainty import (
    apply_degradation,
    box_iou,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_iou_matrix,
    compute_mce,
    match_predictions,
    plot_reliability_diagram,
)
from wheat_mlops.uncertainty.degradation import apply_domain_shift

# -------------------------------------------------------------------------
# Section B: IoU and One-to-One Matching Tests
# -------------------------------------------------------------------------


def test_box_iou_identical():
    box = (10.0, 20.0, 50.0, 80.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_disjoint():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert box_iou(box1, box2) == pytest.approx(0.0)


def test_box_iou_known_geometry():
    box1 = (0.0, 0.0, 10.0, 10.0)  # area 100
    box2 = (5.0, 0.0, 15.0, 10.0)  # area 100, intersection area = 5 * 10 = 50
    # union = 100 + 100 - 50 = 150
    expected = 50.0 / 150.0
    assert box_iou(box1, box2) == pytest.approx(expected)


def test_compute_iou_matrix():
    preds = [[0, 0, 10, 10], [10, 10, 20, 20]]
    gts = [[0, 0, 10, 10], [5, 5, 15, 15], [30, 30, 40, 40]]
    matrix = compute_iou_matrix(preds, gts)
    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 2] == pytest.approx(0.0)
    assert matrix[1, 2] == pytest.approx(0.0)


def test_match_predictions_one_to_one():
    # Two predictions targeting the same ground truth box [0, 0, 10, 10]
    # Prediction 1 has higher confidence (0.95), prediction 2 has lower confidence (0.70)
    predictions = [
        {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "confidence": 0.95},
        {"x1": 1.0, "y1": 1.0, "x2": 10.0, "y2": 10.0, "confidence": 0.70},
    ]
    ground_truths = [(0.0, 0.0, 10.0, 10.0)]

    matches = match_predictions(predictions, ground_truths, iou_threshold=0.5)
    assert len(matches) == 2

    # Prediction 1 should be TP
    assert matches[0].confidence == 0.95
    assert matches[0].is_true_positive is True
    assert matches[0].matched_gt_idx == 0

    # Prediction 2 must be FP due to one-to-one constraint
    assert matches[1].confidence == 0.70
    assert matches[1].is_true_positive is False
    assert matches[1].matched_gt_idx is None


def test_match_predictions_low_iou_is_false_positive():
    predictions = [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "confidence": 0.90}]
    # Box barely overlapping (IoU < 0.5)
    ground_truths = [(8.0, 0.0, 18.0, 10.0)]  # intersection = 2 * 10 = 20, union = 180, iou = 0.111

    matches = match_predictions(predictions, ground_truths, iou_threshold=0.5)
    assert matches[0].is_true_positive is False


def test_match_predictions_empty_ground_truth():
    predictions = [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "confidence": 0.90}]
    matches = match_predictions(predictions, ground_truths=[], iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0].is_true_positive is False


# -------------------------------------------------------------------------
# Section C: Calibration Metrics Tests
# -------------------------------------------------------------------------


def test_compute_brier_score_perfect():
    confs = [1.0, 1.0, 0.0, 0.0]
    targets = [1, 1, 0, 0]
    assert compute_brier_score(confs, targets) == pytest.approx(0.0)


def test_compute_brier_score_worst():
    confs = [1.0, 1.0, 0.0, 0.0]
    targets = [0, 0, 1, 1]
    assert compute_brier_score(confs, targets) == pytest.approx(1.0)


def test_compute_ece_perfect():
    # In bin [0.8, 0.9], mean conf = 0.8, accuracy = 8/10 = 0.8 -> ECE = 0
    confs = [0.8] * 10
    targets = [1] * 8 + [0] * 2
    ece = compute_ece(confs, targets, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_compute_ece_and_mce_miscalibrated():
    # All predicted with 0.9 confidence, but only 50% are correct
    confs = [0.9] * 10
    targets = [1] * 5 + [0] * 5
    ece = compute_ece(confs, targets, n_bins=10)
    mce = compute_mce(confs, targets, n_bins=10)
    assert ece == pytest.approx(0.4, abs=1e-5)
    assert mce == pytest.approx(0.4, abs=1e-5)


def test_compute_calibration_metrics_structure():
    confs = [0.95, 0.85, 0.75, 0.20]
    targets = [1, 1, 0, 0]
    res = compute_calibration_metrics(confs, targets, n_bins=5)
    assert res.total_samples == 4
    assert res.tp_count == 2
    assert res.fp_count == 2
    assert res.accuracy == 0.5
    assert len(res.bin_edges) == 6

    mlflow_dict = res.to_mlflow_metrics(prefix="val")
    assert "val_ece" in mlflow_dict
    assert "val_brier_score" in mlflow_dict
    assert "val_accuracy" in mlflow_dict


def test_plot_reliability_diagram(tmp_path):
    confs = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    targets = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    save_file = tmp_path / "test_diagram.png"
    fig = plot_reliability_diagram(confs, targets, n_bins=5, save_path=save_file)
    assert fig is not None
    assert save_file.is_file()
    assert save_file.stat().st_size > 0


# -------------------------------------------------------------------------
# Section A: Degradation and Domain Shift Tests
# -------------------------------------------------------------------------


def test_apply_degradation_levels():
    # Create test synthetic image
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    clean = apply_degradation(img, severity="clean")
    assert np.array_equal(np.array(clean), np.array(img))

    mild = apply_degradation(img, severity="mild", seed=1)
    moderate = apply_degradation(img, severity="moderate", seed=1)
    heavy = apply_degradation(img, severity="heavy", seed=1)

    assert mild.size == (100, 100)
    assert moderate.size == (100, 100)
    assert heavy.size == (100, 100)

    # Moderate degradation should perturb image more than mild
    orig_arr = np.array(img, dtype=float)
    mild_diff = np.mean(np.abs(np.array(mild, dtype=float) - orig_arr))
    mod_diff = np.mean(np.abs(np.array(moderate, dtype=float) - orig_arr))
    heavy_diff = np.mean(np.abs(np.array(heavy, dtype=float) - orig_arr))

    assert mild_diff > 0
    assert mod_diff > mild_diff
    assert heavy_diff > mod_diff


def test_apply_domain_shift():
    img = Image.new("RGB", (50, 50), color=(100, 100, 100))
    fog = apply_domain_shift(img, shift_type="fog", severity="moderate")
    color = apply_domain_shift(img, shift_type="color_temperature", severity="moderate")
    shadow = apply_domain_shift(img, shift_type="shadow", severity="moderate")

    assert fog.size == (50, 50)
    assert color.size == (50, 50)
    assert shadow.size == (50, 50)

    # Fog should brighten towards 220
    assert np.mean(np.array(fog)) > np.mean(np.array(img))
    # Color temperature should boost red over blue
    color_arr = np.array(color)
    assert color_arr[..., 0].mean() > color_arr[..., 2].mean()
