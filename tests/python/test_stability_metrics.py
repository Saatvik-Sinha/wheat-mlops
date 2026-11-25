from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from wheat_mlops.stability.metrics import (
    compute_stability,
    image_stability,
    iou_matrix,
    match_boxes,
    run,
    summarize,
)


def _frame(boxes):
    return pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2", "conf"])


def test_iou_matrix_basic_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert iou_matrix(a, b)[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_match_is_one_to_one_and_thresholded():
    clean = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    deg = np.array([[1, 0, 11, 10], [200, 200, 210, 210]])
    pairs = match_boxes(clean, deg, min_iou=0.5)
    assert [(p[0], p[1]) for p in pairs] == [(0, 0)]
    # Two clean boxes competing for one degraded box: only one may take it.
    pairs = match_boxes(np.array([[0, 0, 10, 10], [0, 0, 10, 11]]), np.array([[0, 0, 10, 10]]))
    assert len(pairs) == 1 and pairs[0][0] == 0


def test_identical_detections_are_perfectly_stable():
    boxes = _frame([(0, 0, 10, 10, 0.9), (50, 50, 70, 80, 0.6)])
    m = image_stability(boxes, boxes.copy())
    assert m["matched_ratio"] == 1.0 and m["mean_iou"] == pytest.approx(1.0)
    assert m["stability_iou"] == pytest.approx(1.0)
    assert m["center_disp_px"] == 0 and m["width_change_pct"] == 0 and m["height_change_pct"] == 0
    assert m["log_area_ratio"] == 0 and m["count_diff"] == 0 and m["conf_diff"] == 0
    assert m["new_boxes"] == 0 and m["missed_boxes"] == 0


def test_shift_resize_miss_and_new_box():
    clean = _frame([(0, 0, 10, 10, 0.9), (100, 100, 120, 120, 0.8), (300, 300, 310, 310, 0.7)])
    deg = _frame(
        [
            (2, 0, 12, 10, 0.6),  # shifted 2 px right: IoU 8/12
            (100, 100, 124, 120, 0.7),  # 20 % wider
            (500, 500, 510, 510, 0.5),  # new box; third clean box is missed
        ]
    )
    m = image_stability(clean, deg)
    assert m["n_matched"] == 2 and m["missed_boxes"] == 1 and m["new_boxes"] == 1
    assert m["matched_ratio"] == pytest.approx(2 / 3)
    assert m["new_ratio"] == pytest.approx(1 / 3)
    assert m["count_diff"] == 0 and m["abs_count_diff"] == 0  # same count, different places
    assert m["center_disp_px"] == pytest.approx((2 + 2) / 2)
    assert m["width_change_pct"] == pytest.approx((0 + 20) / 2)
    assert m["height_change_pct"] == pytest.approx(0)
    assert m["log_area_ratio"] == pytest.approx(np.mean([0, math.log(1.2)]))
    assert m["matched_conf_abs_diff"] == pytest.approx((0.3 + 0.1) / 2)
    assert m["conf_diff"] == pytest.approx(0.6 - 0.8)
    expected_iou = [8 / 12, 400 / 480]
    assert m["mean_iou"] == pytest.approx(np.mean(expected_iou))
    assert m["stability_iou"] == pytest.approx(sum(expected_iou) / 3)


def test_empty_cases_do_not_crash():
    some = _frame([(0, 0, 10, 10, 0.9)])
    none = _frame([])
    lost = image_stability(some, none)
    assert lost["matched_ratio"] == 0 and lost["missed_boxes"] == 1
    assert math.isnan(lost["mean_iou"]) and lost["stability_iou"] == 0
    assert math.isnan(lost["new_ratio"])
    appeared = image_stability(none, some)
    assert math.isnan(appeared["matched_ratio"]) and appeared["new_boxes"] == 1
    both = image_stability(none, none)
    assert both["count_diff"] == 0 and math.isnan(both["matched_ratio"])


def _detections():
    def rows(image_id, condition, corruption, severity, boxes):
        out = []
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
            out.append(
                dict(
                    image_id=image_id, condition=condition, corruption=corruption,
                    severity=severity, box_idx=i, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                    img_w=100, img_h=100,
                )
            )  # fmt: skip
        return out

    base = [(0, 0, 10, 10, 0.9), (50, 50, 60, 60, 0.8)]
    boxes = (
        rows("a", "clean", "clean", 0, base)
        + rows("a", "gaussian_blur_s1", "gaussian_blur", 1, base)
        + rows("a", "gaussian_blur_s2", "gaussian_blur", 2, base[:1])
        + rows("b", "clean", "clean", 0, base)
        + rows("b", "gaussian_blur_s1", "gaussian_blur", 1, base)
        # image b has no detections at all under blur_s2
    )
    images = pd.DataFrame(
        [
            dict(image_id=i, condition=c, corruption=k, severity=s, n_boxes=0, img_w=100, img_h=100)
            for i in ("a", "b")
            for c, k, s in (
                ("clean", "clean", 0),
                ("gaussian_blur_s1", "gaussian_blur", 1),
                ("gaussian_blur_s2", "gaussian_blur", 2),
            )
        ]
    )
    return pd.DataFrame(boxes), images


def test_compute_and_summarize_over_conditions():
    boxes, images = _detections()
    per_image = compute_stability(boxes, images)
    assert set(per_image["condition"]) == {"gaussian_blur_s1", "gaussian_blur_s2"}
    assert len(per_image) == 4  # images with zero detections still get a row
    summary = summarize(per_image).set_index("condition")
    assert summary.loc["gaussian_blur_s1", "matched_ratio"] == 1.0
    assert summary.loc["gaussian_blur_s2", "matched_ratio"] == pytest.approx((0.5 + 0.0) / 2)
    assert summary.loc["gaussian_blur_s2", "pooled_matched_ratio"] == pytest.approx(1 / 4)
    assert summary.loc["gaussian_blur_s2", "abs_count_diff"] == pytest.approx((1 + 2) / 2)


def test_run_writes_outputs(tmp_path):
    boxes, images = _detections()
    det = tmp_path / "detections"
    det.mkdir()
    for condition, g in boxes.groupby("condition"):
        g.to_csv(det / f"{condition}_boxes.csv", index=False)
    for condition, g in images.groupby("condition"):
        g.to_csv(det / f"{condition}_images.csv", index=False)
    per_image, summary = run(tmp_path, 0.5)
    assert (tmp_path / "metrics" / "per_image_stability.csv").is_file()
    assert (tmp_path / "metrics" / "condition_summary.csv").is_file()
    assert len(per_image) == 4 and len(summary) == 2
