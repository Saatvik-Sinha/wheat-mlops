"""Bounding box IoU calculation and one-to-one matching procedure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MatchedPrediction:
    """A prediction matched against ground-truth boxes."""

    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    is_true_positive: bool
    matched_gt_idx: int | None
    matched_iou: float
    image_id: str = ""


def box_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Calculate Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def compute_iou_matrix(
    pred_boxes: np.ndarray | Sequence[Sequence[float]],
    gt_boxes: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute pairwise IoU matrix between N predicted boxes and M ground-truth boxes.

    Args:
        pred_boxes: Array or list of shape (N, 4) in [x1, y1, x2, y2].
        gt_boxes: Array or list of shape (M, 4) in [x1, y1, x2, y2].

    Returns:
        np.ndarray of shape (N, M) with values in [0, 1].
    """
    preds = np.asarray(pred_boxes, dtype=float)
    gts = np.asarray(gt_boxes, dtype=float)

    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=float)

    # preds: (N, 1, 4), gts: (1, M, 4)
    preds = preds[:, np.newaxis, :]
    gts = gts[np.newaxis, :, :]

    inter_x1 = np.maximum(preds[..., 0], gts[..., 0])
    inter_y1 = np.maximum(preds[..., 1], gts[..., 1])
    inter_x2 = np.minimum(preds[..., 2], gts[..., 2])
    inter_y2 = np.minimum(preds[..., 3], gts[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_preds = np.maximum(0.0, preds[..., 2] - preds[..., 0]) * np.maximum(
        0.0, preds[..., 3] - preds[..., 1]
    )
    area_gts = np.maximum(0.0, gts[..., 2] - gts[..., 0]) * np.maximum(
        0.0, gts[..., 3] - gts[..., 1]
    )

    union = area_preds + area_gts - inter_area
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter_area / union, 0.0)

    return iou


def match_predictions(
    predictions: Sequence[tuple[float, float, float, float, float] | dict],
    ground_truths: Sequence[tuple[float, float, float, float] | Sequence[float]],
    iou_threshold: float = 0.5,
    image_id: str = "",
) -> list[MatchedPrediction]:
    """Match predicted boxes to ground-truth boxes using one-to-one matching procedure.

    Procedure (as defined in standard object detection evaluation):
    1. Sort predictions in descending order of confidence score.
    2. Compute pairwise IoU with all ground-truth boxes.
    3. For each prediction, find the candidate ground-truth box with maximum IoU.
    4. If maximum IoU >= iou_threshold and that ground-truth box has not been matched yet,
       mark the prediction as a True Positive (TP) and reserve that ground-truth box.
    5. Otherwise, mark the prediction as a False Positive (FP).

    Args:
        predictions: List of tuples (x1, y1, x2, y2, confidence) or dicts with corresponding keys.
        ground_truths: List of (x1, y1, x2, y2) tuples/lists.
        iou_threshold: Minimum IoU required to consider a match valid.
        image_id: Optional identifier for logging / debugging.

    Returns:
        List of MatchedPrediction objects corresponding to all predictions.
    """
    if not predictions:
        return []

    # Standardize predictions format into list of (box, conf, original_idx)
    parsed_preds: list[tuple[tuple[float, float, float, float], float, int]] = []
    for idx, p in enumerate(predictions):
        if isinstance(p, dict):
            box = (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"]))
            conf = float(p.get("confidence", p.get("score", 0.0)))
        else:
            box = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
            conf = float(p[4])
        parsed_preds.append((box, conf, idx))

    # Standardize ground truths
    parsed_gts: list[tuple[float, float, float, float]] = []
    for gt in ground_truths:
        if isinstance(gt, dict):
            parsed_gts.append((float(gt["x1"]), float(gt["y1"]), float(gt["x2"]), float(gt["y2"])))
        else:
            parsed_gts.append((float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3])))

    # Sort predictions by confidence in descending order
    # (keeps track of order to return matches or process greedily)
    sorted_preds = sorted(parsed_preds, key=lambda item: item[1], reverse=True)

    matched_gt_indices: set[int] = set()
    results_dict: dict[int, MatchedPrediction] = {}

    if not parsed_gts:
        # No ground truth boxes in this image -> all predictions are false positives
        for box, conf, orig_idx in sorted_preds:
            results_dict[orig_idx] = MatchedPrediction(
                box=box,
                confidence=conf,
                is_true_positive=False,
                matched_gt_idx=None,
                matched_iou=0.0,
                image_id=image_id,
            )
        return [results_dict[i] for i in range(len(predictions))]

    pred_boxes = np.array([sp[0] for sp in sorted_preds], dtype=float)
    gt_boxes = np.array(parsed_gts, dtype=float)
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    for rank, (box, conf, orig_idx) in enumerate(sorted_preds):
        ious = iou_matrix[rank]
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices:
            # Valid one-to-one match: True Positive
            matched_gt_indices.add(best_gt_idx)
            results_dict[orig_idx] = MatchedPrediction(
                box=box,
                confidence=conf,
                is_true_positive=True,
                matched_gt_idx=best_gt_idx,
                matched_iou=best_iou,
                image_id=image_id,
            )
        else:
            # Overlapping duplicate or insufficient IoU: False Positive
            results_dict[orig_idx] = MatchedPrediction(
                box=box,
                confidence=conf,
                is_true_positive=False,
                matched_gt_idx=None,
                matched_iou=best_iou if best_iou >= iou_threshold else 0.0,
                image_id=image_id,
            )

    # Return in original prediction order
    return [results_dict[i] for i in range(len(predictions))]
