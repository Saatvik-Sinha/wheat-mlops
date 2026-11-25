"""Part B: bounding-box stability metrics between clean and degraded detections.

For every image and every degraded condition, the boxes predicted on the clean image are paired
one-to-one with the boxes predicted on the degraded image (Hungarian matching that maximises
total IoU; a pair only counts if IoU >= ``match_iou``). Metrics are then computed per image,
and summarised per condition.

Per-image metrics (one row per image x condition):
    n_clean, n_degraded        number of detections
    count_diff                 n_degraded - n_clean (signed); abs_count_diff is its magnitude
    n_matched                  clean boxes that found a partner
    matched_ratio              n_matched / n_clean              (1.0 = nothing lost)
    new_boxes, new_ratio       degraded boxes with no clean partner, and share of n_degraded
    missed_boxes               clean boxes with no partner (n_clean - n_matched)
    mean_iou                   mean IoU over matched pairs
    stability_iou              sum of matched IoU / n_clean (unmatched clean boxes count as 0)
    center_disp_px             mean distance between matched box centres, in pixels
    center_disp_rel            same, divided by the clean box diagonal (size independent)
    width_change_pct           mean |w_deg - w_clean| / w_clean * 100 over matched pairs
    height_change_pct          same for height
    log_area_ratio             mean ln(area_deg / area_clean); >0 boxes grew, <0 boxes shrank
    mean_conf_clean/degraded   mean confidence of all boxes, and conf_diff = degraded - clean
    matched_conf_abs_diff      mean |conf_deg - conf_clean| over matched pairs

Usage:
    python -m wheat_mlops.stability.metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from wheat_mlops.stability.collect import load_detections

BOX_COLS = ["x1", "y1", "x2", "y2"]

PER_IMAGE_COLUMNS = [
    "image_id",
    "condition",
    "corruption",
    "severity",
    "n_clean",
    "n_degraded",
    "count_diff",
    "abs_count_diff",
    "n_matched",
    "matched_ratio",
    "new_boxes",
    "new_ratio",
    "missed_boxes",
    "mean_iou",
    "stability_iou",
    "center_disp_px",
    "center_disp_rel",
    "width_change_pct",
    "height_change_pct",
    "log_area_ratio",
    "mean_conf_clean",
    "mean_conf_degraded",
    "conf_diff",
    "matched_conf_abs_diff",
]

# Per-image columns that are averaged (over images where they are defined) for the summary.
MEAN_METRICS = [
    "count_diff",
    "abs_count_diff",
    "matched_ratio",
    "new_ratio",
    "mean_iou",
    "stability_iou",
    "center_disp_px",
    "center_disp_rel",
    "width_change_pct",
    "height_change_pct",
    "log_area_ratio",
    "conf_diff",
    "matched_conf_abs_diff",
]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N, 4) and (M, 4) arrays of xyxy boxes -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(
    clean: np.ndarray, degraded: np.ndarray, min_iou: float = 0.5
) -> list[tuple[int, int, float]]:
    """One-to-one matching of clean to degraded boxes, maximising total IoU.

    Returns (clean_index, degraded_index, iou) for each pair whose IoU >= ``min_iou``.
    """
    ious = iou_matrix(clean, degraded)
    if ious.size == 0:
        return []
    rows, cols = linear_sum_assignment(-ious)
    return [
        (int(r), int(c), float(ious[r, c]))
        for r, c in zip(rows, cols, strict=True)
        if ious[r, c] >= min_iou
    ]


def image_stability(
    clean: pd.DataFrame, degraded: pd.DataFrame, min_iou: float = 0.5
) -> dict[str, float]:
    """Stability metrics for one image: its clean boxes vs its boxes under one degradation.

    Both frames need x1,y1,x2,y2,conf columns (may be empty). Undefined values are NaN.
    """
    nan = float("nan")
    n_clean, n_deg = len(clean), len(degraded)
    pairs = match_boxes(clean[BOX_COLS].to_numpy(), degraded[BOX_COLS].to_numpy(), min_iou)
    n_matched = len(pairs)
    out: dict[str, float] = {
        "n_clean": n_clean,
        "n_degraded": n_deg,
        "count_diff": n_deg - n_clean,
        "abs_count_diff": abs(n_deg - n_clean),
        "n_matched": n_matched,
        "matched_ratio": n_matched / n_clean if n_clean else nan,
        "new_boxes": n_deg - n_matched,
        "new_ratio": (n_deg - n_matched) / n_deg if n_deg else nan,
        "missed_boxes": n_clean - n_matched,
        "mean_conf_clean": float(clean["conf"].mean()) if n_clean else nan,
        "mean_conf_degraded": float(degraded["conf"].mean()) if n_deg else nan,
    }
    out["conf_diff"] = out["mean_conf_degraded"] - out["mean_conf_clean"]
    out["stability_iou"] = sum(p[2] for p in pairs) / n_clean if n_clean else nan

    if not pairs:
        for key in (
            "mean_iou",
            "center_disp_px",
            "center_disp_rel",
            "width_change_pct",
            "height_change_pct",
            "log_area_ratio",
            "matched_conf_abs_diff",
        ):
            out[key] = nan
        return out

    ci = [p[0] for p in pairs]
    di = [p[1] for p in pairs]
    c = clean[BOX_COLS].to_numpy(dtype=float)[ci]
    d = degraded[BOX_COLS].to_numpy(dtype=float)[di]
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    dw, dh = d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]
    centre_c = np.stack([(c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2], axis=1)
    centre_d = np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2], axis=1)
    disp = np.linalg.norm(centre_d - centre_c, axis=1)
    diag = np.hypot(cw, ch)
    safe = lambda x: np.where(x > 0, x, np.nan)  # noqa: E731 - degenerate boxes give NaN
    out["mean_iou"] = float(np.mean([p[2] for p in pairs]))
    out["center_disp_px"] = float(disp.mean())
    out["center_disp_rel"] = float(np.nanmean(disp / safe(diag)))
    out["width_change_pct"] = float(np.nanmean(np.abs(dw - cw) / safe(cw)) * 100)
    out["height_change_pct"] = float(np.nanmean(np.abs(dh - ch) / safe(ch)) * 100)
    out["log_area_ratio"] = float(np.nanmean(np.log(safe(dw * dh) / safe(cw * ch))))
    cconf = clean["conf"].to_numpy(dtype=float)[ci]
    dconf = degraded["conf"].to_numpy(dtype=float)[di]
    out["matched_conf_abs_diff"] = float(np.mean(np.abs(dconf - cconf)))
    return out


def compute_stability(
    boxes: pd.DataFrame, images: pd.DataFrame, min_iou: float = 0.5
) -> pd.DataFrame:
    """Per-image metrics for every non-clean condition (see module docstring)."""
    clean_boxes = {k: g for k, g in boxes[boxes["condition"] == "clean"].groupby("image_id")}
    deg_boxes = {
        k: g for k, g in boxes[boxes["condition"] != "clean"].groupby(["image_id", "condition"])
    }
    empty = pd.DataFrame(columns=BOX_COLS + ["conf"])
    clean_ids = set(images.loc[images["condition"] == "clean", "image_id"])
    if not clean_ids:
        raise ValueError("no clean detections; run stability.collect first")

    rows: list[dict] = []
    degraded_images = images[images["condition"] != "clean"]
    for meta in degraded_images.sort_values(["condition", "image_id"]).itertuples(index=False):
        if meta.image_id not in clean_ids:
            continue  # no clean reference for this image
        metrics = image_stability(
            clean_boxes.get(meta.image_id, empty),
            deg_boxes.get((meta.image_id, meta.condition), empty),
            min_iou,
        )
        rows.append(
            {
                "image_id": meta.image_id,
                "condition": meta.condition,
                "corruption": meta.corruption,
                "severity": int(meta.severity),
                **metrics,
            }
        )
    return pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS)


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    """One row per condition: mean metrics plus pooled (micro-averaged) detection ratios.

    Mean metrics skip images where a value is undefined (e.g. no matches). ``pooled_*`` columns
    sum counts over all images first, so images with many wheat heads weigh more.
    """
    records: list[dict] = []
    for (corruption, severity, condition), g in per_image.groupby(
        ["corruption", "severity", "condition"], sort=True
    ):
        total_clean = g["n_clean"].sum()
        total_deg = g["n_degraded"].sum()
        total_matched = g["n_matched"].sum()
        record = {
            "condition": condition,
            "corruption": corruption,
            "severity": int(severity),
            "n_images": len(g),
            "total_clean_boxes": int(total_clean),
            "total_degraded_boxes": int(total_deg),
            "pooled_matched_ratio": total_matched / total_clean if total_clean else np.nan,
            "pooled_new_ratio": (total_deg - total_matched) / total_deg if total_deg else np.nan,
        }
        for metric in MEAN_METRICS:
            record[metric] = float(g[metric].mean())
            record[f"{metric}_std"] = float(g[metric].std())
        records.append(record)
    return pd.DataFrame(records)


def run(out_dir: str | Path, min_iou: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved detections, compute metrics, write CSVs under ``<out_dir>/metrics``."""
    out_dir = Path(out_dir)
    boxes, images = load_detections(out_dir)
    per_image = compute_stability(boxes, images, min_iou)
    summary = summarize(per_image)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(metrics_dir / "per_image_stability.csv", index=False)
    summary.to_csv(metrics_dir / "condition_summary.csv", index=False)
    (metrics_dir / "metrics_info.json").write_text(
        json.dumps(
            {
                "match_iou": min_iou,
                "n_images": int(per_image["image_id"].nunique()),
                "n_conditions": int(summary["condition"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return per_image, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="stability.yaml")
    parser.add_argument("--out-dir", help="directory holding detections/ (default: from config)")
    parser.add_argument("--match-iou", type=float, help="override metrics.match_iou")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = args.out_dir or config["collect"]["out_dir"]
    min_iou = args.match_iou if args.match_iou is not None else config["metrics"]["match_iou"]
    _, summary = run(out_dir, float(min_iou))
    cols = ["condition", "matched_ratio", "mean_iou", "center_disp_px", "abs_count_diff"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nWrote {Path(out_dir) / 'metrics'}")


if __name__ == "__main__":
    main()
