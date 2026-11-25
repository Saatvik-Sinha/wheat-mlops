"""Part A: run YOLO on every clean/degraded image and save all predicted boxes.

Reads the dataset produced by ``wheat_mlops.perturbations generate`` (M1) and writes, for every
condition (``clean``, ``gaussian_blur_s1`` ... ``jpeg_compression_s5``):

    <out_dir>/detections/<condition>_boxes.csv    one row per predicted box
    <out_dir>/detections/<condition>_images.csv   one row per image (also images with 0 boxes)
    <out_dir>/detections/collect_summary.json     run settings and per-condition totals

Usage:
    python -m wheat_mlops.stability.collect
    python -m wheat_mlops.stability.collect --max-images 10      # quick smoke run
    python -m wheat_mlops.stability.collect --overwrite          # redo finished conditions
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import yaml

BOX_COLUMNS = [
    "image_id",
    "condition",
    "corruption",
    "severity",
    "box_idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "img_w",
    "img_h",
]
IMAGE_COLUMNS = [
    "image_id",
    "condition",
    "corruption",
    "severity",
    "n_boxes",
    "img_w",
    "img_h",
]

# A detector takes image paths and returns, per image, (img_w, img_h, boxes) where boxes is a
# list of (x1, y1, x2, y2, conf) in pixels of the image that was fed in.
Box = tuple[float, float, float, float, float]
Detector = Callable[[Sequence[Path]], list[tuple[int, int, list[Box]]]]


def load_stability_config(path: str | Path = "stability.yaml") -> dict:
    """Load stability.yaml and return the ``collect`` section."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config["collect"]


def resolve_checkpoint(candidates: Sequence[str | Path]) -> Path:
    """Return the first existing checkpoint, or raise with a helpful message."""
    for candidate in candidates:
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "No YOLO checkpoint found. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Train a model first or pass --checkpoint."
    )


def load_manifest(perturbed_dir: str | Path, max_images: int | None = None) -> pd.DataFrame:
    """Read M1's manifest.csv, optionally keeping only the first ``max_images`` image ids."""
    path = Path(perturbed_dir) / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Generate the dataset first: "
            "python -m wheat_mlops.perturbations generate"
        )
    manifest = pd.read_csv(path, dtype={"image_id": str})
    if max_images is not None:
        keep = sorted(manifest["image_id"].unique())[: int(max_images)]
        manifest = manifest[manifest["image_id"].isin(keep)]
    return manifest.reset_index(drop=True)


def _condition_of(generated_image: str) -> str:
    """'images/gaussian_blur_s3/abc.jpg' -> 'gaussian_blur_s3'."""
    return Path(generated_image).parent.name


def list_conditions(manifest: pd.DataFrame) -> list[str]:
    """Conditions in a stable order: clean first, then corruptions by name and severity."""
    frame = manifest[["corruption", "severity", "generated_image"]].copy()
    frame["condition"] = frame["generated_image"].map(_condition_of)
    frame = frame.drop_duplicates("condition")
    frame["_not_clean"] = (frame["corruption"] != "clean").astype(int)
    frame = frame.sort_values(["_not_clean", "corruption", "severity"])
    return frame["condition"].tolist()


def make_yolo_detector(
    checkpoint: str | Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    batch: int = 8,
    device: str = "",
) -> Detector:
    """Build a detector backed by Ultralytics YOLO (imported lazily so tests stay light)."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))

    def detect(paths: Sequence[Path]) -> list[tuple[int, int, list[Box]]]:
        results = model.predict(
            [str(p) for p in paths],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            batch=batch,
            device=device or None,
            verbose=False,
        )
        out: list[tuple[int, int, list[Box]]] = []
        for result in results:
            height, width = result.orig_shape
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            boxes = [
                (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
                for b, s in zip(xyxy, scores, strict=True)
            ]
            out.append((int(width), int(height), boxes))
        return out

    return detect


def collect_condition(
    rows: pd.DataFrame,
    perturbed_dir: Path,
    detect: Detector,
    *,
    batch: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the detector on all images of one condition; return (boxes, images) frames."""
    box_rows: list[dict] = []
    image_rows: list[dict] = []
    rows = rows.sort_values("image_id").reset_index(drop=True)
    for start in range(0, len(rows), batch):
        chunk = rows.iloc[start : start + batch]
        paths = [perturbed_dir / rel for rel in chunk["generated_image"]]
        detections = detect(paths)
        if len(detections) != len(chunk):
            raise RuntimeError("detector returned a different number of results than images")
        for row, (width, height, boxes) in zip(
            chunk.itertuples(index=False), detections, strict=True
        ):
            common = {
                "image_id": str(row.image_id),
                "condition": _condition_of(row.generated_image),
                "corruption": row.corruption,
                "severity": int(row.severity),
                "img_w": width,
                "img_h": height,
            }
            # Highest confidence first, so box_idx is a deterministic, meaningful order.
            ordered = sorted(boxes, key=lambda b: -b[4])
            for idx, (x1, y1, x2, y2, score) in enumerate(ordered):
                box_rows.append(
                    {**common, "box_idx": idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    | {"conf": score}
                )
            image_rows.append({**common, "n_boxes": len(ordered)})
    return (
        pd.DataFrame(box_rows, columns=BOX_COLUMNS),
        pd.DataFrame(image_rows, columns=IMAGE_COLUMNS),
    )


def collect_all(
    perturbed_dir: str | Path,
    out_dir: str | Path,
    detect: Detector,
    *,
    batch: int = 8,
    max_images: int | None = None,
    overwrite: bool = False,
    settings: dict | None = None,
) -> dict:
    """Collect detections for every condition and write the CSVs. Returns the summary dict."""
    perturbed_dir, out_dir = Path(perturbed_dir), Path(out_dir)
    det_dir = out_dir / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(perturbed_dir, max_images)
    conditions = list_conditions(manifest)
    if "clean" not in conditions:
        raise ValueError("manifest has no 'clean' condition; stability needs a clean reference")
    manifest["_condition"] = manifest["generated_image"].map(_condition_of)

    totals: dict[str, dict] = {}
    for condition in conditions:
        boxes_path = det_dir / f"{condition}_boxes.csv"
        images_path = det_dir / f"{condition}_images.csv"
        n_expected = int((manifest["_condition"] == condition).sum())
        if boxes_path.is_file() and images_path.is_file() and not overwrite:
            images = pd.read_csv(images_path)
            if len(images) == n_expected:
                totals[condition] = {"images": len(images), "boxes": int(images["n_boxes"].sum())}
                print(f"[skip] {condition}: already collected ({totals[condition]['boxes']} boxes)")
                continue
        rows = manifest[manifest["_condition"] == condition]
        started = time.perf_counter()
        boxes, images = collect_condition(rows, perturbed_dir, detect, batch=batch)
        boxes.to_csv(boxes_path, index=False)
        images.to_csv(images_path, index=False)
        totals[condition] = {"images": len(images), "boxes": len(boxes)}
        print(
            f"[done] {condition}: {len(images)} images, {len(boxes)} boxes "
            f"in {time.perf_counter() - started:.1f}s"
        )

    summary = {
        "perturbed_dir": str(perturbed_dir),
        "n_images": int(manifest["image_id"].nunique()),
        "conditions": conditions,
        "settings": settings or {},
        "totals": totals,
    }
    (det_dir / "collect_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def load_detections(out_dir: str | Path = "data/stability") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every saved condition into two frames: (all boxes, all per-image rows)."""
    det_dir = Path(out_dir) / "detections"
    box_files = sorted(det_dir.glob("*_boxes.csv"))
    if not box_files:
        raise FileNotFoundError(f"no detections in {det_dir}; run stability.collect first")
    boxes = pd.concat([pd.read_csv(p, dtype={"image_id": str}) for p in box_files])
    images = pd.concat(
        [pd.read_csv(p, dtype={"image_id": str}) for p in sorted(det_dir.glob("*_images.csv"))]
    )
    return boxes.reset_index(drop=True), images.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="stability.yaml")
    parser.add_argument("--perturbed-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--checkpoint", help="override the checkpoint list in stability.yaml")
    parser.add_argument("--max-images", type=int, help="only use the first N validation images")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_stability_config(args.config)
    perturbed_dir = args.perturbed_dir or cfg["perturbed_dir"]
    out_dir = args.out_dir or cfg["out_dir"]
    max_images = args.max_images if args.max_images is not None else cfg.get("max_images")
    checkpoint = resolve_checkpoint([args.checkpoint] if args.checkpoint else cfg["checkpoints"])
    print(f"Using checkpoint: {checkpoint}")

    detect = make_yolo_detector(
        checkpoint,
        imgsz=int(cfg["imgsz"]),
        conf=float(cfg["conf"]),
        iou=float(cfg["iou"]),
        batch=int(cfg["batch"]),
        device=str(cfg.get("device", "")),
    )
    settings = {**cfg, "checkpoint": str(checkpoint), "max_images": max_images}
    summary = collect_all(
        perturbed_dir,
        out_dir,
        detect,
        batch=int(cfg["batch"]),
        max_images=max_images,
        overwrite=args.overwrite,
        settings=settings,
    )
    print(json.dumps({k: summary[k] for k in ("n_images", "conditions")}, indent=2))


if __name__ == "__main__":
    main()
