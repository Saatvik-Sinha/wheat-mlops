"""Convert the Kaggle Global Wheat Detection data into an Ultralytics YOLO dataset.

Input  (data/raw):       train.csv + train/*.jpg
Output (data/processed): images/{train,val}, labels/{train,val}, dataset.yaml,
                         split.csv, prepare_stats.json

Design notes
* Boxes are clipped to the image and filtered by area (the raw labels contain a few
  degenerate and oversized boxes).
* Images that exist on disk but have no boxes in train.csv are kept as background images
  (empty label file), which helps the model avoid hallucinating wheat.
* The split is stratified by (source, boxes-per-image bucket) so every field domain and
  every density level appears in both train and val.
* Imports nothing heavy (no torch / ultralytics) so tests and CI can run it quickly.

Usage:  python -m wheat_mlops.prepare_data
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from wheat_mlops.config import load_params


def parse_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the string column ``bbox`` ("[x, y, w, h]") into float columns x, y, w, h."""
    parts = df["bbox"].str.strip("[] ").str.split(",", expand=True).astype(float)
    parts.columns = ["x", "y", "w", "h"]
    return df.drop(columns=["bbox"]).join(parts)


def clip_and_filter(
    boxes: pd.DataFrame, min_area: float, max_area: float
) -> tuple[pd.DataFrame, int]:
    """Clip boxes to the image, drop degenerate / too small / too large ones.

    Returns (kept boxes with x1,y1,x2,y2 columns, number of dropped boxes).
    """
    out = boxes.copy()
    out["x1"] = np.maximum(out["x"], 0)
    out["y1"] = np.maximum(out["y"], 0)
    out["x2"] = np.minimum(out["x"] + out["w"], out["width"])
    out["y2"] = np.minimum(out["y"] + out["h"], out["height"])
    bw = out["x2"] - out["x1"]
    bh = out["y2"] - out["y1"]
    area = bw * bh
    keep = (bw > 1) & (bh > 1) & (area >= min_area) & (area <= max_area)
    return out[keep].reset_index(drop=True), int((~keep).sum())


def add_yolo_columns(boxes: pd.DataFrame) -> pd.DataFrame:
    """Add normalised YOLO columns cx, cy, bw, bh (needs x1,y1,x2,y2,width,height)."""
    out = boxes.copy()
    out["cx"] = (out["x1"] + out["x2"]) / 2 / out["width"]
    out["cy"] = (out["y1"] + out["y2"]) / 2 / out["height"]
    out["bw"] = (out["x2"] - out["x1"]) / out["width"]
    out["bh"] = (out["y2"] - out["y1"]) / out["height"]
    return out


def make_strata(images: pd.DataFrame, bucket: int = 15, max_bucket: int = 6) -> pd.Series:
    """Stratification key = source + boxes-per-image bucket. Rare strata are merged."""
    b = (images["n_boxes"] // bucket).clip(upper=max_bucket)
    key = images["source"].astype(str) + "_" + b.astype(str)
    counts = key.map(key.value_counts())
    key = key.where(counts >= 2, "rare")
    if (key == "rare").sum() == 1:  # a single leftover cannot be stratified
        key = key.where(key != "rare", key.value_counts().idxmax())
    return key


def split_ids(
    images: pd.DataFrame,
    val_fraction: float,
    seed: int,
    bucket: int = 15,
    max_bucket: int = 6,
) -> tuple[set, set]:
    """Stratified train/val split of image ids. Falls back to a plain split if needed."""
    key = make_strata(images, bucket, max_bucket)
    try:
        train_ids, val_ids = train_test_split(
            images["image_id"], test_size=val_fraction, random_state=seed, stratify=key
        )
    except ValueError as err:
        print(f"WARNING: stratified split not possible ({err}); using a plain random split.")
        train_ids, val_ids = train_test_split(
            images["image_id"], test_size=val_fraction, random_state=seed
        )
    return set(train_ids), set(val_ids)


def _place(src: Path, dst: Path, link: bool) -> None:
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different drive / filesystem without hard links -> copy instead
    shutil.copy2(src, dst)


def prepare(
    raw_dir: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    count_bucket: int = 15,
    max_count_bucket: int = 6,
    min_box_area: float = 100,
    max_box_area: float = 100_000,
    link_images: bool = True,
) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    img_dir = raw_dir / "train"
    if not (raw_dir / "train.csv").is_file() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train.csv and train/ in {raw_dir}.\n"
            "Run `python -m wheat_mlops.download_data`."
        )
    # out_dir is wiped on every run, so make sure it can never be (a parent of) raw_dir.
    if out_dir.resolve() == raw_dir.resolve() or out_dir.resolve() in raw_dir.resolve().parents:
        raise ValueError("processed_dir must not be raw_dir or one of its parents.")

    df = pd.read_csv(raw_dir / "train.csv")
    on_disk = {p.stem: p for p in img_dir.glob("*.jpg")}
    missing = set(df["image_id"]) - set(on_disk)
    if missing:
        print(f"WARNING: {len(missing)} images listed in train.csv are missing on disk; skipped.")
        df = df[df["image_id"].isin(on_disk)]

    boxes = parse_bboxes(df)
    n_boxes_raw = len(boxes)
    boxes, n_dropped = clip_and_filter(boxes, min_box_area, max_box_area)
    boxes = add_yolo_columns(boxes)

    source = df.drop_duplicates("image_id").set_index("image_id")["source"]
    counts = boxes.groupby("image_id").size()
    images = pd.DataFrame({"image_id": sorted(on_disk)})
    images["source"] = images["image_id"].map(source).fillna("unknown")
    images["n_boxes"] = images["image_id"].map(counts).fillna(0).astype(int)

    _, val_ids = split_ids(images, val_fraction, seed, count_bucket, max_count_bucket)
    images["split"] = np.where(images["image_id"].isin(val_ids), "val", "train")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    lines = {
        image_id: [f"0 {r.cx:.6f} {r.cy:.6f} {r.bw:.6f} {r.bh:.6f}" for r in grp.itertuples()]
        for image_id, grp in boxes.groupby("image_id")
    }
    for row in images.itertuples():
        _place(
            on_disk[row.image_id],
            out_dir / "images" / row.split / f"{row.image_id}.jpg",
            link_images,
        )
        text = "\n".join(lines.get(row.image_id, []))
        label_path = out_dir / "labels" / row.split / f"{row.image_id}.txt"
        label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    # No `path:` key on purpose: Ultralytics then resolves train/val relative to this file.
    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "wheat"},
    }
    with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write("# auto-generated by src/prepare_data.py - do not edit by hand\n")
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    images[["image_id", "split", "source", "n_boxes"]].to_csv(out_dir / "split.csv", index=False)

    stats = {
        "n_images": len(images),
        "n_train": int((images["split"] == "train").sum()),
        "n_val": int((images["split"] == "val").sum()),
        "n_background_images": int((images["n_boxes"] == 0).sum()),
        "n_boxes_raw": int(n_boxes_raw),
        "n_boxes_kept": int(len(boxes)),
        "n_boxes_dropped": int(n_dropped),
        "boxes_per_image_mean": round(float(images["n_boxes"].mean()), 2),
        "images_per_source": {
            split: {
                str(k): int(v)
                for k, v in images.loc[images["split"] == split, "source"].value_counts().items()
            }
            for split in ("train", "val")
        },
    }
    (out_dir / "prepare_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    p = params["data"]
    out_dir = Path(p["processed_dir"])
    stats = prepare(
        raw_dir=p["raw_dir"],
        out_dir=out_dir,
        val_fraction=p["val_fraction"],
        seed=p["seed"],
        count_bucket=p["count_bucket"],
        max_count_bucket=p["max_count_bucket"],
        min_box_area=p["min_box_area"],
        max_box_area=p["max_box_area"],
        link_images=p["link_images"],
    )
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
