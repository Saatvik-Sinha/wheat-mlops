from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image
from wheat_mlops.perturbations import generate_dataset, load_config
from wheat_mlops.stability.collect import (
    collect_all,
    list_conditions,
    load_detections,
    load_manifest,
    resolve_checkpoint,
)


def _make_perturbed(tmp_path):
    processed = tmp_path / "processed"
    for subdir in ("images/val", "labels/val"):
        (processed / subdir).mkdir(parents=True)
    rows = []
    for image_id in ("v1", "v2", "v3"):
        Image.new("RGB", (32, 24), (90, 100, 110)).save(
            processed / "images/val" / f"{image_id}.jpg"
        )
        (processed / "labels/val" / f"{image_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        rows.append({"image_id": image_id, "split": "val", "source": "a", "n_boxes": 1})
    pd.DataFrame(rows).to_csv(processed / "split.csv", index=False)
    out = tmp_path / "perturbed"
    generate_dataset(
        processed, out, config=load_config("perturbations.yaml"), config_path="perturbations.yaml"
    )
    return out


def _fake_detector(calls):
    def detect(paths):
        calls.append(len(paths))
        results = []
        for p in paths:
            # v2 yields no detections so we can check that empty images are still recorded.
            if p.stem == "v2":
                boxes = []
            else:
                boxes = [(1.0, 2.0, 11.0, 12.0, 0.5), (3.0, 3.0, 9.0, 9.0, 0.9)]
            results.append((32, 24, boxes))
        return results

    return detect


def test_conditions_clean_first_and_complete(tmp_path):
    manifest = load_manifest(_make_perturbed(tmp_path))
    conditions = list_conditions(manifest)
    assert conditions[0] == "clean"
    assert len(conditions) == 26
    assert "gaussian_blur_s3" in conditions


def test_collect_all_writes_boxes_and_empty_images(tmp_path):
    perturbed = _make_perturbed(tmp_path)
    out = tmp_path / "stability"
    summary = collect_all(perturbed, out, _fake_detector([]), batch=2)
    assert summary["n_images"] == 3
    assert summary["totals"]["clean"] == {"images": 3, "boxes": 4}

    boxes, images = load_detections(out)
    assert set(images["condition"]) == set(summary["conditions"])
    assert len(images) == 3 * 26
    assert (images.loc[images["image_id"] == "v2", "n_boxes"] == 0).all()
    clean = boxes[(boxes["condition"] == "clean") & (boxes["image_id"] == "v1")]
    assert clean["box_idx"].tolist() == [0, 1]
    assert clean["conf"].tolist() == [0.9, 0.5]  # sorted by confidence, highest first
    assert set(boxes["image_id"]) == {"v1", "v3"}


def test_collect_is_resumable_and_max_images(tmp_path):
    perturbed = _make_perturbed(tmp_path)
    out = tmp_path / "stability"
    calls: list[int] = []
    collect_all(perturbed, out, _fake_detector(calls), batch=8, max_images=2)
    first_calls = len(calls)
    assert first_calls == 26  # one batch per condition, 2 images each
    summary = collect_all(perturbed, out, _fake_detector(calls), batch=8, max_images=2)
    assert len(calls) == first_calls  # everything skipped
    assert summary["n_images"] == 2
    collect_all(perturbed, out, _fake_detector(calls), batch=8, max_images=2, overwrite=True)
    assert len(calls) == 2 * first_calls
    # A partial earlier run (fewer images) is redone rather than silently reused.
    collect_all(perturbed, out, _fake_detector(calls), batch=8)
    assert len(calls) == 2 * first_calls + 26


def test_missing_inputs_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="perturbations generate"):
        load_manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="No YOLO checkpoint"):
        resolve_checkpoint([tmp_path / "nope.pt"])
    with pytest.raises(FileNotFoundError, match="stability.collect"):
        load_detections(tmp_path)
