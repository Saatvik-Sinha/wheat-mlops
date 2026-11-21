from __future__ import annotations

import pandas as pd
from PIL import Image

from wheat_mlops.perturbations import (
    CORRUPTIONS,
    apply_perturbation,
    generate_dataset,
    load_config,
)


def test_corruptions_preserve_dimensions_and_are_seeded():
    config = load_config("perturbations.yaml")
    image = Image.new("RGB", (32, 24), (120, 90, 60))
    for corruption in CORRUPTIONS:
        first, _ = apply_perturbation(image, corruption, 3, seed=17, config=config)
        second, _ = apply_perturbation(image, corruption, 3, seed=17, config=config)
        assert first.size == image.size
        assert first.mode == "RGB"
        assert first.tobytes() == second.tobytes()


def test_generation_keeps_only_validation_images_and_labels(tmp_path):
    processed = tmp_path / "processed"
    for subdir in ("images/val", "images/train", "labels/val", "labels/train"):
        (processed / subdir).mkdir(parents=True)
    for image_id, split in (("v1", "val"), ("v2", "val"), ("t1", "train")):
        Image.new("RGB", (24, 24), (90, 100, 110)).save(
            processed / "images" / split / f"{image_id}.jpg"
        )
        (processed / "labels" / split / f"{image_id}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
    pd.DataFrame(
        [
            {"image_id": "v1", "split": "val", "source": "a", "n_boxes": 1},
            {"image_id": "v2", "split": "val", "source": "b", "n_boxes": 2},
            {"image_id": "t1", "split": "train", "source": "a", "n_boxes": 1},
        ]
    ).to_csv(processed / "split.csv", index=False)

    config = load_config("perturbations.yaml")
    out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
    generate_dataset(processed, out_a, config=config, config_path="perturbations.yaml")
    generate_dataset(processed, out_b, config=config, config_path="perturbations.yaml")

    manifest = pd.read_csv(out_a / "manifest.csv")
    assert set(manifest["image_id"]) == {"v1", "v2"}
    assert set(manifest["split"]) == {"val"}
    assert len(manifest) == 2 * (1 + len(CORRUPTIONS) * 5)
    assert (out_a / "labels" / "gaussian_noise_s3" / "v1.txt").read_text(encoding="utf-8") == (
        processed / "labels" / "val" / "v1.txt"
    ).read_text(encoding="utf-8")
    assert (out_a / "images" / "gaussian_noise_s3" / "v1.jpg").read_bytes() == (
        out_b / "images" / "gaussian_noise_s3" / "v1.jpg"
    ).read_bytes()
