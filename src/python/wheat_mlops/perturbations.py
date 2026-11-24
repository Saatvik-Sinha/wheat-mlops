"""Generate deterministic, label-preserving wheat image corruptions.

Usage:
    python -m wheat_mlops.perturbations generate
    python -m wheat_mlops.perturbations generate --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageEnhance, ImageFilter

CORRUPTIONS = (
    "gaussian_blur",
    "gaussian_noise",
    "brightness",
    "contrast",
    "jpeg_compression",
)


def load_config(path: str | Path = "perturbations.yaml") -> dict:
    """Load and validate the perturbation configuration."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    generation = config.get("generation", {})
    severities = generation.get("severities", {})
    missing = set(CORRUPTIONS) - set(severities)
    if missing:
        raise ValueError(f"Missing corruption severity definitions: {sorted(missing)}")
    for name in CORRUPTIONS:
        values = severities[name]
        if len(values) != 5:
            raise ValueError(f"{name} must define exactly five severity values")
    parameters = config.get("spsa", {}).get("parameters", {})
    required_parameters = {"blur_sigma", "noise_sigma", "brightness_delta"}
    if set(parameters) != required_parameters:
        raise ValueError(f"SPSA parameters must be exactly {sorted(required_parameters)}")
    for name, bounds in parameters.items():
        if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
            raise ValueError(f"Invalid SPSA bounds for {name}: {bounds}")
    return config


def _image_seed(seed: int, image_id: str, corruption: str, severity: int) -> int:
    key = f"{seed}:{image_id}:{corruption}:{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def apply_perturbation(
    image: Image.Image,
    corruption: str,
    severity: int,
    *,
    seed: int,
    config: dict,
) -> tuple[Image.Image, dict]:
    """Apply one configured corruption and return its effective parameters."""
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    if severity < 1 or severity > 5:
        raise ValueError("severity must be between 1 and 5")
    image = image.convert("RGB")
    value = config["generation"]["severities"][corruption][severity - 1]
    rng = np.random.default_rng(seed)
    parameters: dict[str, float | int] = {"severity_value": value}

    if corruption == "gaussian_blur":
        result = image.filter(ImageFilter.GaussianBlur(radius=float(value)))
        parameters["radius"] = float(value)
    elif corruption == "gaussian_noise":
        pixels = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0.0, float(value), pixels.shape)
        result = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
        parameters["sigma"] = float(value)
    elif corruption in ("brightness", "contrast"):
        sign = -1 if int(rng.integers(0, 2)) == 0 else 1
        factor = max(0.05, 1.0 + sign * float(value))
        enhancer = ImageEnhance.Brightness if corruption == "brightness" else ImageEnhance.Contrast
        result = enhancer(image).enhance(factor)
        parameters["factor"] = factor
        parameters["direction"] = "lower" if sign < 0 else "higher"
    else:
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=int(value), optimize=False)
        stream.seek(0)
        with Image.open(stream) as compressed:
            result = compressed.convert("RGB").copy()
        parameters["quality"] = int(value)

    return result, parameters


def apply_spsa_perturbation(
    image: Image.Image,
    parameters: dict[str, float],
    *,
    seed: int,
) -> Image.Image:
    """Apply the continuous blur/noise/brightness search vector used by SPSA."""
    image = image.convert("RGB")
    result = image.filter(ImageFilter.GaussianBlur(radius=max(0.0, parameters["blur_sigma"])))
    pixels = np.asarray(result, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sigma = max(0.0, parameters["noise_sigma"])
    if sigma:
        pixels += rng.normal(0.0, sigma, pixels.shape)
    pixels = np.clip(pixels, 0, 255)
    brightness = parameters["brightness_delta"]
    pixels = np.clip(pixels * (1.0 + brightness), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def _resolve_images(processed_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    split_path = processed_dir / "split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Expected split.csv in prepared dataset: {processed_dir}")
    split = pd.read_csv(split_path)
    required = {"image_id", "split"}
    if not required.issubset(split.columns):
        raise ValueError(f"split.csv must contain columns {sorted(required)}")
    validation = split.loc[split["split"] == "val"].copy()
    image_dir = processed_dir / "images" / "val"
    label_dir = processed_dir / "labels" / "val"
    images: dict[str, Path] = {}
    for image_id in validation["image_id"].astype(str):
        match = next(
            (
                image_dir / f"{image_id}{ext}"
                for ext in (".jpg", ".jpeg", ".png")
                if (image_dir / f"{image_id}{ext}").is_file()
            ),
            None,
        )
        if match is None:
            raise FileNotFoundError(f"Validation image {image_id} missing in {image_dir}")
        if not (label_dir / f"{image_id}.txt").is_file():
            raise FileNotFoundError(f"Validation label missing for {image_id}")
        images[image_id] = match
    if len(images) < 2:
        raise ValueError("At least two validation images are required")
    return validation, images


def generate_dataset(
    processed_dir: str | Path,
    out_dir: str | Path,
    *,
    config: dict,
    config_path: str | Path,
    seed: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Write clean and all configured variants for the validation images."""
    raw_out_dir = Path(out_dir)
    if raw_out_dir.is_symlink():
        raise ValueError("Refusing to use a symlink output directory")
    processed_dir = Path(processed_dir).resolve()
    out_dir = Path(os.path.abspath(raw_out_dir))
    if out_dir == processed_dir or processed_dir.is_relative_to(out_dir):
        raise ValueError("Output directory must not be the prepared dataset or one of its parents")
    if out_dir.is_symlink():
        raise ValueError("Refusing to overwrite a symlink output directory")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_dir}; pass --overwrite to replace it"
            )
        if not (out_dir / "generation_stats.json").is_file():
            raise ValueError(
                f"Refusing to remove {out_dir}: it does not look like a generated dataset"
            )
        shutil.rmtree(out_dir)

    rows, images = _resolve_images(processed_dir)
    base_seed = config["generation"].get("seed", 42) if seed is None else seed
    quality = int(config["generation"].get("output_quality", 95))
    manifest: list[dict] = []
    for corruption, severity in [("clean", 0)] + [
        (name, level) for name in CORRUPTIONS for level in range(1, 6)
    ]:
        condition = "clean" if corruption == "clean" else f"{corruption}_s{severity}"
        target_images = out_dir / "images" / condition
        target_labels = out_dir / "labels" / condition
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        for row in rows.itertuples(index=False):
            image_id = str(row.image_id)
            source_image = images[image_id]
            source_label = processed_dir / "labels" / "val" / f"{image_id}.txt"
            target_image = target_images / f"{image_id}.jpg"
            target_label = target_labels / f"{image_id}.txt"
            sample_seed = _image_seed(base_seed, image_id, corruption, severity)
            if corruption == "clean":
                shutil.copy2(source_image, target_image)
                effective: dict = {}
            else:
                with Image.open(source_image) as opened:
                    transformed, effective = apply_perturbation(
                        opened, corruption, severity, seed=sample_seed, config=config
                    )
                    transformed.save(target_image, format="JPEG", quality=quality)
            shutil.copy2(source_label, target_label)
            with Image.open(target_image) as generated:
                width, height = generated.size
            manifest.append(
                {
                    "image_id": image_id,
                    "split": "val",
                    "source_image": str(source_image.relative_to(processed_dir)).replace("\\", "/"),
                    "generated_image": str(target_image.relative_to(out_dir)).replace("\\", "/"),
                    "label": str(target_label.relative_to(out_dir)).replace("\\", "/"),
                    "corruption": corruption,
                    "severity": severity,
                    "seed": sample_seed,
                    "source": getattr(row, "source", "unknown"),
                    "n_boxes": int(getattr(row, "n_boxes", 0)),
                    "parameters": json.dumps(effective, sort_keys=True),
                    "width": width,
                    "height": height,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    rows.to_csv(out_dir / "split.csv", index=False)
    dataset = {
        "path": ".",
        # Ultralytics requires a train key even though this dataset is evaluation-only.
        "train": "images/clean",
        "val": "images/clean",
        "nc": 1,
        "names": ["wheat"],
    }
    (out_dir / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(config_path, out_dir / "perturbations.yaml")
    stats = {
        "validation_images": len(images),
        "conditions": 1 + len(CORRUPTIONS) * 5,
        "generated_images": len(manifest),
        "seed": base_seed,
    }
    (out_dir / "generation_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate validation corruption variants")
    generate.add_argument("--processed-dir", default="data/processed")
    generate.add_argument("--out-dir", default="data/perturbed")
    generate.add_argument("--config", default="perturbations.yaml")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    stats = generate_dataset(
        args.processed_dir,
        args.out_dir,
        config=config,
        config_path=args.config,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
