"""Image degradation and domain shift perturbations for uncertainty evaluation."""

from __future__ import annotations

import io
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class DegradationSeverity(str, Enum):
    CLEAN = "clean"
    MILD = "mild"
    MODERATE = "moderate"
    HEAVY = "heavy"


def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur to an image."""
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(image: Image.Image, sigma: float, seed: int | None = None) -> Image.Image:
    """Add zero-mean Gaussian noise with standard deviation sigma in [0, 255]."""
    if sigma <= 0:
        return image
    rng = np.random.default_rng(seed)
    arr = np.array(image, dtype=float)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def apply_brightness_contrast(
    image: Image.Image, brightness_factor: float = 1.0, contrast_factor: float = 1.0
) -> Image.Image:
    """Adjust brightness and contrast."""
    out = image
    if brightness_factor != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness_factor)
    if contrast_factor != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast_factor)
    return out


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Simulate lossy JPEG compression artifacts."""
    if quality >= 100:
        return image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_domain_shift(
    image: Image.Image,
    shift_type: str = "color_temperature",
    severity: str = "moderate",
) -> Image.Image:
    """Simulate realistic field domain shifts (color temp / golden hour, fog/haze, shadowing).

    Args:
        image: Source PIL image.
        shift_type: One of 'color_temperature' (sunset/growth stage), 'fog' (weather), 'shadow'.
        severity: 'mild', 'moderate', or 'heavy'.
    """
    scale = {"mild": 0.5, "moderate": 1.0, "heavy": 1.8}.get(severity, 1.0)
    arr = np.array(image, dtype=float)

    if shift_type == "color_temperature":
        # Simulates different sunlight / sensor white-balance / maturity stage
        # Enhance red/yellow channel, decrease blue channel
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + 0.15 * scale), 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1.0 + 0.05 * scale), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 - 0.20 * scale), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    elif shift_type == "fog":
        # Simulates morning fog / atmospheric haze in field
        haze_intensity = min(0.6, 0.20 * scale)
        haze = np.ones_like(arr) * 220.0
        blended = (1.0 - haze_intensity) * arr + haze_intensity * haze
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    elif shift_type == "shadow":
        # Simulates cloud shadows across field
        h, w = arr.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        gradient = 0.5 * (1 + np.sin(2.5 * xx + 1.5 * yy))
        factor = 1.0 - (0.35 * scale * gradient[:, :, np.newaxis])
        shadowed = np.clip(arr * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(shadowed)

    return image


def apply_degradation(
    image: Image.Image,
    severity: DegradationSeverity | str,
    seed: int | None = None,
) -> Image.Image:
    """Apply compound degradation according to specified severity level.

    Levels:
    - clean: untouched original image
    - mild: slight blur, low noise, slight contrast shift, mild JPEG
    - moderate: medium blur, medium noise, moderate contrast shift, medium JPEG
    - heavy: severe blur, heavy noise, strong contrast shift, aggressive JPEG
    """
    sev = DegradationSeverity(severity)
    if sev == DegradationSeverity.CLEAN:
        return image.copy()

    params = {
        DegradationSeverity.MILD: {
            "blur_radius": 1.0,
            "noise_sigma": 8.0,
            "brightness": 0.95,
            "contrast": 1.05,
            "jpeg_quality": 80,
        },
        DegradationSeverity.MODERATE: {
            "blur_radius": 2.5,
            "noise_sigma": 20.0,
            "brightness": 0.85,
            "contrast": 1.15,
            "jpeg_quality": 45,
        },
        DegradationSeverity.HEAVY: {
            "blur_radius": 5.0,
            "noise_sigma": 40.0,
            "brightness": 0.70,
            "contrast": 1.30,
            "jpeg_quality": 20,
        },
    }[sev]

    out = image.copy()
    out = apply_blur(out, radius=params["blur_radius"])
    out = apply_gaussian_noise(out, sigma=params["noise_sigma"], seed=seed)
    out = apply_brightness_contrast(
        out,
        brightness_factor=params["brightness"],
        contrast_factor=params["contrast"],
    )
    out = apply_jpeg_compression(out, quality=params["jpeg_quality"])
    return out


def create_degraded_dataset(
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    severity: DegradationSeverity | str,
    domain_shift: str | None = None,
    seed: int = 42,
) -> list[Path]:
    """Generate and save degraded versions of given images to output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for idx, img_path in enumerate(image_paths):
        p = Path(img_path)
        img = Image.open(p).convert("RGB")

        # Apply severity degradation
        degraded = apply_degradation(img, severity=severity, seed=seed + idx)

        # Apply optional domain shift
        if domain_shift:
            degraded = apply_domain_shift(degraded, shift_type=domain_shift, severity=str(severity))

        target_path = out_dir / p.name
        degraded.save(target_path, quality=95)
        saved_paths.append(target_path)

    return saved_paths
