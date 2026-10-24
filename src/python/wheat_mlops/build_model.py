"""Build the YOLO model from pretrained weights, prove it runs, save a base checkpoint.

NO TRAINING happens here. This is the Part 1 deliverable: "the model can be created,
is runnable, and a checkpoint is written to disk".

Usage:  python -m wheat_mlops.build_model
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from wheat_mlops.config import load_params


def build(pretrained: str | Path, base_checkpoint: str | Path, imgsz: int) -> dict:
    # Heavy imports live here so the rest of the package (and the tests) stay light.
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    pretrained, base_checkpoint = Path(pretrained), Path(base_checkpoint)
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create the model. If the file does not exist, Ultralytics downloads the official
    #    COCO-pretrained weights to that path (free, one-off).
    model = YOLO(str(pretrained))

    # 2. Prove it is runnable: one forward pass on a blank image (CPU works everywhere).
    blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    results = model.predict(blank, imgsz=imgsz, device="cpu", verbose=False)
    assert len(results) == 1, "unexpected number of results from the smoke prediction"

    # 3. Save the checkpoint, then reload it and run again to prove the file is usable.
    model.save(str(base_checkpoint))
    reloaded = YOLO(str(base_checkpoint))
    reloaded.predict(blank, imgsz=imgsz, device="cpu", verbose=False)

    info = {
        "pretrained_source": str(pretrained),
        "base_checkpoint": str(base_checkpoint),
        "task": model.task,
        "imgsz": imgsz,
        "n_parameters": int(sum(p.numel() for p in model.model.parameters())),
        "checkpoint_size_mb": round(base_checkpoint.stat().st_size / 1e6, 2),
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "python": platform.python_version(),
    }
    (base_checkpoint.parent / "model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return info


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    m = params["model"]
    info = build(m["pretrained"], m["base_checkpoint"], m["imgsz"])
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
