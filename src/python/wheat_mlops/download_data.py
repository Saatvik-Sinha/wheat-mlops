"""Download the Kaggle competition data into data/raw (idempotent).

Loads ``KAGGLE_API_TOKEN`` from the repository-root ``.env`` file before invoking the Kaggle
CLI. The competition rules must also be accepted on Kaggle (click "Join Competition" once).

Usage:  python -m wheat_mlops.download_data [--force]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from wheat_mlops.config import load_params


def is_downloaded(raw_dir: Path) -> bool:
    return (raw_dir / "train.csv").is_file() and (raw_dir / "train").is_dir()


def download(competition: str, raw_dir: str | Path, force: bool = False) -> None:
    raw_dir = Path(raw_dir)
    if is_downloaded(raw_dir) and not force:
        print(f"Data already present in {raw_dir} - nothing to do (use --force to redo).")
        return

    if shutil.which("kaggle") is None:
        sys.exit(
            "The `kaggle` command was not found. Activate your virtualenv and run "
            "install the project dependencies, then set KAGGLE_API_TOKEN in the repository .env."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(raw_dir)]
    print("Running:", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit(
            "Download failed. Usual causes: (1) Kaggle credentials are not set up, "
            "(2) you have not accepted the competition rules on the website, "
            "(3) no internet connection."
        )

    zip_path = raw_dir / f"{competition}.zip"
    if not zip_path.is_file():
        sys.exit(f"Expected {zip_path} after download, but it does not exist.")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()  # save disk space

    n_train = len(list((raw_dir / "train").glob("*.jpg")))
    n_test = len(list((raw_dir / "test").glob("*.jpg")))
    print(f"Done. {n_train} training images, {n_test} test images in {raw_dir}.")


def main() -> None:
    # DVC runs from the repository root, but locate the project .env explicitly so this
    # also works when the command is launched from another working directory.
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args()
    p = load_params()["data"]
    download(p["competition"], p["raw_dir"], force=args.force)


if __name__ == "__main__":
    main()
