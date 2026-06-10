"""Download ISIC 2019 dataset via Kaggle API."""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def download(data_root: str) -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("  2. Place kaggle.json in ~/.kaggle/")
        sys.exit(1)

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "isic-2019.zip"
    if not zip_path.exists():
        print("[DOWNLOAD] Downloading andrewmvd/isic-2019 from Kaggle...")
        os.system(
            f"kaggle datasets download andrewmvd/isic-2019 -p {root} --unzip"
        )
    else:
        print("[DOWNLOAD] Archive already exists, skipping download.")

    # Verify expected files
    expected = [
        root / "ISIC_2019_Training_Metadata.csv",
        root / "ISIC_2019_Training_GroundTruth.csv",
        root / "ISIC_2019_Training_Input",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print("WARNING: Expected files/dirs not found:")
        for m in missing:
            print(f"  {m}")
        print("Check that the Kaggle dataset extracted correctly.")
    else:
        print(f"[DOWNLOAD] Dataset ready at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/isic2019")
    args = parser.parse_args()
    download(args.data_root)
