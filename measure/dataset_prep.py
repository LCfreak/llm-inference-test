# download_en_ne.py

import os
import tarfile
import pandas as pd
import requests
from pathlib import Path

URL = (
    "https://object.pouta.csc.fi/OPUS-100/v1.0/"
    "opus-100-corpus-en-ne-v1.0.tar.gz"
)

DATA_DIR = Path("data/en_ne")
ARCHIVE = DATA_DIR / "en-ne.tar.gz"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def download():
    print("Downloading English-Nepali OPUS-100...")

    response = requests.get(URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(ARCHIVE, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded / total * 100
                    print(f"\r{percent:.1f}%", end="")

    print("\nDownload complete.")


def extract():
    print("Extracting...")

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    print("Extraction complete.")


def find_files():
    print("\nSearching for dataset files...")

    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    download()
    extract()
    find_files()