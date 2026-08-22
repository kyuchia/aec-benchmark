"""Fetch and verify the LibriSpeech test-clean subset.

Downloads the archive from OpenSLR into data/speech/ (gitignored),
verifies its md5 against the published checksum, and extracts it.
Idempotent: an already-verified archive is not re-downloaded, an
already-extracted tree is not re-extracted.

Usage:
    python scripts/fetch_data.py
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
# Published checksum for test-clean.tar.gz (verified against the actual
# OpenSLR download).
ARCHIVE_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEECH_DIR = REPO_ROOT / "data" / "speech"
ARCHIVE_PATH = SPEECH_DIR / "test-clean.tar.gz"
EXTRACTED_DIR = SPEECH_DIR / "LibriSpeech" / "test-clean"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download() -> None:
    print(f"Downloading {ARCHIVE_URL} -> {ARCHIVE_PATH}")
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = ARCHIVE_PATH.with_suffix(".part")
    urllib.request.urlretrieve(ARCHIVE_URL, tmp_path)
    tmp_path.rename(ARCHIVE_PATH)


def ensure_archive() -> None:
    if ARCHIVE_PATH.exists():
        digest = _md5(ARCHIVE_PATH)
        if digest == ARCHIVE_MD5:
            print(f"Archive present, checksum OK: {ARCHIVE_PATH}")
            return
        print(f"Checksum mismatch ({digest}), re-downloading")
        ARCHIVE_PATH.unlink()
    _download()
    digest = _md5(ARCHIVE_PATH)
    if digest != ARCHIVE_MD5:
        sys.exit(
            f"Downloaded archive checksum {digest} != expected {ARCHIVE_MD5}; "
            "aborting."
        )
    print("Download complete, checksum OK")


def ensure_extracted() -> None:
    if EXTRACTED_DIR.is_dir():
        n_flac = sum(1 for _ in EXTRACTED_DIR.rglob("*.flac"))
        if n_flac > 0:
            print(f"Already extracted: {EXTRACTED_DIR} ({n_flac} flac files)")
            return
    print(f"Extracting {ARCHIVE_PATH}")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(SPEECH_DIR, filter="data")
    n_flac = sum(1 for _ in EXTRACTED_DIR.rglob("*.flac"))
    print(f"Extracted {n_flac} flac files to {EXTRACTED_DIR}")


def main() -> None:
    ensure_archive()
    ensure_extracted()


if __name__ == "__main__":
    main()
