from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def prepare_input(path: Path, destination: Path) -> Path:
    if path.is_dir():
        return path

    if path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a directory or .zip file: {path}")

    destination.mkdir(parents=True, exist_ok=True)

    marker = destination / ".extracted"
    if marker.exists():
        return destination

    with zipfile.ZipFile(path, "r") as z:
        z.extractall(destination)

    marker.write_text("ok", encoding="utf-8")
    return destination


def cleanup_prepared(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
