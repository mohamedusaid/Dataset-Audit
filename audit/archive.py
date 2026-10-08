from __future__ import annotations

import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

DEFAULT_MAX_ARCHIVE_MEMBERS = 100_000
DEFAULT_MAX_UNCOMPRESSED_BYTES = 100 * 1024**3


def _archive_fingerprint(path: Path) -> dict[str, int]:
    """A cheap cache key that prevents reusing extraction for a changed archive."""
    metadata = path.stat()
    return {"size_bytes": metadata.st_size, "modified_ns": metadata.st_mtime_ns}


def _validate_archive(
    archive: zipfile.ZipFile,
    destination: Path,
    *,
    max_members: int,
    max_uncompressed_bytes: int,
) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    if len(entries) > max_members:
        raise ValueError(f"Archive has too many members ({len(entries):,}).")

    total_size = sum(entry.file_size for entry in entries)
    if total_size > max_uncompressed_bytes:
        raise ValueError(
            f"Archive exceeds the configured uncompressed-size limit ({total_size:,} bytes)."
        )

    root = destination.resolve()
    for entry in entries:
        member = Path(entry.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe archive member path: {entry.filename!r}")
        if entry.flag_bits & 0x1:
            raise ValueError(f"Encrypted archive member is not supported: {entry.filename!r}")
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlink archive member is not allowed: {entry.filename!r}")
        try:
            (root / member).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe archive member path: {entry.filename!r}") from exc

    return entries


def _safe_extract(
    archive: zipfile.ZipFile, entries: list[zipfile.ZipInfo], destination: Path
) -> None:
    for entry in entries:
        target = destination / entry.filename
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(entry, "r") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def prepare_input(
    path: Path,
    destination: Path,
    *,
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> Path:
    """Return a source directory, securely extracting a ZIP input when needed."""
    if path.is_dir():
        return path

    if path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a directory or .zip file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input archive does not exist: {path}")
    if max_members <= 0 or max_uncompressed_bytes <= 0:
        raise ValueError("Archive limits must be positive.")

    marker = destination / ".extracted"
    fingerprint = _archive_fingerprint(path)
    if destination.exists() and marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == fingerprint:
                return destination
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        raise ValueError(
            f"Prepared input at {destination} belongs to a different or incomplete "
            "archive. Choose another output directory or remove its _prepared "
            "directory."
        )

    if destination.exists():
        if any(destination.iterdir()):
            raise ValueError(
                f"Prepared input directory already exists but is not a matching cache: "
                f"{destination}"
            )
        destination.rmdir()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.extract-", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(path, "r") as z:
            entries = _validate_archive(
                z,
                temporary_destination,
                max_members=max_members,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            _safe_extract(z, entries, temporary_destination)

        (temporary_destination / ".extracted").write_text(json.dumps(fingerprint), encoding="utf-8")
        temporary_destination.replace(destination)
    except BaseException:
        shutil.rmtree(temporary_destination, ignore_errors=True)
        raise

    return destination


def cleanup_prepared(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
