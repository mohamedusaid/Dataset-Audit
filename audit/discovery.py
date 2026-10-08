from __future__ import annotations

from pathlib import Path

from .models import Shard


def infer_split(path: Path, root: Path | None = None) -> str:
    relative = path.relative_to(root) if root is not None else path
    directories = {part.lower() for part in relative.parts[:-1]}
    name = relative.name.lower()

    if directories & {"val", "validation"} or name.startswith(("val_", "validation_")):
        return "val"
    if "test" in directories or name.startswith("test_"):
        return "test"
    if "train" in directories or name.startswith("train_"):
        return "train"
    return "unknown"


def discover_shards(root: Path) -> list[Shard]:
    result = []
    for path in sorted(root.rglob("*.bin")):
        result.append(
            Shard(
                path=path,
                split=infer_split(path, root),
                size_bytes=path.stat().st_size,
            )
        )
    return result
