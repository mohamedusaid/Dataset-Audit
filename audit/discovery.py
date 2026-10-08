from __future__ import annotations

from pathlib import Path

from .models import Shard


def infer_split(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()

    if "val" in parts or name.startswith("val_") or "validation" in name:
        return "val"
    if "test" in parts or name.startswith("test_"):
        return "test"
    if "train" in parts or name.startswith("train_"):
        return "train"
    return "unknown"


def discover_shards(root: Path) -> list[Shard]:
    result = []
    for path in sorted(root.rglob("*.bin")):
        result.append(
            Shard(
                path=path,
                split=infer_split(path),
                size_bytes=path.stat().st_size,
            )
        )
    return result
