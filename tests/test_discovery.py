from pathlib import Path

from audit.discovery import infer_split


def test_split_ignores_directories_above_the_scan_root():
    root = Path("/data/val/export")

    assert infer_split(root / "train" / "shard_0.bin", root) == "train"
    assert infer_split(root / "shard_0.bin", root) == "unknown"
