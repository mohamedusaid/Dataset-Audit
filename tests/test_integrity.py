import numpy as np
import pytest

from audit.discovery import discover_shards
from audit.integrity import inspect_shard
from audit.models import Shard


def test_integrity(tmp_path):
    root = tmp_path / "data" / "train"
    root.mkdir(parents=True)
    path = root / "train_shard_0000.bin"

    np.array([1, 2, 3, 50256], dtype=np.uint16).tofile(path)

    shard = discover_shards(tmp_path / "data")[0]
    stats = inspect_shard(shard)

    assert stats.token_count == 4
    assert stats.min_token_id == 1
    assert stats.max_token_id == 50256
    assert stats.invalid_token_count == 0
    assert stats.eos_count == 1


def test_odd_sized_shard_fails_fast(tmp_path):
    path = tmp_path / "train_bad.bin"
    path.write_bytes(b"\x01\x00\x02")
    shard = Shard(path=path, split="train", size_bytes=path.stat().st_size)

    with pytest.raises(ValueError, match="not divisible"):
        inspect_shard(shard)
