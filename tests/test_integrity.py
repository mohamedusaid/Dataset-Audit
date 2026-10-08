import numpy as np

from audit.discovery import discover_shards
from audit.integrity import inspect_shard


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
