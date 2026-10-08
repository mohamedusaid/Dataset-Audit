import numpy as np

from audit.leakage import train_validation_leakage
from audit.models import Shard


def test_leakage_uses_stitched_documents_and_document_size_filter(tmp_path):
    train_first = tmp_path / "train_0.bin"
    train_second = tmp_path / "train_1.bin"
    validation = tmp_path / "val_0.bin"
    np.array([1, 2], dtype="<u2").tofile(train_first)
    np.array([3, 50256, 8, 9, 10, 11, 50256], dtype="<u2").tofile(train_second)
    np.array([1, 2, 3, 50256, 8, 9, 10, 11, 50256], dtype="<u2").tofile(validation)
    shards = [
        Shard(train_first, "train", train_first.stat().st_size),
        Shard(train_second, "train", train_second.stat().st_size),
        Shard(validation, "val", validation.stat().st_size),
    ]

    result = train_validation_leakage(shards, max_document_tokens=3)

    assert result["train_unique_hashes"] == 1
    assert result["val_unique_hashes"] == 1
    assert result["exact_train_val_overlap"] == 1
