import numpy as np

from audit.document_stats import document_statistics
from audit.models import Shard


def test_document_statistics_stitches_documents_across_shards(tmp_path):
    first = tmp_path / "train_0.bin"
    second = tmp_path / "train_1.bin"
    np.array([1, 2], dtype="<u2").tofile(first)
    np.array([3, 50256], dtype="<u2").tofile(second)
    shards = [
        Shard(first, "train", first.stat().st_size),
        Shard(second, "train", second.stat().st_size),
    ]

    stats = document_statistics(shards)

    assert stats["by_split"]["train"]["documents"] == 1
    assert stats["by_split"]["train"]["tokens_in_documents"] == 3
    assert stats["by_split"]["train"]["max_tokens"] == 3
