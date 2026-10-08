import numpy as np

from audit.models import Shard
from audit.sampling import reservoir_documents


def test_reservoir_sampling_is_deterministic_and_marks_truncation(tmp_path):
    path = tmp_path / "train.bin"
    np.array([1, 50256, 2, 3, 4, 50256, 5, 6, 50256], dtype="<u2").tofile(path)
    shards = [Shard(path=path, split="train", size_bytes=path.stat().st_size)]

    first, metadata = reservoir_documents(
        shards, sample_documents=3, seed=9, max_tokens_per_document=2
    )
    second, _ = reservoir_documents(shards, sample_documents=3, seed=9, max_tokens_per_document=2)

    assert metadata == {
        "documents_seen": 3,
        "reservoir_size": 3,
        "truncated_documents_in_sample": 1,
    }
    assert [record["document_index"] for record in first] == [
        record["document_index"] for record in second
    ]
    assert any(record["tokens_truncated"] for record in first)
