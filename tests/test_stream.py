import numpy as np

from audit.models import Shard
from audit.stream import iter_documents, iter_split_documents


def test_document_stream(tmp_path):
    path = tmp_path / "x.bin"
    arr = np.array([1, 2, 50256, 3, 4, 50256, 50256, 5], dtype=np.uint16)
    arr.tofile(path)

    docs = list(iter_documents(path, eos_token_id=50256))

    assert [d.tolist() for d in docs] == [[1, 2], [3, 4], [5]]


def test_document_stream_handles_block_boundaries_and_empty_documents(tmp_path):
    path = tmp_path / "x.bin"
    arr = np.array([50256, 1, 2, 3, 50256, 50256, 4], dtype="<u2")
    arr.tofile(path)

    docs = list(iter_documents(path, eos_token_id=50256, block_tokens=3, include_empty=True))

    assert [document.tolist() for document in docs] == [[], [1, 2, 3], [], [4]]


def test_documents_are_stitched_across_shards_of_the_same_split(tmp_path):
    first = tmp_path / "train_0.bin"
    second = tmp_path / "train_1.bin"
    np.array([1, 2, 50256, 3, 4], dtype="<u2").tofile(first)
    np.array([5, 50256, 6], dtype="<u2").tofile(second)
    shards = [Shard(path, "train", path.stat().st_size) for path in (first, second)]

    documents = [
        (shard.path.name, index, document.tolist())
        for shard, index, document in iter_split_documents(shards)
    ]

    assert documents == [
        ("train_0.bin", 0, [1, 2]),
        ("train_0.bin", 1, [3, 4, 5]),
        ("train_1.bin", 0, [6]),
    ]


def test_stitching_does_not_cross_splits(tmp_path):
    train = tmp_path / "train_0.bin"
    validation = tmp_path / "val_0.bin"
    np.array([1, 2], dtype="<u2").tofile(train)
    np.array([3, 50256], dtype="<u2").tofile(validation)
    shards = [
        Shard(train, "train", train.stat().st_size),
        Shard(validation, "val", validation.stat().st_size),
    ]

    documents = [
        (shard.split, document.tolist()) for shard, _, document in iter_split_documents(shards)
    ]

    assert documents == [("train", [1, 2]), ("val", [3])]
