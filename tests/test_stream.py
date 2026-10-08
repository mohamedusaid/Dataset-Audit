import numpy as np

from audit.stream import iter_documents


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
