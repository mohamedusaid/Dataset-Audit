import numpy as np

from audit.stream import iter_documents


def test_document_stream(tmp_path):
    path = tmp_path / "x.bin"
    arr = np.array([1, 2, 50256, 3, 4, 50256, 50256, 5], dtype=np.uint16)
    arr.tofile(path)

    docs = list(iter_documents(path, eos_token_id=50256))

    assert [d.tolist() for d in docs] == [[1, 2], [3, 4], [5]]
