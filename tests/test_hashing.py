import hashlib

import numpy as np

from audit.hashing import hash_tokens


def test_token_hash_uses_canonical_little_endian_bytes():
    tokens = np.array([1, 258, 50256], dtype=np.uint16)
    expected = hashlib.sha256(tokens.astype("<u2", copy=False).tobytes()).hexdigest()

    assert hash_tokens(tokens) == expected
