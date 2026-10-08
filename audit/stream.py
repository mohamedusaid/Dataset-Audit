from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

DTYPE = np.dtype(np.uint16)


def open_tokens(path: Path):
    if path.stat().st_size % DTYPE.itemsize != 0:
        raise ValueError(f"Byte size is not divisible by uint16 size: {path}")
    return np.memmap(path, dtype=DTYPE, mode="r")


def iter_token_blocks(path: Path, block_tokens: int = 4_000_000):
    tokens = open_tokens(path)
    for start in range(0, len(tokens), block_tokens):
        yield start, tokens[start:start + block_tokens]


def iter_documents(
    path: Path,
    eos_token_id: int = 50256,
    block_tokens: int = 4_000_000,
) -> Iterator[np.ndarray]:
    """
    Stream documents separated by EOS.

    EOS itself is excluded from the returned document token array.
    Consecutive EOS tokens therefore produce empty documents, which are skipped.
    """
    carry = np.empty(0, dtype=np.uint16)

    for _, block in iter_token_blocks(path, block_tokens):
        arr = block
        if len(carry):
            arr = np.concatenate((carry, arr))

        boundaries = np.flatnonzero(arr == eos_token_id)

        if len(boundaries) == 0:
            carry = arr.copy()
            continue

        start = 0
        for end in boundaries:
            if end > start:
                yield arr[start:end]
            start = int(end) + 1

        carry = arr[start:].copy()

    if len(carry):
        yield carry
