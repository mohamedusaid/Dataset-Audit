from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

# The on-disk format is explicitly little-endian.  Relying on the host-native
# uint16 representation makes hashes and reads non-portable on big-endian hosts.
DTYPE = np.dtype("<u2")


def open_tokens(path: Path):
    if path.stat().st_size % DTYPE.itemsize != 0:
        raise ValueError(f"Byte size is not divisible by uint16 size: {path}")
    return np.memmap(path, dtype=DTYPE, mode="r")


def iter_token_blocks(path: Path, block_tokens: int = 4_000_000):
    tokens = open_tokens(path)
    for start in range(0, len(tokens), block_tokens):
        yield start, tokens[start : start + block_tokens]


def iter_documents(
    path: Path,
    eos_token_id: int = 50256,
    block_tokens: int = 4_000_000,
    include_empty: bool = False,
) -> Iterator[np.ndarray]:
    """
    Stream documents separated by EOS.

    EOS itself is excluded from the returned document token array.
    Consecutive EOS tokens produce empty documents.  They are skipped by default
    but can be returned with ``include_empty=True`` for accounting.

    Chunks are retained as views until a boundary is found.  This avoids repeated
    copying (and quadratic work) for a document spanning multiple input blocks.
    """
    segments: list[np.ndarray] = []

    for _, block in iter_token_blocks(path, block_tokens):
        start = 0
        for end in np.flatnonzero(block == eos_token_id):
            if end > start:
                segments.append(block[start:end])

            if segments:
                document = segments[0] if len(segments) == 1 else np.concatenate(segments)
                yield document
            elif include_empty:
                yield np.empty(0, dtype=DTYPE)

            segments.clear()
            start = int(end) + 1

        if start < len(block):
            segments.append(block[start:])

    if segments:
        yield segments[0] if len(segments) == 1 else np.concatenate(segments)
