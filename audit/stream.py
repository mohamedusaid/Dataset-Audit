from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import Shard

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


def iter_split_documents(
    shards: list[Shard],
    eos_token_id: int = 50256,
    block_tokens: int = 4_000_000,
    include_empty: bool = False,
) -> Iterator[tuple[Shard, int, np.ndarray]]:
    """Yield documents while stitching tails across consecutive shards per split.

    A document that spans shards is attributed to the shard in which it starts.
    ``document_index`` is the ordinal of a document start within that source shard,
    including empty documents.
    """
    by_split: dict[str, list[Shard]] = {}
    for shard in shards:
        by_split.setdefault(shard.split, []).append(shard)

    for split_shards in by_split.values():
        segments: list[np.ndarray] = []
        origin_shard: Shard | None = None
        origin_index: int | None = None

        for shard in split_shards:
            next_local_index = 0
            for _, block in iter_token_blocks(shard.path, block_tokens):
                start = 0
                for end in np.flatnonzero(block == eos_token_id):
                    if end > start:
                        if not segments:
                            origin_shard = shard
                            origin_index = next_local_index
                            next_local_index += 1
                        segments.append(block[start:end])

                    if segments:
                        document = segments[0] if len(segments) == 1 else np.concatenate(segments)
                        assert origin_shard is not None and origin_index is not None
                        yield origin_shard, origin_index, document
                    elif include_empty:
                        yield shard, next_local_index, np.empty(0, dtype=DTYPE)
                        next_local_index += 1

                    segments.clear()
                    origin_shard = None
                    origin_index = None
                    start = int(end) + 1

                if start < len(block):
                    if not segments:
                        origin_shard = shard
                        origin_index = next_local_index
                        next_local_index += 1
                    segments.append(block[start:])

        if segments:
            document = segments[0] if len(segments) == 1 else np.concatenate(segments)
            assert origin_shard is not None and origin_index is not None
            yield origin_shard, origin_index, document
