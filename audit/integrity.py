from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .models import Shard, ShardStats
from .stream import DTYPE, open_tokens


def inspect_shard(
    shard: Shard,
    vocab_size: int = 50257,
    eos_token_id: int = 50256,
    block_tokens: int = 4_000_000,
    compute_sha256: bool = True,
) -> ShardStats:
    path = shard.path

    if path.stat().st_size % DTYPE.itemsize != 0:
        return ShardStats(
            path=str(path),
            split=shard.split,
            size_bytes=path.stat().st_size,
            token_count=0,
            min_token_id=None,
            max_token_id=None,
            invalid_token_count=-1,
            eos_count=0,
        )

    tokens = open_tokens(path)

    min_id = None
    max_id = None
    invalid = 0
    eos = 0
    h = hashlib.sha256() if compute_sha256 else None

    for start in range(0, len(tokens), block_tokens):
        block = np.asarray(tokens[start : start + block_tokens])

        if len(block):
            bmin = int(block.min())
            bmax = int(block.max())
            min_id = bmin if min_id is None else min(min_id, bmin)
            max_id = bmax if max_id is None else max(max_id, bmax)

            invalid += int(np.count_nonzero(block >= vocab_size))
            eos += int(np.count_nonzero(block == eos_token_id))

            if h is not None:
                h.update(block.astype(np.uint16, copy=False).tobytes())

    return ShardStats(
        path=str(path),
        split=shard.split,
        size_bytes=path.stat().st_size,
        token_count=len(tokens),
        min_token_id=min_id,
        max_token_id=max_id,
        invalid_token_count=invalid,
        eos_count=eos,
        sha256=h.hexdigest() if h else None,
    )


def inspect_all(
    shards: list[Shard],
    vocab_size: int,
    eos_token_id: int,
    compute_sha256: bool = True,
    workers: int = 1,
) -> list[ShardStats]:
    if workers < 1:
        raise ValueError("workers must be at least one")

    def inspect(shard: Shard) -> ShardStats:
        return inspect_shard(
            shard,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            compute_sha256=compute_sha256,
        )

    if workers == 1 or len(shards) < 2:
        return [inspect(shard) for shard in tqdm(shards, desc="Integrity")]

    # executor.map preserves the discovery order, so reports remain deterministic.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(inspect, shards),
                total=len(shards),
                desc="Integrity",
            )
        )
