from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .stream import iter_token_blocks


def _shard_token_counts(shard, vocab_size: int) -> tuple[np.ndarray, int]:
    counts = np.zeros(vocab_size, dtype=np.int64)
    total = 0
    for _, block in iter_token_blocks(shard.path):
        arr = np.asarray(block, dtype=np.uint16)
        valid = arr[arr < vocab_size]
        if len(valid):
            counts += np.bincount(valid.astype(np.int64), minlength=vocab_size)
            total += len(valid)
    return counts, total


def token_statistics(
    shards,
    vocab_size: int = 50257,
    eos_token_id: int = 50256,
    top_k: int = 100,
    workers: int = 1,
):
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id must be within the vocabulary")
    if workers < 1:
        raise ValueError("workers must be at least one")

    counts = np.zeros(vocab_size, dtype=np.int64)
    total = 0

    if workers == 1 or len(shards) < 2:
        shard_results = (_shard_token_counts(shard, vocab_size) for shard in shards)
        for shard_counts, shard_total in shard_results:
            counts += shard_counts
            total += shard_total
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for shard_counts, shard_total in executor.map(
                lambda shard: _shard_token_counts(shard, vocab_size), shards
            ):
                counts += shard_counts
                total += shard_total

    order = np.argsort(counts)[::-1][:top_k]

    top = [
        {
            "token_id": int(i),
            "count": int(counts[i]),
            "fraction": float(counts[i] / total) if total else 0.0,
        }
        for i in order
        if counts[i] > 0
    ]

    nonzero = int(np.count_nonzero(counts))
    return {
        "total_valid_tokens": int(total),
        "unique_tokens_observed": nonzero,
        "vocab_size": vocab_size,
        "coverage_fraction": nonzero / vocab_size if vocab_size else 0.0,
        "eos_token_id": eos_token_id,
        "eos_count": int(counts[eos_token_id]),
        "eos_fraction": float(counts[eos_token_id] / total) if total else 0.0,
        "top_tokens": top,
        "counts": counts.tolist(),
    }
