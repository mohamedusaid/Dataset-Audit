from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .stream import iter_token_blocks


def token_statistics(
    shards,
    vocab_size: int = 50257,
    eos_token_id: int = 50256,
    top_k: int = 100,
):
    counts = np.zeros(vocab_size, dtype=np.int64)

    total = 0
    for shard in shards:
        for _, block in iter_token_blocks(shard.path):
            arr = np.asarray(block, dtype=np.uint16)
            valid = arr[arr < vocab_size]
            if len(valid):
                counts += np.bincount(valid.astype(np.int64), minlength=vocab_size)
                total += len(valid)

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
