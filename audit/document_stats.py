from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .stream import iter_documents


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def document_statistics(
    shards,
    eos_token_id=50256,
    min_document_tokens=8,
    max_document_tokens=100000,
):
    by_split = {}
    global_lengths = []

    for shard in tqdm(shards, desc="Documents"):
        split = shard.split
        state = by_split.setdefault(
            split,
            {
                "documents": 0,
                "empty_documents": 0,
                "short_documents": 0,
                "long_documents": 0,
                "tokens_in_documents": 0,
                "lengths": [],
            },
        )

        for doc in iter_documents(shard.path, eos_token_id=eos_token_id):
            n = len(doc)
            if n == 0:
                state["empty_documents"] += 1
                continue

            state["documents"] += 1
            state["tokens_in_documents"] += n

            if n < min_document_tokens:
                state["short_documents"] += 1
            if n > max_document_tokens:
                state["long_documents"] += 1

            # Keep lengths for the distribution. This is only one integer per document.
            state["lengths"].append(n)
            global_lengths.append(n)

    for state in by_split.values():
        lengths = state.pop("lengths")
        state.update(
            {
                "mean_tokens": float(np.mean(lengths)) if lengths else 0.0,
                "median_tokens": percentile(lengths, 50),
                "p90_tokens": percentile(lengths, 90),
                "p95_tokens": percentile(lengths, 95),
                "p99_tokens": percentile(lengths, 99),
                "max_tokens": int(max(lengths)) if lengths else 0,
            }
        )

    return {
        "by_split": by_split,
        "global": {
            "documents": len(global_lengths),
            "mean_tokens": float(np.mean(global_lengths)) if global_lengths else 0.0,
            "median_tokens": percentile(global_lengths, 50),
            "p90_tokens": percentile(global_lengths, 90),
            "p95_tokens": percentile(global_lengths, 95),
            "p99_tokens": percentile(global_lengths, 99),
            "max_tokens": int(max(global_lengths)) if global_lengths else 0,
        },
    }
