from __future__ import annotations

from collections import Counter

from tqdm import tqdm

from .hashing import hash_tokens
from .stream import iter_split_documents


def exact_duplicate_analysis(
    shards,
    eos_token_id=50256,
    max_document_tokens=100000,
):
    """
    Exact SHA-256 document dedup.

    Stores one SHA-256 hash and first location per unique document. This is
    memory-bounded by the number of unique documents, not their token mass.
    """
    occurrences = Counter()
    first_location = {}

    total_docs = 0
    total_tokens = 0

    for shard, doc_index, doc in tqdm(
        iter_split_documents(shards, eos_token_id=eos_token_id),
        desc="Exact dedup",
        unit="doc",
    ):
        n = len(doc)
        if n > max_document_tokens:
            continue

        digest = hash_tokens(doc)
        occurrences[digest] += 1
        total_docs += 1
        total_tokens += n

        if digest not in first_location:
            first_location[digest] = {
                "split": shard.split,
                "shard": str(shard.path),
                "document_index": doc_index,
                "tokens": n,
            }

    duplicate_groups = sum(1 for c in occurrences.values() if c > 1)
    duplicate_docs = sum(c for c in occurrences.values() if c > 1)
    duplicate_excess = sum(c - 1 for c in occurrences.values() if c > 1)
    duplicate_excess_tokens = sum(
        (count - 1) * first_location[digest]["tokens"]
        for digest, count in occurrences.items()
        if count > 1
    )

    top = []
    for digest, count in occurrences.most_common(100):
        if count > 1:
            top.append(
                {
                    "hash": digest,
                    "count": count,
                    "excess_tokens": (count - 1) * first_location[digest]["tokens"],
                    "first_location": first_location[digest],
                }
            )

    return {
        "documents_hashed": total_docs,
        "tokens_hashed": total_tokens,
        "unique_documents": len(occurrences),
        "duplicate_groups": duplicate_groups,
        "duplicate_documents_in_groups": duplicate_docs,
        "duplicate_excess_documents": duplicate_excess,
        "duplicate_document_fraction": (duplicate_excess / total_docs if total_docs else 0.0),
        "duplicate_excess_tokens": duplicate_excess_tokens,
        "duplicate_token_fraction": (
            duplicate_excess_tokens / total_tokens if total_tokens else 0.0
        ),
        "top_duplicate_groups": top,
    }
