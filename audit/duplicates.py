from __future__ import annotations

from collections import Counter, defaultdict

from tqdm import tqdm

from .hashing import hash_tokens
from .stream import iter_documents


def exact_duplicate_analysis(
    shards,
    eos_token_id=50256,
    max_document_tokens=100000,
):
    """
    Exact SHA-256 document dedup.

    Stores one 64-char hash per document plus counters.
    This is substantially smaller than storing document text/tokens.
    """
    occurrences = Counter()
    first_location = {}

    total_docs = 0
    total_tokens = 0

    for shard in tqdm(shards, desc="Exact dedup"):
        for doc_index, doc in enumerate(
            iter_documents(shard.path, eos_token_id=eos_token_id)
        ):
            n = len(doc)
            if n == 0 or n > max_document_tokens:
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

    top = []
    for digest, count in occurrences.most_common(100):
        if count > 1:
            top.append(
                {
                    "hash": digest,
                    "count": count,
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
        "duplicate_document_fraction": (
            duplicate_excess / total_docs if total_docs else 0.0
        ),
        "top_duplicate_groups": top,
    }
