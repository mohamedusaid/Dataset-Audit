from __future__ import annotations

from collections import defaultdict

from tqdm import tqdm

from .hashing import hash_tokens
from .stream import iter_split_documents


def split_document_hashes(shards, eos_token_id=50256, max_document_tokens=100000):
    hashes = defaultdict(set)

    for shard, _, doc in tqdm(
        iter_split_documents(shards, eos_token_id=eos_token_id),
        desc="Split leakage",
        unit="doc",
    ):
        if len(doc) <= max_document_tokens:
            hashes[shard.split].add(hash_tokens(doc))

    return hashes


def train_validation_leakage(shards, eos_token_id=50256, max_document_tokens=100000):
    hashes = split_document_hashes(shards, eos_token_id, max_document_tokens)
    train = hashes.get("train", set())
    val = hashes.get("val", set())

    overlap = train & val

    return {
        "train_unique_hashes": len(train),
        "val_unique_hashes": len(val),
        "exact_train_val_overlap": len(overlap),
        "val_overlap_fraction": len(overlap) / len(val) if val else 0.0,
        "overlap_hashes_sample": sorted(overlap)[:100],
    }
