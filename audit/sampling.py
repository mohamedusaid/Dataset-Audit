from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .stream import iter_documents
from .tokenizer import decode_tokens, get_tokenizer


def reservoir_documents(
    shards,
    *,
    eos_token_id: int = 50256,
    sample_documents: int = 1000,
    seed: int = 42,
    max_tokens_per_document: int = 16_384,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a deterministic, uniform document reservoir.

    Token arrays are only copied for records retained in the reservoir.  The
    previous implementation decoded every document before deciding whether it
    belonged in the sample, which made sampling unnecessarily expensive.
    """
    if sample_documents < 0:
        raise ValueError("sample_documents must be non-negative")
    if max_tokens_per_document <= 0:
        raise ValueError("max_tokens_per_document must be positive")

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0

    for shard in tqdm(shards, desc="Sampling candidates"):
        for doc_index, doc in enumerate(iter_documents(shard.path, eos_token_id=eos_token_id)):
            if not len(doc):
                continue

            seen += 1
            if len(reservoir) < sample_documents:
                selected_index = len(reservoir)
            else:
                selected_index = rng.randrange(seen)
                if selected_index >= sample_documents:
                    continue

            # Copy only after the reservoir algorithm selects this candidate.
            record = {
                "split": shard.split,
                "shard": str(shard.path),
                "document_index": doc_index,
                "token_count": len(doc),
                "tokens": doc[:max_tokens_per_document].copy(),
                "tokens_truncated": len(doc) > max_tokens_per_document,
            }
            if selected_index == len(reservoir):
                reservoir.append(record)
            else:
                reservoir[selected_index] = record

    truncated = sum(int(record["tokens_truncated"]) for record in reservoir)
    return reservoir, {
        "documents_seen": seen,
        "reservoir_size": len(reservoir),
        "truncated_documents_in_sample": truncated,
    }


def write_samples(
    shards,
    output_path: Path,
    eos_token_id=50256,
    sample_documents=1000,
    seed=42,
    max_chars=8000,
    max_tokens_per_document=16_384,
    documents=None,
    sampling_metadata=None,
):
    tokenizer = get_tokenizer()

    if documents is None:
        reservoir, metadata = reservoir_documents(
            shards,
            eos_token_id=eos_token_id,
            sample_documents=sample_documents,
            seed=seed,
            max_tokens_per_document=max_tokens_per_document,
        )
    else:
        reservoir = list(documents[:sample_documents])
        metadata = (sampling_metadata or {}) | {
            "reservoir_size": len(reservoir),
            "truncated_documents_in_sample": sum(
                int(record["tokens_truncated"]) for record in reservoir
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as f:
        temporary_path = Path(f.name)
        for record in reservoir:
            item = {key: value for key, value in record.items() if key != "tokens"}
            item["text"] = decode_tokens(tokenizer, record["tokens"], max_chars=max_chars)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temporary_path, output_path)

    return metadata | {"output": str(output_path)}
