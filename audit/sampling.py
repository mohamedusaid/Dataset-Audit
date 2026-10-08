from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from .stream import iter_documents
from .tokenizer import get_tokenizer, decode_tokens


def write_samples(
    shards,
    output_path: Path,
    eos_token_id=50256,
    sample_documents=1000,
    seed=42,
    max_chars=8000,
):
    rng = random.Random(seed)
    tokenizer = get_tokenizer()

    reservoir = []
    seen = 0

    for shard in tqdm(shards, desc="Sampling"):
        for doc_index, doc in enumerate(
            iter_documents(shard.path, eos_token_id=eos_token_id)
        ):
            if not len(doc):
                continue

            item = {
                "split": shard.split,
                "shard": str(shard.path),
                "document_index": doc_index,
                "token_count": len(doc),
                "text": decode_tokens(tokenizer, doc, max_chars=max_chars),
            }

            seen += 1
            if len(reservoir) < sample_documents:
                reservoir.append(item)
            else:
                j = rng.randrange(seen)
                if j < sample_documents:
                    reservoir[j] = item

    with output_path.open("w", encoding="utf-8") as f:
        for item in reservoir:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "reservoir_size": len(reservoir),
        "documents_seen": seen,
        "output": str(output_path),
    }
