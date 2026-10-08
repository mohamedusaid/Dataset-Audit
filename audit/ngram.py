from __future__ import annotations

import re
from collections import Counter

from .sampling import reservoir_documents
from .tokenizer import decode_tokens, get_tokenizer


def repeated_word_ngrams(
    shards,
    eos_token_id=50256,
    sample_documents=5000,
    n_values=(3, 4, 5),
    seed=42,
    max_tokens_per_document=16_384,
    documents=None,
    sampling_metadata=None,
):
    tokenizer = get_tokenizer()
    counters = {n: Counter() for n in n_values}
    if documents is None:
        documents, sampling_metadata = reservoir_documents(
            shards,
            eos_token_id=eos_token_id,
            sample_documents=sample_documents,
            seed=seed,
            max_tokens_per_document=max_tokens_per_document,
        )
    else:
        documents = list(documents[:sample_documents])
        sampling_metadata = (sampling_metadata or {}) | {
            "reservoir_size": len(documents),
            "truncated_documents_in_sample": sum(
                int(document["tokens_truncated"]) for document in documents
            ),
        }

    for document in documents:
        text = decode_tokens(tokenizer, document["tokens"], max_chars=20000).lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

        for n in n_values:
            if len(words) >= n:
                for index in range(len(words) - n + 1):
                    counters[n][" ".join(words[index : index + n])] += 1

    result = {
        "documents_sampled": len(documents),
        "sampling": sampling_metadata,
    }

    for n, counter in counters.items():
        repeated = [(g, c) for g, c in counter.most_common(100) if c >= 3]
        result[f"{n}gram_top_repeated"] = [{"ngram": g, "count": c} for g, c in repeated]

    return result
