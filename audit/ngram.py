from __future__ import annotations

from collections import Counter
import re

from tqdm import tqdm

from .stream import iter_documents
from .tokenizer import get_tokenizer, decode_tokens


def repeated_word_ngrams(
    shards,
    eos_token_id=50256,
    sample_documents=5000,
    n_values=(3, 4, 5),
):
    tokenizer = get_tokenizer()
    counters = {n: Counter() for n in n_values}
    documents = 0

    for shard in tqdm(shards, desc="N-gram repetition"):
        for doc in iter_documents(shard.path, eos_token_id=eos_token_id):
            if documents >= sample_documents:
                break

            text = decode_tokens(tokenizer, doc, max_chars=20000).lower()
            words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

            for n in n_values:
                if len(words) >= n:
                    for i in range(len(words) - n + 1):
                        counters[n][" ".join(words[i:i+n])] += 1

            documents += 1

        if documents >= sample_documents:
            break

    result = {"documents_sampled": documents}

    for n, counter in counters.items():
        repeated = [(g, c) for g, c in counter.most_common(100) if c >= 3]
        result[f"{n}gram_top_repeated"] = [
            {"ngram": g, "count": c} for g, c in repeated
        ]

    return result
