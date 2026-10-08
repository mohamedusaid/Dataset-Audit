from __future__ import annotations

import tiktoken


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def decode_tokens(tokenizer, token_ids, max_chars: int | None = None) -> str:
    try:
        text = tokenizer.decode(list(map(int, token_ids)))
    except Exception:
        return ""

    if max_chars is not None:
        return text[:max_chars]
    return text
