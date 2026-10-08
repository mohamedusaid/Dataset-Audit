from __future__ import annotations

import tiktoken


class TokenDecodeError(ValueError):
    """Raised when a token sequence cannot be decoded by the configured tokenizer."""


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def decode_tokens(tokenizer, token_ids, max_chars: int | None = None) -> str:
    try:
        text = tokenizer.decode(list(map(int, token_ids)))
    except Exception as exc:
        raise TokenDecodeError("Unable to decode token sequence for audit analysis") from exc

    if max_chars is not None:
        return text[:max_chars]
    return text
