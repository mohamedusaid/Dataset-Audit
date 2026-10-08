from __future__ import annotations

import re
from collections import Counter

from .sampling import reservoir_documents
from .tokenizer import decode_tokens, get_tokenizer

URL_RE = re.compile(r"https?://|www\.", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
HTML_RE = re.compile(r"<[^>]{1,500}>")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
REPEATED_CHAR_RE = re.compile(r"(.)\1{7,}")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{4,}")


def text_metrics(text: str):
    chars = len(text)
    if chars == 0:
        return {
            "chars": 0,
            "words": 0,
            "alpha_fraction": 0.0,
            "digit_fraction": 0.0,
            "whitespace_fraction": 0.0,
            "printable_fraction": 0.0,
            "url": False,
            "email": False,
            "html": False,
            "repeated_chars": False,
            "repeated_punctuation": False,
        }

    words = WORD_RE.findall(text)
    alpha = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    whitespace = sum(c.isspace() for c in text)
    printable = sum(c.isprintable() for c in text)

    return {
        "chars": chars,
        "words": len(words),
        "alpha_fraction": alpha / chars,
        "digit_fraction": digits / chars,
        "whitespace_fraction": whitespace / chars,
        "printable_fraction": printable / chars,
        "url": bool(URL_RE.search(text)),
        "email": bool(EMAIL_RE.search(text)),
        "html": bool(HTML_RE.search(text)),
        "repeated_chars": bool(REPEATED_CHAR_RE.search(text)),
        "repeated_punctuation": bool(REPEATED_PUNCT_RE.search(text)),
    }


def heuristic_quality(metrics, token_count):
    """
    Not a trained quality model.
    This produces flags only; it does not claim semantic quality.
    """
    flags = []

    if token_count < 8:
        flags.append("very_short")
    if metrics["chars"] < 40:
        flags.append("very_short_text")
    if metrics["printable_fraction"] < 0.90:
        flags.append("many_nonprintable")
    if metrics["alpha_fraction"] < 0.03 and metrics["digit_fraction"] < 0.70:
        flags.append("low_alphabetic_content")
    if metrics["repeated_chars"]:
        flags.append("repeated_characters")
    if metrics["repeated_punctuation"]:
        flags.append("repeated_punctuation")
    if metrics["html"]:
        flags.append("html_like")
    if metrics["email"]:
        flags.append("contains_email")
    if metrics["url"]:
        flags.append("contains_url")

    return flags


def sampled_quality_analysis(
    shards,
    eos_token_id=50256,
    sample_documents=20000,
    max_decoded_chars=8000,
    seed=42,
    max_tokens_per_document=16_384,
    documents=None,
    sampling_metadata=None,
):
    tokenizer = get_tokenizer()

    flagged = Counter()
    examples = []

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
        text = decode_tokens(tokenizer, document["tokens"], max_decoded_chars)
        metrics = text_metrics(text)
        flags = heuristic_quality(metrics, document["token_count"])

        flagged.update(flags)

        if flags and len(examples) < 100:
            examples.append(
                {
                    "split": document["split"],
                    "shard": document["shard"],
                    "document_index": document["document_index"],
                    "tokens": document["token_count"],
                    "tokens_truncated": document["tokens_truncated"],
                    "flags": flags,
                    "text": text[:2000],
                }
            )

    return {
        "documents_sampled": len(documents),
        "sampling": sampling_metadata,
        "flag_counts": dict(flagged),
        "flag_fraction": {k: v / len(documents) if documents else 0.0 for k, v in flagged.items()},
        "examples": examples,
    }
