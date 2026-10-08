from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np
from tqdm import tqdm

from .stream import iter_documents
from .tokenizer import get_tokenizer, decode_tokens


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
):
    tokenizer = get_tokenizer()

    total = 0
    flagged = Counter()
    examples = []

    for shard in tqdm(shards, desc="Quality sample"):
        for doc_index, doc in enumerate(
            iter_documents(shard.path, eos_token_id=eos_token_id)
        ):
            if total >= sample_documents:
                break

            text = decode_tokens(tokenizer, doc, max_decoded_chars)
            metrics = text_metrics(text)
            flags = heuristic_quality(metrics, len(doc))

            total += 1
            flagged.update(flags)

            if flags and len(examples) < 100:
                examples.append(
                    {
                        "split": shard.split,
                        "shard": str(shard.path),
                        "document_index": doc_index,
                        "tokens": len(doc),
                        "flags": flags,
                        "text": text[:2000],
                    }
                )

        if total >= sample_documents:
            break

    return {
        "documents_sampled": total,
        "flag_counts": dict(flagged),
        "flag_fraction": {
            k: v / total if total else 0.0 for k, v in flagged.items()
        },
        "examples": examples,
    }
