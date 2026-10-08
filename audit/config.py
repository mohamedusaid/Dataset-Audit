"""Validated configuration for a reproducible audit run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class AuditConfig:
    eos_token_id: int = 50256
    vocab_size: int = 50257
    sequence_length: int = 1024
    sample_documents: int = 20_000
    ngram_sample_documents: int = 5_000
    max_decoded_chars: int = 8_000
    min_document_tokens: int = 8
    max_document_tokens: int = 100_000
    max_sample_tokens_per_document: int = 16_384
    workers: int = 2
    seed: int = 42
    decode: bool = True
    duplicates: bool = True
    quality: bool = True
    ngram: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AuditConfig:
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(sorted(unknown))}")
        config = cls(**dict(values))
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.vocab_size <= 65_536:
            raise ValueError("vocab_size must be between 1 and 65,536 for uint16 shards")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be within the configured vocabulary")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if self.min_document_tokens < 0:
            raise ValueError("min_document_tokens cannot be negative")
        if self.max_document_tokens < self.min_document_tokens:
            raise ValueError("max_document_tokens must be >= min_document_tokens")
        if self.max_sample_tokens_per_document <= 0:
            raise ValueError("max_sample_tokens_per_document must be positive")
        if self.max_decoded_chars <= 0:
            raise ValueError("max_decoded_chars must be positive")
        if self.sample_documents < 0 or self.ngram_sample_documents < 0:
            raise ValueError("sample document counts cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
