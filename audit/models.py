from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    path: Path
    split: str
    size_bytes: int

    def as_dict(self):
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass
class ShardStats:
    path: str
    split: str
    size_bytes: int
    token_count: int
    min_token_id: int | None
    max_token_id: int | None
    invalid_token_count: int
    eos_count: int
    sha256: str | None = None

    def as_dict(self):
        return asdict(self)
