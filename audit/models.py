from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


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
    min_token_id: Optional[int]
    max_token_id: Optional[int]
    invalid_token_count: int
    eos_count: int
    sha256: Optional[str] = None

    def as_dict(self):
        return asdict(self)
