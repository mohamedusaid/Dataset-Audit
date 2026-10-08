from __future__ import annotations

import hashlib
import numpy as np


def hash_tokens(tokens: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(tokens, dtype=np.uint16).tobytes())
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
