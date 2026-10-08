import numpy as np

import audit.ngram
from audit.ngram import repeated_word_ngrams


def test_ngram_analysis_reports_repeated_sequences(monkeypatch):
    monkeypatch.setattr(audit.ngram, "get_tokenizer", lambda: object())
    monkeypatch.setattr(
        audit.ngram,
        "decode_tokens",
        lambda _tokenizer, _tokens, max_chars: "alpha beta gamma delta alpha beta gamma delta",
    )
    documents = [
        {
            "tokens": np.array([1], dtype="<u2"),
            "tokens_truncated": False,
        }
        for _ in range(3)
    ]

    result = repeated_word_ngrams(
        [],
        documents=documents,
        sampling_metadata={"documents_seen": 3},
        n_values=(3,),
    )

    repeated = result["3gram_top_repeated"]
    assert {entry["ngram"] for entry in repeated} >= {"alpha beta gamma"}
