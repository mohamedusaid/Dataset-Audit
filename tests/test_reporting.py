from audit.reporting import build_html, build_markdown


def test_reporting_includes_duplicate_token_metrics_and_renders_html():
    summary = {
        "integrity": [
            {
                "split": "train",
                "path": "train.bin",
                "token_count": 3,
                "min_token_id": 1,
                "max_token_id": 2,
                "invalid_token_count": 0,
                "eos_count": 1,
            }
        ],
        "documents": {
            "by_split": {
                "train": {
                    "documents": 1,
                    "tokens_in_documents": 2,
                    "mean_tokens": 2.0,
                    "median_tokens": 2.0,
                    "p90_tokens": 2.0,
                    "p95_tokens": 2.0,
                    "p99_tokens": 2.0,
                    "max_tokens": 2,
                    "documents_over_sequence_length": 0,
                }
            }
        },
        "duplicates": {
            "documents_hashed": 2,
            "unique_documents": 1,
            "duplicate_groups": 1,
            "duplicate_excess_documents": 1,
            "duplicate_document_fraction": 0.5,
            "duplicate_excess_tokens": 2,
            "duplicate_token_fraction": 0.5,
        },
    }

    markdown = build_markdown(summary)

    assert "Duplicate excess tokens: **2**" in markdown
    assert "<table>" in build_html(markdown)
