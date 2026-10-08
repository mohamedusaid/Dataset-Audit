from __future__ import annotations

import html
import json
from pathlib import Path


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_markdown(summary):
    integrity = summary["integrity"]
    docs = summary["documents"]
    dup = summary.get("duplicates")
    leakage = summary.get("leakage")
    quality = summary.get("quality")

    lines = [
        "# TinyGPT Dataset Audit Report",
        "",
        "## Executive summary",
        "",
        f"- Shards: **{len(integrity)}**",
        f"- Total tokens: **{sum(x['token_count'] for x in integrity):,}**",
        f"- Invalid token IDs: **{sum(x['invalid_token_count'] for x in integrity):,}**",
        f"- EOS tokens: **{sum(x['eos_count'] for x in integrity):,}**",
        "",
        "## Shards",
        "",
        "| Split | Shard | Tokens | Min ID | Max ID | Invalid | EOS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for x in integrity:
        lines.append(
            f"| {x['split']} | `{x['path']}` | "
            f"{x['token_count']:,} | {x['min_token_id']} | "
            f"{x['max_token_id']} | {x['invalid_token_count']:,} | "
            f"{x['eos_count']:,} |"
        )

    lines += ["", "## Document statistics", ""]
    for split, x in docs["by_split"].items():
        lines += [
            f"### {split}",
            "",
            f"- Documents: **{x['documents']:,}**",
            f"- Tokens in documents: **{x['tokens_in_documents']:,}**",
            f"- Mean length: **{x['mean_tokens']:.2f} tokens**",
            f"- Median: **{x['median_tokens']:.2f}**",
            f"- P90: **{x['p90_tokens']:.2f}**",
            f"- P95: **{x['p95_tokens']:.2f}**",
            f"- P99: **{x['p99_tokens']:.2f}**",
            f"- Maximum: **{x['max_tokens']:,}**",
            "",
        ]

    if dup:
        lines += [
            "## Exact duplication",
            "",
            f"- Documents hashed: **{dup['documents_hashed']:,}**",
            f"- Unique documents: **{dup['unique_documents']:,}**",
            f"- Duplicate groups: **{dup['duplicate_groups']:,}**",
            f"- Duplicate excess documents: **{dup['duplicate_excess_documents']:,}**",
            f"- Duplicate excess fraction: **{dup['duplicate_document_fraction']:.4%}**",
            "",
        ]

    if leakage:
        lines += [
            "## Train/validation leakage",
            "",
            f"- Train unique hashes: **{leakage['train_unique_hashes']:,}**",
            f"- Validation unique hashes: **{leakage['val_unique_hashes']:,}**",
            f"- Exact overlap: **{leakage['exact_train_val_overlap']:,}**",
            f"- Validation overlap fraction: **{leakage['val_overlap_fraction']:.4%}**",
            "",
        ]

    if quality:
        lines += ["## Sampled quality heuristics", ""]
        lines.append(f"- Documents sampled: **{quality['documents_sampled']:,}**")
        for k, v in sorted(quality["flag_fraction"].items()):
            lines.append(f"- `{k}`: **{v:.4%}**")
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "This audit distinguishes deterministic integrity from heuristic quality.",
        "A low heuristic flag rate does not prove that the dataset is semantically high quality.",
        "Likewise, a high flag rate does not automatically mean the corresponding documents must be removed.",
        "",
    ]

    return "\n".join(lines)


def build_html(markdown_text):
    # Lightweight HTML wrapper; Markdown remains readable without another dependency.
    escaped = html.escape(markdown_text)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>TinyGPT Dataset Audit</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:40px auto;"
        "padding:0 20px;line-height:1.55;white-space:pre-wrap}</style>"
        f"</head><body>{escaped}</body></html>"
    )


def write_reports(output: Path, summary):
    md = build_markdown(summary)
    (output / "audit_report.md").write_text(md, encoding="utf-8")
    (output / "audit_report.html").write_text(
        build_html(md), encoding="utf-8"
    )
