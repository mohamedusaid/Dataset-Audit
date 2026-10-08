from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path


def _atomic_write_text(path: Path, content: str) -> None:
    """Write complete report artifacts or leave the previous version intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def save_json(path: Path, data):
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


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
            f"- Longer than configured sequence length: "
            f"**{x['documents_over_sequence_length']:,}**",
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
        "Likewise, a high flag rate does not automatically mean the corresponding "
        "documents must be removed.",
        "",
    ]

    return "\n".join(lines)


def _inline_html(value: str) -> str:
    escaped = html.escape(value)
    return escaped.replace("**", "<strong>", 1).replace("**", "</strong>", 1)


def build_html(markdown_text):
    """Render the controlled report Markdown as semantic HTML without extra deps."""
    rendered: list[str] = []
    in_list = False
    in_table = False

    def close_open_blocks() -> None:
        nonlocal in_list, in_table
        if in_list:
            rendered.append("</ul>")
            in_list = False
        if in_table:
            rendered.append("</tbody></table>")
            in_table = False

    for line in markdown_text.splitlines():
        if not line:
            close_open_blocks()
            continue
        if line.startswith("### "):
            close_open_blocks()
            rendered.append(f"<h3>{_inline_html(line[4:])}</h3>")
        elif line.startswith("## "):
            close_open_blocks()
            rendered.append(f"<h2>{_inline_html(line[3:])}</h2>")
        elif line.startswith("# "):
            close_open_blocks()
            rendered.append(f"<h1>{_inline_html(line[2:])}</h1>")
        elif line.startswith("- "):
            if in_table:
                close_open_blocks()
            if not in_list:
                rendered.append("<ul>")
                in_list = True
            rendered.append(f"<li>{_inline_html(line[2:])}</li>")
        elif line.startswith("|") and line.endswith("|"):
            if in_list:
                close_open_blocks()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if not in_table:
                rendered.append("<table><thead><tr>")
                rendered.extend(f"<th>{_inline_html(cell)}</th>" for cell in cells)
                rendered.append("</tr></thead><tbody>")
                in_table = True
            else:
                rendered.append("<tr>")
                rendered.extend(f"<td>{_inline_html(cell)}</td>" for cell in cells)
                rendered.append("</tr>")
        else:
            close_open_blocks()
            rendered.append(f"<p>{_inline_html(line)}</p>")

    close_open_blocks()
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>TinyGPT Dataset Audit</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:40px auto;"
        "padding:0 20px;line-height:1.55}table{border-collapse:collapse;"
        "width:100%;overflow:auto}th,td{border:1px solid #ddd;padding:8px;"
        "text-align:left}th{background:#f6f8fa}</style>"
        f"</head><body>{''.join(rendered)}</body></html>"
    )


def write_reports(output: Path, summary):
    md = build_markdown(summary)
    _atomic_write_text(output / "audit_report.md", md)
    _atomic_write_text(output / "audit_report.html", build_html(md))
