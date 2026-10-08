# TinyGPT Dataset Audit

A standalone, streaming-first audit toolkit for TinyGPT tokenized datasets.

Designed for datasets stored as headerless raw GPT-2 token IDs:

- dtype: `uint16`
- valid token IDs: `0..50256`
- vocabulary: GPT-2 BPE, 50,257 tokens
- EOS/document boundary token: `50256`
- training sequence length: 1024
- shards can be processed with NumPy `memmap` without loading the whole dataset into RAM

## Goals

This project audits the dataset before another TinyGPT training run.

It checks:

1. shard integrity
2. token ID validity
3. token counts
4. token-frequency statistics
5. EOS/document statistics
6. document-length distribution
7. exact duplicate documents
8. sampled low-information quality signals
9. character/token quality heuristics
10. repeated n-gram statistics
11. train/validation leakage at document-hash level
12. sampled decoded examples
13. JSON + Markdown + HTML reports

The project is intentionally modular so expensive checks can be enabled/disabled independently.
It is a dataset diagnostic tool, not a model trainer: its results cannot guarantee
downstream model quality, safety, or benchmark performance.

## Installation

For normal use:

```bash
python -m pip install -r requirements.txt
```

For a reproducible editable development setup, including tests and linting:

```bash
python -m pip install -e ".[dev]"
```

## Expected input

Example:

```text
data_shards/
├── train/
│   ├── train_shard_0000.bin
│   ├── train_shard_0001.bin
│   └── ...
└── val/
    └── val_shard_0000.bin
```

It also accepts a flat directory or a ZIP archive.

## Google Colab

```bash
git clone <your-audit-repo>
cd tinygpt-dataset-audit

pip install -r requirements.txt

python -m audit_cli \
  --data /content/data_shards_download.zip \
  --output /content/audit_results \
  --workers 2
```

If the ZIP is directly downloadable from Hugging Face, this is an example URL
(replace it with the archive you intend to audit):

```bash
wget -O /content/data_shards_download.zip \
  https://huggingface.co/Usaidddddddddddddd/TinyGPT-500M-Archive/resolve/main/data_shards_download.zip
```

Then run the audit.

## Recommended first run

Start with:

```bash
python -m audit_cli \
  --data /content/data_shards_download.zip \
  --output /content/audit_results \
  --sample-documents 20000 \
  --workers 2
```

Memory use grows with the dataset in three places: exact-duplicate detection and
leakage detection keep one SHA-256 hash per unique document (plus a small location
record for duplicates), and document statistics keep one integer length per document.
For a ~135M-token dataset this is a few tens of MB. Token arrays themselves are
streamed through `memmap` and are never held in full.

Use the installed command equivalently:

```bash
tinygpt-audit --data /path/to/data_shards --output ./audit_results
```

For a reproducible audit configuration, copy and edit `config.example.json`:

```bash
python -m audit_cli \
  --data /path/to/data_shards \
  --output ./audit_results \
  --config config.json
```

`--config` deliberately supplies the validated audit-analysis settings from JSON
and takes precedence over audit-specific command-line settings. Input, output,
archive extraction limits, and logging remain command-line options because they
control how the audit is invoked rather than how data is analysed.

For the complete dataset:

```bash
python -m audit_cli \
  --data /content/data_shards_download.zip \
  --output /content/audit_results \
  --sample-documents 100000 \
  --workers 2
```

## Important

The `.bin` files contain token IDs, not original source documents.

Because `50256` is used as EOS, this auditor reconstructs document boundaries by
splitting the token stream on EOS. Documents that straddle consecutive shards of the
**same split** are stitched back together; documents are never joined across splits.
A stitched document is attributed to the shard where it starts.

Counts reconcile as follows, which is a useful sanity check on any run:

- `tokens in documents (all splits) + EOS tokens = total tokens`
- `non-empty documents + empty documents = EOS tokens` when every shard ends with EOS

Document-level metrics depend on EOS being used consistently.

No claim of semantic quality can be made from token statistics alone. The report therefore separates:

- deterministic integrity checks
- statistical quality checks
- heuristic quality checks
- sampled text checks

Quality flags are heuristics, not verdicts:

- `repeated_characters`: 8+ repeats of the same non-whitespace character
- `html_like`: something shaped like a real tag (`<div ...>`, `</p>`), not any `<...>`
- `contains_url`, `contains_email`, `many_nonprintable`, `repeated_punctuation`,
  `very_short`, `low_alphabetic_content`

Flags are measured on a deterministic random sample, and legitimate code or
web-derived text will trigger several of them. Do not delete documents on flags alone.

## Output

```text
audit_results/
├── audit_report.md / audit_report.html
├── audit_summary.json
├── run_manifest.json
├── shard_stats.json, token_stats.json, document_stats.json
├── duplicate_stats.json     (unless --no-duplicates)
├── leakage_stats.json       (only if both train and val shards exist)
├── quality_stats.json       (unless --no-quality or --no-decoding)
├── ngram_stats.json         (unless --no-ngram or --no-decoding)
├── samples.jsonl            (unless --no-decoding)
└── _prepared/               (only when the input is a ZIP)
```

[Dataset audit results (legacy tool v0.1.0)](RESULT_DOCUMENTATION.md) records one
specific prior dataset run. Re-run the current tool before comparing its counts with
that report because v0.3.0 stitches documents across shard boundaries.

## Operational safeguards

- ZIP inputs are checked for path traversal, symlinks, encryption, excessive file
  counts, and excessive decompressed size before extraction.
- Raw shard bytes are read as explicit little-endian `uint16`, making checksums
  consistent across host architectures.
- Independent integrity and token-frequency shard scans use `--workers`; report
  ordering remains deterministic.
- Quality, n-gram, and text-example analysis use deterministic reservoir sampling
  (`--seed`) instead of taking the first documents in shard order.
- Sampled document tokens are capped at `--max-sample-tokens-per-document`; reports
  state when a selected document was truncated for sampling.
- Generated JSON and report artifacts are written atomically. `run_manifest.json`
  records the tool version, validated configuration, Python version, completion
  timestamp, and input shard checksums for traceability.
- Shards whose byte size is not a multiple of 2 abort the audit with a clear error
  instead of producing partial statistics.
- Splits are inferred from paths **relative to the data root**, so a parent directory
  named `val` or `test` cannot misclassify every shard.
- Duplicate and leakage analyses apply the same `max_document_tokens` filter.
- Integrity checksums use canonical little-endian bytes.

## Limitations

- Only **exact** duplicates are detected; near-duplicates are not.
- Quality, n-gram, and text checks run on a sample, not the full dataset.
- `duplicate_stats.json` lists only the top 100 duplicate groups. It is a diagnostic,
  not a removal list.
- Leakage compares `train` against `val` only.
- Token statistics cannot establish semantic quality, safety, or benchmark performance.

Run the quality gate locally with:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## Architecture

```text
audit_cli.py
    |
    +-- pipeline.py
    +-- config.py
    +-- archive.py
    +-- discovery.py
    +-- models.py
    +-- tokenizer.py
    +-- stream.py
    +-- hashing.py
    +-- integrity.py
    +-- token_stats.py
    +-- document_stats.py
    +-- duplicates.py
    +-- leakage.py
    +-- quality.py
    +-- sampling.py
    +-- reporting.py
```

The implementation is intentionally independent from the TinyGPT training code.
