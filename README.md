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
8. repeated/low-information documents
9. character/token quality heuristics
10. language heuristics on sampled documents
11. repeated n-gram statistics
12. train/validation leakage at document-hash level
13. sampled decoded examples
14. JSON + Markdown + HTML reports

The project is intentionally modular so expensive checks can be enabled/disabled independently.

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

If the ZIP is directly downloadable from Hugging Face:

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

This runs the full non-neural audit while keeping memory usage bounded.

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

Because `50256` is used as EOS, this auditor reconstructs approximate document boundaries by splitting the token stream on EOS. That is appropriate for this dataset format, but any document-level metric depends on EOS being used consistently.

No claim of semantic quality can be made from token statistics alone. The report therefore separates:

- deterministic integrity checks
- statistical quality checks
- heuristic quality checks
- sampled text checks

## Output

```text
audit_results/
├── audit_report.md
├── audit_report.html
├── audit_summary.json
├── shard_stats.json
├── token_stats.json
├── document_stats.json
├── duplicate_stats.json
├── leakage_stats.json
├── quality_stats.json
├── samples.jsonl
└── logs/
```

## Architecture

```text
audit_cli.py
    |
    +-- archive.py
    +-- discovery.py
    +-- tokenizer.py
    +-- stream.py
    +-- integrity.py
    +-- token_stats.py
    +-- document_stats.py
    +-- duplicates.py
    +-- leakage.py
    +-- quality.py
    +-- sampling.py
    +-- reporting.py
    |
    +-- report/
         +-- markdown.py
         +-- html.py
```

The implementation is intentionally independent from the TinyGPT training code.
