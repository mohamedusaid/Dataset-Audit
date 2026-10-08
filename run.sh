#!/usr/bin/env bash
set -euo pipefail

ZIP_URL="https://huggingface.co/Usaidddddddddddddd/TinyGPT-500M-Archive/resolve/main/data_shards_download.zip"
ZIP_PATH="/content/data_shards_download.zip"
OUTPUT="/content/audit_results"

echo "== TinyGPT Dataset Audit =="
echo "Downloading dataset archive..."

wget -c -O "$ZIP_PATH" "$ZIP_URL"

python -m pip install -q -r requirements.txt

python -m audit_cli \
  --data "$ZIP_PATH" \
  --output "$OUTPUT" \
  --sample-documents 20000 \
  --ngram-sample-documents 5000 \
  --workers 2

echo
echo "Audit complete:"
echo "$OUTPUT/audit_report.md"
echo "$OUTPUT/audit_report.html"
