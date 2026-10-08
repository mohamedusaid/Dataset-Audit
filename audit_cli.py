from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

from audit.archive import prepare_input
from audit.config import AuditConfig
from audit.discovery import discover_shards
from audit.pipeline import run_audit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit TinyGPT raw uint16 GPT-2 token shards.")
    p.add_argument("--data", required=True, help="ZIP file or extracted data directory")
    p.add_argument("--output", default="./audit_results")
    p.add_argument(
        "--config",
        help=(
            "Optional JSON audit configuration. When provided, it replaces "
            "audit-specific CLI settings."
        ),
    )
    p.add_argument("--eos-token-id", type=int, default=50256)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--sample-documents", type=int, default=20000)
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for independent shard integrity and token scans.",
    )
    p.add_argument("--max-decoded-chars", type=int, default=8000)
    p.add_argument("--min-document-tokens", type=int, default=8)
    p.add_argument("--max-document-tokens", type=int, default=100000)
    p.add_argument("--ngram-sample-documents", type=int, default=5000)
    p.add_argument("--max-sample-tokens-per-document", type=int, default=16384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-archive-members", type=int, default=100000)
    p.add_argument(
        "--max-archive-uncompressed-bytes",
        type=int,
        default=100 * 1024**3,
    )
    p.add_argument("--no-decoding", action="store_true")
    p.add_argument("--no-duplicates", action="store_true")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--no-ngram", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        input_path = Path(args.data).expanduser().resolve()
        output = Path(args.output).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        output.mkdir(parents=True, exist_ok=True)

        cli_config = {
            "eos_token_id": args.eos_token_id,
            "vocab_size": args.vocab_size,
            "sequence_length": args.sequence_length,
            "sample_documents": args.sample_documents,
            "workers": args.workers,
            "max_decoded_chars": args.max_decoded_chars,
            "min_document_tokens": args.min_document_tokens,
            "max_document_tokens": args.max_document_tokens,
            "ngram_sample_documents": args.ngram_sample_documents,
            "max_sample_tokens_per_document": args.max_sample_tokens_per_document,
            "seed": args.seed,
            "decode": not args.no_decoding,
            "duplicates": not args.no_duplicates,
            "quality": not args.no_quality,
            "ngram": not args.no_ngram,
        }
        if args.config:
            config_path = Path(args.config).expanduser().resolve()
            with config_path.open(encoding="utf-8") as config_file:
                loaded_config = json.load(config_file)
            if not isinstance(loaded_config, dict):
                raise ValueError("Configuration JSON must contain an object.")
            config = AuditConfig.from_mapping(loaded_config).as_dict()
        else:
            config = AuditConfig.from_mapping(cli_config).as_dict()

        logging.info("Preparing input: %s", input_path)
        data_root = prepare_input(
            input_path,
            output / "_prepared",
            max_members=args.max_archive_members,
            max_uncompressed_bytes=args.max_archive_uncompressed_bytes,
        )

        shards = discover_shards(data_root)
        if not shards:
            logging.error("No .bin shards found under %s", data_root)
            return 2

        logging.info("Discovered %d shard(s)", len(shards))
        for shard in shards:
            logging.info(
                "%s | split=%s | %.2f MB",
                shard.path,
                shard.split,
                shard.size_bytes / 1024**2,
            )

        summary = run_audit(shards, output, config)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logging.error("Audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
