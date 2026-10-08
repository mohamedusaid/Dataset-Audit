from __future__ import annotations

import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import AuditConfig
from .document_stats import document_statistics
from .duplicates import exact_duplicate_analysis
from .integrity import inspect_all
from .leakage import train_validation_leakage
from .ngram import repeated_word_ngrams
from .quality import sampled_quality_analysis
from .reporting import save_json, write_reports
from .sampling import reservoir_documents, write_samples
from .token_stats import token_statistics

log = logging.getLogger(__name__)


def run_audit(shards, output: Path, config: dict):
    config = AuditConfig.from_mapping(config).as_dict()
    output.mkdir(parents=True, exist_ok=True)

    integrity = inspect_all(
        shards,
        vocab_size=config["vocab_size"],
        eos_token_id=config["eos_token_id"],
        compute_sha256=True,
        workers=config["workers"],
    )
    integrity_dict = [x.as_dict() for x in integrity]
    save_json(output / "shard_stats.json", integrity_dict)

    token_stats = token_statistics(
        shards,
        vocab_size=config["vocab_size"],
        eos_token_id=config["eos_token_id"],
        workers=config["workers"],
    )
    save_json(output / "token_stats.json", token_stats)

    docs = document_statistics(
        shards,
        eos_token_id=config["eos_token_id"],
        min_document_tokens=config["min_document_tokens"],
        max_document_tokens=config["max_document_tokens"],
        sequence_length=config["sequence_length"],
    )
    save_json(output / "document_stats.json", docs)

    summary = {
        "config": config,
        "integrity": integrity_dict,
        "token_stats": token_stats,
        "documents": docs,
    }

    sampled_documents = None
    sampling_metadata = None
    if config.get("decode", True):
        shared_sample_size = max(
            config["sample_documents"] if config.get("quality", True) else 0,
            config["ngram_sample_documents"] if config.get("ngram", True) else 0,
            min(config["sample_documents"], 5000),
        )
        sampled_documents, sampling_metadata = reservoir_documents(
            shards,
            eos_token_id=config["eos_token_id"],
            sample_documents=shared_sample_size,
            seed=config["seed"],
            max_tokens_per_document=config["max_sample_tokens_per_document"],
        )

    if config.get("duplicates", True):
        dup = exact_duplicate_analysis(
            shards,
            eos_token_id=config["eos_token_id"],
            max_document_tokens=config["max_document_tokens"],
        )
        summary["duplicates"] = dup
        save_json(output / "duplicate_stats.json", dup)

    if any(s.split == "train" for s in shards) and any(s.split == "val" for s in shards):
        leakage = train_validation_leakage(
            shards,
            eos_token_id=config["eos_token_id"],
            max_document_tokens=config["max_document_tokens"],
        )
        summary["leakage"] = leakage
        save_json(output / "leakage_stats.json", leakage)

    if config.get("quality", True) and config.get("decode", True):
        quality = sampled_quality_analysis(
            shards,
            eos_token_id=config["eos_token_id"],
            sample_documents=config["sample_documents"],
            max_decoded_chars=config["max_decoded_chars"],
            seed=config["seed"],
            max_tokens_per_document=config["max_sample_tokens_per_document"],
            documents=sampled_documents,
            sampling_metadata=sampling_metadata,
        )
        summary["quality"] = quality
        save_json(output / "quality_stats.json", quality)

    if config.get("ngram", True) and config.get("decode", True):
        ngram = repeated_word_ngrams(
            shards,
            eos_token_id=config["eos_token_id"],
            sample_documents=config["ngram_sample_documents"],
            seed=config["seed"],
            max_tokens_per_document=config["max_sample_tokens_per_document"],
            documents=sampled_documents,
            sampling_metadata=sampling_metadata,
        )
        summary["ngram"] = ngram
        save_json(output / "ngram_stats.json", ngram)

    if config.get("decode", True):
        samples = write_samples(
            shards,
            output / "samples.jsonl",
            eos_token_id=config["eos_token_id"],
            sample_documents=min(config["sample_documents"], 5000),
            max_chars=config["max_decoded_chars"],
            seed=config["seed"],
            max_tokens_per_document=config["max_sample_tokens_per_document"],
            documents=sampled_documents,
            sampling_metadata=sampling_metadata,
        )
        summary["samples"] = samples

    save_json(output / "audit_summary.json", summary)
    write_reports(output, summary)

    save_json(
        output / "run_manifest.json",
        {
            "tool": "tinygpt-dataset-audit",
            "version": __version__,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "config": config,
            "shards": integrity_dict,
        },
    )

    return {
        "output": str(output),
        "shards": len(shards),
        "total_tokens": sum(x["token_count"] for x in integrity_dict),
        "invalid_token_ids": sum(x["invalid_token_count"] for x in integrity_dict),
        "report": str(output / "audit_report.md"),
    }
