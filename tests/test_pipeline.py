import json

import numpy as np

from audit.config import AuditConfig
from audit.discovery import discover_shards
from audit.pipeline import run_audit


def test_pipeline_writes_reproducible_report_artifacts(tmp_path):
    train = tmp_path / "data" / "train"
    validation = tmp_path / "data" / "val"
    train.mkdir(parents=True)
    validation.mkdir()
    np.array([1, 2, 50256, 3, 50256], dtype="<u2").tofile(train / "train.bin")
    np.array([1, 2, 50256], dtype="<u2").tofile(validation / "val.bin")

    output = tmp_path / "output"
    config = AuditConfig(
        decode=False,
        duplicates=False,
        quality=False,
        ngram=False,
        workers=2,
    ).as_dict()
    result = run_audit(discover_shards(tmp_path / "data"), output, config)

    assert result["total_tokens"] == 8
    summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["leakage"]["exact_train_val_overlap"] == 1
    assert manifest["config"]["workers"] == 2
    assert "<table>" in (output / "audit_report.html").read_text(encoding="utf-8")
