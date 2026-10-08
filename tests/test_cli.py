import json
import sys

import numpy as np

import audit_cli


def test_cli_uses_validated_json_config(tmp_path, monkeypatch):
    data = tmp_path / "data" / "train"
    data.mkdir(parents=True)
    np.array([1, 50256], dtype="<u2").tofile(data / "train.bin")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "decode": False,
                "duplicates": False,
                "quality": False,
                "ngram": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_cli",
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(output),
            "--config",
            str(config),
        ],
    )

    assert audit_cli.main() == 0
    assert (output / "run_manifest.json").exists()


def test_cli_keeps_archive_limits_outside_the_audit_config(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"decode": False}), encoding="utf-8")
    received = {}

    def capture_prepare_input(path, destination, *, max_members, max_uncompressed_bytes):
        received.update(
            {
                "path": path,
                "destination": destination,
                "max_members": max_members,
                "max_uncompressed_bytes": max_uncompressed_bytes,
            }
        )
        return path

    monkeypatch.setattr(audit_cli, "prepare_input", capture_prepare_input)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_cli",
            "--data",
            str(data),
            "--config",
            str(config),
            "--max-archive-members",
            "7",
            "--max-archive-uncompressed-bytes",
            "99",
        ],
    )

    assert audit_cli.main() == 2
    assert received["max_members"] == 7
    assert received["max_uncompressed_bytes"] == 99
