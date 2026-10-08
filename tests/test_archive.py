import zipfile

import pytest

import audit.archive
from audit.archive import prepare_input


def test_prepare_input_extracts_a_safe_archive_and_reuses_matching_cache(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("data/train/train_000.bin", b"\x01\x00")

    destination = tmp_path / "prepared"
    extracted = prepare_input(archive, destination)

    assert extracted == destination
    extracted_file = destination / "data" / "train" / "train_000.bin"
    assert extracted_file.read_bytes() == b"\x01\x00"
    assert prepare_input(archive, destination) == destination


def test_prepare_input_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../escape.bin", b"bad")

    with pytest.raises(ValueError, match="Unsafe archive member path"):
        prepare_input(archive, tmp_path / "prepared")

    assert not (tmp_path / "escape.bin").exists()


def test_prepare_input_publishes_only_a_complete_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("data/train/train_000.bin", b"\x01\x00")

    destination = tmp_path / "prepared"

    def interrupted_extract(_archive, _entries, temporary_destination):
        (temporary_destination / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(audit.archive, "_safe_extract", interrupted_extract)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare_input(archive, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".prepared.extract-*"))
