import pytest

from audit.config import AuditConfig


def test_config_rejects_invalid_token_format_and_unknown_settings():
    with pytest.raises(ValueError, match="within the configured vocabulary"):
        AuditConfig.from_mapping({"vocab_size": 10, "eos_token_id": 10})

    with pytest.raises(ValueError, match="Unknown configuration"):
        AuditConfig.from_mapping({"unrecognised": True})


def test_config_rejects_invalid_document_bounds():
    with pytest.raises(ValueError, match="max_document_tokens"):
        AuditConfig.from_mapping({"min_document_tokens": 10, "max_document_tokens": 9})
