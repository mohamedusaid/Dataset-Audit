import pytest

from audit.tokenizer import TokenDecodeError, decode_tokens


class FailingTokenizer:
    def decode(self, token_ids):
        raise ValueError(f"invalid tokens: {token_ids}")


def test_decode_failure_is_surfaced_to_the_audit_caller():
    with pytest.raises(TokenDecodeError, match="Unable to decode") as error:
        decode_tokens(FailingTokenizer(), [1, 2, 3])

    assert isinstance(error.value.__cause__, ValueError)
