import numpy as np

from audit.models import Shard
from audit.token_stats import token_statistics


def test_token_statistics_counts_valid_ids_and_eos(tmp_path):
    path = tmp_path / "train.bin"
    np.array([0, 1, 2, 5, 5], dtype="<u2").tofile(path)
    shard = Shard(path, "train", path.stat().st_size)

    result = token_statistics([shard], vocab_size=5, eos_token_id=2)

    assert result["total_valid_tokens"] == 3
    assert result["unique_tokens_observed"] == 3
    assert result["eos_count"] == 1
