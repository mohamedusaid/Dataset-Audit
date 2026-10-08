import numpy as np

from audit.duplicates import exact_duplicate_analysis
from audit.models import Shard


def test_duplicate_token_mass_is_reported(tmp_path):
    path = tmp_path / "train.bin"
    np.array(
        [1, 2, 3, 50256, 1, 2, 3, 50256, 1, 2, 3, 50256, 9, 50256],
        dtype="<u2",
    ).tofile(path)

    result = exact_duplicate_analysis([Shard(path, "train", path.stat().st_size)])

    assert result["duplicate_excess_documents"] == 2
    assert result["duplicate_excess_tokens"] == 6
    assert result["tokens_hashed"] == 10
    assert result["duplicate_token_fraction"] == 0.6
