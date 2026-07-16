from pathlib import Path

import pytest

from src.data.load_data import DEFAULT_DATA_PATH, load_data


@pytest.mark.skipif(
    not Path(DEFAULT_DATA_PATH).exists(),
    reason="dataset CSV is gitignored and not available in CI",
)
def test_load_data_returns_nonempty_dataframe():
    df = load_data()
    assert df.shape[0] > 0
    assert "Burnout_Risk_Level" in df.columns
