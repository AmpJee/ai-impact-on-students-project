from src.data import load_data


def test_load_data_returns_nonempty_dataframe():
    df = load_data()
    assert df.shape[0] > 0
    assert "Burnout_Risk_Level" in df.columns
