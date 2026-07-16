import pandas as pd

from src.data.preprocess import boolean_to_int


def test_boolean_to_int_converts_bool_columns(raw_df):
    out = boolean_to_int(raw_df)
    assert pd.api.types.is_integer_dtype(out["Paid_Subscription"])


def test_preprocess_leaves_no_boolean_columns(clean_df):
    assert clean_df.select_dtypes(include="bool").empty
