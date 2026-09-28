from .clean_data import clean
import pandas as pd


def boolean_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to integer (0 and 1)."""
    df = df.copy()
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset: clean and prepare for modeling."""
    prep_df = clean(df)
    prep_df = boolean_to_int(prep_df)

    return prep_df
