from .clean_data import clean
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset: clean and prepare for modeling."""
    prep_df = clean(df)
    return prep_df
