from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and testing sets."""

    return train_test_split(df, test_size=test_size, random_state=random_state)


def split_data_with_stratification(
    df: pd.DataFrame,
    target_column: str = "Burnout_Risk_Level_Enc",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into stratified train/test sets, keeping the target out of the features."""

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
