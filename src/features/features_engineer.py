import pandas as pd

NOMINAL_FEATURES = [
    "Major_Category",
    "Year_of_Study",
    "Primary_Use_Case",
    "Institutional_Policy",
]


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Effort_Hours"] = df["Weekly_GenAI_Hours"] + df["Traditional_Study_Hours"]
    df["GenAI_to_Traditional_Ratio"] = df["Weekly_GenAI_Hours"] / (
        df["Traditional_Study_Hours"] + 1e-6
    )
    return df


def add_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ORDINAL_MAP = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    df["Prompt_Skill_Ordinal"] = df["Prompt_Engineering_Skill"].map(ORDINAL_MAP)
    df = df.drop(columns=["Prompt_Engineering_Skill"])
    return df


def one_hot_encode(
    df: pd.DataFrame, columns: list[str] = NOMINAL_FEATURES
) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ratio_features(df)
    df = add_ordinal_features(df)
    df = one_hot_encode(df)
    return df
