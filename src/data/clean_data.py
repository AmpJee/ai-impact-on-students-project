import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = [
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Tool_Diversity",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
]

LEAKAGE_FEATURES = ["Post_Semester_GPA"]

DROP_FEATURES = ["Student_ID"]


def drop_leakage_features(
    df: pd.DataFrame, leakage_features: list[str] = LEAKAGE_FEATURES
) -> pd.DataFrame:
    """Drop features that could lead to data leakage in modeling."""

    df = df.copy()
    df = df.drop(columns=leakage_features, errors="ignore")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate Student_ID rows, if any."""

    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def drop_columns(df: pd.DataFrame, columns: list[str] = DROP_FEATURES) -> pd.DataFrame:
    """Drop specified columns from the DataFrame."""
    df = df.copy()
    df = df.drop(columns=columns, errors="ignore")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that numeric columns are within expected ranges. Raises ValueError if any anomalies are found."""

    not_gpa = df[(df["Pre_Semester_GPA"] < 0) | (df["Pre_Semester_GPA"] > 4)]
    if len(not_gpa) > 0:
        raise ValueError(f"{len(not_gpa)} rows have Pre_Semester_GPA outside [0,4]")

    not_hours = df[(df["Weekly_GenAI_Hours"] < 0) | (df["Traditional_Study_Hours"] < 0)]
    if len(not_hours) > 0:
        raise ValueError(f"{len(not_hours)} rows have negative study hours")

    return df


def add_anomaly_flag(
    df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42, remove=False
) -> pd.DataFrame:
    """Flag unusual student profiles for monitoring or further investigation using Isolation Forest."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ANOMALY_FEATURES])
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df = df.copy()
    df["anomaly_flag"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    df["anomaly_score_raw"] = iso.decision_function(X_scaled)

    if remove:
        df = df[df["anomaly_flag"] == 1].reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_leakage_features(df)
    df = drop_duplicates(df)
    df = validate_ranges(df)
    df = add_anomaly_flag(df, remove=True)
    return df
