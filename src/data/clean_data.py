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


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate Student_ID rows, if any."""

    return df.drop_duplicates(subset=["Student_ID"]).reset_index(drop=True)


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
    df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42
) -> pd.DataFrame:
    """Flag unusual student profiles for monitoring or further investigation using Isolation Forest."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ANOMALY_FEATURES])
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df = df.copy()
    df["anomaly_flag"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    df["anomaly_score_raw"] = iso.decision_function(X_scaled)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping duplicates, validating ranges, and adding anomaly flags."""

    df = drop_duplicates(df)
    df = validate_ranges(df)
    df = add_anomaly_flag(df)
    return df
