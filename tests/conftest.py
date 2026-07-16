import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from src.features import engineer_features


@pytest.fixture
def raw_df():
    """Small synthetic dataset with the same schema as the real CSV."""
    rng = np.random.default_rng(42)
    n = 90
    return pd.DataFrame(
        {
            "Student_ID": range(1, n + 1),
            "Major_Category": np.tile(
                ["STEM", "Business", "Arts", "Humanities", "Medical"], n // 5
            ),
            "Year_of_Study": np.tile(
                ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"], n // 5
            ),
            "Pre_Semester_GPA": rng.uniform(2.0, 4.0, n).round(3),
            "Weekly_GenAI_Hours": rng.uniform(0.0, 30.0, n).round(2),
            "Primary_Use_Case": np.tile(
                [
                    "Copywriting/Drafting",
                    "Ideation",
                    "Summarizing_Reading",
                    "Debugging/Troubleshooting",
                    "Direct_Answer_Generation",
                ],
                n // 5,
            ),
            "Prompt_Engineering_Skill": np.tile(
                ["Beginner", "Intermediate", "Advanced"], n // 3
            ),
            "Tool_Diversity": rng.integers(1, 6, n),
            "Paid_Subscription": rng.integers(0, 2, n).astype(bool),
            "Traditional_Study_Hours": rng.uniform(1.0, 20.0, n).round(2),
            "Perceived_AI_Dependency": rng.integers(1, 6, n),
            "Institutional_Policy": np.tile(
                ["Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged"], n // 3
            ),
            "Anxiety_Level_During_Exams": rng.integers(1, 11, n),
            "Post_Semester_GPA": rng.uniform(2.0, 4.0, n).round(3),
            "Skill_Retention_Score": rng.uniform(50.0, 100.0, n).round(2),
            "Burnout_Risk_Level": np.tile(["Low", "Medium", "High"], n // 3),
        }
    )


@pytest.fixture
def clean_df(raw_df):
    """Cleaned + preprocessed frame (still has the raw categorical/target columns)."""
    return preprocess_data(raw_df)


@pytest.fixture
def engineered_df(clean_df):
    """Fully engineered frame ready for modelling (target = Burnout_Risk_Level_Enc)."""
    return engineer_features(clean_df)
