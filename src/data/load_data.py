from pathlib import Path
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ai_student_impact_dataset.csv"
)


def load_data(file_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""

    df = pd.read_csv(file_path)
    return df
