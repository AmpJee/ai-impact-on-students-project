import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> dict:
    """Load the project config.json."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_xgboost_params() -> dict:
    """Return the tuned XGBoost hyperparameters from config.json."""
    return load_config()["xgboost"]
