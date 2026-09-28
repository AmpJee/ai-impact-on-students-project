import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


def load_config() -> dict:
    """Load the project config.json."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_model_params(name: str) -> dict:
    """Return the tuned hyperparameters for a model from config.json."""
    return load_config()[name]


def get_xgboost_params() -> dict:
    """Return the tuned XGBoost hyperparameters from config.json."""
    return get_model_params("xgboost")


def get_lightgbm_params() -> dict:
    """Return the tuned LightGBM hyperparameters from config.json."""
    return get_model_params("lightgbm")


def get_catboost_params() -> dict:
    """Return the tuned CatBoost hyperparameters from config.json."""
    return get_model_params("catboost")
