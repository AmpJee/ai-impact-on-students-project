from src.models import (
    BaseModel,
    CatBoostBurnoutModel,
    LightGBMBurnoutModel,
    XGBoostBurnoutModel,
)

MODELS = {
    "xgboost": XGBoostBurnoutModel,
    "lightgbm": LightGBMBurnoutModel,
    "catboost": CatBoostBurnoutModel,
}


def get_model(name: str, **params) -> BaseModel:
    """Create a model by name (see MODELS). Extra kwargs override its config params."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}.")
    return MODELS[name](**params)
