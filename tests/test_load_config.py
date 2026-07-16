from src.services.load_config import (
    get_catboost_params,
    get_lightgbm_params,
    get_xgboost_params,
    load_config,
)


def test_load_config_has_model_sections():
    config = load_config()
    assert {"xgboost", "lightgbm", "catboost"} <= set(config)


def test_model_param_getters_return_hyperparameters():
    assert "n_estimators" in get_xgboost_params()
    assert "n_estimators" in get_lightgbm_params()
    assert "iterations" in get_catboost_params()
