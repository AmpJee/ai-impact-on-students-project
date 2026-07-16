import numpy as np
import pytest

from src.data.data_split import split_data_with_stratification
from src.models import BaseModel, XGBoostBurnoutModel


@pytest.fixture
def split(engineered_df):
    return split_data_with_stratification(engineered_df)


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel()


def test_fit_returns_self(split):
    X_train, X_test, y_train, y_test = split
    model = XGBoostBurnoutModel()
    assert model.fit(X_train, y_train) is model


def test_predict_returns_valid_labels(split):
    X_train, X_test, y_train, y_test = split
    model = XGBoostBurnoutModel().fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)) <= {0, 1, 2}


def test_predict_proba_shape(split):
    X_train, X_test, y_train, y_test = split
    model = XGBoostBurnoutModel().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)


def test_evaluate_returns_metrics(split):
    X_train, X_test, y_train, y_test = split
    model = XGBoostBurnoutModel().fit(X_train, y_train)
    m = model.evaluate(X_test, y_test, target_names=["Low", "Medium", "High"])
    assert set(m) == {"accuracy", "macro_f1", "report"}
