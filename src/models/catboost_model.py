from catboost import CatBoostClassifier

from ..services.load_config import get_catboost_params
from .base_model import BaseModel


class CatBoostBurnoutModel(BaseModel):
    def __init__(self, **params):
        default_params = get_catboost_params()
        default_params.update(params)
        self.model = CatBoostClassifier(**default_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X).ravel().astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
