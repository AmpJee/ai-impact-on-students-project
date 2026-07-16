from lightgbm import LGBMClassifier

from ..services.load_config import get_lightgbm_params
from .base_model import BaseModel


class LightGBMBurnoutModel(BaseModel):
    def __init__(self, **params):
        default_params = get_lightgbm_params()
        default_params.update(params)
        self.model = LGBMClassifier(**default_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
