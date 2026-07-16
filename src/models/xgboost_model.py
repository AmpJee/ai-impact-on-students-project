from xgboost import XGBClassifier

from services.load_config import get_xgboost_params

from .base_model import BaseModel


class XGBoostBurnoutModel(BaseModel):
    def __init__(self, **params):
        default_params = get_xgboost_params()
        default_params.update(params)
        self.model = XGBClassifier(**default_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
