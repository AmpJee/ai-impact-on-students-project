from xgboost import XGBClassifier
from .base_model import BaseModel


class XGBoostBurnoutModel(BaseModel):
    def __init__(self, **params):
        default_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "n_estimators": 370,
            "learning_rate": 0.03275494231675825,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 0.8995177458059798,
            "colsample_bytree": 0.6003744287092915,
            "reg_alpha": 0.19136141848829258,
            "reg_lambda": 4.1037113192886325,
        }
        default_params.update(params)
        self.model = XGBClassifier(**default_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
