from abc import ABC, abstractmethod
import pandas as pd
from .metrics import classification_metrics


class BaseModel(ABC):
    """Common interface so the burnout classifier can be trained and evaluated
    the same way regardless of underlying algorithm."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame): ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame): ...

    def evaluate(self, X_test: pd.DataFrame, y_test, target_names=None) -> dict:
        y_pred = self.predict(X_test)
        return classification_metrics(y_test, y_pred, target_names=target_names)
