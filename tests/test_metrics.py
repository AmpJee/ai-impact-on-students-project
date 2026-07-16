from src.models.metrics import classification_metrics


def test_classification_metrics_returns_expected_keys():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    m = classification_metrics(y_true, y_pred)
    assert set(m) == {"accuracy", "macro_f1", "report"}
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_classification_metrics_perfect_prediction():
    y = [0, 1, 2]
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_classification_metrics_uses_target_names():
    m = classification_metrics([0, 1, 2], [0, 1, 2], target_names=["Low", "Medium", "High"])
    assert "Low" in m["report"]
