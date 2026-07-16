from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(y_true, y_pred, target_names=None) -> dict:
    """Standard multiclass metrics shared across models."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        ),
    }
