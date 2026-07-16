import shap
import matplotlib.pyplot as plt


def _estimator(model):
    """Unwrap a BaseModel wrapper to its underlying estimator if needed."""
    return getattr(model, "model", model)


def get_shap_values(model, X):
    """Compute SHAP values for a fitted tree-based model.

    Returns an array of shape (n_samples, n_features, n_classes) for multiclass.
    """
    explainer = shap.TreeExplainer(_estimator(model))
    return explainer.shap_values(X)


def plot_shap_summary(model, X, class_index, class_name=None):
    """Plot a SHAP summary for one class of a multiclass tree model."""
    shap_values = get_shap_values(model, X)
    shap.summary_plot(shap_values[:, :, class_index], X, show=False)
    title = f"SHAP Summary — {class_name}" if class_name else "SHAP Summary"

    plt.title(title)
    plt.tight_layout()
    plt.show()
