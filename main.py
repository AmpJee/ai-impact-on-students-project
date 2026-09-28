from src.data import load_data, preprocess_data, split_data_with_stratification
from src.features import engineer_features, select_features
from src.services.model_service import get_model
from src.services.shap_service import plot_shap_summary

BURNOUT_ORDER = ["Low", "Medium", "High"]

# Change the model -> "xgboost", "lightgbm", or "catboost".
MODEL_NAME = "xgboost"


def run_pipeline(model_name: str = MODEL_NAME, interpret: bool = True):
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data_with_stratification(df)

    cols = select_features(X_train, y_train, k=26)
    X_train, X_test = X_train[cols], X_test[cols]

    model = get_model(model_name)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test, target_names=BURNOUT_ORDER)
    print(
        f"{model_name}\n"
        f"Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}"
    )
    if interpret:
        plot_shap_summary(model, X_test, class_index=2, class_name="High Burnout Risk")
    return model


if __name__ == "__main__":
    run_pipeline(model_name=MODEL_NAME)
