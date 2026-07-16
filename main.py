from src.data import load_data, preprocess_data, split_data_with_stratification
from src.features import engineer_features
from src.models import XGBoostBurnoutModel

BURNOUT_ORDER = ["Low", "Medium", "High"]


def run_pipeline():
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data_with_stratification(
        df, target_column="Burnout_Risk_Level_Enc", test_size=0.2, random_state=42
    )

    model = XGBoostBurnoutModel()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test, target_names=BURNOUT_ORDER)
    print(
        f"XGBoost \nAccuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}"
    )
    return model


if __name__ == "__main__":
    run_pipeline()
