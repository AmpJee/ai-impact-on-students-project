from src.data import load_data, clean, split_data, preprocess_data
from src.features import engineer_features


def run_pipeline():
    df_raw = load_data()
    df = clean(df_raw)

    train_df, test_df = split_data(df)

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = run_pipeline()
    print(f"Training: {train_df.shape}")
    print(train_df.head())
    print(f"Testing: {test_df.shape}")
