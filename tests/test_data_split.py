from src.data.data_split import split_data, split_data_with_stratification


def test_split_data_preserves_row_count(engineered_df):
    train, test = split_data(engineered_df, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(engineered_df)


def test_stratified_split_excludes_target_from_features(engineered_df):
    X_train, X_test, y_train, y_test = split_data_with_stratification(engineered_df)
    assert "Burnout_Risk_Level_Enc" not in X_train.columns
    assert "Burnout_Risk_Level_Enc" not in X_test.columns
    assert y_train.name == "Burnout_Risk_Level_Enc"


def test_stratified_split_sizes_are_consistent(engineered_df):
    X_train, X_test, y_train, y_test = split_data_with_stratification(
        engineered_df, test_size=0.2
    )
    assert len(X_train) + len(X_test) == len(engineered_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_stratified_split_preserves_class_distribution(engineered_df):
    _, _, y_train, y_test = split_data_with_stratification(engineered_df, test_size=0.2)
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    for cls in train_dist.index:
        assert abs(train_dist[cls] - test_dist.get(cls, 0)) < 0.15
