from src.features.features_engineer import (
    add_interactions,
    add_ordinal_features,
    add_ratio_features,
    one_hot_encode,
    select_features,
)


def test_add_ratio_features(clean_df):
    out = add_ratio_features(clean_df)
    assert "Total_Effort_Hours" in out.columns
    assert "GenAI_to_Traditional_Ratio" in out.columns


def test_add_ordinal_features_encodes_target_and_skill(clean_df):
    out = add_ordinal_features(clean_df)
    assert "Burnout_Risk_Level_Enc" in out.columns
    assert "Burnout_Risk_Level" not in out.columns
    assert "Prompt_Skill_Ordinal" in out.columns
    assert set(out["Burnout_Risk_Level_Enc"].unique()) <= {0, 1, 2}


def test_add_interactions(clean_df):
    out = add_interactions(clean_df)
    for col in ["GenAI_x_Dependency", "GenAI_x_Anxiety", "log_GenAI_Hours"]:
        assert col in out.columns


def test_one_hot_encode_expands_nominal_features(clean_df):
    out = one_hot_encode(clean_df)
    assert "Major_Category" not in out.columns
    assert any(c.startswith("Major_Category_") for c in out.columns)


def test_engineer_features_produces_numeric_frame(engineered_df):
    assert "Burnout_Risk_Level_Enc" in engineered_df.columns
    assert engineered_df.select_dtypes(include="object").empty


def test_select_features_returns_k_columns(engineered_df):
    X = engineered_df.drop(columns=["Burnout_Risk_Level_Enc"])
    y = engineered_df["Burnout_Risk_Level_Enc"]
    cols = select_features(X, y, k=5)
    assert len(cols) == 5
    assert set(cols) <= set(X.columns)
