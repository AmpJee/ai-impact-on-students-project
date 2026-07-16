import pandas as pd
import pytest

from src.data.clean_data import (
    add_anomaly_flag,
    clean,
    drop_duplicates,
    drop_leakage_features,
    validate_ranges,
)


def test_drop_leakage_removes_post_semester_gpa(raw_df):
    out = drop_leakage_features(raw_df)
    assert "Post_Semester_GPA" not in out.columns


def test_drop_duplicates_removes_exact_duplicates(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    out = drop_duplicates(dup)
    assert len(out) == len(raw_df)


def test_validate_ranges_passes_on_valid_data(raw_df):
    # Should not raise.
    validate_ranges(raw_df)


def test_validate_ranges_raises_on_out_of_range_gpa(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Pre_Semester_GPA"] = 5.0
    with pytest.raises(ValueError):
        validate_ranges(bad)


def test_validate_ranges_raises_on_negative_hours(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Weekly_GenAI_Hours"] = -1.0
    with pytest.raises(ValueError):
        validate_ranges(bad)


def test_add_anomaly_flag_can_remove_rows(raw_df):
    out = add_anomaly_flag(raw_df, remove=True)
    assert "anomaly_flag" in out.columns
    assert len(out) <= len(raw_df)


def test_clean_drops_leakage_and_helper_columns(raw_df):
    out = clean(raw_df)
    for col in ["Post_Semester_GPA", "Student_ID", "anomaly_flag", "anomaly_score_raw"]:
        assert col not in out.columns
    # target and categoricals are kept for later encoding
    assert "Burnout_Risk_Level" in out.columns
