import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
import torch

from neural_ode import generate_synthetic_data

def test_synthetic_generation_schema_counts_and_reproducibility():
    train, test = generate_synthetic_data(
        n_train=8, n_test=5, n_classes=3, n_features=2, length=4,
        irregularity=0.3, noise=0.1, missing_rate=0.3, seed=13,
    )
    train_again, test_again = generate_synthetic_data(
        n_train=8, n_test=5, n_classes=3, n_features=2, length=4,
        irregularity=0.3, noise=0.1, missing_rate=0.3, seed=13,
    )
    expected_columns = ["trajectory_id", "time", "feature_0", "feature_1", "label"]

    assert train.columns.tolist() == expected_columns
    assert test.columns.tolist() == expected_columns
    assert len(train) == 8 * 4
    assert len(test) == 5 * 4
    assert train["trajectory_id"].str.startswith("train_").all()
    assert test["trajectory_id"].str.startswith("test_").all()
    pd.testing.assert_frame_equal(train, train_again)
    pd.testing.assert_frame_equal(test, test_again)
    assert train[["feature_0", "feature_1"]].isna().any().any()
    different_seed, _ = generate_synthetic_data(
        n_train=8, n_test=5, n_classes=3, n_features=2, length=4,
        irregularity=0.3, noise=0.1, missing_rate=0.3, seed=14,
    )
    assert not train.equals(different_seed)


def test_synthetic_generation_variable_lengths_and_validation():
    train, test = generate_synthetic_data(
        n_train=6, n_test=4, n_classes=2, n_features=1, length=5,
        min_length=2, max_length=7, irregularity=0.4, seed=2,
    )
    assert train.groupby("trajectory_id").size().between(2, 7).all()
    assert test.groupby("trajectory_id").size().between(2, 7).all()
    assert train.groupby("trajectory_id")["time"].apply(lambda times: times.is_monotonic_increasing).all()
    assert any(
        group["time"].diff().dropna().nunique() > 1
        for _, group in train.groupby("trajectory_id")
    )
    with pytest.raises(ValueError, match="Set both min_length and max_length"):
        generate_synthetic_data(min_length=3)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"n_train": 0}, "n_train and n_test must be positive"),
        ({"n_classes": 1}, "n_classes must be at least 2"),
        ({"irregularity": -0.1}, "irregularity must be nonnegative"),
        ({"missing_rate": 1.0}, r"missing_rate must be in \[0, 1\)"),
        ({"time_end": 0}, "time_end must be positive"),
    ],
)
def test_synthetic_generation_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        generate_synthetic_data(**options)
