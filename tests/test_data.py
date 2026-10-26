import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
import torch

from neural_ode import prep_batches, prep_data

def sample_frame():
    return pd.DataFrame({
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
        "time": [0, 1, 2] * 4,
        "x": [1, 2, 3, 10, 11, 12, 20, 21, 22, 30, 31, 32],
        "y": [2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "label": [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3,
    })


def test_prep_data_cutoff_masks_and_labels():
    data = prep_data(sample_frame(), feature_cols=["x", "y"], label_col="label",
                     cutoff=0.5, cutoff_mode="fraction")
    trajectory = data.trajectories[0]
    assert trajectory.cutoff_index == 2
    assert trajectory.values.shape == (3, 2)
    assert not trajectory.mask[1, 1]
    assert trajectory.label == 0


def test_prep_data_validates_duplicate_id_time_and_absolute_cutoff():
    frame = sample_frame()
    data = prep_data(frame, feature_cols=["x"], cutoff=1, cutoff_mode="time")
    assert data.trajectories[0].cutoff_index == 2
    with pytest.raises(ValueError, match="at most one row"):
        prep_data(pd.concat([frame, frame.iloc[[0]]]), feature_cols=["x"])


def test_prep_batches_splits_ids_fits_normalization_on_train_only():
    data = prep_data(sample_frame(), feature_cols=["x"], cutoff=0.5, cutoff_mode="fraction")
    loaders = prep_batches(data, batch_size=2, val_split=0.25, test_split=0.25,
                          shuffle=False, normalize=True)
    ids = [set(batch["ids"]) for loader in (loaders.train, loaders.val, loaders.test)
           for batch in loader]
    assert len(set.union(*ids)) == 4
    train_values = torch.cat([batch["observed_data"] for batch in loaders.train])
    assert torch.isfinite(train_values).all()
    assert loaders.mean.shape == (1,)
    assert {loaders.train.dataset[i]["id"] for i in range(len(loaders.train.dataset))} == {"c", "d"}


def test_normalized_missing_values_stay_zero_under_mask():
    data = prep_data(sample_frame(), feature_cols=["x", "y"], cutoff=0.5,
                     cutoff_mode="fraction")
    loaders = prep_batches(data, val_split=0, normalize=True, shuffle=False)
    batch = next(iter(loaders.train))
    missing = ~batch["observed_mask"].bool()
    assert torch.all(batch["observed_data"][missing] == 0)

