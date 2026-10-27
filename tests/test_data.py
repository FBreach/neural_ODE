import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
import torch
import numpy as np

from neural_ode import prep_batches, prep_data
from neural_ode.data import _TrajectoryDataset

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

def test_prep_data_raises_errors_with_invalid_data():
    frame = sample_frame()
    with pytest.raises(TypeError, match = "must be a path or pandas DataFrame"):
        prep_data(np.zeros(5), feature_cols=["x"], cutoff=1, cutoff_mode="time")
    with pytest.raises(ValueError, match = "Missing required columns"):
        prep_data(frame, id_col = "identifier", feature_cols=["x"], cutoff=1, cutoff_mode="time")
    with pytest.raises(ValueError, match = "at least one existing column"):
        prep_data(frame, feature_cols=["z"], cutoff=1, cutoff_mode="time")

def test_prep_data_raises_errors_with_invalid_cutoff():
    frame = sample_frame()
    with pytest.raises(ValueError, match = "must be 'time' or 'fraction'"):
        prep_data(frame, feature_cols=["x"], cutoff=1, cutoff_mode="mixed")
    with pytest.raises(ValueError, match= "strictly between 0 and 1"):
        prep_data(frame, feature_cols=["x"], cutoff=1.5, cutoff_mode="fraction")


@pytest.mark.parametrize("cutoff, mode, observed, target", [
    (None, "interp", [0, 1, 2], [0, 1, 2]),
    (1, "extrap", [0, 1], [2]),
])
def test_trajectory_dataset_slices_context_and_targets(cutoff, mode, observed, target):
    data = prep_data(sample_frame(), feature_cols=["x", "y"], label_col="label",
                     cutoff=cutoff)
    dataset = _TrajectoryDataset(data.trajectories)
    trajectory = data.trajectories[0]
    sample = dataset[0]

    assert len(dataset) == 4
    assert sample["id"] == "a"
    assert sample["labels"] == 0
    assert sample["mode"] == mode
    assert sample["observed_length"] == len(observed)
    torch.testing.assert_close(sample["observed_data"], trajectory.values[observed])
    torch.testing.assert_close(sample["observed_tp"], trajectory.times[observed])
    torch.testing.assert_close(sample["observed_mask"], trajectory.mask[observed].float())
    torch.testing.assert_close(sample["data_to_predict"], trajectory.values[target])
    torch.testing.assert_close(sample["tp_to_predict"], trajectory.times[target])
    torch.testing.assert_close(sample["mask_predicted_data"], trajectory.mask[target].float())

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

def test_prep_batches_raises_errors_for_invalid_input():
    frame = sample_frame()
    data = prep_data(sample_frame(), feature_cols=["x"], cutoff=0.5, cutoff_mode="fraction")
    with pytest.raises(TypeError, match = "must be returned by prep_data"):
        prep_batches(frame, batch_size=2, val_split=0.25, test_split=0.25,
                          shuffle=False, normalize=True)
    with pytest.raises(ValueError, match = "must be positive"):
        prep_batches(data, batch_size=0, val_split=0.25, test_split=0.25,
                          shuffle=False, normalize=True)
    with pytest.raises(ValueError, match = "sum to less than 1"):
            prep_batches(data, batch_size=2, val_split=0.25, test_split=0.85,
                              shuffle=False, normalize=True)
    

def test_normalized_missing_values_stay_zero_under_mask():
    data = prep_data(sample_frame(), feature_cols=["x", "y"], cutoff=0.5,
                     cutoff_mode="fraction")
    loaders = prep_batches(data, val_split=0, normalize=True, shuffle=False)
    batch = next(iter(loaders.train))
    missing = ~batch["observed_mask"].bool()
    assert torch.all(batch["observed_data"][missing] == 0)



