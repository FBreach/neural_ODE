import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch

from neural_ode import (
    plot_input_trajectories, plot_latent_space, plot_predicted_trajectories,
    plot_training_loss, predict, prep_batches, prep_data,
)

def sample_frame():
    return pd.DataFrame({
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
        "time": [0, 1, 2] * 4,
        "x": [1, 2, 3, 10, 11, 12, 20, 21, 22, 30, 31, 32],
        "y": [2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "label": [0] * 3 + [1] * 3 + [0] * 3 + [1] * 3,
    })

def _irregular_frame():
    times_by_id = {
    "a": [0, 1, 2],
    "b": [0, 2, 5],
    "c": [0, 4, 8],
    "d": [0, 6, 12],
    }
    rows = []
    for trajectory_id, times in times_by_id.items():
        rows.extend(
    {"id": trajectory_id, "time": time, "x": index}
    for index, time in enumerate(times)
    )
    return pd.DataFrame(rows)


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


def test_plotters_return_figures():
    data = prep_data(sample_frame(), feature_cols=["x", "y"], cutoff=1,
                     cutoff_mode="time")
    assert set(plot_input_trajectories(data)) == {"x", "y"}
    fig, ax = plot_training_loss([{"epoch": 1, "train_loss": 2., "val_loss": 3.}])
    assert ax.get_xlabel() == "Epoch"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_input_trajectory_plot_groups_labels_and_shows_one_cutoff():
    from matplotlib import pyplot as plt

    frame = sample_frame()
    frame["label"] = frame["label"].map({0: "control", 1: "case"})
    frame.loc[frame["id"] == "d", "label"] = None
    data = prep_data(
        frame, feature_cols=["x"], label_col="label",
        cutoff=0.5, cutoff_mode="fraction",
    )
    fig, ax = plot_input_trajectories(data)["x"]

    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert {
        "control", "case", "Unlabeled", "Cutoff", "Before cutoff", "After cutoff"
    } <= set(legend_labels)
    assert len(ax.collections) == 3
    group_colors = {
        line.get_label(): line.get_color()
        for line in ax.lines
        if line.get_label() in {"control", "case", "Unlabeled"}
    }
    assert len(group_colors) == 3
    assert len(set(group_colors.values())) == 3
    assert len([line for line in ax.lines if line.get_label() == "Cutoff"]) == 1
    plt.close(fig)


def test_input_trajectory_plot_unlabeled_data_uses_one_group_and_keeps_gaps():
    from matplotlib import pyplot as plt

    frame = sample_frame()
    frame.loc[(frame["id"] == "a") & (frame["time"] == 1), "x"] = None
    data = prep_data(frame, feature_cols=["x"])
    fig, ax = plot_input_trajectories(data)["x"]

    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["All trajectories"]
    raw_colors = [line.get_color() for line in ax.lines[: len(data.trajectories)]]
    assert len(set(raw_colors)) == 1
    assert np.isnan(ax.lines[0].get_ydata()).any()
    assert len(ax.collections) == 1
    plt.close(fig)


def test_input_trajectory_plot_scales_to_many_trajectories():
    from matplotlib import pyplot as plt

    base = sample_frame()
    frames = []
    for offset in range(30):
        frame = base.copy()
        frame["id"] = frame["id"].map(lambda value: f"{offset}-{value}")
        frames.append(frame)
    data = prep_data(pd.concat(frames, ignore_index=True), feature_cols=["x"])
    figures = plot_input_trajectories(data, max_trajectories=40)
    fig, ax = figures["x"]

    assert len([line for line in ax.lines if line.get_label() == "All trajectories"]) == 1
    assert len(ax.collections) == 1
    raw_lines = [line for line in ax.lines if line.get_label().startswith("_child")]
    assert len(raw_lines) == 40
    plt.close(fig)


class _PlotPredictionModel(torch.nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.feature_count = feature_count

    def get_reconstruction(self, target_times, observed, observed_times, observed_mask,
                           mode=None, use_sampling=False):
        batch, steps = target_times.shape[:2]
        prediction = torch.zeros(1, batch, steps, self.feature_count)
        return prediction, None, None

    def pred_cluster(self, observed, observed_times, observed_mask):
        return torch.zeros(observed.shape[0], dtype=torch.long)

def test_prediction_plot_uses_one_shared_absolute_cutoff_line():
    from matplotlib import pyplot as plt

    model = _PlotPredictionModel(1)
    data = prep_data(_irregular_frame(), feature_cols=["x"], cutoff=1, cutoff_mode="time")
    predictions = predict(model, data)
    _, ax = plot_predicted_trajectories(predictions)["x"]
    cutoff_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(cutoff_lines) == 1
    assert cutoff_lines[0].get_xdata()[0] == 1
    plt.close("all")


@pytest.mark.parametrize("as_loaders", [False, True])
def test_fractional_prediction_plot_uses_one_median_cutoff_line(as_loaders):
    from matplotlib import pyplot as plt

    model = _PlotPredictionModel(1)
    data = prep_data(_irregular_frame(), feature_cols=["x"], cutoff=0.5,
                     cutoff_mode="fraction")
    prediction_input = prep_batches(data, val_split=0, shuffle=False) if as_loaders else data
    predictions = predict(model, prediction_input)
    assert predictions.cutoff_time == 3.0

    _, ax = plot_predicted_trajectories(predictions)["x"]
    cutoff_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(cutoff_lines) == 1
    assert cutoff_lines[0].get_xdata()[0] == 3.0

    _, ax_without_observed = plot_predicted_trajectories(
        predictions, show_observed=False
    )["x"]
    assert not any(line.get_linestyle() == "--" for line in ax_without_observed.lines)
    plt.close("all")

def test_plot_latent_space_without_color_does_not_require_cluster_predictions():
    from matplotlib import pyplot as plt

    class EncoderOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode(self, observed, times, mask):
            return observed[:, 0, :2].unsqueeze(0), None

    batch = {
        "observed_data": torch.tensor([[[0., 0.]], [[1., 0.]], [[0., 1.]], [[1., 1.]]]),
        "observed_tp": torch.zeros(4, 1, 1),
        "observed_mask": torch.ones(4, 1, 2),
        "labels": torch.tensor([-1, -1, -1, -1]),
    }
    fig, ax, _ = plot_latent_space(EncoderOnlyModel(), {"train": [batch]}, color_by=None)
    assert len(ax.collections) == 1
    assert ax.get_legend() is None
    plt.close(fig)

def test_plot_latent_space_without_color_by_defaults_to_clusters():

    class EncoderOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def encode(self, observed, times, mask):
            return observed[:, 0, :2].unsqueeze(0), None

    batch = {
        "observed_data": torch.tensor([[[0., 0.]], [[1., 0.]], [[0., 1.]], [[1., 1.]]]),
        "observed_tp": torch.zeros(4, 1, 1),
        "observed_mask": torch.ones(4, 1, 2),
        "labels": torch.tensor([-1, -1, -1, -1]),
    }
    with pytest.raises(ValueError, match = "No colours were found"):
        _fig, _ax, _ = plot_latent_space(EncoderOnlyModel(), {"train": [batch]})

@pytest.mark.parametrize("color_by", ["cluster", "ground_truth"])
def test_plot_latent_space_groups_points_by_color(color_by):
    from matplotlib import pyplot as plt

    class LatentModel(torch.nn.Module):
        def encode(self, observed, times, mask):
            return observed[:, 0, :2].unsqueeze(0), None

        def pred_cluster(self, observed, times, mask):
            return (observed[:, 0, 0] > 0).long()

    batch = {
        "observed_data": torch.tensor([
            [[0., 0.]], [[1., 0.]], [[0., 1.]], [[1., 1.]],
        ]),
        "observed_tp": torch.zeros(4, 1, 1),
        "observed_mask": torch.ones(4, 1, 2),
        "labels": torch.tensor([-1, 0, -1, 0]),
    }

    fig, ax, _ = plot_latent_space(
        LatentModel(), {"train": [batch]}, color_by=color_by,
    )
    try:
        expected_labels = (
            ["0", "1"] if color_by == "cluster" else ["Unlabeled", "0"]
        )
        assert len(ax.collections) == 2
        assert [len(group.get_offsets()) for group in ax.collections] == [2, 2]
        assert [text.get_text() for text in ax.get_legend().get_texts()] == expected_labels
        assert ax.get_legend().get_title().get_text() == color_by
        assert ax.get_title() == f"Latent space by {color_by}"
    finally:
        plt.close(fig)


def test_plot_training_loss():
    from matplotlib import pyplot as plt

    history = [
        {"epoch": 1, "train_loss": 3.0, "val_loss": 4.0},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 2.0},
    ]
    fig, ax = plot_training_loss(history)
    try:
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Loss"
        assert [line.get_label() for line in ax.lines] == ["train_loss", "val_loss"]
        for line, values in zip(ax.lines, ([3.0, 1.0], [4.0, 2.0])):
            np.testing.assert_array_equal(line.get_xdata(), [1, 2])
            np.testing.assert_array_equal(line.get_ydata(), values)
    finally:
        plt.close(fig)
