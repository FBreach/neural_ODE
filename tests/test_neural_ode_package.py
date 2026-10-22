import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch

from neural_ode import (
    plot_input_trajectories,
    plot_latent_space,
    plot_predicted_trajectories,
    plot_training_loss,
    predict,
    evaluate_performance,
    prep_batches,
    prep_data,
    generate_synthetic_data,
)


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


class _EvaluationModel(torch.nn.Module):
    def __init__(self, *, phase="pretrain", single_cluster=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self._neural_ode_phase = phase
        self.single_cluster = single_cluster

    def get_reconstruction(self, target_times, observed, observed_times, observed_mask,
                           mode=None, use_sampling=False):
        batch, steps = target_times.shape[:2]
        return torch.zeros(1, batch, steps, 1, device=target_times.device), None, None

    def encode(self, observed, observed_times, observed_mask):
        return observed[:, 0, :1].unsqueeze(0), torch.ones(1, observed.shape[0], 1)

    def pred_cluster(self, observed, observed_times, observed_mask):
        if self.single_cluster:
            return torch.zeros(observed.shape[0], dtype=torch.long, device=observed.device)
        return (observed[:, 0, 0] > 0).long()


def _evaluation_batch(values, labels):
    values = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    batch_size = len(values)
    return {
        "observed_data": values[:, None, :],
        "observed_tp": torch.zeros(batch_size, 1, 1),
        "observed_mask": torch.ones(batch_size, 1, 1),
        "tp_to_predict": torch.zeros(batch_size, 2, 1),
        "data_to_predict": torch.tensor(
            [[[1.0], [3.0]]] * batch_size, dtype=torch.float32
        ),
        "mask_predicted_data": torch.tensor(
            [[[1.0], [1.0]]] * batch_size, dtype=torch.float32
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
        "mode": "interp",
    }


def test_evaluate_performance_pools_masked_forecast_mse_and_skips_pre_gmm_metrics():
    first = _evaluation_batch([-1.0], [-1])
    first["data_to_predict"] = torch.tensor([[[1.0], [3.0]]])
    second = _evaluation_batch([1.0, 2.0], [-1, -1])
    second["data_to_predict"] = torch.tensor([[[2.0], [0.0]], [[4.0], [0.0]]])
    second["mask_predicted_data"] = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])

    scores = evaluate_performance(_EvaluationModel(), [first, second], device="cpu")

    assert list(scores) == [
        "mse", "nmi", "ari", "acc", "silhouette", "davies_bouldin",
        "calinski_harabasz",
    ]
    # Valid errors are 1^2 + 3^2 + 2^2 + 4^2 = 30 across four target values.
    assert scores["mse"] == pytest.approx(7.5)
    assert all(scores[name] is None for name in list(scores)[1:])


def test_evaluate_performance_accepts_separately_normalized_loader():
    class NormalizedBatches(list):
        pass

    test_loader = NormalizedBatches([_evaluation_batch([0.0], [-1])])
    test_loader._neural_ode_normalization = (torch.tensor([20.0]), torch.tensor([4.0]))
    model = _EvaluationModel()
    model._neural_ode_normalization = (torch.tensor([10.0]), torch.tensor([2.0]))

    scores = evaluate_performance(model, test_loader)

    # Test values [1, 3] map to training scale [7, 11], whose mean square is 85.
    assert scores["mse"] == pytest.approx(85.0)


def test_evaluate_performance_reports_label_and_internal_cluster_metrics():
    batches = [
        _evaluation_batch([-2.0, -1.0], [1, 1]),
        _evaluation_batch([1.0, 2.0], [0, -1]),
    ]

    scores = evaluate_performance(_EvaluationModel(phase="clustered"), batches)

    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["acc"] == pytest.approx(1.0)
    assert scores["silhouette"] is not None
    assert scores["davies_bouldin"] is not None
    assert scores["calinski_harabasz"] is not None


def test_evaluate_performance_supports_unlabeled_data_and_degenerate_clusters():
    batches = [
        _evaluation_batch([-2.0, -1.0], [-1, -1]),
        _evaluation_batch([1.0, 2.0], [-1, -1]),
    ]
    scores = evaluate_performance(_EvaluationModel(phase="finetune"), batches)
    assert all(scores[name] is None for name in ("nmi", "ari", "acc"))
    assert scores["silhouette"] is not None

    degenerate = evaluate_performance(
        _EvaluationModel(phase="clustered", single_cluster=True),
        [_evaluation_batch([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1])],
    )
    assert degenerate["nmi"] is not None
    assert all(degenerate[name] is None for name in (
        "silhouette", "davies_bouldin", "calinski_harabasz"
    ))


def _irregular_frame():
    times_by_id = {"a": [0, 1, 2], "b": [0, 2, 5], "c": [0, 4, 8], "d": [0, 6, 12]}
    rows = []
    for trajectory_id, times in times_by_id.items():
        rows.extend({"id": trajectory_id, "time": time, "x": index}
                    for index, time in enumerate(times))
    return pd.DataFrame(rows)


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


def test_create_model_uses_private_core_implementation():
    from neural_ode import create_model

    model = create_model(1, {"latent_dim": 2, "k": 2})
    assert isinstance(model, torch.nn.Module)
    assert model.k.item() == 2


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
