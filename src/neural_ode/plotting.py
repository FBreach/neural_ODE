"""Plotting helpers for raw, predicted, latent, and training trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import torch

from .data import TrajectoryData
from .training import PredictionResult


_NEUTRAL_COLOR = "#718096"
_PALETTE = plt.get_cmap("tab20")


def _group_colors(groups):
    """Return stable colors for encoded groups and neutral missing groups."""
    colors = {}
    fallback_index = 0
    for group in groups:
        if group is None or group == "unknown" or group == -1:
            colors[group] = _NEUTRAL_COLOR
        elif isinstance(group, (int, np.integer)) and group >= 0:
            colors[group] = _PALETTE(int(group) % _PALETTE.N)
        else:
            colors[group] = _PALETTE(fallback_index % _PALETTE.N)
            fallback_index += 1
    return colors


def _style_axes(ax):
    """Apply the shared light grid and uncluttered frame."""
    ax.set_axisbelow(True)
    ax.grid(color="#d9dee7", linewidth=0.7, alpha=0.65)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _interpolate_observed_segments(times, values, mask, grid):
    """Interpolate within contiguous observed runs, leaving gaps as NaN."""
    result = np.full(grid.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(mask & np.isfinite(values))
    if not len(valid_indices):
        return result

    runs = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1)
    for run in runs:
        if len(run) < 2:
            continue
        in_run = (grid >= times[run[0]]) & (grid <= times[run[-1]])
        result[in_run] = np.interp(grid[in_run], times[run], values[run])
    return result


def _summary_quantiles(values):
    """Return pointwise quartiles without warnings for unsupported times."""
    lower = np.full(values.shape[1], np.nan)
    median = np.full(values.shape[1], np.nan)
    upper = np.full(values.shape[1], np.nan)
    for index in range(values.shape[1]):
        observed = values[:, index]
        observed = observed[np.isfinite(observed)]
        if observed.size:
            lower[index], median[index], upper[index] = np.percentile(
                observed, [25, 50, 75]
            )
    return lower, median, upper


def plot_input_trajectories(data, *, feature_cols=None, cutoff=None, max_trajectories=None):
    """Plot feature-wise trajectories, colored by label when available.

    Individual paths are shown faintly with a median and interquartile band for
    each label. Unlabeled data uses one neutral color and an overall summary.
    Missing observations remain visible as gaps in both paths and summaries.
    """
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be returned by prep_data")
    feature_names = list(feature_cols or data.feature_names)
    unknown = set(feature_names) - set(data.feature_names)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    selected = data.trajectories[:max_trajectories]
    labeled = any(trajectory.label is not None for trajectory in selected)
    group_keys = list(dict.fromkeys(
        trajectory.label for trajectory in selected if trajectory.label is not None
    ))
    if labeled and any(trajectory.label is None for trajectory in selected):
        group_keys.append(None)
    if not labeled:
        group_keys = [None]

    colors = _group_colors(group_keys)

    def group_name(group):
        if not labeled:
            return "All trajectories"
        if group is None:
            return "Unlabeled"
        if data.label_values is not None and 0 <= int(group) < len(data.label_values):
            return str(data.label_values[int(group)])
        return str(group)

    def trajectory_split(trajectory):
        if cutoff is not None:
            return int(np.searchsorted(trajectory.times.numpy(), float(cutoff), side="left"))
        if trajectory.cutoff_index is not None:
            return trajectory.cutoff_index
        return len(trajectory.times)

    cutoff_times = []
    if cutoff is not None:
        cutoff_time = float(cutoff)
    elif data.cutoff is not None and data.cutoff_mode == "time":
        cutoff_time = float(data.cutoff)
    elif data.cutoff is not None:
        cutoff_time = None
        for trajectory in selected:
            split = trajectory.cutoff_index
            if split is not None and split < len(trajectory.times):
                cutoff_times.append(float(trajectory.times[split]))
        if cutoff_times:
            cutoff_time = float(np.median(cutoff_times))
    else:
        cutoff_time = None

    figures = {}
    for name in feature_names:
        feature_idx = data.feature_names.index(name)
        fig, ax = plt.subplots(figsize=(8, 5))
        _style_axes(ax)

        if selected:
            all_times = np.concatenate(
                [trajectory.times.numpy() for trajectory in selected]
            )
            grid = np.linspace(float(all_times.min()), float(all_times.max()), 240)
            group_values = {group: [] for group in group_keys}

        for trajectory in selected:
            times = trajectory.times.numpy()
            values = trajectory.values[:, feature_idx].numpy()
            mask = trajectory.mask[:, feature_idx].numpy().astype(bool)
            values = np.where(mask, values, np.nan)
            split = trajectory_split(trajectory)
            group = trajectory.label if labeled else None
            color = colors[group]
            ax.plot(times[:split], values[:split], color=color, linewidth=0.8, alpha=0.16)
            if split < len(times):
                ax.plot(
                    times[split:], values[split:], color=color,
                    linewidth=0.8, alpha=0.2, linestyle="--",
                )
            group_values[group].append(
                _interpolate_observed_segments(times, values, mask, grid)
            )

        if selected:
            for group in group_keys:
                if not group_values[group]:
                    continue
                lower, median, upper = _summary_quantiles(
                    np.stack(group_values[group])
                )
                color = colors[group]
                supported = np.isfinite(lower) & np.isfinite(upper)
                ax.fill_between(
                    grid, lower, upper, where=supported, color=color,
                    alpha=0.18, linewidth=0,
                )
                name_for_group = group_name(group)
                split_at = cutoff_time
                if split_at is None:
                    ax.plot(
                        grid, median, color=color, linewidth=2.3,
                        label=name_for_group, zorder=4,
                    )
                else:
                    before = grid < split_at
                    after = grid >= split_at
                    ax.plot(
                        grid[before], median[before], color=color,
                        linewidth=2.3, label=name_for_group, zorder=4,
                    )
                    ax.plot(
                        grid[after], median[after], color=color,
                        linewidth=2.3, linestyle="--", zorder=4,
                    )

        if cutoff_time is not None:
            ax.axvline(
                cutoff_time, linestyle=":", color="#374151", linewidth=1.2,
                label="Cutoff", zorder=3,
            )
        ax.set(title=name, xlabel="Time", ylabel=name)
        ax.margins(x=0.02, y=0.08)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if cutoff_time is not None:
                handles.extend([
                    Line2D([], [], color="#4b5563", linewidth=1.6, label="Before cutoff"),
                    Line2D([], [], color="#4b5563", linewidth=1.6, linestyle="--", label="After cutoff"),
                ])
            ax.legend(handles=handles, title="Label" if labeled else None,
                      frameon=False, loc="best")
        fig.tight_layout()
        figures[name] = (fig, ax)
    return figures


def plot_predicted_trajectories(
    predictions, *, color_by="auto", max_trajectories_per_group=3,
    show_observed=True,
):
    """Plot forecasts, choosing a meaningful default grouping for the model phase.

    Pretraining results use ground-truth labels when available and a neutral
    color otherwise. Results after GMM fitting default to cluster colors.
    Pass ``color_by="cluster"``, ``"ground_truth"``, or ``None`` to override.
    """
    if not isinstance(predictions, PredictionResult):
        raise TypeError("predictions must be returned by predict")
    if color_by not in {"auto", None, "cluster", "ground_truth"}:
        raise ValueError("color_by must be 'auto', None, 'cluster', or 'ground_truth'")
    if not predictions.trajectories:
        return {}
    if color_by == "auto":
        has_labels = any(trajectory.label is not None for trajectory in predictions.trajectories)
        color_by = (
            "ground_truth" if predictions.phase == "pretrain" and has_labels
            else None if predictions.phase == "pretrain"
            else "cluster"
        )
    feature_names = predictions.feature_names or [f"feature_{i}" for i in range(predictions.trajectories[0].prediction.shape[-1])]
    figs = {}
    for feature_idx, name in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(8, 5))
        _style_axes(ax)
        if show_observed and predictions.cutoff_time is not None:
            ax.axvline(predictions.cutoff_time, linestyle="--", color="#374151",
                       linewidth=1.2, alpha=.8, label="Cutoff")
        trajectory_groups = [
            (trajectory.cluster if color_by == "cluster" else trajectory.label)
            if color_by is not None else None
            for trajectory in predictions.trajectories
        ]
        group_order = list(dict.fromkeys(trajectory_groups))
        colors = _group_colors(group_order)
        counts = {}
        for trajectory in predictions.trajectories:
            group = (
                trajectory.cluster if color_by == "cluster" else trajectory.label
            ) if color_by is not None else None
            if counts.get(group, 0) >= max_trajectories_per_group:
                continue
            counts[group] = counts.get(group, 0) + 1
            color = colors[group]
            group_name = "Unlabeled" if group is None or group == -1 else str(group)
            mask = trajectory.mask[:, feature_idx].bool()
            if show_observed and trajectory.observed_times is not None:
                context_mask = trajectory.observed_mask[:, feature_idx].bool()
                full_times = torch.cat((trajectory.observed_times, trajectory.times))
                full_values = torch.cat((trajectory.observed_values, trajectory.truth))
                full_mask = torch.cat((context_mask, mask))
                ax.plot(
                    full_times[full_mask],
                    full_values[full_mask, feature_idx],
                    ":", color=color, alpha=.65,
                )
            else:
                ax.plot(trajectory.times[mask], trajectory.truth[mask, feature_idx], ":", color=color, alpha=.65)
            ax.plot(trajectory.times[mask], trajectory.prediction[mask, feature_idx], "-", color=color,
                    label=(f"{color_by} {group_name}"
                           if color_by is not None and counts[group] == 1 else None))
        ax.set(title=f"Predicted trajectories: {name}", xlabel="Time", ylabel=name)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, frameon=False, loc="best")
        ax.margins(x=0.02, y=0.08)
        fig.tight_layout()
        figs[name] = (fig, ax)
    return figs


def plot_latent_space(model, loaders, *, color_by="cluster", device="cpu"):
    """Plot PCA projection of posterior means with optional group coloring."""
    from .training import _get_loaders
    train_loader, _ = _get_loaders(loaders)
    if color_by not in {None, "cluster", "ground_truth"}:
        raise ValueError("color_by must be None, 'cluster', or 'ground_truth'")
    latent, labels, clusters = [], [], []
    model.to(device).eval()
    with torch.no_grad():
        for batch in train_loader:
            observed = batch["observed_data"].to(device)
            times = batch["observed_tp"].to(device)
            mask = batch["observed_mask"].to(device)
            mu, _ = model.encode(observed, times, mask)
            latent.append(mu.squeeze(0).cpu().numpy())
            if color_by == "ground_truth":
                labels.extend(batch["labels"].tolist())
            elif color_by == "cluster":
                clusters.extend(model.pred_cluster(observed, times, mask).cpu().tolist())
    latent = np.concatenate(latent)
    if latent.shape[0] < 2:
        raise ValueError("PCA needs at least two trajectories")
    projection = PCA(n_components=2).fit(latent)
    coordinates = projection.transform(latent)
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_axes(ax)
    if color_by is None:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], color=_NEUTRAL_COLOR,
                   alpha=.8, edgecolors="white", linewidths=.35)
        title = "Latent space"
    else:
        colors = np.asarray(clusters if color_by == "cluster" else labels)
        groups = list(np.unique(colors))
        palette = _group_colors(groups)
        for label in groups:
            selection = colors == label
            label_name = "Unlabeled" if label == -1 else str(label)
            ax.scatter(coordinates[selection, 0], coordinates[selection, 1],
                       color=[palette[label]], label=label_name, alpha=.82,
                       edgecolors="white", linewidths=.35)
        ax.legend(title=color_by, frameon=False, loc="best")
        title = f"Latent space by {color_by}"
    ax.set(xlabel="PC1", ylabel="PC2", title=title)
    fig.tight_layout()
    return fig, ax, projection


def plot_training_loss(history, *, metrics=None):
    """Plot per-epoch metrics, restricting pretraining to reconstruction loss."""
    phase = getattr(history, "phase", None)
    if hasattr(history, "history"):
        history = history.history
    if not history:
        raise ValueError("history cannot be empty")
    phase = phase or history[0].get("phase")
    available = [key for key in history[0] if key not in {"epoch", "kl_coef", "phase"}]
    if phase == "pretrain":
        available = [key for key in available if key.endswith("reconstruction_loss")]
    selected = metrics or [key for key in available if key.endswith("loss")]
    if phase == "pretrain":
        selected = [key for key in selected if key.endswith("reconstruction_loss")]
    selected = [key for key in selected if any(key in row for row in history)]
    if not selected:
        raise ValueError("No requested metrics are present in history")
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_axes(ax)
    epochs = [row["epoch"] for row in history]
    for index, metric in enumerate(selected):
        values = [row.get(metric, np.nan) for row in history]
        ax.plot(epochs, values, marker=".", color=_PALETTE(index % _PALETTE.N),
                linewidth=1.8, label=metric)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training history")
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    return fig, ax
