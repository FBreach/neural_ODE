"""CSV preparation and variable-length PyTorch batching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Trajectory:
    """Store one time-ordered trajectory and its observation mask.

    Parameters
    ----------
    id : object
        Original trajectory identifier.
    times : torch.Tensor of shape (n_times,)
        Observation times in ascending order.
    values : torch.Tensor of shape (n_times, n_features)
        Feature values, with missing entries filled with zero by ``prep_data``.
    mask : torch.Tensor of shape (n_times, n_features)
        Boolean mask indicating observed feature values.
    label : int or None, optional
        Encoded trajectory label, or ``None`` for an unlabelled trajectory.
    cutoff_index : int or None, optional
        Index of the first forecast observation. If ``None``, all observations
        are used as both context and reconstruction targets.

    """

    id: Any
    times: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    label: Any = None
    cutoff_index: int | None = None


@dataclass
class TrajectoryData:
    """Collect trajectories with their feature, label and cutoff metadata.

    Parameters
    ----------
    trajectories : list of Trajectory
        Prepared trajectories.
    feature_names : list of str
        Feature names in tensor column order.
    label_values : list of object or None, optional
        Original labels indexed by their encoded integer labels.
    cutoff : float or None, optional
        Absolute time or context fraction used to split each trajectory.
    cutoff_mode : {'time', 'fraction'}, optional
        Interpretation of ``cutoff``. Default is ``'time'``.

    """

    trajectories: list[Trajectory]
    feature_names: list[str]
    label_values: list[Any] | None = None
    cutoff: float | None = None
    cutoff_mode: str = "time"

    def __len__(self):
        """Return the number of trajectories.

        Returns
        -------
        int
            Number of stored trajectories.

        """
        return len(self.trajectories)


@dataclass
class DataLoaders:
    """Bundle trajectory loaders and their shared preparation metadata.

    Parameters
    ----------
    train : torch.utils.data.DataLoader
        Training batches.
    val : torch.utils.data.DataLoader or None
        Validation batches, or ``None`` when no validation split is requested.
    test : torch.utils.data.DataLoader or None
        Test batches, or ``None`` when no test split is requested.
    mean : torch.Tensor of shape (n_features,)
        Training feature means used for scaling, or zeros without scaling.
    std : torch.Tensor of shape (n_features,)
        Training feature standard deviations used for scaling, or ones
        without scaling.
    feature_names : list of str
        Feature names in tensor column order.
    label_values : list of object or None
        Original labels indexed by their encoded integer labels.
    cutoff : float or None, optional
        Absolute time or context fraction used to prepare the trajectories.
    cutoff_mode : {'time', 'fraction'}, optional
        Interpretation of ``cutoff``. Default is ``'time'``.

    """

    train: DataLoader
    val: DataLoader | None
    test: DataLoader | None
    mean: torch.Tensor
    std: torch.Tensor
    feature_names: list[str]
    label_values: list[Any] | None
    cutoff: float | None = None
    cutoff_mode: str = "time"


def prep_data(
    csv_path,
    *,
    id_col="id",
    time_col="time",
    feature_cols=None,
    label_col=None,
    cutoff=None,
    cutoff_mode="time",
) -> TrajectoryData:
    """Read long-format data into time-ordered, masked trajectories.

    Parameters
    ----------
    csv_path : str, pathlib.Path or pandas.DataFrame
        CSV file or in-memory table with one row per trajectory observation.
        A supplied DataFrame is copied.
    id_col : str, optional
        Trajectory identifier column. Default is ``'id'``.
    time_col : str, optional
        Numeric observation-time column. Default is ``'time'``.
    feature_cols : sequence of str or None, optional
        Feature columns in the desired tensor order. If ``None``, use all
        columns except the ID, time and specified label columns.
    label_col : str or None, optional
        Optional trajectory label column. Non-missing labels must be constant
        within each trajectory and are encoded as consecutive integers.
    cutoff : float or None, optional
        Context/forecast split. For ``'time'``, context includes times less
        than or equal to this value. For ``'fraction'``, context contains
        ``ceil(n_times * cutoff)`` observations, clipped to leave at least
        one context and one forecast observation. If ``None``, reconstruct
        all observations from the complete trajectory.
    cutoff_mode : {'time', 'fraction'}, optional
        Interpretation of ``cutoff``. Default is ``'time'``.

    Returns
    -------
    TrajectoryData
        Trajectories with float32 times and values, Boolean observation masks,
        and feature and label metadata. Missing feature values become zero
        with a false mask entry.

    Raises
    ------
    TypeError
        If the input is not a supported table or path.
    ValueError
        If columns, times, features or labels are invalid, ID/time pairs are
        duplicated, no trajectories exist, or a cutoff cannot produce a
        valid context/forecast split.

    """
    frame = pd.read_csv(csv_path) if isinstance(csv_path, (str, Path)) else csv_path.copy()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("csv_path must be a path or pandas DataFrame")
    required = [id_col, time_col] + ([label_col] if label_col else [])
    missing_columns = [name for name in required if name not in frame]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if feature_cols is None:
        feature_cols = [c for c in frame.columns if c not in required]
    feature_cols = list(feature_cols)
    if not feature_cols or any(c not in frame for c in feature_cols):
        raise ValueError("feature_cols must name at least one existing column")
    if len(set(feature_cols)) != len(feature_cols) or set(feature_cols) & set(required):
        raise ValueError("Feature columns must be unique and distinct from ID, time, and label columns")
    if frame[id_col].isna().any() or frame[time_col].isna().any():
        raise ValueError("ID and time columns cannot contain missing values")
    times = pd.to_numeric(frame[time_col], errors="coerce")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError("Time values must be finite numbers")
    frame[time_col] = times
    for column in feature_cols:
        original = frame[column]
        frame[column] = pd.to_numeric(original, errors="coerce")
        if (original.notna() & frame[column].isna()).any():
            raise ValueError(f"Feature {column!r} contains non-numeric values")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {column!r} contains infinite values")
    if frame.duplicated([id_col, time_col]).any():
        raise ValueError("Each trajectory must have at most one row per time value")
    if cutoff_mode not in {"time", "fraction"}:
        raise ValueError("cutoff_mode must be 'time' or 'fraction'")
    if cutoff is not None and cutoff_mode == "fraction" and not 0 < float(cutoff) < 1:
        raise ValueError("Fractional cutoff must be strictly between 0 and 1")

    trajectories = []
    label_values = None
    if label_col:
        label_values = list(pd.unique(frame[label_col].dropna()))
        label_to_index = {value: i for i, value in enumerate(label_values)}
    for trajectory_id, group in frame.sort_values(time_col).groupby(id_col, sort=False):
        group = group.sort_values(time_col)
        t = torch.tensor(group[time_col].to_numpy(dtype=np.float32))
        array = group[feature_cols].to_numpy(dtype=np.float32)
        mask = torch.tensor(np.isfinite(array), dtype=torch.bool)
        values = torch.tensor(np.nan_to_num(array, nan=0.0), dtype=torch.float32)
        if len(t) == 0:
            continue
        cut_index = None
        if cutoff is not None:
            if len(t) < 2:
                raise ValueError(f"Trajectory {trajectory_id!r} needs at least two observations for a cutoff")
            if cutoff_mode == "fraction":
                cut_index = min(max(int(np.ceil(len(t) * float(cutoff))), 1), len(t) - 1)
            else:
                cut_index = int(torch.searchsorted(t, torch.tensor(float(cutoff)), right=True).item())
                if cut_index == 0 or cut_index == len(t):
                    raise ValueError(f"Cutoff must leave context and forecast observations for trajectory {trajectory_id!r}")
        label = None
        if label_col:
            labels = group[label_col].dropna().unique()
            if len(labels) > 1:
                raise ValueError(f"Trajectory {trajectory_id!r} has inconsistent labels")
            if len(labels):
                label = label_to_index[labels[0]]
        trajectories.append(Trajectory(trajectory_id, t, values, mask, label, cut_index))
    if not trajectories:
        raise ValueError("CSV contains no trajectories")
    return TrajectoryData(trajectories, feature_cols, label_values, cutoff, cutoff_mode)


class _TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        trajectory = self.trajectories[index]
        cut = trajectory.cutoff_index
        if cut is None:
            observed = slice(None)
            target = slice(None)
            mode = "interp"
        else:
            observed = slice(0, cut)
            target = slice(cut, None)
            mode = "extrap"
        return {
            "id": trajectory.id,
            "labels": trajectory.label,
            "observed_length": len(trajectory.times[observed]),
            "observed_data": trajectory.values[observed],
            "observed_mask": trajectory.mask[observed].float(),
            "observed_tp": trajectory.times[observed],
            "data_to_predict": trajectory.values[target],
            "mask_predicted_data": trajectory.mask[target].float(),
            "tp_to_predict": trajectory.times[target],
            "mode": mode,
        }


def _collate(samples):
    def pad(key):
        return torch.nn.utils.rnn.pad_sequence([s[key] for s in samples], batch_first=True)

    labels = [s["labels"] for s in samples]
    return {
        "ids": [s["id"] for s in samples],
        "labels": torch.tensor([-1 if x is None else x for x in labels], dtype=torch.long),
        "observed_lengths": torch.tensor([s["observed_length"] for s in samples]),
        "observed_data": pad("observed_data"),
        "observed_mask": pad("observed_mask"),
        "observed_tp": pad("observed_tp").unsqueeze(-1),
        "data_to_predict": pad("data_to_predict"),
        "mask_predicted_data": pad("mask_predicted_data"),
        "tp_to_predict": pad("tp_to_predict").unsqueeze(-1),
        "target_lengths": torch.tensor([len(s["tp_to_predict"]) for s in samples]),
        "mode": samples[0]["mode"],
    }


def prep_batches(
    data: TrajectoryData,
    *,
    batch_size=32,
    val_split=0.2,
    test_split=0.0,
    shuffle=True,
    seed=0,
    num_workers=0,
    normalize=False,
) -> DataLoaders:
    """Split trajectories and build padded batches with optional scaling.

    Parameters
    ----------
    data : TrajectoryData
        Trajectories returned by ``prep_data``.
    batch_size : int, optional
        Maximum trajectories per batch. Default is 32.
    val_split : float, optional
        Fraction reserved for validation. Default is 0.2.
    test_split : float, optional
        Fraction reserved for testing. Default is 0.0. Split sizes are rounded;
        each positive split receives at least one trajectory.
    shuffle : bool, optional
        Randomise split membership and training-batch order. Default is True.
        If False, assign test, validation and training splits in input order.
    seed : int, optional
        Seed for split membership and training shuffling. Default is 0.
    num_workers : int, optional
        Number of PyTorch loader worker processes. Default is 0.
    normalize : bool, optional
        Standardise features using observed values across the complete
        training trajectories, including forecast observations. Default is
        False. Standard deviations are clamped to at least ``1e-6``.

    Returns
    -------
    DataLoaders
        Loaders and scaling metadata. Batches are dictionaries containing
        ``ids``, integer ``labels`` (-1 for missing labels), ``mode``
        (``'interp'`` or ``'extrap'``), and observed/target lengths.
        ``observed_data`` and ``observed_mask`` have shape
        ``(batch, observed_steps, n_features)``; ``data_to_predict`` and
        ``mask_predicted_data`` have shape
        ``(batch, target_steps, n_features)``. ``observed_tp`` and
        ``tp_to_predict`` have matching step counts and a final dimension of
        one. Padding and missing values have zero masks.

    Raises
    ------
    TypeError
        If ``data`` is not a ``TrajectoryData`` instance.
    ValueError
        If batch or worker counts are invalid, split fractions are invalid,
        or the requested splits leave no training trajectories.

    Notes
    -----
    Input trajectories are not modified. Each loader carries its scaling
    metadata for use by ``evaluate_performance``.

    """
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be returned by prep_data")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if val_split < 0 or test_split < 0 or val_split + test_split >= 1:
        raise ValueError("val_split and test_split must be nonnegative and sum to less than 1")
    n = len(data)
    if n < 2 and (val_split or test_split):
        raise ValueError("At least two trajectories are required when using validation or test splits")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist() if shuffle else list(range(n))
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    if test_split and n_test == 0:
        n_test = 1
    if val_split and n_val == 0:
        n_val = 1
    if n_test + n_val >= n:
        raise ValueError("Splits leave no training trajectories")
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]

    train_values = torch.cat([data.trajectories[i].values for i in train_idx])
    train_masks = torch.cat([data.trajectories[i].mask for i in train_idx])
    count = train_masks.sum(0).clamp_min(1)
    mean = (train_values * train_masks).sum(0) / count
    variance = (((train_values - mean) * train_masks) ** 2).sum(0) / count
    std = variance.sqrt().clamp_min(1e-6)
    if not normalize:
        mean = torch.zeros_like(mean)
        std = torch.ones_like(std)
    transformed = []
    for trajectory in data.trajectories:
        values = torch.where(trajectory.mask, (trajectory.values - mean) / std, 0.0)
        transformed.append(Trajectory(trajectory.id, trajectory.times, values, trajectory.mask,
                                      trajectory.label, trajectory.cutoff_index))

    def loader(indices, do_shuffle=False):
        result = DataLoader(
            _TrajectoryDataset([transformed[i] for i in indices]),
            batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers,
            collate_fn=_collate,
            generator=torch.Generator().manual_seed(seed) if do_shuffle else None,
        )
        result._neural_ode_normalization = (mean.detach().cpu(), std.detach().cpu())
        return result

    return DataLoaders(loader(train_idx, shuffle), loader(val_idx) if val_idx else None,
                       loader(test_idx) if test_idx else None, mean, std,
                       data.feature_names, data.label_values, data.cutoff, data.cutoff_mode)
