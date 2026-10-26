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
    id: Any
    times: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    label: Any = None
    cutoff_index: int | None = None


@dataclass
class TrajectoryData:
    trajectories: list[Trajectory]
    feature_names: list[str]
    label_values: list[Any] | None = None
    cutoff: float | None = None
    cutoff_mode: str = "time"

    def __len__(self):
        return len(self.trajectories)


@dataclass
class DataLoaders:
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
    """Read long-format CSV into one masked tensor pair per trajectory."""
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
    """Split by trajectory ID and return padded loaders and train-fit scaling."""
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
