"""In-memory generation of labeled synthetic trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_timepoints(
    length: int, time_end: float, irregularity: float, rng: np.random.Generator
) -> np.ndarray:
    if irregularity == 0:
        return np.linspace(0.0, time_end, length, dtype=np.float64)
    shape = 1.0 / (irregularity**2)
    gaps = rng.gamma(shape=shape, scale=1.0, size=length - 1)
    gaps *= time_end / gaps.sum()
    return np.concatenate(([0.0], np.cumsum(gaps)))


def _class_features(
    times: np.ndarray,
    label: int,
    n_classes: int,
    n_features: int,
    class_variability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized_time = times / times[-1]
    centered_time = normalized_time - 0.5
    class_frequency = 0.8 + 0.55 * label
    class_trend = 0.7 * (label - (n_classes - 1) / 2.0) / max(n_classes - 1, 1)
    features = np.empty((len(times), n_features), dtype=np.float64)

    for feature_idx in range(n_features):
        amplitude = max(0.05, 1.0 + rng.normal(0.0, class_variability))
        frequency = class_frequency * (1.0 + feature_idx * 0.35)
        frequency *= max(0.1, 1.0 + rng.normal(0.0, class_variability * 0.25))
        phase = rng.normal(0.0, class_variability * np.pi)
        features[:, feature_idx] = (
            amplitude * np.sin(2.0 * np.pi * frequency * normalized_time + phase)
            + 0.25
            * np.cos(2.0 * np.pi * (frequency + 0.2) * normalized_time - phase / 2.0)
            + class_trend * centered_time
        )
    return features


def _generate_split(
    *,
    split: str,
    n_trajectories: int,
    length: int,
    min_length: int | None,
    max_length: int | None,
    time_end: float,
    irregularity: float,
    noise: float,
    n_classes: int,
    n_features: int,
    class_variability: float,
    missing_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    labels = np.arange(n_trajectories, dtype=np.int64) % n_classes
    rng.shuffle(labels)
    columns = ["trajectory_id", "time"]
    columns.extend(f"feature_{i}" for i in range(n_features))
    columns.append("label")
    rows = []

    for trajectory_idx, label in enumerate(labels):
        n_points = (
            length
            if min_length is None
            else int(rng.integers(min_length, max_length + 1))
        )
        times = _make_timepoints(n_points, time_end, irregularity, rng)
        values = _class_features(
            times, int(label), n_classes, n_features, class_variability, rng
        )
        if noise:
            values += rng.normal(0.0, noise, size=values.shape)
        values[rng.random(values.shape) < missing_rate] = np.nan

        for time_idx, time_value in enumerate(times):
            row = {
                "trajectory_id": f"{split}_{trajectory_idx:06d}",
                "time": float(time_value),
            }
            row.update(
                {
                    f"feature_{feature_idx}": float(value)
                    for feature_idx, value in enumerate(values[time_idx])
                }
            )
            row["label"] = int(label)
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def generate_synthetic_data(
    *,
    n_train: int = 300,
    n_test: int = 100,
    n_classes: int = 3,
    n_features: int = 3,
    length: int = 50,
    irregularity: float = 0.0,
    noise: float = 0.1,
    min_length: int | None = None,
    max_length: int | None = None,
    time_end: float = 5.0,
    class_variability: float = 0.08,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train and test DataFrames matching the synthetic-data CLI.

    Returns ``(train_df, test_df)`` with columns ``trajectory_id``, ``time``,
    ``feature_0`` through ``feature_{n_features-1}``, and ``label``. Missing
    feature values are represented by ``NaN``. No files are written.
    """
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if length < 2:
        raise ValueError("length must be at least 2")
    if (min_length is None) != (max_length is None):
        raise ValueError("Set both min_length and max_length, or neither")
    if min_length is not None and (min_length < 2 or max_length < min_length):
        raise ValueError("Require 2 <= min_length <= max_length")
    if irregularity < 0:
        raise ValueError("irregularity must be nonnegative")
    if noise < 0 or class_variability < 0:
        raise ValueError("noise and class_variability must be nonnegative")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    if time_end <= 0:
        raise ValueError("time_end must be positive")

    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
    common = {
        "length": length,
        "min_length": min_length,
        "max_length": max_length,
        "time_end": time_end,
        "irregularity": irregularity,
        "noise": noise,
        "n_classes": n_classes,
        "n_features": n_features,
        "class_variability": class_variability,
        "missing_rate": missing_rate,
    }
    train_df = _generate_split(
        split="train",
        n_trajectories=n_train,
        rng=np.random.default_rng(train_seed),
        **common,
    )
    test_df = _generate_split(
        split="test",
        n_trajectories=n_test,
        rng=np.random.default_rng(test_seed),
        **common,
    )
    return train_df, test_df
