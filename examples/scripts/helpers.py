"""Private file and checkpoint adapters for the example workflow."""

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402
import pandas as pd  # noqa: E402

from neural_ode import create_model, prep_data  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(len(checkpoint["feature_names"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model._neural_ode_normalization = checkpoint["normalization"]
    model._neural_ode_phase = checkpoint["phase"]
    return model.eval(), checkpoint


def load_test_data(path, checkpoint):
    preparation = dict(checkpoint["preparation"])
    preparation["feature_cols"] = checkpoint["feature_names"]
    return prep_data(
        path, **preparation
    )


def split_weather_windows(frame, held_out_fraction):
    """Reserve the latest windows, discarding earlier windows overlapping them."""
    if not 0 < held_out_fraction < 1:
        raise ValueError("Weather split fractions must be strictly between 0 and 1")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any() or frame["window_id"].isna().any():
        raise ValueError("Weather dates and window IDs must be present on every row")
    bounds = (frame.assign(date=dates).groupby("window_id")["date"]
              .agg(["min", "max"]).sort_values("min", kind="stable"))
    count = max(1, round(len(bounds) * held_out_fraction))
    held_out = bounds.iloc[-count:]
    earlier = bounds.iloc[:-count]
    earlier = earlier[earlier["max"] < held_out["min"].min()]
    if earlier.empty or held_out.empty:
        raise ValueError("Not enough non-overlapping weather windows for a chronological split")
    return (
        frame[frame["window_id"].isin(earlier.index)].copy(),
        frame[frame["window_id"].isin(held_out.index)].copy(),
    )


def run_logged(task, job):
    for path in [*job.output, *job.log]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(job.log[0], "w") as log, redirect_stdout(log), redirect_stderr(log):
        torch.set_num_threads(job.threads)
        task(job)
