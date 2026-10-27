"""Fetch 30 complete Oxford context days and forecast the following seven days.

Run after training with ``python examples/forecast/predict_oxford_weather.py``.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neural_ode import create_model, predict, prep_data  # noqa: E402
from fetch_oxford_weather import CONTEXT_DAYS, FEATURES, TARGET_DAYS, fetch_weather  # noqa: E402


FORECAST_OUTPUT = Path(__file__).resolve().parent / "results" / "oxford_weather_forecast.csv"
MODEL_OUTPUT = FORECAST_OUTPUT.parent / "oxford_weather_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--output", type=Path, default=FORECAST_OUTPUT)
    parser.add_argument("--forecast-date", type=date.fromisoformat,
                        help="first forecast date, YYYY-MM-DD; defaults to today in Europe/London")
    args = parser.parse_args()
    forecast_date = args.forecast_date or datetime.now(ZoneInfo("Europe/London")).date()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "model_config", "feature_names", "normalization_mean", "normalization_std", "context_days"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Invalid weather checkpoint: use the standalone weather trainer's checkpoint")
    if checkpoint["feature_names"] != FEATURES:
        raise ValueError("The trained model feature list does not match the current fetcher")
    if checkpoint["context_days"] != CONTEXT_DAYS:
        raise ValueError(f"The weather checkpoint must use {CONTEXT_DAYS} context days")
    context_days = CONTEXT_DAYS
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    if (mean.shape != (len(FEATURES),) or std.shape != (len(FEATURES),)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Checkpoint scaling must contain one finite mean and positive standard deviation per feature")
    end_date = forecast_date - timedelta(days=1)
    start_date = end_date - timedelta(days=context_days - 1)
    future_dates = pd.date_range(forecast_date, periods=TARGET_DAYS, freq="D")
    print(f"Context dates: {start_date} to {end_date}", flush=True)
    print(f"Forecast dates: {forecast_date} to {future_dates[-1].date()}", flush=True)
    weather = fetch_weather(start_date.isoformat(), end_date.isoformat()).sort_values("date")
    expected_dates = pd.date_range(start_date, end_date, freq="D")
    weather_dates = pd.DatetimeIndex(pd.to_datetime(weather["date"]))
    if len(weather) != context_days or not weather_dates.equals(expected_dates):
        raise ValueError(f"Open-Meteo did not return {context_days} consecutive context days")
    weather[FEATURES] = weather[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(weather[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Recent weather context contains missing or non-finite feature values")

    rows = weather[["date", *FEATURES]].copy()
    rows.insert(0, "window_id", "oxford_live")
    rows["time"] = np.arange(context_days)
    future = pd.DataFrame({
        "window_id": "oxford_live",
        "date": future_dates,
        "time": np.arange(context_days, context_days + TARGET_DAYS),
        **{feature: np.nan for feature in FEATURES},
    })
    forecast_input = pd.concat([rows, future], ignore_index=True)
    data = prep_data(forecast_input, id_col="window_id", time_col="time",
                     feature_cols=FEATURES, cutoff=context_days - 1, cutoff_mode="time")

    model = create_model(len(FEATURES), checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model._neural_ode_normalization = (torch.from_numpy(mean), torch.from_numpy(std))
    model._neural_ode_phase = "finetune"

    result = predict(model, data)
    if len(result.trajectories) != 1:
        raise ValueError("Expected exactly one Oxford prediction trajectory")
    predicted = result.trajectories[0].prediction.numpy()
    if predicted.shape != (TARGET_DAYS, len(FEATURES)):
        raise ValueError(f"Expected {TARGET_DAYS} finite forecast rows with {len(FEATURES)} features")
    predicted = predicted * std + mean
    if not np.isfinite(predicted).all():
        raise ValueError(f"Expected {TARGET_DAYS} finite forecast rows with {len(FEATURES)} features")
    forecast_columns = {feature: predicted[:, index]
                        for index, feature in enumerate(FEATURES)}
    # Precipitation is optional in DAILY_VARIABLES and is nonnegative when used.
    if "precipitation_sum_mm" in forecast_columns:
        forecast_columns["precipitation_sum_mm"] = np.maximum(
            forecast_columns["precipitation_sum_mm"], 0.0
        )
    output = pd.DataFrame({"date": future_dates.date, **forecast_columns})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Replace only a complete CSV, leaving any previous successful output intact on failure.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=args.output.name + ".", suffix=".tmp",
                                         encoding="utf-8", newline="", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            output.to_csv(temporary, index=False, float_format="%.3f")
        temporary_path.replace(args.output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(output.to_string(index=False))
    print(f"Saved {TARGET_DAYS}-day forecast to {args.output}")


if __name__ == "__main__":
    main()
