"""Fetch 30 recent Oxford days and forecast temperature and precipitation.

Run after training with ``python examples/forecast/predict_oxford_weather.py``.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neural_ode import create_model, predict, prep_data
from fetch_oxford_weather import FEATURES, TARGET_DAYS, fetch_weather
from train_oxford_weather import MODEL_OUTPUT


FORECAST_OUTPUT = Path(__file__).resolve().parent / "results" / "oxford_weather_forecast.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--output", type=Path, default=FORECAST_OUTPUT)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if checkpoint["feature_names"] != FEATURES:
        raise ValueError("The trained model feature list does not match the current fetcher")
    context_days = int(checkpoint["context_days"])
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=context_days - 1)
    weather = fetch_weather(start_date.isoformat(), end_date.isoformat()).sort_values("date")
    expected_dates = pd.date_range(start_date, end_date, freq="D")
    weather_dates = pd.DatetimeIndex(pd.to_datetime(weather["date"]))
    if len(weather) != context_days or not weather_dates.equals(expected_dates):
        raise ValueError(f"Open-Meteo did not return {context_days} consecutive context days")
    if weather[FEATURES].isna().any().any():
        raise ValueError("Recent weather context contains missing feature values")

    rows = weather[["date", *FEATURES]].copy()
    rows.insert(0, "window_id", "oxford_live")
    rows["time"] = np.arange(context_days)
    future_dates = pd.date_range(end_date + timedelta(days=1), periods=TARGET_DAYS, freq="D")
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
    mean = torch.tensor(checkpoint["normalization_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalization_std"], dtype=torch.float32)
    model._neural_ode_normalization = (mean, std)
    model._neural_ode_phase = "finetune"

    result = predict(model, data)
    predicted = result.trajectories[0].prediction.numpy()
    predicted = predicted * np.asarray(checkpoint["normalization_std"]) + np.asarray(
        checkpoint["normalization_mean"]
    )
    forecast_columns = {feature: predicted[:, index]
                        for index, feature in enumerate(FEATURES)}
    # Precipitation is optional in DAILY_VARIABLES and is nonnegative when used.
    if "precipitation_sum_mm" in forecast_columns:
        forecast_columns["precipitation_sum_mm"] = np.maximum(
            forecast_columns["precipitation_sum_mm"], 0.0
        )
    output = pd.DataFrame({"date": future_dates.date, **forecast_columns})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.3f")
    print(output.to_string(index=False))
    print(f"Saved {TARGET_DAYS}-day forecast to {args.output}")


if __name__ == "__main__":
    main()
