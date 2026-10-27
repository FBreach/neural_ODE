"""Fetch Oxford weather and save only the prepared rolling-window CSV.

Run from the repository root with ``python examples/forecast/fetch_oxford_weather.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
WINDOW_OUTPUT = DATA_DIR / "oxford_weather_windows.csv"
LATITUDE, LONGITUDE = 51.7520, -1.2577
SOURCE_TRAJECTORY_ID = "oxford"
# Keep the API variable names and units together so the request and output schema
# cannot drift apart. Open-Meteo returns wind speed in km/h by default.
DAILY_VARIABLES = {
    "temperature_2m_mean": "temperature_2m_mean_c",
    "temperature_2m_max": "temperature_2m_max_c",
    "temperature_2m_min": "temperature_2m_min_c",
    # "precipitation_sum": "precipitation_sum_mm",
    # "precipitation_hours": "precipitation_hours",
    # "relative_humidity_2m_mean": "relative_humidity_2m_mean_percent",
    # "dew_point_2m_mean": "dew_point_2m_mean_c",
    # "cloud_cover_mean": "cloud_cover_mean_percent",
    # "wind_speed_10m_mean": "wind_speed_10m_mean_kmh",
    # "shortwave_radiation_sum": "shortwave_radiation_sum_mj_m2",
    # "sunshine_duration": "sunshine_duration_seconds",
}
FEATURES = list(DAILY_VARIABLES.values())
CONTEXT_DAYS, TARGET_DAYS, STRIDE_DAYS = 30, 7, 7
WINDOW_DAYS = CONTEXT_DAYS + TARGET_DAYS


def fetch_weather(start_date: str = "2021-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Fetch daily weather into memory for an inclusive date range."""
    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    params = urlencode({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/London",
    })
    try:
        with urlopen(f"https://archive-api.open-meteo.com/v1/archive?{params}", timeout=30) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch Open-Meteo weather data: {exc}") from exc
    if "daily" not in payload:
        raise RuntimeError(f"Open-Meteo returned an error: {payload.get('reason', payload)}")

    daily = payload["daily"]
    print(f"Open-Meteo selected grid cell: {payload['latitude']}, {payload['longitude']}")
    data = {
        "trajectory_id": SOURCE_TRAJECTORY_ID,
        "date": daily["time"],
    }
    data.update({column: daily[variable] for variable, column in DAILY_VARIABLES.items()})
    return pd.DataFrame(data)


def prepare_windows(weather: pd.DataFrame) -> pd.DataFrame:
    """Return one long-format trajectory per rolling window."""
    required = {"trajectory_id", "date", *FEATURES}
    if not required.issubset(weather.columns):
        raise ValueError(f"Input CSV must contain columns: {sorted(required)}")
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    if weather["date"].isna().any() or weather["trajectory_id"].isna().any():
        raise ValueError("trajectory_id and date must be present on every row")
    if weather.duplicated(["trajectory_id", "date"]).any():
        raise ValueError("Input must contain at most one row per trajectory_id and date")
    for feature in FEATURES:
        weather[feature] = pd.to_numeric(weather[feature], errors="coerce")

    rows = []
    for source_id, series in weather.groupby("trajectory_id", sort=True):
        series = series.sort_values("date").reset_index(drop=True)
        for start in range(0, len(series) - WINDOW_DAYS + 1, STRIDE_DAYS):
            window = series.iloc[start:start + WINDOW_DAYS]
            dates = window["date"].reset_index(drop=True)
            if (window[FEATURES].isna().any().any()
                    or not dates.diff().dropna().eq(pd.Timedelta(days=1)).all()):
                continue
            window_id = f"{source_id}_{dates.iloc[0]:%Y%m%d}"
            for time, (_, row) in enumerate(window.iterrows()):
                rows.append({
                    "trajectory_id": source_id,
                    "window_id": window_id,
                    "date": dates.iloc[time].date().isoformat(),
                    "time": time,
                    "segment": "context" if time < CONTEXT_DAYS else "target",
                    **{feature: row[feature] for feature in FEATURES},
                })
    if not rows:
        raise ValueError(f"No complete {WINDOW_DAYS}-day windows found in the input CSV")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-output", type=Path, default=WINDOW_OUTPUT, help="prepared window CSV path")
    parser.add_argument("--start-date", default="2021-01-01", help="inclusive first archive date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="inclusive final archive date; defaults to yesterday")
    args = parser.parse_args()
    windows = prepare_windows(fetch_weather(args.start_date, args.end_date))
    args.window_output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.window_output, index=False, float_format="%.6f")
    print(f"Saved {windows['window_id'].nunique()} windows ({WINDOW_DAYS} rows each) to {args.window_output}")
    print(f"Each window has {CONTEXT_DAYS} context rows and {TARGET_DAYS} target rows; "
          f"windows advance by {STRIDE_DAYS} days.")


if __name__ == "__main__":
    main()
