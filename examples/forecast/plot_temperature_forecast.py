"""Plot daily mean temperature bars with separate minimum and maximum markers."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_forecast(frame):
    """Return a figure for seven consecutive days of finite temperature forecasts."""
    columns = ["temperature_2m_mean_c", "temperature_2m_min_c", "temperature_2m_max_c"]
    if not {"date", *columns}.issubset(frame.columns) or len(frame) != 7:
        raise ValueError("Expected seven forecast rows with date, mean, minimum and maximum temperature columns")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    if dates.isna().any() or not dates.equals(pd.date_range(dates[0], periods=7, freq="D")):
        raise ValueError("Forecast dates must be seven consecutive days in order")
    temperatures = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(temperatures).all():
        raise ValueError("Forecast temperatures must be finite")

    figure, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    days = np.arange(7)
    ax.bar(days, temperatures[:, 0], width=0.55, color="#a8becd", label="Mean", zorder=2)
    ax.plot(days, temperatures[:, 1], color="#2563eb", marker="_", markersize=22,
            markeredgewidth=2.5, linestyle="none", label="Minimum", zorder=3)
    ax.plot(days, temperatures[:, 2], color="#ea580c", marker="_", markersize=22,
            markeredgewidth=2.5, linestyle="none", label="Maximum", zorder=3)
    ax.set_xticks(days, dates.strftime("%a\n%d %b"))
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Oxford seven-day temperature forecast\n{dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y}")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", length=0, pad=8)
    ax.margins(y=0.2)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    return figure


def main():
    """Read a forecast CSV and save its temperature chart as a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure = plot_forecast(pd.read_csv(args.input))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, format="png", dpi=160)
    finally:
        plt.close(figure)
    print(f"Saved temperature forecast chart to {args.output}")


if __name__ == "__main__":
    main()
