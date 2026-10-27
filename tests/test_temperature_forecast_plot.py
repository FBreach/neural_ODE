"""Check the forecast chart against its source temperatures."""

import importlib.util
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def plotter():
    path = Path(__file__).resolve().parents[1] / "examples/forecast/plot_temperature_forecast.py"
    spec = importlib.util.spec_from_file_location("temperature_plotter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def forecast():
    mean = np.array([-2, 0, 3, 4, 7, 5, 1])
    return pd.DataFrame({
        "date": pd.date_range("2026-10-27", periods=7),
        "temperature_2m_mean_c": mean,
        "temperature_2m_min_c": mean - 2,
        "temperature_2m_max_c": mean + 3,
    })


def test_bars_and_lines_match_forecast_values(plotter, forecast):
    figure = plotter.plot_forecast(forecast)
    try:
        ax = figure.axes[0]
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches], forecast["temperature_2m_mean_c"])
        assert len(ax.lines) == 2
        np.testing.assert_allclose(ax.lines[0].get_ydata(), forecast["temperature_2m_min_c"])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), forecast["temperature_2m_max_c"])
        assert "°C" in ax.get_ylabel()
        assert len(ax.get_xticklabels()) == 7
        assert {text.get_text() for text in ax.get_legend().get_texts()} == {"Mean", "Minimum", "Maximum"}
    finally:
        plt.close(figure)


def test_plot_cli_saves_png(plotter, forecast, tmp_path, monkeypatch):
    csv = tmp_path / "forecast.csv"
    png = tmp_path / "models/forecast.png"
    forecast.to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv", ["plot", "--input", str(csv), "--output", str(png)])
    plotter.main()
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.imread(png).shape[:2] == (800, 1440)


@pytest.mark.parametrize("problem", ["missing_column", "missing_day", "duplicate_day", "nan", "inf"])
def test_bad_forecast_fails_before_plotting(plotter, forecast, problem):
    if problem == "missing_column":
        forecast = forecast.drop(columns="temperature_2m_max_c")
    elif problem == "missing_day":
        forecast = forecast.iloc[:-1]
    elif problem == "duplicate_day":
        forecast.loc[0, "date"] = forecast.loc[1, "date"]
    else:
        forecast["temperature_2m_mean_c"] = forecast["temperature_2m_mean_c"].astype(float)
        forecast.loc[0, "temperature_2m_mean_c"] = float(problem)
    with pytest.raises(ValueError):
        plotter.plot_forecast(forecast)
