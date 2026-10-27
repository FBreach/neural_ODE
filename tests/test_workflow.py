"""Exercise the example DAG, including reuse of completed training."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_example(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def weather_fetcher():
    return load_example("weather_fetcher", "examples/forecast/fetch_oxford_weather.py")


@pytest.fixture
def weather_days(weather_fetcher):
    dates = pd.date_range("2021-01-01", periods=300)
    temperature = 10 + 8 * np.sin(np.arange(len(dates)) / 30)
    return pd.DataFrame({
        "trajectory_id": "oxford",
        "date": dates,
        **{feature: temperature + offset for offset, feature in enumerate(weather_fetcher.FEATURES)},
    })


def test_weather_split_has_no_overlapping_dates(weather_fetcher, weather_days):
    helpers = load_example("workflow_helpers", "examples/scripts/helpers.py")
    windows = weather_fetcher.prepare_windows(weather_days)
    train, test = helpers.split_weather_windows(windows.sample(frac=1, random_state=3), 0.2)
    assert pd.to_datetime(train["date"]).max() < pd.to_datetime(test["date"]).min()
    assert len(train) % weather_fetcher.WINDOW_DAYS == 0
    assert len(test) % weather_fetcher.WINDOW_DAYS == 0
    assert set(train["window_id"]).isdisjoint(test["window_id"])
    with pytest.raises(ValueError, match="Not enough non-overlapping"):
        helpers.split_weather_windows(windows.iloc[:weather_fetcher.WINDOW_DAYS * 2], 0.2)
    for fraction in (0, 1, -0.1):
        with pytest.raises(ValueError, match="split fractions"):
            helpers.split_weather_windows(windows, fraction)


def test_weather_fetch_cli_passes_configured_dates(weather_fetcher, weather_days, tmp_path, monkeypatch):
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return weather_days

    output = tmp_path / "windows.csv"
    monkeypatch.setattr(weather_fetcher, "fetch_weather", fetch)
    monkeypatch.setattr(sys, "argv", [
        "fetch_oxford_weather.py", "--start-date", "2021-01-01", "--end-date", "2021-10-27",
        "--window-output", str(output),
    ])
    weather_fetcher.main()
    assert calls == [("2021-01-01", "2021-10-27")]
    assert pd.read_csv(output)["window_id"].nunique() > 1


@pytest.mark.skipif(importlib.util.find_spec("snakemake") is None, reason="requires the workflow extra")
def test_weather_workflow_outputs_and_incremental_runs(tmp_path, weather_fetcher, weather_days):
    # Supply prepared archive-shaped data so this test never contacts the API.
    window_csv = tmp_path / "weather.csv"
    weather_fetcher.prepare_windows(weather_days).to_csv(window_csv, index=False)
    settings = {
        "output_dir": "results",
        "weather": {"window_csv": str(window_csv)},
        "training": {"epochs": 1, "batch_size": 6},
        "model": {
            "latent_dim": 2,
            "n_clusters": 2,
            "networks": {
                "z0_encoder_odernn": {
                    "state_dim": 4,
                    "recognition_ode_hidden_dims": [4],
                    "gru_update_hidden_dims": [4],
                    "posterior_hidden_dims": [4],
                },
                "generative_ode": {"hidden_dims": [4]},
                "emission": {"hidden_dims": [4]},
            },
        },
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    command = [
        sys.executable, "-m", "snakemake", "--snakefile", str(ROOT / "Snakefile"),
        "--directory", str(tmp_path), "--configfile", str(config), "--cores", "1",
    ]

    def run(*arguments):
        result = subprocess.run(command + list(arguments), capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr

    assert "localrule fetch_weather:" not in run()
    results = tmp_path / "results"
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["mse"] >= 0
    assert all(metrics[key] is None for key in ("nmi", "ari", "acc"))
    checkpoint = torch.load(results / "model.pt", weights_only=True)
    frame = pd.read_csv(results / "data/train.csv")
    test = pd.read_csv(results / "data/test.csv")
    train = frame[frame["window_id"].isin(checkpoint["training_window_ids"])]
    validation = frame[frame["window_id"].isin(checkpoint["validation_window_ids"])]
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()
    assert checkpoint["feature_names"] == weather_fetcher.FEATURES
    histories = json.loads((results / "history.json").read_text())
    assert len(histories["pretrain"]) == len(histories["finetune"]) == 1
    for name in ("input", "predictions", "training_loss"):
        assert (results / f"figures/{name}.pdf").read_bytes().startswith(b"%PDF")
    for name in ("gather_data", "train", "evaluate", "plot"):
        assert (results / f"logs/{name}.log").stat().st_size > 0

    model_mtime = (results / "model.pt").stat().st_mtime_ns
    assert "Nothing to be done" in run()
    (results / "figures/predictions.pdf").unlink()
    run()
    assert (results / "figures/predictions.pdf").exists()
    assert (results / "model.pt").stat().st_mtime_ns == model_mtime

    settings["training"]["epochs"] = 2
    config.write_text(json.dumps(settings))
    plan = run("--dry-run")
    assert "rule train:" in plan
    assert "rule evaluate:" in plan
    assert "rule plot:" in plan
    assert "rule gather_data:" not in plan
    assert "rule fetch_weather:" not in plan
