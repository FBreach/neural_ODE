"""Check daily inference and its Snakemake target without contacting Open-Meteo."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def predictor(monkeypatch):
    directory = ROOT / "examples/forecast"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("weather_predictor", directory / "predict_oxford_weather.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(predictor):
    return {
        "feature_names": predictor.FEATURES.copy(),
        "context_days": 30,
        "normalization_mean": [10.0, 12.0, 8.0],
        "normalization_std": [2.0, 3.0, 4.0],
        "model_config": {},
        "model_state_dict": {},
    }


@pytest.fixture
def inference(predictor, checkpoint, monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    torch.save(checkpoint, model_path)
    model = SimpleNamespace(load_state_dict=lambda state: None)
    monkeypatch.setattr(predictor, "create_model", lambda *args: model)
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return pd.DataFrame({
            "date": pd.date_range(start, end),
            **{feature: np.arange(30, dtype=float) for feature in predictor.FEATURES},
        })

    monkeypatch.setattr(predictor, "fetch_weather", fetch)
    monkeypatch.setattr(predictor, "predict", lambda model, data: SimpleNamespace(
        trajectories=[SimpleNamespace(prediction=torch.ones(7, 3))]
    ))
    output = tmp_path / "forecast.csv"

    def run(forecast_date=None):
        arguments = ["predict", "--model", str(model_path), "--output", str(output)]
        if forecast_date is not None:
            arguments += ["--forecast-date", forecast_date]
        monkeypatch.setattr(sys, "argv", arguments)
        predictor.main()
        return pd.read_csv(output)

    return SimpleNamespace(run=run, calls=calls, output=output, model_path=model_path, model=model)


@pytest.mark.parametrize(("forecast_date", "context_start", "context_end", "forecast_end"), [
    ("2026-01-01", "2025-12-02", "2025-12-31", "2026-01-07"),
    ("2026-03-29", "2026-02-27", "2026-03-28", "2026-04-04"),
    ("2026-10-25", "2026-09-25", "2026-10-24", "2026-10-31"),
    ("2026-04-01", "2026-03-02", "2026-03-31", "2026-04-07"),
])
def test_forecast_dates_and_scaling(inference, predictor, forecast_date, context_start, context_end, forecast_end, capsys):
    frame = inference.run(forecast_date)
    assert inference.calls == [(context_start, context_end)]
    assert list(frame.columns) == ["date", *predictor.FEATURES]
    assert frame["date"].tolist() == pd.date_range(forecast_date, forecast_end).strftime("%Y-%m-%d").tolist()
    np.testing.assert_allclose(frame[predictor.FEATURES], np.tile([12.0, 15.0, 12.0], (7, 1)))
    assert "Context dates:" in capsys.readouterr().out
    assert not list(inference.output.parent.glob("*.tmp"))


def test_default_date_uses_london_when_utc_is_previous_day(inference, predictor, monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert str(tz) == "Europe/London"
            return datetime(2026, 3, 29, 23, 30, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(predictor, "datetime", FixedDatetime)
    assert inference.run().iloc[0]["date"] == "2026-03-30"


@pytest.mark.parametrize("problem", ["features", "context", "mean_shape", "mean_nan", "std_zero", "std_negative", "std_inf", "format"])
def test_bad_checkpoint_never_publishes_csv(inference, checkpoint, problem):
    if problem == "features":
        checkpoint["feature_names"] = ["wrong"]
    elif problem == "context":
        checkpoint["context_days"] = 29
    elif problem == "mean_shape":
        checkpoint["normalization_mean"] = [1]
    elif problem == "mean_nan":
        checkpoint["normalization_mean"][0] = float("nan")
    elif problem.startswith("std_"):
        checkpoint["normalization_std"][0] = {"std_zero": 0, "std_negative": -1, "std_inf": float("inf")}[problem]
    else:
        del checkpoint["normalization_mean"]
    torch.save(checkpoint, inference.model_path)
    with pytest.raises(ValueError):
        inference.run("2026-10-27")
    assert not inference.output.exists()
    assert inference.calls == []


@pytest.mark.parametrize("problem", ["missing_day", "duplicate_day", "nan", "inf"])
def test_incomplete_context_preserves_previous_output(inference, predictor, monkeypatch, problem):
    fetch = predictor.fetch_weather

    def bad_fetch(start, end):
        frame = fetch(start, end)
        if problem == "missing_day":
            return frame.iloc[:-1]
        if problem == "duplicate_day":
            frame.loc[0, "date"] = frame.loc[1, "date"]
        else:
            frame.loc[0, predictor.FEATURES[0]] = float(problem)
        return frame

    monkeypatch.setattr(predictor, "fetch_weather", bad_fetch)
    inference.output.write_text("previous success\n")
    with pytest.raises(ValueError):
        inference.run("2026-10-27")
    assert inference.output.read_text() == "previous success\n"


@pytest.mark.parametrize("prediction", [torch.ones(6, 3), torch.ones(7, 2), torch.full((7, 3), float("nan")), torch.full((7, 3), float("inf"))])
def test_invalid_prediction_never_publishes(inference, predictor, monkeypatch, prediction):
    monkeypatch.setattr(predictor, "predict", lambda *args: SimpleNamespace(
        trajectories=[SimpleNamespace(prediction=prediction)]
    ))
    with pytest.raises(ValueError, match="finite forecast rows"):
        inference.run("2026-10-27")
    assert not inference.output.exists()


def test_interrupted_write_preserves_previous_csv(inference, monkeypatch):
    inference.output.write_text("previous success\n")

    def failed_write(frame, file, **kwargs):
        file.write("partial")
        raise OSError("disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_write)
    with pytest.raises(OSError, match="disk failure"):
        inference.run("2026-10-27")
    assert inference.output.read_text() == "previous success\n"
    assert not list(inference.output.parent.glob("*.tmp"))


def test_real_model_checkpoint_inference(inference, predictor, checkpoint, monkeypatch):
    from neural_ode import create_model, predict

    settings = {
        "latent_dim": 2, "n_clusters": 2,
        "networks": {
            "z0_encoder_odernn": {
                "state_dim": 4, "recognition_ode_hidden_dims": [4],
                "gru_update_hidden_dims": [4], "posterior_hidden_dims": [4],
            },
            "generative_ode": {"hidden_dims": [4]}, "emission": {"hidden_dims": [4]},
        },
    }
    model = create_model(3, settings)
    checkpoint.update(model_config=settings, model_state_dict=model.state_dict())
    torch.save(checkpoint, inference.model_path)
    monkeypatch.setattr(predictor, "create_model", create_model)
    monkeypatch.setattr(predictor, "predict", predict)
    frame = inference.run("2026-10-27")
    assert frame.shape == (7, 4)
    assert np.isfinite(frame[predictor.FEATURES]).all().all()


@pytest.mark.skipif(importlib.util.find_spec("snakemake") is None, reason="requires the workflow extra")
def test_live_target_has_no_training_and_reuses_daily_csv(tmp_path):
    # Exercise the real Snakefile with a tiny local predictor instead of an API call.
    shutil.copy(ROOT / "Snakefile", tmp_path / "Snakefile")
    examples = tmp_path / "examples"
    examples.mkdir()
    shutil.copy(ROOT / "examples/config.yaml", examples / "config.yaml")
    scripts = examples / "forecast"
    scripts.mkdir()
    (scripts / "fetch_oxford_weather.py").write_text("# Offline fetch fixture\n")
    (scripts / "predict_oxford_weather.py").write_text(
        "import argparse, csv\nfrom datetime import date, timedelta\nfrom pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "for name in ('model', 'forecast-date', 'output'): parser.add_argument('--' + name, required=True)\n"
        "args = parser.parse_args()\nassert Path(args.model).is_file()\n"
        "first = date.fromisoformat(args.forecast_date)\n"
        "with open(args.output, 'w') as file:\n"
        "    writer = csv.writer(file)\n    writer.writerow(['date', 'temperature'])\n"
        "    writer.writerows([(first + timedelta(days=i), 10) for i in range(7)])\n"
    )
    model = tmp_path / "external-model.pt"
    model.write_bytes(b"external checkpoint")
    initial_mtime = model.stat().st_mtime_ns
    config = tmp_path / "live.json"
    settings = {"live_forecast": {"model_path": str(model), "output_dir": "live", "forecast_date": "2026-10-27"}}
    config.write_text(json.dumps(settings))
    command = [
        sys.executable, "-m", "snakemake", "live_forecast", "--cores", "1",
        "--snakefile", str(tmp_path / "Snakefile"), "--directory", str(tmp_path), "--configfile", str(config),
    ]

    def run(*args):
        result = subprocess.run(command + list(args), capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr

    plan = run("--dry-run")
    assert "rule live_forecast:" in plan
    assert "rule train:" not in plan
    assert "rule fetch_weather:" not in plan
    run()
    csv = tmp_path / "live/oxford_weather_forecast_2026-10-27.csv"
    assert len(pd.read_csv(csv)) == 7
    assert (tmp_path / "live/logs/forecast_2026-10-27.log").exists()
    mtime = csv.stat().st_mtime_ns
    assert "Nothing to be done" in run()
    assert csv.stat().st_mtime_ns == mtime
    settings["live_forecast"]["forecast_date"] = "2026-10-28"
    config.write_text(json.dumps(settings))
    run()
    assert (tmp_path / "live/oxford_weather_forecast_2026-10-28.csv").exists()
    assert model.stat().st_mtime_ns == initial_mtime
    model.unlink()
    result = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "MissingInputException" in result.stdout + result.stderr


def test_bad_forecast_date_is_rejected(inference):
    with pytest.raises(SystemExit):
        inference.run("2026-02-30")
    assert not inference.output.exists()
