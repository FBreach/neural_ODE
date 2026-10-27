"""Fetch Oxford weather, train a forecaster and save evaluation and figures."""

from pathlib import Path
from datetime import date, datetime
import sys
from zoneinfo import ZoneInfo

ROOT = Path(workflow.basedir)
configfile: str(ROOT / "examples/config.yaml")

RESULTS = config["output_dir"]
SOURCES = sorted(str(path) for path in (ROOT / "src/neural_ode").rglob("*.py"))
HELPERS = str(ROOT / "examples/scripts/helpers.py")
LIVE = config.get("live_forecast", {})
LIVE_DATE = date.fromisoformat(str(
    LIVE.get("forecast_date") or datetime.now(ZoneInfo("Europe/London")).date()
)).isoformat()
LIVE_RESULTS = LIVE.get("output_dir", "results/live")


rule all:
    input:
        f"{RESULTS}/metrics.json",
        f"{RESULTS}/figures/input.pdf",
        f"{RESULTS}/figures/predictions.pdf",
        f"{RESULTS}/figures/training_loss.pdf",


rule live_forecast:
    input:
        model=LIVE.get("model_path", "results/live/oxford_weather_model.pt"),
        predictor=str(ROOT / "examples/forecast/predict_oxford_weather.py"),
        fetcher=str(ROOT / "examples/forecast/fetch_oxford_weather.py"),
        sources=SOURCES,
    output:
        csv=f"{LIVE_RESULTS}/oxford_weather_forecast_{LIVE_DATE}.csv",
    params:
        python=sys.executable,
        forecast_date=LIVE_DATE,
    threads: 1
    log:
        f"{LIVE_RESULTS}/logs/forecast_{LIVE_DATE}.log",
    shell:
        "{params.python:q} {input.predictor:q} --model {input.model:q} "
        "--forecast-date {params.forecast_date:q} --output {output.csv:q} > {log:q} 2>&1"


rule fetch_weather:
    input:
        fetcher=str(ROOT / "examples/forecast/fetch_oxford_weather.py"),
    output:
        windows=config["weather"]["window_csv"],
    params:
        python=sys.executable,
        start=config["weather"]["start_date"],
        end=config["weather"]["end_date"],
    log:
        f"{RESULTS}/logs/fetch_weather.log",
    shell:
        "{params.python:q} {input.fetcher:q} --start-date {params.start:q} "
        "--end-date {params.end:q} --window-output {output.windows:q} > {log:q} 2>&1"


rule gather_data:
    input:
        windows=rules.fetch_weather.output.windows,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        train=f"{RESULTS}/data/train.csv",
        test=f"{RESULTS}/data/test.csv",
    params:
        test_fraction=config["weather"]["test_fraction"],
    log:
        f"{RESULTS}/logs/gather_data.log",
    script:
        "examples/scripts/gather_data.py"


rule train:
    input:
        train=rules.gather_data.output.train,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        model=f"{RESULTS}/model.pt",
        history=f"{RESULTS}/history.json",
    params:
        preparation=config["preparation"],
        training=config["training"],
        model=config["model"],
    threads: 1
    log:
        f"{RESULTS}/logs/train.log",
    script:
        "examples/scripts/train.py"


rule evaluate:
    input:
        test=rules.gather_data.output.test,
        model=rules.train.output.model,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        metrics=f"{RESULTS}/metrics.json",
    threads: 1
    log:
        f"{RESULTS}/logs/evaluate.log",
    script:
        "examples/scripts/evaluate.py"


rule plot:
    input:
        test=rules.gather_data.output.test,
        model=rules.train.output.model,
        history=rules.train.output.history,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        inputs=f"{RESULTS}/figures/input.pdf",
        predictions=f"{RESULTS}/figures/predictions.pdf",
        loss=f"{RESULTS}/figures/training_loss.pdf",
    threads: 1
    log:
        f"{RESULTS}/logs/plot.log",
    script:
        "examples/scripts/plot.py"
