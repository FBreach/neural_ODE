# Oxford weather forecast

Run these commands from the repository root. The fetch and prediction stages use the Open-Meteo archive API.

## Workflow

To run the root Snakemake workflow with this weather source:

```bash
python -m pip install -e '.[workflow]'
snakemake --cores 1
```

This fetches the fixed archive range in [`../config.yaml`](../config.yaml),
caches `data/oxford_weather_windows.csv`, and saves the model, held-out test
metrics and PDF plots under `results/oxford_weather/` at the repository root.
Training, validation and test windows are split chronologically; windows that
would share observation dates across splits are discarded. The workflow's
checkpoint is used by its own evaluation and plotting scripts.

The standalone commands below use their own checkpoint format and produce a
live 7-day forecast. They remain a separate entry point:

1. Fetch daily weather and prepare rolling 30-day context plus 7-day target windows:

   ```bash
   python examples/forecast/fetch_oxford_weather.py
   ```

2. Train the model on the prepared windows:

   ```bash
   python examples/forecast/train_oxford_weather.py
   ```

3. Fetch the latest 30 complete context days, load the trained model, and predict the next 7 days:

   ```bash
   python examples/forecast/predict_oxford_weather.py
   ```

## Outputs

- `examples/forecast/data/oxford_weather_windows.csv` contains the prepared rolling windows and weather features.
- `examples/forecast/results/oxford_weather_model.pt` contains the trained model and feature scaling metadata.
- `examples/forecast/results/oxford_weather_forecast.csv` contains the 7-day forecast for all model features.
- `examples/forecast/results/` also contains the training checkpoints and PNG plots for latent space, predicted trajectories, and training loss.

The `data/` and `results/` directories are Git-ignored, so their generated files are not included when staging changes with `git add .`.

## Daily inference with Snakemake

The repository includes a small trained checkpoint at `examples/forecast/models/oxford_weather_model.pt`.
From the repository root, install the workflow dependencies and run inference:

```bash
python -m pip install -e '.[workflow]'
python -m snakemake --cores 1 live_forecast
```

This uses the preceding 30 complete days of Open-Meteo model/reanalysis data and
predicts seven days starting today in `Europe/London`. It writes
`results/live/oxford_weather_forecast_YYYY-MM-DD.csv` and a log at
`results/live/logs/forecast_YYYY-MM-DD.log`. The CSV contains date, mean, maximum,
and minimum temperature in degrees Celsius. Missing data or invalid predictions
fail the run rather than publishing an incomplete CSV.

Snakemake reuses the output for the same day. To fetch data and predict again:

```bash
python -m snakemake --cores 1 live_forecast --forcerun live_forecast
```

The explicit target performs inference only. The default Snakemake target still
runs the historical training/evaluation workflow described above.

The **Daily weather forecast** Action runs this same command on pushes to `felix`,
on manual dispatch, and daily at **08:17 UTC** after it reaches the default branch.
Download the CSV from the successful run's **Artifacts** section. Artifacts are
retained for up to 90 days; generated CSVs are not committed. No model Release,
repository variables or model downloads need to be configured.

To update the included model, fetch training data and run the standalone trainer
as described above, specifying `--model-output examples/forecast/models/oxford_weather_model.pt`.
Run a live forecast, inspect the result, then commit the replacement checkpoint
with any related code changes. The historical root workflow's checkpoint uses a
different format; use the standalone weather trainer for this file. Daily runs
do not retrain automatically. Recent API data must be complete for inference.
