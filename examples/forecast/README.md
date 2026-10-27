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

The explicit `live_forecast` target loads a previously trained model. It does not
run any of the historical training, evaluation or plotting rules. The default
`snakemake --cores 1` command still runs the historical workflow described above.

### Train and publish the model once

Run from the repository root on the committed version of the code you want to
use for training. The following uses a separate ignored directory and leaves
the standalone example's previous outputs in place:

```bash
python -m pip install -e '.[workflow]'
python examples/forecast/fetch_oxford_weather.py \
  --start-date 2021-01-01 --end-date 2025-12-31 \
  --window-output results/live/training_windows.csv
python examples/forecast/train_oxford_weather.py \
  --window-csv results/live/training_windows.csv \
  --model-output results/live/oxford_weather_model.pt
snakemake --cores 1 live_forecast
```

The forecast is saved as `results/live/oxford_weather_forecast_YYYY-MM-DD.csv`,
with a log at `results/live/logs/forecast_YYYY-MM-DD.log`. Inspect the CSV before
publishing the model: it must contain exactly seven consecutive dates starting
today in `Europe/London`, followed by the mean, maximum and minimum temperature
columns in degrees Celsius. Values must be finite. The model generates these
predictions from the preceding 30 complete context days.

Use the checkpoint from the **standalone weather trainer** above. The historical
root workflow's `results/oxford_weather/model.pt` has a different checkpoint
format and is not accepted by this predictor.

After a successful live forecast, use an authenticated [GitHub CLI](https://cli.github.com/)
to publish the checkpoint and select it for Actions:

```bash
git rev-parse HEAD
```

Copy that full commit SHA into `--target` in the next command. The commit must be
pushed to GitHub, and it must be the version used for training.

```bash
gh release create oxford-weather-model-v1 \
  results/live/oxford_weather_model.pt \
  --target FULL_TRAINING_COMMIT_SHA \
  --title "Oxford weather model v1" \
  --notes "Standalone Oxford weather checkpoint; training interval 2021-01-01 to 2025-12-31; 30 context days and 7 forecast days." \
  --latest=false
gh variable set WEATHER_MODEL_RELEASE --body oxford-weather-model-v1
```

This creates a versioned model release without replacing the latest software
release. Keep the published checkpoint fixed. To intentionally retrain later,
publish a new release such as `oxford-weather-model-v2`, run its live smoke check,
and update `WEATHER_MODEL_RELEASE`. Daily inference never retrains automatically.

### Configure and reproduce a forecast

The `live_forecast` section of `examples/config.yaml` defines:

| Setting | Default | Meaning |
| --- | --- | --- |
| `model_path` | `results/live/oxford_weather_model.pt` | Existing standalone checkpoint |
| `output_dir` | `results/live` | Dated CSVs and forecast logs |
| `forecast_date` | `null` | First predicted date; null resolves today in Europe/London |

Set `forecast_date` to a quoted `YYYY-MM-DD` value in a config file to select a
specific forecast date. This fixes the date window; the upstream reanalysis may
still be revised. The standalone command also accepts this date explicitly:

```bash
python examples/forecast/predict_oxford_weather.py \
  --model results/live/oxford_weather_model.pt \
  --forecast-date 2026-10-27 \
  --output results/live/oxford_weather_forecast_2026-10-27.csv
```

Snakemake reuses a completed daily output while declared inputs remain unchanged.
To regenerate that date deliberately:

```bash
snakemake --cores 1 live_forecast --forcerun live_forecast
```

Dates follow the London calendar, including daylight-saving changes. Context is
the preceding 30 calendar days; the seven forecast dates start on the selected
date. The source is Open-Meteo's [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
which supplies model/reanalysis estimates rather than a stream of station
observations. If recent days are unavailable, incomplete or non-finite, the job
fails without publishing a new CSV. It does not silently substitute older dates
or fill gaps. The predictor validates checkpoint features and scaling, verifies
the prediction shape, and atomically replaces the CSV only after completion.

### Daily GitHub Actions run

`.github/workflows/daily-forecast.yml` runs at **08:17 UTC every day** and also
supports **Run workflow** in the Actions tab. This is 08:17 in London during GMT
and 09:17 during BST. Each run downloads `oxford_weather_model.pt` from the
release named by the repository's `WEATHER_MODEL_RELEASE` Actions variable,
captures today's London date once, and runs only `live_forecast` on Python 3.12.

The built-in GitHub token needs only repository `contents: read`; no personal
access token or third-party storage is required. A missing variable, release or
checkpoint is reported as a failed job. The forecast log is printed on inference
failure. Active runs are allowed to finish and have a 30-minute timeout.

After success, download the CSV from the run's **Artifacts** section. The artifact
name contains its forecast date, run ID and attempt. It is retained for 90 days
(subject to the repository's retention limit), so this is not permanent storage
or a public endpoint for a web app. The run summary lists forecast dates, UTC
generation time, model release, code commit and artifact name. Generated files
are not committed to the repository.

### Rollout

1. Develop and validate on `felix`, push it, and open a PR targeting `main`.
2. Run the existing PR checks and publish the validated model release and Actions
   variable as described above.
3. Merge after review, then manually run **Daily weather forecast** from Actions
   on `main`. Download its CSV and inspect the seven rows and run summary.
4. Confirm the following daily run succeeds. Review failed runs in Actions;
   fixing data availability or replacing the model remains a manual action.

[GitHub schedules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
run only from the default branch and may be delayed. The workflow must be on
`main` before scheduling and manual dispatch are available. Public repositories'
schedules can be disabled after 60 days without activity. CSV artifacts also
[expire under the configured retention policy](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/remove-workflow-artifacts).

Validation commands (install the development extras first):

```bash
python -m pip install -e '.[dev,workflow]'
python -m pytest tests/test_live_forecast.py tests/test_workflow.py
ruff check .
snakemake --cores 1 live_forecast --dry-run
```

The tests use offline weather fixtures. The separate live smoke run above checks
the API, trained checkpoint and CSV together. Forecast accuracy monitoring,
automatic retraining and web hosting are outside this workflow.
