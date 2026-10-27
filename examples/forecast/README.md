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

3. Fetch the latest 30 observed days, load the trained model, and predict the next 7 days:

   ```bash
   python examples/forecast/predict_oxford_weather.py
   ```

## Outputs

- `examples/forecast/data/oxford_weather_windows.csv` contains the prepared rolling windows and weather features.
- `examples/forecast/results/oxford_weather_model.pt` contains the trained model and feature scaling metadata.
- `examples/forecast/results/oxford_weather_forecast.csv` contains the 7-day forecast for all model features.
- `examples/forecast/results/` also contains the training checkpoints and PNG plots for latent space, predicted trajectories, and training loss.

The `data/` and `results/` directories are Git-ignored, so their generated files are not included when staging changes with `git add .`.
