# Oxford weather forecast

Run these commands from the repository root. The fetch and prediction stages use the Open-Meteo archive API.

## Workflow

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
