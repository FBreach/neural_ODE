"""Train the Oxford weather forecaster from prepared rolling windows.

Run after ``fetch_oxford_weather.py`` with
``python examples/forecast/train_oxford_weather.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neural_ode import (  # noqa: E402
    create_model,
    prep_batches,
    prep_data,
    plot_latent_space,
    plot_predicted_trajectories,
    plot_training_loss,
    predict,
    train_fit_gmm,
    train_finetune,
    train_pretrain,
)
from fetch_oxford_weather import FEATURES, WINDOW_OUTPUT  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_OUTPUT = RESULTS_DIR / "oxford_weather_model.pt"
CONTEXT_DAYS = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-csv", type=Path, default=WINDOW_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_csv(args.window_csv)
    required = {"window_id", "time", "segment", *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Window CSV is missing columns: {sorted(required - set(frame.columns))}")

    windows = (frame.groupby("window_id", sort=False)["date"].min()
               .sort_values(kind="stable"))
    window_ids = windows.index.tolist()
    if len(window_ids) < 2:
        raise ValueError("At least two weather windows are required for chronological validation")
    validation_count = max(1, round(len(window_ids) * 0.2))
    validation_ids = window_ids[-validation_count:]
    training_ids = window_ids[:-validation_count]

    data = prep_data(frame, id_col="window_id", time_col="time", feature_cols=FEATURES,
                     cutoff=CONTEXT_DAYS - 1, cutoff_mode="time")
    trajectory_by_id = {trajectory.id: trajectory for trajectory in data.trajectories}
    # prep_batches reserves its first fraction as validation; put the latest
    # forecast weeks first so validation remains strictly later than training.
    data.trajectories = [trajectory_by_id[window_id]
                         for window_id in validation_ids + training_ids]
    loaders = prep_batches(
        data,
        batch_size=16,
        val_split=validation_count / len(window_ids),
        shuffle=False,
        normalize=True,
        seed=args.seed,
    )

    model_config = {
        "latent_dim": 6,
        "k": 3,
        "networks": {
            "z0_encoder_odernn": {
                "state_dim": 32,
                "recognition_ode_hidden_dims": [16],
                "gru_update_hidden_dims": [32],
                "posterior_hidden_dims": [32],
            },
            "generative_ode": {"hidden_dims": [16]},
            "emission": {"hidden_dims": [16]},
        },
    }
    model = create_model(len(FEATURES), model_config)
    checkpoint_dir = args.model_output.parent
    pretrained = train_pretrain(
        loaders,
        model_config=model,
        epochs=args.pretrain_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
    )
    clustered = train_fit_gmm(pretrained.model, loaders, n_components=3, seed=args.seed)
    finetuned = train_finetune(
        clustered.model,
        loaders,
        epochs=args.finetune_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = args.model_output.parent
    torch.save({
        "model_state_dict": finetuned.model.state_dict(),
        "model_config": model_config,
        "feature_names": FEATURES,
        "normalization_mean": loaders.mean.tolist(),
        "normalization_std": loaders.std.tolist(),
        "context_days": CONTEXT_DAYS,
        "validation_window_ids": validation_ids,
        "training_window_ids": training_ids,
        "pretrain_best_epoch": min(pretrained.history,
                                    key=lambda item: item.get("val_loss", item["train_loss"]))["epoch"],
        "finetune_best_epoch": min(finetuned.history,
                                   key=lambda item: item.get("val_loss", item["train_loss"]))["epoch"],
    }, args.model_output)

    latent_fig, _, _ = plot_latent_space(finetuned.model, loaders, color_by="cluster")
    latent_fig.savefig(results_dir / "latent_space.png", dpi=160, bbox_inches="tight")
    plt.close(latent_fig)

    prediction_result = predict(finetuned.model, loaders)
    mean = loaders.mean
    std = loaders.std
    for trajectory in prediction_result.trajectories:
        trajectory.truth = trajectory.truth * std + mean
        trajectory.prediction = trajectory.prediction * std + mean
        if trajectory.observed_values is not None:
            trajectory.observed_values = trajectory.observed_values * std + mean
    for feature, (fig, _) in plot_predicted_trajectories(
        prediction_result, max_trajectories_per_group=8, show_observed=True
    ).items():
        safe_feature = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        fig.savefig(results_dir / f"predicted_trajectories_{safe_feature}.png",
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

    for name, result in (("pretrain", pretrained), ("finetune", finetuned)):
        fig, _ = plot_training_loss(result)
        fig.savefig(results_dir / f"{name}_loss.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"Training windows: {len(training_ids)}; latest validation windows: {len(validation_ids)}")
    print(f"Saved model and PNG plots to {results_dir}")


if __name__ == "__main__":
    main()
