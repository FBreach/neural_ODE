"""Public API for neural ODE trajectory forecasting and clustering."""

from .data import DataLoaders, Trajectory, TrajectoryData, prep_batches, prep_data
from .model import create_model
from .plotting import (
    plot_input_trajectories,
    plot_latent_space,
    plot_predicted_trajectories,
    plot_training_loss,
)
from .synthetic import generate_synthetic_data
from .training import (
    GMMResult,
    PredictionResult,
    PredictionTrajectory,
    TrainingResult,
    evaluate_performance,
    predict,
    train_fit_gmm,
    train_finetune,
    train_pretrain,
)

__all__ = [
    "DataLoaders", "GMMResult", "PredictionResult", "PredictionTrajectory",
    "TrainingResult", "Trajectory", "TrajectoryData", "create_model",
    "plot_input_trajectories", "plot_latent_space", "plot_predicted_trajectories",
    "plot_training_loss", "predict", "evaluate_performance", "prep_batches", "prep_data",
    "generate_synthetic_data",
    "train_fit_gmm", "train_finetune", "train_pretrain",
]
