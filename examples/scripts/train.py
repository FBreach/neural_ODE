"""Run pretraining, GMM fitting and fine-tuning in one process."""

from helpers import run_logged, split_weather_windows, write_json
import pandas as pd
import torch

from neural_ode import (
    create_model, prep_batches, prep_data, train_finetune, train_fit_gmm,
    train_pretrain,
)


def main(job):
    settings = job.params.training
    frame = pd.read_csv(job.input.train)
    training, validation = split_weather_windows(frame, settings["val_split"])
    training_ids = training["window_id"].drop_duplicates().tolist()
    validation_ids = validation["window_id"].drop_duplicates().tolist()
    data = prep_data(pd.concat([training, validation]), **job.params.preparation)
    by_id = {trajectory.id: trajectory for trajectory in data.trajectories}
    # prep_batches reserves the first trajectories for validation when shuffle=False.
    data.trajectories = [by_id[window_id] for window_id in validation_ids + training_ids]
    loaders = prep_batches(
        data, batch_size=settings["batch_size"],
        val_split=len(validation_ids) / len(data.trajectories),
        shuffle=False, normalize=True, seed=settings["seed"],
    )
    torch.manual_seed(settings["seed"])
    model = create_model(len(data.feature_names), job.params.model)
    pretrained = train_pretrain(
        loaders, model_config=model, epochs=settings["epochs"], seed=settings["seed"],
    )
    gmm = train_fit_gmm(pretrained.model, loaders, seed=settings["seed"])
    finetuned = train_finetune(
        gmm.model, loaders, epochs=settings["epochs"], seed=settings["seed"],
    )
    torch.save({
        "model_state_dict": finetuned.model.state_dict(),
        "model_config": dict(job.params.model),
        "feature_names": data.feature_names,
        "preparation": dict(job.params.preparation),
        "normalization": (loaders.mean, loaders.std),
        "phase": finetuned.phase,
        "training_window_ids": training_ids,
        "validation_window_ids": validation_ids,
    }, job.output.model)
    write_json(job.output.history, {
        "pretrain": pretrained.history, "finetune": finetuned.history,
    })


run_logged(main, snakemake)  # noqa: F821
