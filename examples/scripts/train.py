"""Run pretraining, GMM fitting and fine-tuning in one process."""

from helpers import run_logged, write_json
import torch

from neural_ode import (
    create_model, prep_batches, prep_data, train_finetune, train_fit_gmm,
    train_pretrain,
)


def main(job):
    settings = job.params.training
    data = prep_data(job.input.train, **job.params.preparation)
    loaders = prep_batches(
        data, batch_size=settings["batch_size"], val_split=settings["val_split"],
        normalize=True, seed=settings["seed"],
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
    }, job.output.model)
    write_json(job.output.history, {
        "pretrain": pretrained.history, "finetune": finetuned.history,
    })


run_logged(main, snakemake)  # noqa: F821
