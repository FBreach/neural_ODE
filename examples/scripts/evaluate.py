"""Evaluate the saved model on the held-out test trajectories."""

from helpers import load_model, load_test_data, run_logged, write_json
from neural_ode import evaluate_performance, prep_batches


def main(job):
    model, checkpoint = load_model(job.input.model)
    data = load_test_data(job.input.test, checkpoint)
    loader = prep_batches(data, val_split=0, shuffle=False, normalize=True).train
    scores = evaluate_performance(model, loader)
    write_json(job.output.metrics, scores)
    print(scores)


run_logged(main, snakemake)  # noqa: F821
