"""Produce the workflow's input CSVs using the synthetic data generator."""

from helpers import run_logged
from neural_ode import generate_synthetic_data


def main(job):
    train, test = generate_synthetic_data(**job.params.generation)
    train.to_csv(job.output.train, index=False)
    test.to_csv(job.output.test, index=False)
    print(f"Wrote {len(train)} training rows and {len(test)} test rows.")


run_logged(main, snakemake)  # noqa: F821
