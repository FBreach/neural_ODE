"""Split Oxford weather windows into chronological training and test CSVs."""

from helpers import run_logged, split_weather_windows
import pandas as pd


def main(job):
    windows = pd.read_csv(job.input.windows)
    train, test = split_weather_windows(windows, job.params.test_fraction)
    train.to_csv(job.output.train, index=False)
    test.to_csv(job.output.test, index=False)
    print(f"Wrote {len(train)} training rows and {len(test)} test rows.")


run_logged(main, snakemake)  # noqa: F821
