"""Private file and checkpoint adapters for the example workflow."""

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from neural_ode import create_model, prep_data  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(len(checkpoint["feature_names"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model._neural_ode_normalization = checkpoint["normalization"]
    model._neural_ode_phase = checkpoint["phase"]
    return model.eval(), checkpoint


def load_test_data(path, checkpoint):
    return prep_data(
        path, feature_cols=checkpoint["feature_names"], **checkpoint["preparation"]
    )


def run_logged(task, job):
    for path in [*job.output, *job.log]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(job.log[0], "w") as log, redirect_stdout(log), redirect_stderr(log):
        torch.set_num_threads(job.threads)
        task(job)
