"""Save the basic example's plots as PDFs without requiring a display."""

import json

from helpers import load_model, load_test_data, run_logged
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from neural_ode import (
    plot_input_trajectories, plot_predicted_trajectories, plot_training_loss, predict,
)


def save_features(path, figures):
    with PdfPages(path) as pdf:
        for figure, _ in figures.values():
            pdf.savefig(figure)
            plt.close(figure)


def main(job):
    model, checkpoint = load_model(job.input.model)
    data = load_test_data(job.input.test, checkpoint)
    save_features(job.output.inputs, plot_input_trajectories(data))
    save_features(job.output.predictions, plot_predicted_trajectories(predict(model, data)))
    with open(job.input.history) as stream:
        histories = json.load(stream)
    with PdfPages(job.output.loss) as pdf:
        for phase, history in histories.items():
            figure, ax = plot_training_loss(history)
            ax.set_title(phase.capitalize())
            pdf.savefig(figure)
            plt.close(figure)
    print("Saved input, prediction and training-loss figures.")


run_logged(main, snakemake)  # noqa: F821
