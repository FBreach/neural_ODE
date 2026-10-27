# neural_ode

A Python package for forecasting and clustering irregular trajectories with a
Latent ODE and Gaussian mixture prior. This project is part of the Research Software Engineering Course at the University of Oxford and builds upon work done by Felix Breach at the Adaptive Brain Lab (University of Cambridge).


[Read the documentation](https://neural-ode.readthedocs.io/en/latest/index.html)
for the getting started guide and API reference.

## Install locally

Requires Python 3.11 or later. From the repository root, create a virtual
environment and install the package:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The distribution name is `neural-ode`; the import name is `neural_ode`.
The Snakemake example and documentation build also require Python 3.11 or later.

## Quick start

To train the model, the user must provide an appropriate trajectory or csv path to prep_data.
The input data must contain one row per trajectory observation with an ID, a
numeric time, and one column per feature. The final 30% of each trajectory is reserved for forecasting.
Labels are optional metadata for plotting and evaluation. Missing feature values are supported. 

`evaluate_performance` reports masked forecast MSE, label-based clustering
scores (NMI, ARI and matched accuracy), and label-free clustering scores
(silhouette, Davies–Bouldin and Calinski–Harabasz). Unavailable metrics are
`None`; clustering metrics are evaluated after GMM fitting. Evaluation converts
separately normalised test batches to the model's training scale. Predictions
and their plots use that training scale.

The synthetic generator returns two DataFrames without writing files. The user can
specify the trajectory length, observation times, noise, any missing values, and random 
seeds for generation. 

For a command-line example using your own CSV, see
[`examples/basic_workflow.py`](examples/basic_workflow.py). It expects `id`,
`time` and `label` columns, with the remaining columns treated as features:

```sh
python examples/basic_workflow.py path/to/trajectories.csv --epochs 20
```

## Run the example with Snakemake

With Python 3.11 or later, install the optional workflow dependencies and run
these commands from the repository root:

```sh
python -m pip install -e '.[workflow]'
snakemake --cores 1 --dry-run
snakemake --cores 1
```

The [Snakefile](Snakefile) fetches Oxford weather using
[`examples/forecast/fetch_oxford_weather.py`](examples/forecast/fetch_oxford_weather.py),
splits the prepared windows, trains a model on weather data from 30 context days,
evaluates it and plots the predicted weather over the next 7 days.
The first run requires internet access to the Open-Meteo archive API. The DAG for the
Snakemake workflow is shown below.

![Weather workflow DAG: fetch_weather supplies gather_data; gather_data supplies train, evaluate and plot; train supplies evaluate and plot; all collects the final outputs.](examples/figures/dag.svg)

Evaluation and plotting both depend on the trained model and can run
independently once training finishes. The `all` target collects their final
outputs. The diagram's Mermaid source is available in
[`examples/figures/dag.mmd`](examples/figures/dag.mmd).

Edit [`examples/config.yaml`](examples/config.yaml) to set archive dates,
preparation, model, and training parameters. Outputs go to
`results/oxford_weather/`:

| Output | Contents |
| --- | --- |
| `data/train.csv`, `data/test.csv` | Chronologically split weather windows |
| `model.pt` | Trained model weights and reconstruction metadata |
| `history.json` | Per-epoch pretraining and fine-tuning metrics |
| `metrics.json` | Evaluation on the separate test set |
| `figures/input.pdf`, `figures/predictions.pdf` | Test inputs and forecasts, one page per feature |
| `figures/training_loss.pdf` | Training loss, one page per training phase |
| `logs/` | One log per workflow stage |

The fetched windows are cached at
`examples/forecast/data/oxford_weather_windows.csv`, shared with the standalone
[weather example](examples/forecast/README.md). The default archive dates are
1 January 2021 to 31 December 2025; change `weather.end_date` to extend the
dataset. Existing cached windows are reused when their fetch inputs and
parameters are unchanged.

The latest 20% of windows form the test set. Within the earlier data, the latest
20% form validation. Windows overlapping a later split's dates are discarded,
so training, validation and test sets share no observation dates. Consequently,
the retained proportions differ from the requested fractions. Weather has no
class labels, so NMI, ARI and matched accuracy are unavailable.

Completed stages are reused when their inputs and declared parameters are
unchanged. Changing training parameters reruns training and downstream jobs;
changing only `examples/scripts/plot.py` reruns plotting. To use another output
directory or configuration file:

```sh
snakemake --cores 1 --config output_dir=results/another_run
snakemake --cores 1 --configfile path/to/config.yaml
```

## Documentation

The [Read the Docs site](https://neural-ode.readthedocs.io/en/latest/index.html)
contains the getting-started guide and API reference generated from the
NumPy-style docstrings. To build the Sphinx documentation locally:

```sh
python -m pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the local build.

## Development

Install the development and workflow dependencies, then run the checks used
by the test CI job:

```sh
python -m pip install -e '.[dev,workflow]'
ruff check .
python -m pytest
```

The workflow is tested using an offline proxy for the weather data; the integration test
checks the basic functionality of the workflow including saving outputs, plotting, and running with
varying parameters. It also checks that chronological splits share
no observation dates. The integration test is skipped when Snakemake is unavailable.

## Source and licence

The package is extracted from research code in the parent workspace. See
[`NOTICE.md`](NOTICE.md) for provenance and the current restrictions on
redistribution, and [`LICENSE.md`](LICENSE.md) for the licence status.
