# neural_ode

A Python package for forecasting and clustering irregular trajectories with a
Latent ODE and Gaussian mixture prior. This project is part of the Research Software Engineering Course at the University of Oxford and builds upon work done by Felix Breach at the Adaptive Brain Lab (University of Cambridge).


[Read the documentation](https://neural-ode.readthedocs.io/en/latest/index.html)
for the getting started guide and API reference.

## Install locally

Requires Python 3.10 or later. From the repository root, create a virtual
environment and install the package:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The distribution name is `neural-ode`; the import name is `neural_ode`.
The Snakemake example and documentation build require Python 3.11 or later.

## Quick start

Input data is long format: one row per trajectory observation, with an ID, a
numeric time and one column per feature. Labels are optional metadata for
plotting and evaluation. Missing feature values are supported.

This example generates training and test data in memory, trains a model and
evaluates forecasts on the separate test set:

```python
from neural_ode import (
    evaluate_performance, generate_synthetic_data,
    plot_predicted_trajectories, predict, prep_batches, prep_data,
    train_fit_gmm, train_finetune, train_pretrain,
)

train_df, test_df = generate_synthetic_data(
    n_train=60, n_test=18, n_classes=3, n_features=2,
    length=20, irregularity=0.4, noise=0.1, seed=42,
)
data_options = {
    "id_col": "trajectory_id",
    "time_col": "time",
    "feature_cols": ["feature_0", "feature_1"],
    "label_col": "label",
    "cutoff": 0.7,
    "cutoff_mode": "fraction",
}
data = prep_data(train_df, **data_options)
loaders = prep_batches(
    data, batch_size=16, val_split=0.2, normalize=True, seed=42,
)
pretrained = train_pretrain(
    loaders, model_config={"latent_dim": 6, "n_clusters": 3},
    epochs=20, seed=42,
)
gmm = train_fit_gmm(pretrained.model, loaders, seed=42)
finetuned = train_finetune(gmm.model, loaders, epochs=20, seed=42)

test_data = prep_data(test_df, **data_options)
test_loader = prep_batches(
    test_data, batch_size=16, val_split=0, normalize=True, shuffle=False
).train
scores = evaluate_performance(finetuned.model, test_loader)
predictions = predict(finetuned.model, test_data)
figures = plot_predicted_trajectories(predictions)
print(scores)
```

`prep_data` also accepts a CSV path. The cutoff reserves the final 30% of each
trajectory for forecasting. Training consists of reconstruction pretraining,
GMM fitting and fine-tuning with the mixture prior.

`evaluate_performance` reports masked forecast MSE, label-based clustering
scores (NMI, ARI and matched accuracy), and label-free clustering scores
(silhouette, Davies–Bouldin and Calinski–Harabasz). Unavailable metrics are
`None`; clustering metrics are evaluated after GMM fitting. Evaluation converts
separately normalised test batches to the model's training scale. Predictions
and their plots use that training scale.

The synthetic generator returns two DataFrames without writing files. Its
options include variable trajectory lengths, irregular observation times,
noise, missing values and seeded generation.

For a command-line example using your own CSV, see
[`examples/basic_workflow.py`](examples/basic_workflow.py). It expects `id`,
`time` and `label` columns, with the remaining columns treated as features:

```sh
python examples/basic_workflow.py path/to/trajectories.csv --epochs 20
```

Plotting helpers
`plot_input_trajectories`, `plot_predicted_trajectories`, `plot_latent_space`,
and `plot_training_loss` are also exported from `neural_ode`. Input trajectory
plots colour and summarise by label when labels are available; without labels,
they use one neutral colour and an overall median with an interquartile band.
`plot_predicted_trajectories` automatically uses ground-truth labels after
pretraining when labels are available, and a neutral colour otherwise. After
GMM fitting it defaults to cluster colours; pass `color_by=None` to always use
one neutral colour.

## Run the example with Snakemake

With Python 3.11 or later, install the optional workflow dependencies and run
these commands from the repository root:

```sh
python -m pip install -e '.[workflow]'
snakemake --cores 1 --dry-run
snakemake --cores 1
```

The [Snakefile](Snakefile) connects four stages: gather data, train, evaluate
and plot. For now, data gathering calls `generate_synthetic_data` and saves
training and test CSVs. A future real-data acquisition function can replace
that call in [`gather_data.py`](examples/scripts/gather_data.py), producing
the same two long-format CSV inputs.

![Example workflow DAG: gather_data supplies train, evaluate and plot; train supplies evaluate and plot; all collects the final outputs.](examples/figures/dag.svg)

Evaluation and plotting both depend on the trained model and can run
independently once training finishes. The `all` target collects their final
outputs. The diagram's Mermaid source is available in
[`examples/figures/dag.mmd`](examples/figures/dag.mmd).

Edit [`examples/config.yaml`](examples/config.yaml) to set generation,
preparation, model and training parameters. Outputs go to
`results/basic_workflow/`:

| Output | Contents |
| --- | --- |
| `data/train.csv`, `data/test.csv` | Generated training and test trajectories |
| `model.pt` | Trained model weights and reconstruction metadata |
| `history.json` | Per-epoch pretraining and fine-tuning metrics |
| `metrics.json` | Evaluation on the separate test set |
| `figures/input.pdf`, `figures/predictions.pdf` | Test inputs and forecasts, one page per feature |
| `figures/training_loss.pdf` | Training loss, one page per training phase |
| `logs/` | One log per workflow stage |

The workflow keeps pretraining, GMM fitting and fine-tuning in one training
job. Its private checkpoint includes the model configuration, weights,
training normalisation, feature names and phase for downstream jobs; it is
not an optimiser checkpoint for resuming interrupted training. Evaluation and
plotting load that checkpoint in separate processes.

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

The workflow integration test uses a tiny generated dataset to check saved
outputs, reuse of completed work, plotting without retraining and planned
reruns after parameter changes. It is skipped when Snakemake is unavailable.

## Source and licence

The package is extracted from research code in the parent workspace. See
[`NOTICE.md`](NOTICE.md) for provenance and the current restrictions on
redistribution, and [`LICENSE.md`](LICENSE.md) for the licence status.
