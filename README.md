# neural_ode

A small Python package for preparing irregular trajectory data, training a
Latent ODE with a mixture prior, forecasting, and plotting results.

## Install locally

From this directory, install the package into the active Python environment:

```sh
python -m pip install -e .
```

The import name is `neural_ode`.

## Generate synthetic trajectories

`generate_synthetic_data` returns in-memory `(train_df, test_df)` DataFrames
using the same defaults, seeded split generation, feature columns, labels, and
missing-value behavior as the research CSV generator. It does not write files.

```python
from neural_ode import generate_synthetic_data

train_data, test_data = generate_synthetic_data(
    n_train=300, n_test=100, n_classes=3, n_features=4,
    length=60, irregularity=0.4, noise=0.1, seed=42,
)
```

Optional `min_length` and `max_length` enable variable trajectory lengths;
`time_end`, `class_variability`, and `missing_rate` provide the remaining
controls from the research generator.

## Quick start

Input data is long format: one row per trajectory observation, with an ID, a
numeric time, and one column per feature. The optional label is metadata used
for plotting and evaluation.

```python
from neural_ode import (
    create_model, evaluate_performance, prep_batches, prep_data, predict,
    train_fit_gmm, train_finetune, train_pretrain,
)

data = prep_data(
    "trajectories.csv", id_col="trajectory_id", time_col="time",
    feature_cols=["feature_0", "feature_1"], label_col="label",
    cutoff=0.7, cutoff_mode="fraction",
)
loaders = prep_batches(data, batch_size=16, val_split=0.2, normalize=True)
model = create_model(len(data.feature_names), {"latent_dim": 6, "k": 3})
pretrained = train_pretrain(loaders, model_config=model, epochs=20)
gmm = train_fit_gmm(pretrained.model, loaders)
finetuned = train_finetune(gmm.model, loaders, epochs=20)
predictions = predict(finetuned.model, loaders)

test_data = prep_data(
    "test_trajectories.csv", id_col="trajectory_id", time_col="time",
    feature_cols=["feature_0", "feature_1"], label_col="label",
    cutoff=0.7, cutoff_mode="fraction",
)
test_loader = prep_batches(
    test_data, batch_size=16, val_split=0, normalize=True, shuffle=False
).train
scores = evaluate_performance(finetuned.model, test_loader)
```

`evaluate_performance` returns masked forecast MSE, supervised clustering
scores (NMI, ARI, and Hungarian-matched accuracy) when labels are available,
and label-free silhouette, Davies–Bouldin, and Calinski–Harabasz scores after
GMM fitting. It returns `None` for unavailable metrics; before GMM fitting,
only forecast MSE is evaluated. The dictionary keys are `mse`, `nmi`, `ari`,
`acc`, `silhouette`, `davies_bouldin`, and `calinski_harabasz`. A test loader
created separately with `prep_batches` carries its scaling metadata, so its
values are converted to the model's training scale during evaluation.

See [`examples/basic_workflow.py`](examples/basic_workflow.py) for a runnable
version and the function docstrings for argument details. Plotting helpers
`plot_input_trajectories`, `plot_predicted_trajectories`, `plot_latent_space`,
and `plot_training_loss` are also exported from `neural_ode`. Input trajectory
plots color and summarize by label when labels are available; without labels,
they use one neutral color and an overall median with an interquartile band.
`plot_predicted_trajectories` automatically uses ground-truth labels after
pretraining when labels are available, and a neutral color otherwise. After
GMM fitting it defaults to cluster colors; pass `color_by=None` to always use
one neutral color.

## Development

Run the focused package tests with:

```sh
python -m pytest
```

This private package is extracted from the research code in the parent
workspace. See [`NOTICE.md`](NOTICE.md) for source provenance and license
status.
