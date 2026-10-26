Getting started
===============

Installation
------------

From the repository root, install the package into your active Python
environment (Python 3.10 or later):

.. code-block:: sh

   python -m pip install -e .

Prepare and train
-----------------

Input tables use one row per observation, with a trajectory identifier,
numeric time and one column per feature. Labels are optional metadata.
Missing feature values are supported.

This example generates data in memory, trains a model, forecasts held-out
observations and evaluates a separate test set:

.. code-block:: python

   from neural_ode import (
       create_model, evaluate_performance, generate_synthetic_data,
       plot_predicted_trajectories, predict, prep_batches, prep_data,
       train_finetune, train_fit_gmm, train_pretrain,
   )

   train_df, test_df = generate_synthetic_data(
       n_train=60, n_test=20, n_classes=3, n_features=2, seed=42,
   )
   data_options = {
       "id_col": "trajectory_id",
       "feature_cols": ["feature_0", "feature_1"],
       "label_col": "label",
       "cutoff": 0.7,
       "cutoff_mode": "fraction",
   }
   data = prep_data(train_df, **data_options)
   loaders = prep_batches(data, batch_size=16, normalize=True)
   model = create_model(2, {"latent_dim": 6, "n_clusters": 3})
   pretrained = train_pretrain(loaders, model_config=model, epochs=20)
   clustered = train_fit_gmm(pretrained.model, loaders)
   trained = train_finetune(clustered.model, loaders, epochs=20)

   predictions = predict(trained.model, loaders)
   figures = plot_predicted_trajectories(predictions)

   test_data = prep_data(test_df, **data_options)
   test_loader = prep_batches(
       test_data, val_split=0, normalize=True, shuffle=False,
   ).train
   scores = evaluate_performance(trained.model, test_loader)

``predict`` selects the test split from a loader bundle when present,
otherwise validation, then training. Predictions retain the model's input
scale. ``evaluate_performance`` converts separately normalised loader values
to the model's training scale using the stored scaling metadata.

With no cutoff, the model reconstructs complete trajectories. With a cutoff,
it uses the context segment to forecast the remaining observations.

See :doc:`api` for parameter definitions and result types.
