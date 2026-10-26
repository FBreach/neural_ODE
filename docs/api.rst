API reference
=============

The supported API is exported from ``neural_ode``. This reference reads its
NumPy-style docstrings directly from the installed package. Private
implementation modules under ``_core`` are excluded.

.. py:module:: neural_ode

Data preparation
----------------

.. autofunction:: prep_data
.. autofunction:: prep_batches

.. autoclass:: Trajectory
.. autoclass:: TrajectoryData
.. autoclass:: DataLoaders

Synthetic data
--------------

.. autofunction:: generate_synthetic_data

Model construction
------------------

.. autofunction:: create_model

Training
--------

.. autofunction:: train_pretrain
.. autofunction:: train_fit_gmm
.. autofunction:: train_finetune

.. autoclass:: TrainingResult
.. autoclass:: GMMResult

Prediction and evaluation
-------------------------

.. autofunction:: predict
.. autofunction:: evaluate_performance

.. autoclass:: PredictionTrajectory
.. autoclass:: PredictionResult

Plotting
--------

.. autofunction:: plot_input_trajectories
.. autofunction:: plot_predicted_trajectories
.. autofunction:: plot_latent_space
.. autofunction:: plot_training_loss
