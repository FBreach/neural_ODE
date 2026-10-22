"""Masked reconstruction error used by the Latent ODE objective."""

import torch
from torch import nn

from .utils import get_device


def compute_masked_likelihood(mu, data, mask, likelihood_func):
    n_samples, n_trajectories, _, n_features = data.size()
    values = []
    for sample_idx in range(n_samples):
        for trajectory_idx in range(n_trajectories):
            for feature_idx in range(n_features):
                observed = mask[sample_idx, trajectory_idx, :, feature_idx].bool()
                values.append(
                    likelihood_func(
                        mu[sample_idx, trajectory_idx, observed, feature_idx],
                        data[sample_idx, trajectory_idx, observed, feature_idx],
                        indices=(sample_idx, trajectory_idx, feature_idx),
                    )
                )
    result = torch.stack(values, 0).to(get_device(data))
    return result.reshape(n_samples, n_trajectories, n_features).mean(-1).transpose(0, 1)


def mse(mu, data, indices=None):
    if mu.size(-1) > 0:
        return nn.MSELoss()(mu, data)
    return torch.zeros([1], device=get_device(data)).squeeze()


def compute_mse(mu, data, mask=None):
    if mu.ndim == 3:
        mu = mu.unsqueeze(0)
    if data.ndim == 2:
        data = data.unsqueeze(0).unsqueeze(2)
    elif data.ndim == 3:
        data = data.unsqueeze(0)

    n_samples, n_trajectories, n_timepoints, n_features = mu.size()
    if data.size(-1) != n_features:
        raise ValueError("mu and data must have the same feature dimension")
    if mask is None:
        return mse(
            mu.reshape(n_samples * n_trajectories, n_timepoints * n_features),
            data.reshape(n_samples * n_trajectories, n_timepoints * n_features),
        )
    return compute_masked_likelihood(mu, data, mask, mse)
