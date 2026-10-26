"""Helpers used by the vendored Latent ODE implementation."""

import torch
from torch import nn


def split_last_dim(data):
    return data.chunk(2, dim=-1)


def init_network_weights(net, std=0.1):
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=std)
            nn.init.constant_(module.bias, val=0)


def get_device(tensor):
    return tensor.device


def sample_standard_gaussian(mu, sigma):
    mu, sigma = mu.float(), sigma.float()
    return torch.randn_like(mu) * sigma + mu


def reverse(tensor):
    return tensor.flip(0)


def hidden_dims_list(hidden_dims, name="hidden_dims", *, allow_empty=True):
    if hidden_dims is None:
        raise ValueError(f"{name} must be provided explicitly.")
    dims = [int(dim) for dim in hidden_dims]
    if not allow_empty and not dims:
        raise ValueError(f"{name} must contain at least one hidden dimension.")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"{name} must contain positive dimensions, got {dims}.")
    return dims


def create_mlp(
    n_inputs,
    hidden_dims,
    n_outputs,
    *,
    nonlinear=nn.Tanh,
    final_activation=None,
    name="hidden_dims",
):
    layers = []
    previous_dim = int(n_inputs)
    for hidden_dim in hidden_dims_list(hidden_dims, name):
        layers.extend((nn.Linear(previous_dim, hidden_dim), nonlinear()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, int(n_outputs)))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


def check_mask(data, mask):
    """Check that masked-out values are zero and the mask is binary."""
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only zero and one values")
    if torch.any(data[mask == 0] != 0):
        raise ValueError("masked-out data values must be zero")
