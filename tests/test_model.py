import matplotlib
matplotlib.use("Agg")

#import pandas as pd
#import pytest
import torch

#from neural_ode import create_model

def test_create_model_uses_private_core_implementation():
    from neural_ode import create_model

    model = create_model(1, {"latent_dim": 2, "k": 2})
    assert isinstance(model, torch.nn.Module)
    assert model.k.item() == 2

