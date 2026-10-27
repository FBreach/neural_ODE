import pytest
import torch
import numpy.testing as npt

DEFAULT_NETWORKS = {
    "z0_encoder_odernn": {
        "state_dim": 100,
        "recognition_ode_hidden_dims": [32],
        "gru_update_hidden_dims": [100],
        "posterior_hidden_dims": [100],
    },
    "generative_ode": {"hidden_dims": [32]},
    "emission": {"hidden_dims": [20]},
}



def test_cfg_value_non_dict():
    from neural_ode._core.create_latent_ode_model import _cfg_value

    class Non_dictionary_class:
        def __init__(self, attribute1, attribute2):
            self.attribute1 = attribute1
            self.attribute2 = attribute2

    test_section = Non_dictionary_class("First Attribute", 2)

    result1 = _cfg_value(test_section, "attribute1")
    result2 = _cfg_value(test_section, "attribute2")

    assert result1 == "First Attribute"
    assert result2 == 2

def test_cfg_value__dict():
    from neural_ode._core.create_latent_ode_model import _cfg_value

   

    test_section = {"attribute1": "First Attribute", "attribute2" : 2}

    result1 = _cfg_value(test_section, "attribute1")
    result2 = _cfg_value(test_section, "attribute2")

    assert result1 == "First Attribute"
    assert result2 == 2

def test_builds_small_odernn_latent_ode():
    NETWORKS = {
        "z0_encoder_odernn": {
            "state_dim": 4,
            "recognition_ode_hidden_dims": [4],
            "gru_update_hidden_dims": [4],
            "posterior_hidden_dims": [4],
        },
        "generative_ode": {"hidden_dims": [4]},
        "emission": {"hidden_dims": [4]},
    }

    from neural_ode._core.create_latent_ode_model import create_LatentODE_model
    from neural_ode._core.encoder_decoder import Encoder_z0_ODE_RNN
    from neural_ode._core.latent_ode import LatentODE

    
    model = create_LatentODE_model(
        input_dim=2,
        model_name="LatentODE_odernn",
        latent_dim=3,
        networks=NETWORKS,
        k=2,
    )

    assert isinstance(model, LatentODE)
    assert isinstance(model.encoder_z0, Encoder_z0_ODE_RNN)
    assert model.input_dim == 2
    assert model.k.item() == 2


RNN_NETWORKS = {
    "z0_encoder_rnn": {
        "hidden_dims": [4],
        "posterior_hidden_dims": [4],
    },
    "generative_ode": {"hidden_dims": [4]},
    "emission": {"hidden_dims": [4]},
}
def test_builds_small_rnn_latent_ode():   

    from neural_ode._core.create_latent_ode_model import create_LatentODE_model
    from neural_ode._core.encoder_decoder import Encoder_z0_RNN
    from neural_ode._core.latent_ode import LatentODE

    
    model = create_LatentODE_model(
        input_dim=2,
        model_name="LatentODE_rnn",
        latent_dim=3,
        networks=RNN_NETWORKS,
    )

    assert isinstance(model, LatentODE)
    assert isinstance(model.encoder_z0, Encoder_z0_RNN)
    assert model.k.item() == 2

from neural_ode._core.latent_ode import LatentODE, LatentODEADNIMixture, LatentODEPostHocGMM
@pytest.mark.parametrize(
    "model_name", "expected_type",
    [("LatentODE_odernn_vae_gmm", LatentODEPostHocGMM),
     ("LatentODE_odernn_adni_mixture", LatentODEADNIMixture),
     ("LatentODE_odernn", LatentODE)
    ]
)

def test_makes_correct_model_variant(model_name, expected_type):
    from neural_ode._core.create_latent_ode_model import create_LatentODE_model
    model = create_LatentODE_model(input_dim=2,
            model_name=model_name,
            latent_dim=3,
            networks=RNN_NETWORKS,
        )
    assert isinstance(model, expected_type)