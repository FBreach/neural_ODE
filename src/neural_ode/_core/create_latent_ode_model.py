###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################


import torch.nn as nn

from .utils import create_mlp
# from .latent_ode import LatentODE, LatentODEADNIMixture, LatentODEPostHocGMM
from .latent_ode import LatentODE
from .encoder_decoder import Encoder_z0_ODE_RNN, Encoder_z0_RNN
from .diffeq_solver import DiffeqSolverParallel

from .ode_func import ODEFunc


SUPPORTED_ODE_METHODS = ("euler", "rk2", "rk4")
# POSTHOC_GMM_LATENTODE_ODERNN = "LatentODE_odernn_vae_gmm"
# ADNI_MIXTURE_LATENTODE_ODERNN = "LatentODE_odernn_adni_mixture"


def _cfg_value(section, key):
    if isinstance(section, dict):
        return section[key]
    return getattr(section, key)


def _encoder_name_for_model(model_name):
    if model_name == "LatentODE_rnn":
        return "rnn"
    if model_name in (
        "LatentODE_odernn",
#         POSTHOC_GMM_LATENTODE_ODERNN,
#         ADNI_MIXTURE_LATENTODE_ODERNN,
    ):
        return "odernn"
    raise ValueError(f"Unknown LatentODE model_name: {model_name}")


def create_LatentODE_model(
    input_dim,
    model_name,
    latent_dim=6,
    networks=None,
#     poisson=False,
    use_delta_t=True,
    method="rk4",
    odeint_rtol=1e-3,
    odeint_atol=1e-4,
    k=2,
    run_backwards=True,
#     lambda_dx=1.0,
#     dx_decoder_type="categorical",
#     feature_names=None,
):
    """Create a Latent ODE model from the explicit model.networks schema."""
    if networks is None:
        raise ValueError("model.networks must be provided explicitly.")
    if method not in SUPPORTED_ODE_METHODS:
        supported = ", ".join(SUPPORTED_ODE_METHODS)
        raise ValueError(
            f"Unsupported ODE method '{method}'. Use one of: {supported}."
        )

    encoder_name = _encoder_name_for_model(model_name)

    dim = latent_dim + 1
    generative_ode_cfg = _cfg_value(networks, "generative_ode")
    ode_func_net = create_mlp(
        dim,
        _cfg_value(generative_ode_cfg, "hidden_dims"),
        latent_dim,
        nonlinear=nn.Tanh,
        name="generative_ode.hidden_dims",
    )

    gen_ode_func = ODEFunc(
        input_dim=input_dim,
        latent_dim=latent_dim,
        ode_func_net=ode_func_net,
    )

    z0_diffeq_solver = None
    enc_input_dim = int(input_dim) * 2  # we concatenate the mask
    gen_data_dim = input_dim

    z0_dim = latent_dim
#     if poisson:
#         z0_dim += latent_dim  # predict the initial poisson rate

    if encoder_name == "odernn":
        encoder_cfg = _cfg_value(networks, "z0_encoder_odernn")
        recognition_state_dim = int(_cfg_value(encoder_cfg, "state_dim"))
        ode_func_net = create_mlp(
            recognition_state_dim + 1,
            _cfg_value(encoder_cfg, "recognition_ode_hidden_dims"),
            recognition_state_dim,
            nonlinear=nn.Tanh,
            name="z0_encoder_odernn.recognition_ode_hidden_dims",
        )

        rec_ode_func = ODEFunc(
            input_dim=enc_input_dim,
            latent_dim=recognition_state_dim,
            ode_func_net=ode_func_net,
        )

        z0_diffeq_solver = DiffeqSolverParallel(
            enc_input_dim,
            rec_ode_func,
            method,
            latent_dim,
            odeint_rtol=1e-3,
            odeint_atol=1e-4,
        )

        encoder_z0 = Encoder_z0_ODE_RNN(
            recognition_state_dim,
            enc_input_dim,
            z0_diffeq_solver,
            z0_dim=z0_dim,
            gru_update_hidden_dims=_cfg_value(
                encoder_cfg,
                "gru_update_hidden_dims",
            ),
            posterior_hidden_dims=_cfg_value(
                encoder_cfg,
                "posterior_hidden_dims",
            ),
        )

    elif encoder_name == "rnn":
        encoder_cfg = _cfg_value(networks, "z0_encoder_rnn")
        encoder_z0 = Encoder_z0_RNN(
            z0_dim,
            enc_input_dim,
            hidden_dims=_cfg_value(encoder_cfg, "hidden_dims"),
            posterior_hidden_dims=_cfg_value(
                encoder_cfg,
                "posterior_hidden_dims",
            ),
            use_delta_t=use_delta_t,
        )
    diffeq_solver = DiffeqSolverParallel(
        gen_data_dim,
        gen_ode_func,
        method,
        latent_dim,
        odeint_rtol=odeint_rtol,
        odeint_atol=odeint_atol,
    )

#     if model_name == POSTHOC_GMM_LATENTODE_ODERNN:
#         model_cls = LatentODEPostHocGMM
#         extra_model_kwargs = {}
#     elif model_name == ADNI_MIXTURE_LATENTODE_ODERNN:
#         model_cls = LatentODEADNIMixture
#         extra_model_kwargs = {
#             "lambda_dx": lambda_dx,
#             "dx_decoder_type": dx_decoder_type,
#             "feature_names": feature_names,
#         }
#     else:
#         model_cls = LatentODE
#         extra_model_kwargs = {}
#
    model = LatentODE(
        input_dim=gen_data_dim,
        latent_dim=latent_dim,
        encoder_z0=encoder_z0,
        diffeq_solver=diffeq_solver,
        k=k,
        run_backwards=run_backwards,
        emission_hidden_dims=_cfg_value(
            _cfg_value(networks, "emission"),
            "hidden_dims",
        ),
#         **extra_model_kwargs,
    )

    return model
