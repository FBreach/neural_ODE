"""Model construction helpers."""

from ._compat import to_plain_dict
from ._core.create_latent_ode_model import create_LatentODE_model


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


def create_model(input_dim, model_config=None):
    """Construct a latent ODE model with a Gaussian mixture prior.

    Parameters
    ----------
    input_dim : int
        Number of observed features per time point.
    model_config : mapping or None, optional
        Model options. Supported keys and defaults are ``model_name``
        (``'LatentODE_odernn'``), ``latent_dim`` (10), ``n_clusters`` (2,
        with ``k`` accepted as a fallback), ``use_delta_t`` (True),
        ``method`` (``'rk4'``), ``odeint_rtol`` (``1e-3``),
        ``odeint_atol`` (``1e-4``) and ``run_backwards`` (True).
        ``networks`` replaces the complete network configuration; by default
        it uses ``DEFAULT_NETWORKS``. The default ODE-RNN configuration has
        ``z0_encoder_odernn`` keys ``state_dim`` (100),
        ``recognition_ode_hidden_dims`` ([32]), ``gru_update_hidden_dims``
        ([100]) and ``posterior_hidden_dims`` ([100]);
        ``generative_ode.hidden_dims`` is [32] and
        ``emission.hidden_dims`` is [20].

    Returns
    -------
    neural_ode._core.latent_ode.LatentODE
        Untrained PyTorch model on the CPU, with encoder, ODE solver,
        decoder and mixture-prior parameters.

    Raises
    ------
    ValueError
        If the model name, solver method or network configuration is invalid.

    """
    cfg = to_plain_dict(model_config or {})
    return create_LatentODE_model(
        input_dim=input_dim,
        model_name=cfg.get("model_name", "LatentODE_odernn"),
        latent_dim=int(cfg.get("latent_dim", 10)),
        networks=cfg.get("networks", DEFAULT_NETWORKS),
        use_delta_t=cfg.get("use_delta_t", True),
        method=cfg.get("method", "rk4"),
        odeint_rtol=cfg.get("odeint_rtol", 1e-3),
        odeint_atol=cfg.get("odeint_atol", 1e-4),
        k=int(cfg.get("n_clusters", cfg.get("k", 2))),
        run_backwards=cfg.get("run_backwards", True),
    )
