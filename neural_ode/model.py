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
    """Create LatentODE with a small, documented config dictionary."""
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
