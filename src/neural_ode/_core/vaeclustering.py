from dataclasses import dataclass
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from torch.distributions.normal import Normal
from torch.nn import functional as F

from .likelihood_eval import compute_mse
from .utils import create_mlp, hidden_dims_list

RECON_STD_MIN = 1e-3
RECON_STD_MAX = 1e3
ADNI_FEATURE_NAMES = ("gm_score", "CENTILOID", "ADAS13", "MMSE")
ADNI_CONTINUOUS_FEATURES = frozenset(("gm_score", "ADAS13"))
ADNI_DISCRETIZED_GAUSSIAN_FEATURES = frozenset(("CENTILOID",))
ADNI_GAUSSIAN_FEATURES = (
    ADNI_CONTINUOUS_FEATURES | ADNI_DISCRETIZED_GAUSSIAN_FEATURES
)


def validate_adni_feature_names(feature_names):
    if feature_names is None:
        return tuple(ADNI_FEATURE_NAMES)
    if isinstance(feature_names, str):
        names = (feature_names,)
    else:
        names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("ADNI feature_names must not be empty.")
    unknown = sorted(set(names) - set(ADNI_FEATURE_NAMES))
    if unknown:
        supported = ", ".join(ADNI_FEATURE_NAMES)
        raise ValueError(
            f"Unsupported ADNI feature_names: {unknown}. "
            f"Supported features: {supported}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"ADNI feature_names must be unique, got {list(names)}.")
    return names


@dataclass
class Results:
    """Loss output returned by VAEClustering training/evaluation steps."""
    loss: torch.Tensor | None = None
    likelihood: torch.Tensor | None = None
    mse: torch.Tensor | None = None
    kl_first_p: torch.Tensor | None = None
    std_first_p: torch.Tensor | None = None
    recon_std_mean: torch.Tensor | None = None
    recon_std_min: torch.Tensor | None = None
    recon_std_max: torch.Tensor | None = None
    metrics: dict | None = None


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims):
        super().__init__()
        self.proj = create_mlp(
            latent_dim,
            hidden_dims,
            output_dim * 2,
            name="emission.hidden_dims",
        )

    def forward(self, zt):
        out = self.proj(zt)
        mu, logvar = out.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return mu, std


class ADNIMixedDecoder(nn.Module):
    """Shared emission trunk with ADNI feature-specific likelihood heads."""

    def __init__(self, latent_dim, hidden_dims, feature_names=None):
        super().__init__()
        self.feature_names = validate_adni_feature_names(feature_names)
        dims = hidden_dims_list(
            hidden_dims,
            "emission.hidden_dims",
            allow_empty=False,
        )
        trunk_dim = dims[-1]
        self.trunk = create_mlp(
            latent_dim,
            dims[:-1],
            trunk_dim,
            name="emission.hidden_dims",
        )
        self.gaussian_heads = nn.ModuleDict(
            {
                feature_name: nn.Linear(trunk_dim, 2)
                for feature_name in self.feature_names
                if feature_name in ADNI_GAUSSIAN_FEATURES
            }
        )
        if "MMSE" in self.feature_names:
            self.mmse_head = nn.Linear(trunk_dim, 1)
            gap_init = math.log(math.exp(1.0) - 1.0)
            self.mmse_threshold_deltas = nn.Parameter(torch.full((29,), gap_init))
        else:
            self.mmse_head = None
            self.register_parameter("mmse_threshold_deltas", None)

    @staticmethod
    def _positive_std(raw_std):
        return F.softplus(raw_std).clamp(min=RECON_STD_MIN, max=RECON_STD_MAX)

    def mmse_thresholds(self):
        if self.mmse_threshold_deltas is None:
            raise RuntimeError("MMSE thresholds are unavailable because MMSE is not decoded.")
        tau0 = self.mmse_threshold_deltas.new_zeros(1)
        gaps = F.softplus(self.mmse_threshold_deltas) + 1e-4
        return torch.cat((tau0, tau0 + torch.cumsum(gaps, dim=0)), dim=0)

    def mmse_probabilities(self, eta):
        thresholds = self.mmse_thresholds().to(device=eta.device, dtype=eta.dtype)
        cdf = torch.sigmoid(thresholds - eta.unsqueeze(-1))
        p0 = cdf[..., :1]
        pmid = cdf[..., 1:] - cdf[..., :-1]
        p30 = 1.0 - cdf[..., -1:]
        return torch.cat((p0, pmid, p30), dim=-1).clamp_min(1e-12)

    def forward(self, zt):
        h = self.trunk(zt)

        mean_parts = []
        std_parts = []
        feature_params = {}
        for feature_name in self.feature_names:
            if feature_name in ADNI_GAUSSIAN_FEATURES:
                mu, raw_std = self.gaussian_heads[feature_name](h).chunk(2, dim=-1)
                std = self._positive_std(raw_std)
                mean_parts.append(mu)
                std_parts.append(std)
                feature_params[feature_name] = {
                    "mu": mu.squeeze(-1),
                    "std": std.squeeze(-1),
                }
            elif feature_name == "MMSE":
                eta = self.mmse_head(h).squeeze(-1)
                probs = self.mmse_probabilities(eta)
                categories = torch.arange(
                    31,
                    device=zt.device,
                    dtype=zt.dtype,
                )
                expected = (probs * categories).sum(dim=-1, keepdim=True)
                mean_parts.append(expected)
                std_parts.append(torch.ones_like(expected))
                feature_params[feature_name] = {"eta": eta}
            else:
                raise RuntimeError(f"Unsupported ADNI feature: {feature_name}")

        return {
            "mean": torch.cat(mean_parts, dim=-1),
            "std": torch.cat(std_parts, dim=-1),
            "feature_params": feature_params,
        }

    def feature_log_probs(self, params, normalized_targets, raw_targets):
        normalized_targets = normalized_targets.unsqueeze(0).to(
            params["mean"].device
        )
        raw_targets = raw_targets.unsqueeze(0).to(params["mean"].device)
        log_probs = {}
        feature_params = params["feature_params"]
        for feature_idx, feature_name in enumerate(self.feature_names):
            feature_param = feature_params[feature_name]
            if feature_name in ADNI_CONTINUOUS_FEATURES:
                target = normalized_targets[..., feature_idx]
                log_probs[feature_name] = Normal(
                    feature_param["mu"],
                    feature_param["std"],
                ).log_prob(target)
            elif feature_name in ADNI_DISCRETIZED_GAUSSIAN_FEATURES:
                target = raw_targets[..., feature_idx]
                dist = Normal(feature_param["mu"], feature_param["std"])
                prob = dist.cdf(target + 0.5) - dist.cdf(target - 0.5)
                log_probs[feature_name] = prob.clamp_min(1e-12).log()
            elif feature_name == "MMSE":
                target = raw_targets[..., feature_idx].round().long().clamp(0, 30)
                probs = self.mmse_probabilities(feature_param["eta"])
                prob = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                log_probs[feature_name] = prob.clamp_min(1e-12).log()
            else:
                raise RuntimeError(f"Unsupported ADNI feature: {feature_name}")
        return log_probs


class VAEClustering(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        k=2,
        run_backwards=True,
        emission_hidden_dims=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        # self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        # self.train_classif_w_reconstr = train_classif_w_reconstr

        if emission_hidden_dims is None:
            raise ValueError("emission.hidden_dims must be provided explicitly.")
        self.decoder = Decoder(latent_dim, input_dim, emission_hidden_dims)

        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("latent_dim", torch.tensor(latent_dim, dtype=torch.int))

        # Mixture prior p(z|c)
        self.prior_mu = nn.Parameter(torch.randn(k, latent_dim) * 0.5)
        self.log_prior_var = nn.Parameter(torch.zeros(k, latent_dim))

        # p(c) learnable, these are logits. Start from a finite uniform prior.
        self.log_pi = nn.Parameter(torch.zeros(k))
        self.run_backwards = run_backwards

    def prepare_recon_std(self, recon_std):
        return recon_std.clamp(min=RECON_STD_MIN, max=RECON_STD_MAX)

    def recon_std_stats(self, recon_std, mask=None):
        values = recon_std
        if mask is not None:
            mask_bool = mask.bool()
            while mask_bool.ndim < recon_std.ndim:
                mask_bool = mask_bool.unsqueeze(0)
            try:
                mask_bool = torch.broadcast_to(mask_bool, recon_std.shape)
                masked_values = recon_std[mask_bool]
                if masked_values.numel() > 0:
                    values = masked_values
            except RuntimeError:
                values = recon_std
        return (
            values.mean().detach(),
            values.min().detach(),
            values.max().detach(),
        )

    def observed_value_count(self, data_to_predict, mask_predicted_data=None):
        if mask_predicted_data is not None:
            return mask_predicted_data.sum((-2, -1)).clamp_min(1.0)
        return data_to_predict.new_full(
            data_to_predict.shape[:-2],
            data_to_predict.size(-2) * data_to_predict.size(-1),
        )

    def normalize_trajectory_term(self, values, obs_count):
        while obs_count.ndim < values.ndim:
            obs_count = obs_count.unsqueeze(0)
        return values / obs_count

    def normalized_reconstruction_log_prob(
        self,
        pred_y,
        recon_std,
        data_to_predict,
        mask_predicted_data=None,
    ):
        log_px = Normal(pred_y, recon_std).log_prob(data_to_predict)
        if mask_predicted_data is not None:
            log_px = log_px * mask_predicted_data
        obs_count = self.observed_value_count(data_to_predict, mask_predicted_data)
        return self.normalize_trajectory_term(log_px.sum((-2, -1)), obs_count)

    def get_mse(self, truth, pred_y, mask=None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        # n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        # In case more sample have been taken from the posterior
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)
        # shape: [1]
        return torch.mean(log_density_data)

    def q_c_given_z(self, z):
        """
        z: (B,D)
        returns:
        log_qc: (B,K)
        qc:     (B,K)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B, D = z.shape

        z_exp = z.unsqueeze(1)  # (B,1,D)
        mu = self.prior_mu.unsqueeze(0)  # (1,K,D)
        logvar = self.log_prior_var.unsqueeze(0)  # (1,K,D)
        var = torch.exp(logvar)  # (1,K,D)
        inv_var = 1.0 / var  # (1,K,D)

        const = D * math.log(2.0 * math.pi)
        log_det = logvar.sum(-1)  # (1,K)
        sq = (z_exp - mu) ** 2
        quad = (sq * inv_var).sum(-1)  # (B,K)

        log_p_z_c = -0.5 * (const + log_det + quad)  # (B,K)
        log_p_c = F.log_softmax(self.log_pi, dim=0)  # (K,)

        log_q = log_p_z_c + log_p_c.unsqueeze(0)  # (B,K)
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
        qc = log_q.exp()
        return log_q, qc

    def mixture_prior_kl(self, mu, std, z0, obs_count=None):
        """Analytic KL(q(z0,c|x) || p(z0,c)) under the diagonal GMM prior."""
        mu = mu.squeeze(0) if mu.dim() == 3 and mu.size(0) == 1 else mu
        std = std.squeeze(0) if std.dim() == 3 and std.size(0) == 1 else std
        std = std.abs().clamp_min(1e-6)
        D = mu.size(-1)

        log_qc, qc = self.q_c_given_z(mu)
        mu_c = self.prior_mu.unsqueeze(0)
        logvar_c = self.log_prior_var.unsqueeze(0)
        inv_var_c = 1.0 / logvar_c.exp()

        mu_z = mu.unsqueeze(1)
        var_z = std.pow(2).unsqueeze(1)
        const = D * math.log(2.0 * math.pi)
        log_det_c = logvar_c.sum(-1)
        sq_term = var_z + (mu_z - mu_c) ** 2
        mah_dist = (sq_term * inv_var_c).sum(-1)
        e_log_p_z_c = -0.5 * (const + log_det_c + mah_dist)

        log_pi = F.log_softmax(self.log_pi, dim=0).unsqueeze(0)
        e_log_pzc = (qc * (e_log_p_z_c + log_pi)).sum(-1)

        log_qz = Normal(mu, std).log_prob(z0).sum(-1)
        e_log_qc = (qc * log_qc).sum(-1)
        kldiv_z0 = log_qz + e_log_qc - e_log_pzc

        if obs_count is not None:
            kldiv_z0 = self.normalize_trajectory_term(kldiv_z0, obs_count)
        return kldiv_z0

    def standard_normal_kl(self, mu, std, obs_count=None):
        """Analytic KL(q(z0|x) || N(0, I))."""
        mu = mu.squeeze(0) if mu.dim() == 3 and mu.size(0) == 1 else mu
        std = std.squeeze(0) if std.dim() == 3 and std.size(0) == 1 else std
        std = std.abs().clamp_min(1e-6)
        var = std.pow(2)
        logvar = torch.log(var)
        kl = 0.5 * (mu.pow(2) + var - logvar - 1.0).sum(-1)
        if obs_count is not None:
            kl = self.normalize_trajectory_term(kl, obs_count)
        return kl

    @abstractmethod
    def get_reconstruction_det(
        self,
        time_steps_to_predict,
        truth,
        truth_time_steps,
        mask=None,
        n_traj_samples=1,
        run_backwards=True,
        mode=None,
    ):
        pass

    @abstractmethod
    def get_reconstruction(
        self,
        time_steps_to_predict,
        truth,
        truth_time_steps,
        mask=None,
        n_traj_samples=1,
        # TODO: run_backwards
        run_backwards=True,
        mode=None,
        use_sampling=True,
    ):
        """
        Returns:
            recon_m: _description_
            recon_std:
            all_extra_info:  "first_point": (first_point_mu, first_point_std, first_point_enc), "latent_traj": sol_y.detach(),
        """
        pass

    def compute_all_losses(
        self, batch_dict, n_traj_samples=1, kl_coef=1.0, device="cpu", use_sampling=True
    )->Results:
        tp_to_predict = batch_dict["tp_to_predict"].to(device)
        observed_data = batch_dict["observed_data"].to(device)
        observed_tp = batch_dict["observed_tp"].to(device)
        observed_mask = batch_dict["observed_mask"].to(device)
        data_to_predict = batch_dict["data_to_predict"].to(device)
        mask_predicted_data = batch_dict["mask_predicted_data"].to(device)
        mode = batch_dict.get("mode")

        # Condition on subsampled points
        # Make predictions for all the points
        # pred_y: n_traj_samples, B, T, F
        # recon_std: n_traj_samples, B, T, F
        pred_y, recon_std, info = self.get_reconstruction(
            tp_to_predict,
            observed_data,
            observed_tp,
            mask=observed_mask,
            n_traj_samples=n_traj_samples,
            mode=mode,
            use_sampling=use_sampling,
        )

        D = self.latent_dim
        # KL(q(z|x_ctx) || N(0, I))
        mu, std, z0 = info["first_point"]
        mu, std = mu.squeeze(0), std.squeeze(0)
        std = std.abs()
        assert torch.sum(std < 0) == 0.0

        # ----- q(c|z) with z = μ(x_ctx) -----
        log_qc, qc = self.q_c_given_z(mu)  # (B,K), (B,K)

        # ----- analytic KL(q(z,c|x_ctx) || p(z,c)) -----
        mu_c = self.prior_mu.unsqueeze(0)  # (1,K,D)
        logvar_c = self.log_prior_var.unsqueeze(0)  # (1,K,D)
        var_c = logvar_c.exp()
        inv_var_c = 1.0 / var_c

        mu_z = mu.unsqueeze(1)  # (B,1,D)
        var_z = (std**2).unsqueeze(1)  # (B,1,D)

        const = D * math.log(2.0 * math.pi)
        log_det_c = logvar_c.sum(-1)  # (1,K)
        sq_term = var_z + (mu_z - mu_c) ** 2  # (B,K,D)
        mah_dist = (sq_term * inv_var_c).sum(-1)  # (B,K)
        E_log_p_z_c = -0.5 * (const + log_det_c + mah_dist)  # (B,K)

        log_pi = F.log_softmax(self.log_pi, dim=0)  # (K,)
        log_pi_exp = log_pi.unsqueeze(0)  # (1,K)

        E_log_pzc = (qc * (E_log_p_z_c + log_pi_exp)).sum(-1)  # (B,)

        # log q(z|x_ctx)
        log_qz = Normal(mu, std).log_prob(z0).sum(-1)  # (B,)

        # E_q(z,c|x_ctx)[log q(c|z)]
        E_log_qc = (qc * log_qc).sum(-1)  # (B,)
        E_log_qzc = log_qz + E_log_qc  # (B,)
        kldiv_z0 = E_log_qzc - E_log_pzc  # (B,)

        if torch.isnan(kldiv_z0).any():
            print(mu)
            print(std)
            raise Exception("kldiv_z0 is Nan!")

        # Match the per-observed-value scale used for the reconstruction term.
        obs_count = self.observed_value_count(data_to_predict, mask_predicted_data)
        kldiv_z0 = self.normalize_trajectory_term(kldiv_z0, obs_count)

        recon_std = self.prepare_recon_std(recon_std)
        recon_std_mean, recon_std_min, recon_std_max = self.recon_std_stats(
            recon_std,
            mask=mask_predicted_data,
        )
        log_px = self.normalized_reconstruction_log_prob(
            pred_y,
            recon_std,
            data_to_predict,
            mask_predicted_data=mask_predicted_data,
        )

        mse = self.get_mse(data_to_predict, pred_y, mask=mask_predicted_data)

        ################################
        # Compute CE loss for binary classification on Physionet
        # IWAE loss
        loss = -torch.logsumexp(log_px - kl_coef * kldiv_z0, 0)
        # if torch.isnan(loss):
        # loss = - torch.mean(log_px - kl_coef * kldiv_z0)

        results = Results(
            loss=torch.mean(loss),
            likelihood=torch.mean(log_px).detach(),
            mse=torch.mean(mse).detach(),
            kl_first_p=torch.mean(kldiv_z0).detach(),
            std_first_p=torch.mean(std).detach(),
            recon_std_mean=recon_std_mean,
            recon_std_min=recon_std_min,
            recon_std_max=recon_std_max,
        )
        return results

    def compute_all_losses_pretrain_auto(
            self, batch_dict, n_traj_samples=1, kl_coef=1.0, device="cpu"
        )->Results:
        tp_to_predict = batch_dict["tp_to_predict"].to(device)
        observed_data = batch_dict["observed_data"].to(device)
        observed_tp = batch_dict["observed_tp"].to(device)
        observed_mask = batch_dict["observed_mask"].to(device)
        data_to_predict = batch_dict["data_to_predict"].to(device)
        mask_predicted_data = batch_dict["mask_predicted_data"].to(device)
        mode = batch_dict.get("mode")

        # Condition on subsampled points
        # Make predictions for all the points
        # pred_y: n_traj_samples, B, T, F
        # recon_std: n_traj_samples, B, T, F
        pred_y, _, _ = self.get_reconstruction_det(
            tp_to_predict,
            observed_data,
            observed_tp,
            mask=observed_mask,
            n_traj_samples=n_traj_samples,
            mode=mode,
        )

        mse = self.get_mse(data_to_predict, pred_y, mask=mask_predicted_data)

        nan = torch.tensor(float("nan"), device=device)
        results = Results(
            loss=mse,
            likelihood=nan,
            mse=mse.detach(),
            kl_first_p=nan,
            std_first_p=nan,
            recon_std_mean=nan,
            recon_std_min=nan,
            recon_std_max=nan,
        )

        return results

    def compute_all_losses_standard_vae(
            self, batch_dict, n_traj_samples=1, kl_coef=1.0, device="cpu", use_sampling=True
        )->Results:
        tp_to_predict = batch_dict["tp_to_predict"].to(device)
        observed_data = batch_dict["observed_data"].to(device)
        observed_tp = batch_dict["observed_tp"].to(device)
        observed_mask = batch_dict["observed_mask"].to(device)
        data_to_predict = batch_dict["data_to_predict"].to(device)
        mask_predicted_data = batch_dict["mask_predicted_data"].to(device)
        mode = batch_dict.get("mode")

        pred_y, recon_std, info = self.get_reconstruction(
            tp_to_predict,
            observed_data,
            observed_tp,
            mask=observed_mask,
            n_traj_samples=n_traj_samples,
            mode=mode,
            use_sampling=use_sampling,
        )

        mu, std, _z0 = info["first_point"]
        mu, std = mu.squeeze(0), std.squeeze(0).abs()
        assert torch.sum(std < 0) == 0.0

        recon_std = self.prepare_recon_std(recon_std)
        recon_std_mean, recon_std_min, recon_std_max = self.recon_std_stats(
            recon_std,
            mask=mask_predicted_data,
        )
        log_px = self.normalized_reconstruction_log_prob(
            pred_y,
            recon_std,
            data_to_predict,
            mask_predicted_data=mask_predicted_data,
        )

        var = std.pow(2).clamp_min(1e-12)
        logvar = torch.log(var)
        kldiv_z0 = 0.5 * (mu.pow(2) + var - logvar - 1.0).sum(-1)
        obs_count = self.observed_value_count(data_to_predict, mask_predicted_data)
        kldiv_z0 = self.normalize_trajectory_term(kldiv_z0, obs_count)

        if torch.isnan(kldiv_z0).any():
            print(mu)
            print(std)
            raise Exception("kldiv_z0 is Nan!")

        mse = self.get_mse(data_to_predict, pred_y, mask=mask_predicted_data)

        loss = -torch.logsumexp(log_px - kl_coef * kldiv_z0, 0)

        results = Results(
            loss=torch.mean(loss),
            likelihood=torch.mean(log_px).detach(),
            mse=torch.mean(mse).detach(),
            kl_first_p=torch.mean(kldiv_z0).detach(),
            std_first_p=torch.mean(std).detach(),
            recon_std_mean=recon_std_mean,
            recon_std_min=recon_std_min,
            recon_std_max=recon_std_max,
        )

        return results

    def compute_all_losses_pretrain(
            self, batch_dict, n_traj_samples=1, kl_coef=1.0, device="cpu", use_sampling=True
        )->Results:
        tp_to_predict = batch_dict["tp_to_predict"].to(device)
        observed_data = batch_dict["observed_data"].to(device)
        observed_tp = batch_dict["observed_tp"].to(device)
        observed_mask = batch_dict["observed_mask"].to(device)
        data_to_predict = batch_dict["data_to_predict"].to(device)
        mask_predicted_data = batch_dict["mask_predicted_data"].to(device)
        mode = batch_dict.get("mode")

        # Condition on subsampled points
        # Make predictions for all the points
        # pred_y: n_traj_samples, B, T, F
        # recon_std: n_traj_samples, B, T, F
        pred_y, recon_std, info = self.get_reconstruction(
            tp_to_predict,
            observed_data,
            observed_tp,
            mask=observed_mask,
            n_traj_samples=n_traj_samples,
            mode=mode,
            use_sampling=use_sampling,
        )

        D = self.latent_dim
        # KL(q(z|x_ctx) || N(0, I))
        mu, std, z0 = info["first_point"]
        mu, std = mu.squeeze(0), std.squeeze(0)
        std = std.abs()
        assert torch.sum(std < 0) == 0.0

        recon_std = self.prepare_recon_std(recon_std)
        recon_std_mean, recon_std_min, recon_std_max = self.recon_std_stats(
            recon_std,
            mask=mask_predicted_data,
        )
        log_px = self.normalized_reconstruction_log_prob(
            pred_y,
            recon_std,
            data_to_predict,
            mask_predicted_data=mask_predicted_data,
        )


        log_qz = Normal(mu, std).log_prob(z0).sum(-1)  # (B,)
        log_pz = Normal(torch.zeros_like(mu), torch.ones_like(std)).log_prob(z0).sum(-1)
        klz0 = log_pz - log_qz
        # Match the per-observed-value scale used for the reconstruction term.
        obs_count = self.observed_value_count(data_to_predict, mask_predicted_data)
        klz0 = self.normalize_trajectory_term(klz0, obs_count)

        elbo = log_px + kl_coef * klz0
        loss = -elbo.mean()

        mse = self.get_mse(data_to_predict, pred_y, mask=mask_predicted_data)

        results = Results(
            loss=torch.mean(loss),
            likelihood=torch.mean(log_px).detach(),
            mse=torch.mean(mse).detach(),
            kl_first_p=torch.mean(klz0).detach(),
            std_first_p=torch.mean(std).detach(),
            recon_std_mean=recon_std_mean,
            recon_std_min=recon_std_min,
            recon_std_max=recon_std_max,
        )

        return results

    @torch.no_grad()
    def pred_cluster(self, observed_data, observed_tp, observed_mask, n_traj_samples=1, use_sampling=False):
        self.eval()
        # TODO: Issue here should I squeeeze or not

        mu, std = self.encode(
            observed_data, observed_tp, observed_mask
        )

        mu = mu.squeeze(0)
        std = std.squeeze(0)

        if use_sampling:
            eps = torch.randn_like(std)
            z0 = mu + eps * std
        else:
            z0 = mu

        _, qc = self.q_c_given_z(z0)  # (B,K),(B,K)
        labels = qc.argmax(dim=-1)  # (B,)
        return labels

    def encode(self, truth, truth_time_steps, mask=None):
        truth_w_mask = truth
        if mask is not None:
            truth_w_mask = torch.cat((truth, mask), -1)
        first_point_mu, first_point_std = self.encoder_z0(
            truth_w_mask, truth_time_steps, run_backwards=self.run_backwards
        )
        return first_point_mu, torch.clamp(first_point_std, min=0.1)

    @torch.no_grad()
    def init_prior_from_gmm(self, centers: np.ndarray, sigma2: np.ndarray, weights: np.ndarray):
        """
        Initialize model.prior_mu, model.log_prior_var, model.log_pi
        from a diagonal-covariance GMM fitted on encoder means μ(x).

        Args:
            centers:  (K,D) numpy array of GMM component means
            sigma2:   (K,D) numpy array of diagonal GMM component variances
            weights:  (K,) numpy array of GMM component weights
        """
        device = self.prior_mu.device
        # Get latent spaces
        centers_t = torch.from_numpy(centers).to(device).float()
        log_var = torch.log(torch.from_numpy(sigma2).to(device).float())

        # (K,D)
        self.prior_mu.copy_(centers_t)
        # all clusters, all dims
        self.log_prior_var.copy_(log_var)

        # avoid log(0)
        eps = 1e-12
        logits = torch.log(torch.from_numpy(weights) + eps).to(device=device)
        self.log_pi.copy_(logits)


@torch.no_grad()
def sample_latent_space(model: VAEClustering, train_loader, return_pred=False, add_noise=False, device="cpu"):
    all_mu = []
    gt_labels = []
    pred_y_labels = []
    model.eval()
    for batch in train_loader:
        observed_data, observed_tp, observed_mask = (
            batch["observed_data"],
            batch["observed_tp"],
            batch["observed_mask"],
        )
        observed_data, observed_tp, observed_mask = (
            observed_data.to(device),
            observed_tp.to(device),
            observed_mask.to(device),
        )
        mu, std = model.encode(
            observed_data, observed_tp, observed_mask
        )
        if add_noise:
            mu = mu + std * torch.randn_like(mu)
        if return_pred:
            pred_y = model.pred_cluster(observed_data, observed_tp, observed_mask)
            pred_y_labels.append(pred_y.cpu())
        all_mu.append(mu.cpu().squeeze(0))
        labels = batch.get("labels", None)
        if labels is None:
            labels = torch.full(
                (observed_data.shape[0],),
                -1,
                dtype=torch.long,
            )
        gt_labels.append(labels.cpu())
    gt_labels = torch.concat(gt_labels)
    if return_pred:
        pred_y_labels = torch.concat(pred_y_labels)
    Z = torch.cat(all_mu, dim=0)
    return (Z, gt_labels, pred_y_labels) if return_pred else (Z, gt_labels)


@torch.no_grad()
def kmeans(Z, gt_labels, K=4):
    """
    Run KMeans on Z (μ(x)), compare cluster labels to ground truth, and plot:

      - left:  latent space colored by ground-truth labels
      - right: latent space colored by KMeans-predicted labels

    Also prints NMI, ARI, and Hungarian-matched accuracy.

    Args:
        Z         : torch.Tensor or np.ndarray, shape (N,D)
        gt_labels : 1D array-like of ints, shape (N,)
        K         : number of clusters for KMeans
    """
    Z_np = Z.numpy(force=True)

    gt_labels = np.asarray(gt_labels)

    # ---- KMeans ----
    km = KMeans(n_clusters=K, n_init=20)
    pred_labels = km.fit_predict(Z_np)  # (N,)
    centers = km.cluster_centers_  # (K,D)

    return km, pred_labels, centers


@torch.no_grad()
def init_gmm_from_kmeans(model, centers, device="cuda", sigma2=0.05):
    """
    Initialize model.prior_mu, model.log_prior_var, model.log_pi
    from KMeans centers on μ(x).

    Args:
        model:    LatentODE_VaDE instance
        centers:  (K,D) numpy array from KMeans.cluster_centers_
        sigma2:   scalar variance for all dims/clusters
    """
    # Get latent spaces
    model.to(device)
    K = model.k

    # assert centers.shape == (K, D), f"centers shape {centers.shape} != ({K},{D})"

    centers_t = torch.from_numpy(centers).to(device).float()

    model.prior_mu.copy_(centers_t)  # (K,D)
    model.log_prior_var.fill_(math.log(sigma2))  # all clusters, all dims
    model.log_pi.copy_(torch.log(torch.full((K,), 1.0 / K, device=device)))
