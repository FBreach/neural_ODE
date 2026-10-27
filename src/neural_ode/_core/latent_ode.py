###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
# import torch.nn as nn
# import torch.nn.functional as F


# from .vaeclustering import (
#     ADNI_CONTINUOUS_FEATURES,
#     ADNIMixedDecoder,
#     Results,
#     VAEClustering,
#     validate_adni_feature_names,
# )
from .vaeclustering import VAEClustering
from .utils import sample_standard_gaussian

class LatentODE(VAEClustering):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_z0,
        diffeq_solver,
        k=2,
        run_backwards=True,
        emission_hidden_dims=None,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            k,
            run_backwards=run_backwards,
            emission_hidden_dims=emission_hidden_dims,
        )
        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver

    def _last_observed_time(self, truth_time_steps, mask, batch_size):
        if truth_time_steps.ndim == 1:
            return truth_time_steps[-1]

        if truth_time_steps.ndim == 2:
            truth_time_steps = truth_time_steps.unsqueeze(-1)

        if mask is None:
            valid = torch.ones(
                truth_time_steps.shape[:2],
                dtype=torch.bool,
                device=truth_time_steps.device,
            )
        else:
            valid = mask.any(dim=-1)

        counts = valid.sum(dim=1).clamp_min(1)
        last_idx = counts - 1
        batch_idx = torch.arange(batch_size, device=truth_time_steps.device)
        return truth_time_steps[batch_idx, last_idx]

    def _prediction_time_grid(self, time_steps_to_predict, truth_time_steps, mask, batch_size, mode):
        if time_steps_to_predict.ndim == 2:
            time_steps_to_predict = time_steps_to_predict.unsqueeze(-1)

        if mode != "extrap":
            return time_steps_to_predict, False

        anchor_t = self._last_observed_time(truth_time_steps, mask, batch_size)
        if time_steps_to_predict.ndim == 1:
            solver_times = torch.cat((anchor_t.reshape(1), time_steps_to_predict), dim=0)
        else:
            if anchor_t.ndim == 1:
                anchor_t = anchor_t.unsqueeze(-1)
            solver_times = torch.cat((anchor_t.unsqueeze(1), time_steps_to_predict), dim=1)

        return solver_times, True

    def _decode_latent_traj(
        self,
        first_point_enc,
        time_steps_to_predict,
        truth_time_steps,
        mask,
        mode,
    ):
        batch_size = first_point_enc.size(1)
        solver_times, drop_anchor = self._prediction_time_grid(
            time_steps_to_predict,
            truth_time_steps,
            mask,
            batch_size,
            mode,
        )

        assert not torch.isnan(solver_times).any()
        assert not torch.isnan(first_point_enc).any()

        sol_y = self.diffeq_solver(first_point_enc, solver_times)
        if drop_anchor:
            sol_y = sol_y[:, :, 1:, :]
        return sol_y

#     def get_reconstruction_det(
#             self,
#             time_steps_to_predict,
#             truth,
#             truth_time_steps,
#             mask=None,
#             n_traj_samples=1,
#             run_backwards=True,
#             mode=None
#     ):
#         first_point_mu, first_point_std = self.encode(
#             truth, truth_time_steps, mask
#         )
#         means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
#         first_point_enc = means_z0
#
#         # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
#         sol_y = self._decode_latent_traj(
#             first_point_enc,
#             time_steps_to_predict,
#             truth_time_steps,
#             mask,
#             mode,
#         )
#
#         recon_m, recon_std = self.decoder(sol_y)
#
#         all_extra_info = {
#             "first_point": (first_point_mu, first_point_std, first_point_enc),
#             "latent_traj": sol_y.detach(),
#             "latent_traj_grad": sol_y,
#         }
#         return recon_m, recon_std, all_extra_info

    def get_reconstruction(
        self,
        time_steps_to_predict,
        truth,
        truth_time_steps,
        mask=None,
        n_traj_samples=1,
        run_backwards=True,
        mode=None,
        use_sampling=True,
    ):
        first_point_mu, first_point_std = self.encode(
            truth, truth_time_steps, mask
        )

        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        if use_sampling:
            first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
        else:
            first_point_enc = means_z0

        first_point_std = first_point_std.abs()
        assert torch.sum(first_point_std < 0) == 0.0

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self._decode_latent_traj(
            first_point_enc,
            time_steps_to_predict,
            truth_time_steps,
            mask,
            mode,
        )

        recon_m, recon_std = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach(),
            "latent_traj_grad": sol_y,
        }

        return recon_m, recon_std, all_extra_info

    # q(c|z) ∝ p(z|c)p(c)

#     def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples=1):
#         raise NotImplementedError()
#         # input_dim = starting_point.size()[-1]
#         # starting_point = starting_point.view(1,1,input_dim)
#
#         # Sample z0 from prior
#         starting_point_enc = self.z0_prior.sample(
#             [n_traj_samples, 1, self.latent_dim]
#         ).squeeze(-1)
#
#         starting_point_enc_aug = starting_point_enc
#
#         sol_y = self.diffeq_solver.sample_traj_from_prior(
#             starting_point_enc_aug, time_steps_to_predict, n_traj_samples=3
#         )
#
#         return self.decoder(sol_y)


# class LatentODEADNIMixture(LatentODE):
#     """Latent ODE with GMM prior, mixed ADNI likelihoods, and a DX head."""
#
#     adni_mixture_model = True
#     dx_decoder_types = ("categorical", "ordinal_logistic")
#
#     def __init__(
#         self,
#         input_dim,
#         latent_dim,
#         encoder_z0,
#         diffeq_solver,
#         k=2,
#         run_backwards=True,
#         emission_hidden_dims=None,
#         lambda_dx=1.0,
#         dx_decoder_type="categorical",
#         feature_names=None,
#     ):
#         adni_feature_names = validate_adni_feature_names(feature_names)
#         if dx_decoder_type not in self.dx_decoder_types:
#             supported = ", ".join(self.dx_decoder_types)
#             raise ValueError(
#                 f"Unsupported dx_decoder_type '{dx_decoder_type}'. "
#                 f"Use one of: {supported}."
#             )
#         super().__init__(
#             input_dim=input_dim,
#             latent_dim=latent_dim,
#             encoder_z0=encoder_z0,
#             diffeq_solver=diffeq_solver,
#             k=k,
#             run_backwards=run_backwards,
#             emission_hidden_dims=emission_hidden_dims,
#         )
#         self.feature_names = adni_feature_names
#         if input_dim != len(self.feature_names):
#             raise ValueError(
#                 f"ADNI mixed model expects {len(self.feature_names)} features "
#                 f"from feature_names={list(self.feature_names)}, got input_dim={input_dim}."
#             )
#         self.decoder = ADNIMixedDecoder(
#             latent_dim,
#             emission_hidden_dims,
#             feature_names=self.feature_names,
#         )
#         self.dx_decoder_type = dx_decoder_type
#         dx_output_dim = 1 if self.dx_decoder_type == "ordinal_logistic" else 3
#         self.dx_head = nn.Linear(latent_dim, dx_output_dim)
#         if self.dx_decoder_type == "ordinal_logistic":
#             self.dx_threshold_delta = nn.Parameter(torch.tensor(1.0))
#         else:
#             self.register_parameter("dx_threshold_delta", None)
#         self.lambda_dx = float(lambda_dx)
#         self.register_buffer("dx_class_weights", torch.ones(3))
#         self.register_buffer("adni_norm_mean", torch.zeros(len(self.feature_names)))
#         self.register_buffer("adni_norm_std", torch.ones(len(self.feature_names)))
#
#     def set_dx_class_weights(self, weights):
#         weights = torch.as_tensor(
#             weights,
#             dtype=self.dx_class_weights.dtype,
#             device=self.dx_class_weights.device,
#         )
#         if weights.shape != self.dx_class_weights.shape:
#             raise ValueError(
#                 f"DX class weights must have shape {tuple(self.dx_class_weights.shape)}, "
#                 f"got {tuple(weights.shape)}."
#             )
#         self.dx_class_weights.copy_(weights)
#
#     def set_adni_normalization(self, mean, std):
#         mean = torch.as_tensor(
#             mean,
#             dtype=self.adni_norm_mean.dtype,
#             device=self.adni_norm_mean.device,
#         )
#         std = torch.as_tensor(
#             std,
#             dtype=self.adni_norm_std.dtype,
#             device=self.adni_norm_std.device,
#         )
#         if (
#             mean.shape != self.adni_norm_mean.shape
#             or std.shape != self.adni_norm_std.shape
#         ):
#             expected = tuple(self.adni_norm_mean.shape)
#             raise ValueError(
#                 f"ADNI normalization stats must have shape {expected}, "
#                 f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}."
#             )
#         self.adni_norm_mean.copy_(mean)
#         self.adni_norm_std.copy_(std.clamp_min(1e-8))
#
#     def adni_prediction_to_raw(self, pred_y):
#         pred_raw = pred_y.clone()
#         for feature_idx, feature_name in enumerate(self.feature_names):
#             if feature_name in ADNI_CONTINUOUS_FEATURES:
#                 pred_raw[..., feature_idx] = (
#                     pred_raw[..., feature_idx] * self.adni_norm_std[feature_idx]
#                     + self.adni_norm_mean[feature_idx]
#                 )
#         return pred_raw
#
#     def _reconstruct_from_latent(
#         self,
#         time_steps_to_predict,
#         truth,
#         truth_time_steps,
#         mask,
#         n_traj_samples,
#         mode,
#         use_sampling,
#     ):
#         first_point_mu, first_point_std = self.encode(
#             truth,
#             truth_time_steps,
#             mask,
#         )
#         means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
#         sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
#         if use_sampling:
#             first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)
#         else:
#             first_point_enc = means_z0
#
#         first_point_std = first_point_std.abs()
#         sol_y = self._decode_latent_traj(
#             first_point_enc,
#             time_steps_to_predict,
#             truth_time_steps,
#             mask,
#             mode,
#         )
#         decoder_out = self.decoder(sol_y)
#         all_extra_info = {
#             "first_point": (first_point_mu, first_point_std, first_point_enc),
#             "latent_traj": sol_y.detach(),
#             "latent_traj_grad": sol_y,
#             "decoder_out": decoder_out,
#         }
#         return decoder_out["mean"], decoder_out["std"], all_extra_info
#
#     def get_reconstruction_det(
#         self,
#         time_steps_to_predict,
#         truth,
#         truth_time_steps,
#         mask=None,
#         n_traj_samples=1,
#         run_backwards=True,
#         mode=None,
#     ):
#         return self._reconstruct_from_latent(
#             time_steps_to_predict,
#             truth,
#             truth_time_steps,
#             mask,
#             n_traj_samples,
#             mode,
#             use_sampling=False,
#         )
#
#     def get_reconstruction(
#         self,
#         time_steps_to_predict,
#         truth,
#         truth_time_steps,
#         mask=None,
#         n_traj_samples=1,
#         run_backwards=True,
#         mode=None,
#         use_sampling=True,
#     ):
#         return self._reconstruct_from_latent(
#             time_steps_to_predict,
#             truth,
#             truth_time_steps,
#             mask,
#             n_traj_samples,
#             mode,
#             use_sampling=use_sampling,
#         )
#
#     def mixed_reconstruction_log_prob(
#         self,
#         decoder_out,
#         normalized_targets,
#         raw_targets,
#         mask,
#     ):
#         feature_log_probs = self.decoder.feature_log_probs(
#             decoder_out,
#             normalized_targets,
#             raw_targets,
#         )
#         total = None
#         metrics = {}
#         for feature_idx, feature_name in enumerate(self.feature_names):
#             feature_key = feature_name.lower()
#             feature_mask = mask[..., feature_idx].unsqueeze(0)
#             contrib = feature_log_probs[feature_name] * feature_mask
#             total = contrib.sum(-1) if total is None else total + contrib.sum(-1)
#             count = feature_mask.sum().clamp_min(1.0)
#             metrics[f"{feature_key}_nll"] = (-contrib.sum() / count).detach()
#
#         obs_count = self.observed_value_count(normalized_targets, mask)
#         log_px = self.normalize_trajectory_term(total, obs_count)
#         return log_px, metrics
#
#     def forecast_error_metrics(self, pred_y, targets, mask):
#         pred_mean = pred_y.mean(dim=0)
#         metrics = {}
#         total_sq = targets.new_tensor(0.0)
#         total_count = targets.new_tensor(0.0)
#         for feature_idx, feature_name in enumerate(self.feature_names):
#             feature_key = feature_name.lower()
#             feature_mask = mask[..., feature_idx]
#             err = (pred_mean[..., feature_idx] - targets[..., feature_idx]) * feature_mask
#             raw_count = feature_mask.sum()
#             count = raw_count.clamp_min(1.0)
#             sq_sum = err.pow(2).sum()
#             metrics[f"{feature_key}_rmse"] = torch.sqrt(sq_sum / count).detach()
#             metrics[f"{feature_key}_rmse__sq_sum"] = sq_sum.detach()
#             metrics[f"{feature_key}_rmse__count"] = raw_count.detach()
#             total_sq = total_sq + sq_sum
#             total_count = total_count + raw_count
#         metrics["forecast_rmse"] = torch.sqrt(
#             total_sq / total_count.clamp_min(1.0)
#         ).detach()
#         metrics["forecast_rmse__sq_sum"] = total_sq.detach()
#         metrics["forecast_rmse__count"] = total_count.detach()
#         return metrics
#
#     def dx_class_log_probs(self, logits):
#         if self.dx_decoder_type == "categorical":
#             return F.log_softmax(logits, dim=-1)
#
#         if logits.size(-1) != 1:
#             raise ValueError(
#                 "Ordinal-logistic DX decoder expects one scalar score per "
#                 f"timepoint, got last dimension {logits.size(-1)}."
#             )
#         eta = logits.squeeze(-1)
#         tau0 = eta.new_tensor(0.0)
#         tau1 = tau0 + F.softplus(self.dx_threshold_delta) + 1e-4
#         cdf0 = torch.sigmoid(tau0 - eta)
#         cdf1 = torch.sigmoid(tau1 - eta)
#         probs = torch.stack(
#             (
#                 cdf0,
#                 cdf1 - cdf0,
#                 1.0 - cdf1,
#             ),
#             dim=-1,
#         )
#         return probs.clamp_min(1e-8).log()
#
#     def dx_class_probs(self, logits):
#         return self.dx_class_log_probs(logits).exp()
#
#     @staticmethod
#     def _zero_dx_scores(zero):
#         metrics = {
#             "dx_acc": zero,
#             "dx_macro_f1": zero,
#             "dx_balanced_acc": zero,
#         }
#         for true_cls in range(3):
#             for pred_cls in range(3):
#                 metrics[f"dx__cm_{true_cls}_{pred_cls}"] = zero
#         return metrics
#
#     def _dx_scores(self, logits, labels, mask):
#         if mask.sum() <= 0:
#             return self._zero_dx_scores(logits.new_tensor(0.0))
#         logits_mean = self.dx_class_probs(logits).mean(dim=0)
#         valid = mask.bool()
#         pred = logits_mean.argmax(dim=-1)[valid]
#         truth = labels[valid].long()
#         confusion = logits.new_zeros((3, 3))
#         for true_cls in range(3):
#             for pred_cls in range(3):
#                 confusion[true_cls, pred_cls] = (
#                     (truth == true_cls) & (pred == pred_cls)
#                 ).sum()
#         total = confusion.sum().clamp_min(1.0)
#         acc = confusion.diag().sum() / total
#         f1_values = []
#         recall_values = []
#         for class_idx in range(3):
#             tp = confusion[class_idx, class_idx]
#             fp = confusion[:, class_idx].sum() - tp
#             fn = confusion[class_idx].sum() - tp
#             support = confusion[class_idx].sum()
#             f1_denom = 2 * tp + fp + fn
#             f1_values.append(torch.where(f1_denom > 0, 2 * tp / f1_denom, tp))
#             recall_values.append(torch.where(support > 0, tp / support, support))
#         metrics = {
#             "dx_acc": acc.detach(),
#             "dx_macro_f1": torch.stack(f1_values).mean().detach(),
#             "dx_balanced_acc": torch.stack(recall_values).mean().detach(),
#         }
#         for true_cls in range(3):
#             for pred_cls in range(3):
#                 metrics[f"dx__cm_{true_cls}_{pred_cls}"] = (
#                     confusion[true_cls, pred_cls].detach()
#                 )
#         return metrics
#
#     def dx_logits_at_times(self, info, dx_times, observed_tp, observed_mask, mode):
#         first_point_enc = info["first_point"][2]
#         dx_latent = self._decode_latent_traj(
#             first_point_enc,
#             dx_times,
#             observed_tp,
#             observed_mask,
#             mode,
#         )
#         return self.dx_head(dx_latent)
#
#     def dx_loss_and_metrics(self, info, batch_dict, observed_tp, observed_mask, mode, device):
#         if "dx_times" not in batch_dict or "dx_mask" not in batch_dict:
#             zero = observed_tp.new_tensor(0.0)
#             metrics = {"dx_ce": zero.detach()}
#             metrics.update(self._zero_dx_scores(zero.detach()))
#             return zero, metrics
#
#         dx_times = batch_dict["dx_times"].to(device)
#         dx_labels = batch_dict["dx_labels"].to(device)
#         dx_mask = batch_dict["dx_mask"].to(device)
#         if dx_times.size(1) == 0 or dx_mask.sum() <= 0:
#             zero = observed_tp.new_tensor(0.0)
#             metrics = {"dx_ce": zero.detach()}
#             metrics.update(self._zero_dx_scores(zero.detach()))
#             return zero, metrics
#
#         logits = self.dx_logits_at_times(
#             info,
#             dx_times,
#             observed_tp,
#             observed_mask,
#             mode,
#         )
#         valid = dx_mask.bool().unsqueeze(0).expand(logits.shape[:-1])
#         labels = dx_labels.unsqueeze(0).expand(logits.shape[:-1])
#         if self.dx_decoder_type == "categorical":
#             dx_loss = F.cross_entropy(
#                 logits[valid],
#                 labels[valid].long(),
#                 weight=self.dx_class_weights,
#                 reduction="mean",
#             )
#         else:
#             log_probs = self.dx_class_log_probs(logits)
#             flat_labels = labels[valid].long()
#             flat_log_probs = log_probs[valid]
#             nll = -flat_log_probs.gather(
#                 dim=-1,
#                 index=flat_labels.unsqueeze(-1),
#             ).squeeze(-1)
#             weights = self.dx_class_weights[flat_labels]
#             dx_loss = (nll * weights).sum() / weights.sum().clamp_min(1e-8)
#         metrics = {"dx_ce": dx_loss.detach()}
#         metrics.update(self._dx_scores(logits.detach(), dx_labels, dx_mask))
#         return dx_loss, metrics
#
#     def _compute_adni_losses(
#         self,
#         batch_dict,
#         n_traj_samples=1,
#         kl_coef=1.0,
#         recon_nll_weight=1.0,
#         lambda_dx=None,
#         device="cpu",
#         use_sampling=True,
#         pretrain=False,
#     ):
#         recon_nll_weight = float(recon_nll_weight)
#         if recon_nll_weight < 0:
#             raise ValueError(
#                 f"recon_nll_weight must be non-negative, got {recon_nll_weight}."
#             )
#         phase_lambda_dx = self.lambda_dx if lambda_dx is None else float(lambda_dx)
#         if phase_lambda_dx < 0:
#             raise ValueError(f"lambda_dx must be non-negative, got {phase_lambda_dx}.")
#         tp_to_predict = batch_dict["tp_to_predict"].to(device)
#         observed_data = batch_dict["observed_data"].to(device)
#         observed_tp = batch_dict["observed_tp"].to(device)
#         observed_mask = batch_dict["observed_mask"].to(device)
#         normalized_data_to_predict = batch_dict["data_to_predict"].to(device)
#         raw_data_to_predict = batch_dict["raw_data_to_predict"].to(device)
#         mask_predicted_data = batch_dict["mask_predicted_data"].to(device)
#         mode = batch_dict.get("mode")
#
#         pred_y, recon_std, info = self.get_reconstruction(
#             tp_to_predict,
#             observed_data,
#             observed_tp,
#             mask=observed_mask,
#             n_traj_samples=n_traj_samples,
#             mode=mode,
#             use_sampling=use_sampling,
#         )
#
#         mu, std, z0 = info["first_point"]
#         mu, std = mu.squeeze(0), std.squeeze(0).abs()
#         obs_count = self.observed_value_count(
#             normalized_data_to_predict,
#             mask_predicted_data,
#         )
#         if pretrain:
#             kl_z0 = self.standard_normal_kl(mu, std, obs_count=obs_count)
#             kl_metric_name = "standard_kl"
#         else:
#             kl_z0 = self.mixture_prior_kl(mu, std, z0, obs_count=obs_count)
#             kl_metric_name = "mixture_kl"
#
#         log_px, feature_nll_metrics = self.mixed_reconstruction_log_prob(
#             info["decoder_out"],
#             normalized_data_to_predict,
#             raw_data_to_predict,
#             mask_predicted_data,
#         )
#         mixed_recon_nll = -torch.mean(log_px)
#         weighted_log_px = recon_nll_weight * log_px
#         nelbo = -torch.logsumexp(weighted_log_px - kl_coef * kl_z0, dim=0).mean()
#
#         dx_ce = normalized_data_to_predict.new_tensor(0.0)
#         dx_metrics = {
#             "dx_ce": dx_ce.detach(),
#             "dx_acc": dx_ce.detach(),
#             "dx_macro_f1": dx_ce.detach(),
#             "dx_balanced_acc": dx_ce.detach(),
#         }
#         dx_metrics.update(self._zero_dx_scores(dx_ce.detach()))
#         dx_ce, dx_metrics = self.dx_loss_and_metrics(
#             info,
#             batch_dict,
#             observed_tp,
#             observed_mask,
#             mode,
#             device,
#         )
#         total_loss = nelbo + phase_lambda_dx * dx_ce
#
#         recon_std_mean, recon_std_min, recon_std_max = self.recon_std_stats(
#             recon_std,
#             mask=mask_predicted_data,
#         )
#         pred_y_raw = self.adni_prediction_to_raw(pred_y)
#         metrics = {
#             "total_loss": total_loss.detach(),
#             "nelbo_mixed": nelbo.detach(),
#             "mixed_recon_nll": mixed_recon_nll.detach(),
#             kl_metric_name: torch.mean(kl_z0).detach(),
#             "dx_ce": dx_metrics["dx_ce"],
#             "dx_acc": dx_metrics["dx_acc"],
#             "dx_macro_f1": dx_metrics["dx_macro_f1"],
#             "dx_balanced_acc": dx_metrics["dx_balanced_acc"],
#             "lambda_dx": normalized_data_to_predict.new_tensor(phase_lambda_dx),
#             "recon_nll_weight": normalized_data_to_predict.new_tensor(
#                 recon_nll_weight
#             ),
#         }
#         for true_cls in range(3):
#             for pred_cls in range(3):
#                 key = f"dx__cm_{true_cls}_{pred_cls}"
#                 metrics[key] = dx_metrics[key]
#         metrics.update(feature_nll_metrics)
#         metrics.update(
#             self.forecast_error_metrics(
#                 pred_y_raw,
#                 raw_data_to_predict,
#                 mask_predicted_data,
#             )
#         )
#
#         return Results(
#             loss=total_loss,
#             likelihood=torch.mean(log_px).detach(),
#             kl_first_p=torch.mean(kl_z0).detach(),
#             std_first_p=torch.mean(std).detach(),
#             recon_std_mean=recon_std_mean,
#             recon_std_min=recon_std_min,
#             recon_std_max=recon_std_max,
#             metrics=metrics,
#         )
#
#     def compute_all_losses(
#         self,
#         batch_dict,
#         n_traj_samples=1,
#         kl_coef=1.0,
#         recon_nll_weight=1.0,
#         lambda_dx=None,
#         device="cpu",
#         use_sampling=True,
#     ):
#         return self._compute_adni_losses(
#             batch_dict,
#             n_traj_samples=n_traj_samples,
#             kl_coef=kl_coef,
#             recon_nll_weight=recon_nll_weight,
#             lambda_dx=lambda_dx,
#             device=device,
#             use_sampling=use_sampling,
#             pretrain=False,
#         )
#
#     def compute_all_losses_pretrain(
#         self,
#         batch_dict,
#         n_traj_samples=1,
#         kl_coef=1.0,
#         recon_nll_weight=1.0,
#         lambda_dx=None,
#         device="cpu",
#         use_sampling=True,
#     ):
#         return self._compute_adni_losses(
#             batch_dict,
#             n_traj_samples=n_traj_samples,
#             kl_coef=kl_coef,
#             recon_nll_weight=recon_nll_weight,
#             lambda_dx=lambda_dx,
#             device=device,
#             use_sampling=use_sampling,
#             pretrain=True,
#         )


# class LatentODEPostHocGMM(LatentODE):
#     """Latent ODE trained as a standard VAE, then clustered with a fitted GMM."""
#
#     standard_vae_prior = True
#     posthoc_gmm_clustering = True
#
#     def compute_all_losses(
#         self,
#         batch_dict,
#         n_traj_samples=1,
#         kl_coef=1.0,
#         device="cpu",
#         use_sampling=True,
#     ):
#         return self.compute_all_losses_standard_vae(
#             batch_dict,
#             n_traj_samples=n_traj_samples,
#             kl_coef=kl_coef,
#             device=device,
#             use_sampling=use_sampling,
#         )
#
#     def compute_all_losses_pretrain(
#         self,
#         batch_dict,
#         n_traj_samples=1,
#         kl_coef=1.0,
#         device="cpu",
#         use_sampling=True,
#     ):
#         return self.compute_all_losses_standard_vae(
#             batch_dict,
#             n_traj_samples=n_traj_samples,
#             kl_coef=kl_coef,
#             device=device,
#             use_sampling=use_sampling,
#         )
