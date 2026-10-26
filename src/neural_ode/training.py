"""Config-light training, GMM initialization, and prediction APIs."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .data import DataLoaders, TrajectoryData, prep_batches


@dataclass
class TrainingResult:
    model: torch.nn.Module
    history: list[dict]
    checkpoint_path: Path | None = None
    phase: str | None = None


@dataclass
class GMMResult:
    model: torch.nn.Module
    estimator: GaussianMixture


@dataclass
class PredictionTrajectory:
    id: object
    times: torch.Tensor
    truth: torch.Tensor
    prediction: torch.Tensor
    mask: torch.Tensor
    cluster: int
    label: int | None
    observed_times: torch.Tensor | None = None
    observed_values: torch.Tensor | None = None
    observed_mask: torch.Tensor | None = None
    cutoff_time: float | None = None


@dataclass
class PredictionResult:
    trajectories: list[PredictionTrajectory]
    feature_names: list[str] | None = None
    cutoff_time: float | None = None
    phase: str | None = None


def _get_loaders(loaders):
    if isinstance(loaders, DataLoaders):
        return loaders.train, loaders.val
    if isinstance(loaders, dict) and "train" in loaders:
        return loaders["train"], loaders.get("val")
    raise TypeError("loaders must be DataLoaders or a mapping with 'train' and optional 'val'")


def _loss(model, batch, phase, device, kl_coef):
    fn = model.compute_all_losses_pretrain if phase == "pretrain" else model.compute_all_losses
    return fn(batch_dict=batch, n_traj_samples=1, kl_coef=kl_coef, device=device, use_sampling=True)


def _mean_terms(results):
    return {
        "loss": float(results.loss.detach().cpu()),
        "reconstruction_loss": float(-results.likelihood.detach().cpu()),
        "kl_loss": float(results.kl_first_p.detach().cpu()),
    }


def _run_phase(model, loaders, *, phase, epochs, learning_rate, weight_decay, device,
               kl_coef, kl_warmup_epochs=0, checkpoint_dir=None):
    train_loader, val_loader = _get_loaders(loaders)
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("epochs and learning_rate must be positive; weight_decay must be nonnegative")
    model.to(device)
    if isinstance(loaders, DataLoaders):
        model._neural_ode_normalization = (loaders.mean.detach().cpu(), loaders.std.detach().cpu())
    for parameter in (model.prior_mu, model.log_prior_var, model.log_pi):
        parameter.requires_grad_(phase == "finetune")
    optimizer = torch.optim.Adamax((p for p in model.parameters() if p.requires_grad),
                                   lr=learning_rate, weight_decay=weight_decay)
    history = []
    best = float("inf")
    best_state = None
    checkpoint = None
    for epoch in range(epochs):
        model.train()
        train_terms = []
        current_kl = kl_coef if phase == "pretrain" or not kl_warmup_epochs else kl_coef * min(1.0, (epoch + 1) / kl_warmup_epochs)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            result = _loss(model, batch, phase, device, current_kl)
            result.loss.backward()
            optimizer.step()
            train_terms.append(_mean_terms(result))
        record = {"epoch": epoch + 1, "phase": phase, "kl_coef": current_kl}
        for key in ("loss", "reconstruction_loss", "kl_loss"):
            record[f"train_{key}"] = float(np.mean([item[key] for item in train_terms]))
        if val_loader is not None:
            model.eval()
            val_terms = []
            with torch.no_grad():
                for batch in val_loader:
                    val_terms.append(_mean_terms(_loss(model, batch, phase, device, current_kl)))
            for key in ("loss", "reconstruction_loss", "kl_loss"):
                record[f"val_{key}"] = float(np.mean([item[key] for item in val_terms]))
        history.append(record)
        log_parts = [
            f"{phase} epoch {epoch + 1}/{epochs}",
            f"train_loss={record['train_loss']:.6g}",
        ]
        if "val_loss" in record:
            log_parts.append(f"val_loss={record['val_loss']:.6g}")
        print(" | ".join(log_parts), flush=True)
        score = record.get("val_loss", record["train_loss"])
        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_dir is not None:
                checkpoint = Path(checkpoint_dir) / f"best_{phase}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                            "metrics": record,
                            "normalization": getattr(model, "_neural_ode_normalization", None)}, checkpoint)
    if best_state is not None:
        model.load_state_dict(best_state)
    model._neural_ode_phase = phase
    return TrainingResult(model, history, checkpoint, phase)


def train_pretrain(loaders, *, model_config=None, epochs=100, learning_rate=1e-3,
                   weight_decay=0.0, device="cpu", seed=0, checkpoint_dir=None):
    """Train model reconstruction objective; create model from config if supplied."""
    torch.manual_seed(seed)
    train_loader, _ = _get_loaders(loaders)
    model = model_config if isinstance(model_config, torch.nn.Module) else None
    if model is None:
        if model_config is None:
            raise ValueError("Pass a model instance as model_config or use create_model before training")
        from .model import create_model
        model = create_model(train_loader.dataset[0]["observed_data"].shape[-1], model_config)
    return _run_phase(model, loaders, phase="pretrain", epochs=epochs, learning_rate=learning_rate,
                      weight_decay=weight_decay, device=device, kl_coef=0.0,
                      checkpoint_dir=checkpoint_dir)


def train_fit_gmm(model, loaders, *, n_components=None, seed=0, reg_covar=1e-6):
    """Fit a diagonal Gaussian mixture to training posterior means."""
    train_loader, _ = _get_loaders(loaders)
    device = next(model.parameters()).device
    means = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            mu, _ = model.encode(batch["observed_data"].to(device), batch["observed_tp"].to(device),
                                 batch["observed_mask"].to(device))
            means.append(mu.squeeze(0).cpu().numpy())
    latent = np.concatenate(means)
    components = int(n_components or model.k.item())
    if components > len(latent):
        raise ValueError(f"n_components={components} exceeds {len(latent)} training trajectories")
    estimator = GaussianMixture(n_components=components, covariance_type="diag", n_init=10,
                                random_state=seed, reg_covar=reg_covar).fit(latent)
    model.init_prior_from_gmm(estimator.means_, estimator.covariances_, estimator.weights_)
    model._neural_ode_phase = "clustered"
    return GMMResult(model, estimator)


def train_finetune(model, loaders, *, epochs=100, learning_rate=1e-3, weight_decay=0.0,
                   kl_coef=1.0, kl_warmup_epochs=0, device="cpu", seed=0,
                   checkpoint_dir=None):
    """Fine-tune mixture-prior VAE objective after GMM initialization."""
    torch.manual_seed(seed)
    return _run_phase(model, loaders, phase="finetune", epochs=epochs,
                      learning_rate=learning_rate, weight_decay=weight_decay,
                      device=device, kl_coef=kl_coef, kl_warmup_epochs=kl_warmup_epochs,
                      checkpoint_dir=checkpoint_dir)


def predict(model, data_or_loader, *, device="cpu"):
    """Forecast each trajectory and return values with IDs and optional labels."""
    feature_names = None
    cutoff_time = None
    if isinstance(data_or_loader, TrajectoryData):
        feature_names = data_or_loader.feature_names
        if data_or_loader.cutoff_mode == "time":
            cutoff_time = data_or_loader.cutoff
        mean, std = getattr(model, "_neural_ode_normalization", (torch.zeros(len(feature_names)), torch.ones(len(feature_names))))
        transformed = TrajectoryData(
            [type(t)(t.id, t.times, torch.where(t.mask, (t.values - mean) / std, 0.0),
                        t.mask, t.label, t.cutoff_index)
             for t in data_or_loader.trajectories],
            data_or_loader.feature_names, data_or_loader.label_values,
            data_or_loader.cutoff, data_or_loader.cutoff_mode)
        data_or_loader = prep_batches(transformed, val_split=0, shuffle=False).train
    elif isinstance(data_or_loader, DataLoaders):
        if data_or_loader.cutoff_mode == "time":
            cutoff_time = data_or_loader.cutoff
        feature_names = data_or_loader.feature_names
        data_or_loader = data_or_loader.test or data_or_loader.val or data_or_loader.train
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for batch in data_or_loader:
            observed = batch["observed_data"].to(device)
            observed_t = batch["observed_tp"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            target_t = batch["tp_to_predict"].to(device)
            pred, _, _ = model.get_reconstruction(target_t, observed, observed_t,
                                                  observed_mask, mode=batch.get("mode"),
                                                  use_sampling=False)
            pred = pred.squeeze(0).cpu()
            clusters = model.pred_cluster(observed, observed_t, observed_mask).cpu()
            for i, trajectory_id in enumerate(batch["ids"]):
                length = int(batch["target_lengths"][i])
                observed_length = int(batch["observed_lengths"][i])
                valid = torch.arange(batch["mask_predicted_data"].shape[1]) < length
                label = int(batch["labels"][i])
                observed_times = batch["observed_tp"][i, :observed_length].squeeze(-1)
                observed_values = batch["observed_data"][i, :observed_length]
                observed_mask = batch["observed_mask"][i, :observed_length].bool()
                predictions.append(PredictionTrajectory(
                    trajectory_id, batch["tp_to_predict"][i][valid].squeeze(-1),
                    batch["data_to_predict"][i][valid], pred[i][valid],
                    batch["mask_predicted_data"][i][valid], int(clusters[i]),
                    None if label < 0 else label,
                    observed_times=observed_times,
                    observed_values=observed_values,
                    observed_mask=observed_mask,
                    cutoff_time=(float(observed_times[-1])
                                 if batch.get("mode") == "extrap" and observed_length else None)))
    if cutoff_time is None:
        observed_cutoffs = [trajectory.cutoff_time for trajectory in predictions
                            if trajectory.cutoff_time is not None]
        if observed_cutoffs:
            cutoff_time = float(np.median(observed_cutoffs))
    return PredictionResult(
        predictions, feature_names, cutoff_time,
        phase=getattr(model, "_neural_ode_phase", None),
    )


def _hungarian_accuracy(y_true, y_pred):
    """Compute clustering accuracy after optimally matching cluster IDs."""
    true_values, true_indices = np.unique(y_true, return_inverse=True)
    pred_values, pred_indices = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((len(pred_values), len(true_values)), dtype=np.int64)
    np.add.at(contingency, (pred_indices, true_indices), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[rows, columns].sum() / len(y_true))


def _batch_values_in_model_scale(values, mask, source_stats, model_stats):
    """Convert prepared loader values back to the model's training scale."""
    if source_stats is None or model_stats is None:
        return values
    source_mean, source_std = (
        torch.as_tensor(stat, dtype=values.dtype, device=values.device).reshape(-1)
        for stat in source_stats
    )
    model_mean, model_std = (
        torch.as_tensor(stat, dtype=values.dtype, device=values.device).reshape(-1)
        for stat in model_stats
    )
    feature_count = values.shape[-1]
    if any(stat.numel() != feature_count for stat in
           (source_mean, source_std, model_mean, model_std)):
        raise ValueError("Loader and model normalization must match the feature count")
    converted = (values * source_std + source_mean - model_mean) / model_std
    return torch.where(mask.bool(), converted, torch.zeros_like(converted))


def evaluate_performance(model, test_loader, *, device="cpu"):
    """Evaluate masked forecast error and available clustering metrics.

    Returns a fixed-key dictionary containing forecast ``mse``; supervised
    clustering metrics ``nmi``, ``ari``, and Hungarian-matched ``acc``; and
    label-free latent cluster scores ``silhouette``, ``davies_bouldin``, and
    ``calinski_harabasz``. Clustering metrics are ``None`` until GMM fitting
    has completed, and any inapplicable score is also ``None``. The loader is
    consumed directly; it need not be wrapped in a ``DataLoaders`` object.
    """
    metric_names = (
        "mse", "nmi", "ari", "acc", "silhouette", "davies_bouldin",
        "calinski_harabasz",
    )
    metrics = dict.fromkeys(metric_names)
    phase = getattr(model, "_neural_ode_phase", None)
    clustering_ready = phase in {"clustered", "finetune"}
    squared_error_sum = 0.0
    valid_target_count = 0
    latent_batches = []
    cluster_batches = []
    label_batches = []
    loader_normalization = getattr(test_loader, "_neural_ode_normalization", None)
    model_normalization = getattr(model, "_neural_ode_normalization", None)
    model.to(device).eval()

    with torch.no_grad():
        for batch in test_loader:
            observed_mask = batch["observed_mask"].to(device)
            observed = _batch_values_in_model_scale(
                batch["observed_data"].to(device), observed_mask,
                loader_normalization, model_normalization,
            )
            observed_times = batch["observed_tp"].to(device)
            target_times = batch["tp_to_predict"].to(device)
            target_mask = batch["mask_predicted_data"].to(device).bool()
            target = _batch_values_in_model_scale(
                batch["data_to_predict"].to(device), target_mask,
                loader_normalization, model_normalization,
            )
            prediction, _, _ = model.get_reconstruction(
                target_times, observed, observed_times, observed_mask,
                mode=batch.get("mode"), use_sampling=False,
            )
            if prediction.ndim == 4 and prediction.shape[0] == 1:
                prediction = prediction.squeeze(0)
            valid = target_mask & torch.isfinite(target) & torch.isfinite(prediction)
            squared_error = torch.where(valid, (prediction - target).square(), 0.0)
            squared_error_sum += float(squared_error.sum().cpu())
            valid_target_count += int(valid.sum().cpu())

            if clustering_ready:
                latent_mean, _ = model.encode(observed, observed_times, observed_mask)
                if latent_mean.ndim == 3 and latent_mean.shape[0] == 1:
                    latent_mean = latent_mean.squeeze(0)
                clusters = model.pred_cluster(observed, observed_times, observed_mask)
                latent_batches.append(latent_mean.detach().cpu().numpy())
                cluster_batches.append(torch.as_tensor(clusters).detach().cpu().numpy().reshape(-1))
                labels = batch.get("labels")
                if labels is None:
                    labels = torch.full((observed.shape[0],), -1, dtype=torch.long)
                label_batches.append(torch.as_tensor(labels).cpu().numpy().reshape(-1))

    if valid_target_count:
        metrics["mse"] = squared_error_sum / valid_target_count

    if clustering_ready and cluster_batches:
        latent = np.concatenate(latent_batches, axis=0)
        clusters = np.concatenate(cluster_batches)
        labels = np.concatenate(label_batches)
        labeled = labels >= 0
        if labeled.any():
            true_labels = labels[labeled]
            predicted_clusters = clusters[labeled]
            metrics["nmi"] = float(normalized_mutual_info_score(true_labels, predicted_clusters))
            metrics["ari"] = float(adjusted_rand_score(true_labels, predicted_clusters))
            metrics["acc"] = _hungarian_accuracy(true_labels, predicted_clusters)

        n_groups = np.unique(clusters).size
        if 1 < n_groups < len(clusters) and len(clusters) > 2 and np.isfinite(latent).all():
            for key, scorer in (
                ("silhouette", silhouette_score),
                ("davies_bouldin", davies_bouldin_score),
                ("calinski_harabasz", calinski_harabasz_score),
            ):
                try:
                    metrics[key] = float(scorer(latent, clusters))
                except ValueError:
                    metrics[key] = None

    return metrics
