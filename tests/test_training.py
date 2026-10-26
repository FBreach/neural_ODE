from types import SimpleNamespace

import pytest
import torch

from neural_ode import evaluate_performance, predict, train_finetune, train_fit_gmm, train_pretrain
from neural_ode.training import _run_phase


class _EvaluationModel(torch.nn.Module):
    def __init__(self, *, phase="pretrain", single_cluster=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self._neural_ode_phase = phase
        self.single_cluster = single_cluster

    def get_reconstruction(self, target_times, observed, observed_times, observed_mask,
                           mode=None, use_sampling=False):
        batch, steps = target_times.shape[:2]
        return torch.zeros(1, batch, steps, 1, device=target_times.device), None, None

    def encode(self, observed, observed_times, observed_mask):
        return observed[:, 0, :1].unsqueeze(0), torch.ones(1, observed.shape[0], 1)

    def pred_cluster(self, observed, observed_times, observed_mask):
        if self.single_cluster:
            return torch.zeros(observed.shape[0], dtype=torch.long, device=observed.device)
        return (observed[:, 0, 0] > 0).long()


def _evaluation_batch(values, labels):
    values = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    batch_size = len(values)
    return {
        "observed_data": values[:, None, :],
        "observed_tp": torch.zeros(batch_size, 1, 1),
        "observed_mask": torch.ones(batch_size, 1, 1),
        "tp_to_predict": torch.zeros(batch_size, 2, 1),
        "data_to_predict": torch.tensor(
            [[[1.0], [3.0]]] * batch_size, dtype=torch.float32
        ),
        "mask_predicted_data": torch.tensor(
            [[[1.0], [1.0]]] * batch_size, dtype=torch.float32
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
        "mode": "interp",
    }


class _TrainingModel(_EvaluationModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("k", torch.tensor(2))
        self.prior_mu = torch.nn.Parameter(torch.ones(2, 1))
        self.log_prior_var = torch.nn.Parameter(torch.ones(2, 1))
        self.log_pi = torch.nn.Parameter(torch.ones(2))

    def compute_all_losses_pretrain(self, batch_dict, n_traj_samples, kl_coef, device,
                                    use_sampling):
        reconstruction = (self.anchor - batch_dict["target"]).square()
        kl = self.prior_mu.square().mean() + self.log_prior_var.square().mean() + self.log_pi.square().mean()
        return SimpleNamespace(loss=reconstruction + kl_coef * kl,
                               likelihood=-reconstruction, kl_first_p=kl)

    def compute_all_losses(self, **kwargs):
        return self.compute_all_losses_pretrain(**kwargs)

    def init_prior_from_gmm(self, centers, sigma2, weights):
        with torch.no_grad():
            self.prior_mu.copy_(torch.as_tensor(centers))
            self.log_prior_var.copy_(torch.as_tensor(sigma2).log())
            self.log_pi.copy_(torch.as_tensor(weights).log())


def test_train_pretrain_updates_model_and_freezes_prior():
    model = _TrainingModel()
    prior = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
             if name != "anchor"}

    result = train_pretrain({"train": [{"target": 1.0}]}, model_config=model, epochs=1)

    assert result.model is model
    assert model.anchor.item() > 0
    assert result.phase == model._neural_ode_phase == "pretrain"
    assert result.checkpoint_path is None
    assert result.history[0]["kl_coef"] == 0.0
    assert "val_loss" not in result.history[0]
    for name, original in prior.items():
        parameter = dict(model.named_parameters())[name]
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter, original)


def test_train_finetune_updates_prior_and_warms_up_kl():
    model = _TrainingModel()
    for parameter in (model.prior_mu, model.log_prior_var, model.log_pi):
        parameter.requires_grad_(False)

    result = train_finetune(model, {"train": [{"target": 1.0}]}, epochs=3,
                            kl_coef=0.8, kl_warmup_epochs=2)

    assert result.model is model
    assert result.phase == model._neural_ode_phase == "finetune"
    assert [record["kl_coef"] for record in result.history] == pytest.approx([0.4, 0.8, 0.8])
    for parameter in (model.prior_mu, model.log_prior_var, model.log_pi):
        assert parameter.requires_grad
        assert torch.all(parameter < 1)


def test_run_phase_restores_best_validation_checkpoint(tmp_path):
    model = _TrainingModel()
    loaders = {"train": [{"target": 1.0}], "val": [{"target": 0.0}]}

    result = _run_phase(model, loaders, phase="pretrain", epochs=2, learning_rate=0.1,
                        weight_decay=0.0, device="cpu", kl_coef=0.0,
                        checkpoint_dir=tmp_path / "checkpoints")

    assert [record["epoch"] for record in result.history] == [1, 2]
    assert result.history[0]["val_loss"] < result.history[1]["val_loss"]
    assert result.history[1]["train_loss"] < result.history[0]["train_loss"]
    assert result.checkpoint_path == tmp_path / "checkpoints" / "best_pretrain.pt"
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["metrics"] == result.history[0]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model_state_dict"][name])


def test_train_fit_gmm_copies_fitted_parameters_to_prior():
    model = _TrainingModel()
    loaders = {"train": [_evaluation_batch([-3.0, -2.0, 2.0, 3.0], [0, 0, 1, 1])]}

    result = train_fit_gmm(model, loaders)

    assert result.model is model
    assert model._neural_ode_phase == "clustered"
    assert result.estimator.n_components == 2
    assert result.estimator.covariance_type == "diag"
    torch.testing.assert_close(model.prior_mu, torch.tensor(result.estimator.means_, dtype=torch.float32))
    torch.testing.assert_close(model.log_prior_var.exp(),
                               torch.tensor(result.estimator.covariances_, dtype=torch.float32))
    torch.testing.assert_close(model.log_pi.exp(), torch.tensor(result.estimator.weights_, dtype=torch.float32))


def test_predict_preserves_metadata_and_removes_padding():
    model = _EvaluationModel(phase="finetune")
    batch = _evaluation_batch([-1.0, 1.0], [0, -1])
    batch.update(ids=["first", "second"], target_lengths=torch.tensor([2, 1]),
                 observed_lengths=torch.tensor([1, 1]), mode="extrap")

    result = predict(model, [batch])

    assert result.phase == "finetune"
    assert result.cutoff_time == 0.0
    assert not model.training
    assert [trajectory.id for trajectory in result.trajectories] == batch["ids"]
    assert [trajectory.label for trajectory in result.trajectories] == [0, None]
    assert [trajectory.cluster for trajectory in result.trajectories] == [0, 1]
    for trajectory, length in zip(result.trajectories, [2, 1]):
        assert trajectory.prediction.shape == trajectory.truth.shape == trajectory.mask.shape == (length, 1)
        assert trajectory.times.shape == (length,)
        torch.testing.assert_close(trajectory.prediction, torch.zeros(length, 1))
        torch.testing.assert_close(trajectory.truth, batch["data_to_predict"][0, :length])
        assert trajectory.observed_values.shape == trajectory.observed_mask.shape == (1, 1)
        assert trajectory.cutoff_time == 0.0


def test_evaluate_performance_pools_masked_forecast_mse_and_skips_pre_gmm_metrics():
    first = _evaluation_batch([-1.0], [-1])
    first["data_to_predict"] = torch.tensor([[[1.0], [3.0]]])
    second = _evaluation_batch([1.0, 2.0], [-1, -1])
    second["data_to_predict"] = torch.tensor([[[2.0], [0.0]], [[4.0], [0.0]]])
    second["mask_predicted_data"] = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])

    scores = evaluate_performance(_EvaluationModel(), [first, second], device="cpu")

    assert list(scores) == [
        "mse", "nmi", "ari", "acc", "silhouette", "davies_bouldin",
        "calinski_harabasz",
    ]
    # Valid errors are 1^2 + 3^2 + 2^2 + 4^2 = 30 across four target values.
    assert scores["mse"] == pytest.approx(7.5)
    assert all(scores[name] is None for name in list(scores)[1:])


def test_evaluate_performance_accepts_separately_normalized_loader():
    class NormalizedBatches(list):
        pass

    test_loader = NormalizedBatches([_evaluation_batch([0.0], [-1])])
    test_loader._neural_ode_normalization = (torch.tensor([20.0]), torch.tensor([4.0]))
    model = _EvaluationModel()
    model._neural_ode_normalization = (torch.tensor([10.0]), torch.tensor([2.0]))

    scores = evaluate_performance(model, test_loader)

    # Test values [1, 3] map to training scale [7, 11], whose mean square is 85.
    assert scores["mse"] == pytest.approx(85.0)


def test_evaluate_performance_reports_label_and_internal_cluster_metrics():
    batches = [
        _evaluation_batch([-2.0, -1.0], [1, 1]),
        _evaluation_batch([1.0, 2.0], [0, -1]),
    ]

    scores = evaluate_performance(_EvaluationModel(phase="clustered"), batches)

    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["acc"] == pytest.approx(1.0)
    assert scores["silhouette"] is not None
    assert scores["davies_bouldin"] is not None
    assert scores["calinski_harabasz"] is not None


def test_evaluate_performance_supports_unlabeled_data_and_degenerate_clusters():
    batches = [
        _evaluation_batch([-2.0, -1.0], [-1, -1]),
        _evaluation_batch([1.0, 2.0], [-1, -1]),
    ]
    scores = evaluate_performance(_EvaluationModel(phase="finetune"), batches)
    assert all(scores[name] is None for name in ("nmi", "ari", "acc"))
    assert scores["silhouette"] is not None

    degenerate = evaluate_performance(
        _EvaluationModel(phase="clustered", single_cluster=True),
        [_evaluation_batch([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1])],
    )
    assert degenerate["nmi"] is not None
    assert all(degenerate[name] is None for name in (
        "silhouette", "davies_bouldin", "calinski_harabasz"
    ))
