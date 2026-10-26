import pytest
import torch

from neural_ode import evaluate_performance

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
