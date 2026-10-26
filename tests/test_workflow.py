"""Exercise the example DAG, including reuse of completed training."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("snakemake") is None, reason="requires the workflow extra")
def test_basic_workflow_outputs_and_incremental_runs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    settings = {
        "output_dir": "results",
        "generation": {"n_train": 12, "n_test": 6, "n_classes": 2, "n_features": 1, "length": 5},
        "training": {"epochs": 1, "batch_size": 6},
        "model": {
            "latent_dim": 2,
            "n_clusters": 2,
            "networks": {
                "z0_encoder_odernn": {
                    "state_dim": 4,
                    "recognition_ode_hidden_dims": [4],
                    "gru_update_hidden_dims": [4],
                    "posterior_hidden_dims": [4],
                },
                "generative_ode": {"hidden_dims": [4]},
                "emission": {"hidden_dims": [4]},
            },
        },
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    command = [
        sys.executable, "-m", "snakemake", "--snakefile", str(root / "Snakefile"),
        "--directory", str(tmp_path), "--configfile", str(config), "--cores", "1",
    ]

    def run(*arguments):
        result = subprocess.run(command + list(arguments), capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr

    run()
    results = tmp_path / "results"
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["mse"] >= 0
    assert metrics["nmi"] is not None
    histories = json.loads((results / "history.json").read_text())
    assert len(histories["pretrain"]) == len(histories["finetune"]) == 1
    for name in ("input", "predictions", "training_loss"):
        assert (results / f"figures/{name}.pdf").read_bytes().startswith(b"%PDF")
    for name in ("gather_data", "train", "evaluate", "plot"):
        assert (results / f"logs/{name}.log").stat().st_size > 0

    model_mtime = (results / "model.pt").stat().st_mtime_ns
    assert "Nothing to be done" in run()
    (results / "figures/predictions.pdf").unlink()
    run()
    assert (results / "figures/predictions.pdf").exists()
    assert (results / "model.pt").stat().st_mtime_ns == model_mtime

    settings["training"]["epochs"] = 2
    config.write_text(json.dumps(settings))
    plan = run("--dry-run")
    assert "rule train:" in plan
    assert "rule evaluate:" in plan
    assert "rule plot:" in plan
    assert "rule gather_data:" not in plan
