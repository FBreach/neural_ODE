"""Run the basic example from generated data to saved evaluation and figures."""

from pathlib import Path

ROOT = Path(workflow.basedir)
configfile: str(ROOT / "examples/config.yaml")

RESULTS = config["output_dir"]
SOURCES = sorted(str(path) for path in (ROOT / "src/neural_ode").rglob("*.py"))
HELPERS = str(ROOT / "examples/scripts/helpers.py")


rule all:
    input:
        f"{RESULTS}/metrics.json",
        f"{RESULTS}/figures/input.pdf",
        f"{RESULTS}/figures/predictions.pdf",
        f"{RESULTS}/figures/training_loss.pdf",


rule gather_data:
    input:
        sources=SOURCES,
        helpers=HELPERS,
    output:
        train=f"{RESULTS}/data/train.csv",
        test=f"{RESULTS}/data/test.csv",
    params:
        generation=config["generation"],
    log:
        f"{RESULTS}/logs/gather_data.log",
    script:
        "examples/scripts/gather_data.py"


rule train:
    input:
        train=rules.gather_data.output.train,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        model=f"{RESULTS}/model.pt",
        history=f"{RESULTS}/history.json",
    params:
        preparation=config["preparation"],
        training=config["training"],
        model=config["model"],
    threads: 1
    log:
        f"{RESULTS}/logs/train.log",
    script:
        "examples/scripts/train.py"


rule evaluate:
    input:
        test=rules.gather_data.output.test,
        model=rules.train.output.model,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        metrics=f"{RESULTS}/metrics.json",
    threads: 1
    log:
        f"{RESULTS}/logs/evaluate.log",
    script:
        "examples/scripts/evaluate.py"


rule plot:
    input:
        test=rules.gather_data.output.test,
        model=rules.train.output.model,
        history=rules.train.output.history,
        sources=SOURCES,
        helpers=HELPERS,
    output:
        inputs=f"{RESULTS}/figures/input.pdf",
        predictions=f"{RESULTS}/figures/predictions.pdf",
        loss=f"{RESULTS}/figures/training_loss.pdf",
    threads: 1
    log:
        f"{RESULTS}/logs/plot.log",
    script:
        "examples/scripts/plot.py"
