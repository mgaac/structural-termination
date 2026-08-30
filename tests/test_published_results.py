import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "historical_single_seed"


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def test_fixed_threshold_diagnostic_matches_readme() -> None:
    evaluation = read_json("eval_only.json")
    assert evaluation["termination"]["mode"] == "distance"
    assert evaluation["termination"]["threshold"] == 7.5
    assert evaluation["val"]["acc/bf_termination"] == 0.9259526133537292
    assert evaluation["test"]["acc/bf_termination"] == 0.8092063069343567
    assert evaluation["val"]["acc/bfs_termination"] == 0.6949048638343811
    assert evaluation["test"]["acc/bfs_termination"] == 0.6316666603088379


def test_historical_run_is_single_seed() -> None:
    metadata = read_json("meta.json")
    assert metadata["seed"] == 42
    assert metadata["config"]["model"]["termination_mode"] == "distance"
    assert metadata["config"]["model"]["embed_dim"] == 32


def test_sweeps_are_not_misrepresented_as_locked_protocol() -> None:
    validation = read_json("val_threshold_sweep.json")
    test = read_json("test_threshold_sweep.json")
    assert validation["thresholds"] != test["thresholds"]
    assert min(validation["thresholds"]) == 25.0
    assert min(test["thresholds"]) == 74.0


def test_convergence_curve_preserves_changing_cohort_counts() -> None:
    convergence = read_json("latent_convergence_stats.json")
    assert convergence["metadata"]["num_graphs"] == 100
    assert convergence["counts"][0] == 99.0
    assert convergence["counts"][-1] == 7.0
    assert len(convergence["mean"]) == len(convergence["counts"])
