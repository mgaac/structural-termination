import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISED = ROOT / "results" / "locked_multiseed"
NO_SUPERVISION = ROOT / "results" / "locked_multiseed_no_supervision"
COMPARISON = json.loads((NO_SUPERVISION / "comparison_to_supervised.json").read_text())


def _metric(split: str, algorithm: str, metric: str) -> dict:
    return COMPARISON["structural_policy"]["rms"][split][algorithm][metric]


def test_latest_result_bundle_is_complete() -> None:
    for result_dir in (SUPERVISED, NO_SUPERVISION):
        assert (result_dir / "protocol_manifest.json").is_file()
        assert (result_dir / "aggregate.json").is_file()
        for seed in (11, 22, 33, 44, 55):
            result = result_dir / "seeds" / f"seed_{seed}" / "locked_results.json"
            assert result.is_file()


def test_ablation_protocol_is_matched() -> None:
    assert COMPARISON["seeds"] == [11, 22, 33, 44, 55]
    assert COMPARISON["num_seeds"] == 5
    assert all(COMPARISON["protocol_checks"].values())
    supervised = COMPARISON["treatments"]["supervised"]
    no_supervision = COMPARISON["treatments"]["no_supervision"]
    assert supervised["termination_supervision_weight"] == 1.0
    assert supervised["termination_balance_loss"] is True
    assert no_supervision["termination_supervision_weight"] == 0.0
    assert no_supervision["termination_balance_loss"] is False


def test_readme_ablation_values_match_comparison() -> None:
    expected = {
        ("test_id", "bf"): (0.9052, 0.565),
        ("test_id", "bfs"): (0.7971, 0.677),
        ("test_ood", "bf"): (0.8793, 1.200),
        ("test_ood", "bfs"): (0.5524, 1.866),
    }
    for (split, algorithm), (accuracy, mae) in expected.items():
        actual_accuracy = _metric(split, algorithm, "balanced_accuracy")[
            "no_supervision"
        ]["mean"]
        actual_mae = _metric(split, algorithm, "stopping_mae")["no_supervision"][
            "mean"
        ]
        assert math.isclose(actual_accuracy, accuracy, abs_tol=5e-5)
        assert math.isclose(actual_mae, mae, abs_tol=5e-4)


def test_latest_figures_exist() -> None:
    assert (ROOT / "figures" / "latent_trajectory_pca.png").is_file()
    assert (ROOT / "figures" / "locked_multiseed_summary.png").is_file()
