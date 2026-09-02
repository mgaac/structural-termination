import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "locked_multiseed"
MANIFEST = json.loads((RESULTS / "protocol_manifest.json").read_text())
AGGREGATE = json.loads((RESULTS / "aggregate.json").read_text())


def nested_get(payload: dict, dotted_path: str):
    value = payload
    for part in dotted_path.split("."):
        value = value[part]
    return value


def test_locked_protocol_contract() -> None:
    assert MANIFEST["protocol"] == "validation-locked-v1"
    assert MANIFEST["model_seeds"] == [11, 22, 33, 44, 55]
    assert MANIFEST["split_specs"] == {
        "train": {"graphs": 1500, "nodes": 20, "generator_seed": 1101},
        "val": {"graphs": 200, "nodes": 20, "generator_seed": 2202},
        "test_id": {"graphs": 200, "nodes": 20, "generator_seed": 3303},
        "test_ood": {"graphs": 100, "nodes": 200, "generator_seed": 4404},
    }
    assert set(MANIFEST["exact_graph_overlap_counts"].values()) == {0}
    assert MANIFEST["training"]["termination_supervision_weight"] == 1.0
    assert MANIFEST["training"]["termination_balance_loss"] is True
    assert MANIFEST["training"]["evaluation_rollout"] == "autoregressive"


def test_seed_records_match_protocol() -> None:
    for seed in MANIFEST["model_seeds"]:
        path = RESULTS / "seeds" / f"seed_{seed}" / "locked_results.json"
        record = json.loads(path.read_text())
        assert record["seed"] == seed
        assert record["checkpoint_step"] == 100
        assert record["rollout_mode"] == "autoregressive"
        assert record["termination_supervision"] == {
            "enabled": True,
            "weight": 1.0,
            "balanced": True,
        }


def test_aggregate_is_recomputed_from_all_seed_records() -> None:
    records = [
        json.loads(
            (RESULTS / "seeds" / f"seed_{seed}" / "locked_results.json").read_text()
        )
        for seed in MANIFEST["model_seeds"]
    ]
    summary = AGGREGATE["scalar_summary"]
    paths = [
        f"distance_metrics.rms.test.{split}.{algorithm}.{policy}.{metric}"
        for split in ("test_id", "test_ood")
        for algorithm in ("bf", "bfs")
        for policy in ("distance", "fixed_step", "always_continue")
        for metric in (
            "classification.balanced_accuracy",
            "stopping.mean_absolute_error",
            "stopping.exact_stop_accuracy",
            "stopping.mean_signed_error",
        )
    ]
    for path in paths:
        values = [float(nested_get(record, path)) for record in records]
        assert summary[path]["n"] == 5
        assert math.isclose(summary[path]["mean"], mean(values), abs_tol=1e-12)
        assert math.isclose(summary[path]["std"], stdev(values), abs_tol=1e-12)


def test_readme_headline_values_are_fixed() -> None:
    summary = AGGREGATE["scalar_summary"]
    expected = {
        "distance_metrics.rms.test.test_id.bf.distance.classification.balanced_accuracy": (
            0.9133398486759142,
            0.005730159995736714,
        ),
        "distance_metrics.rms.test.test_id.bfs.distance.classification.balanced_accuracy": (
            0.9628571428571429,
            0.0072996636597874994,
        ),
        "distance_metrics.rms.test.test_ood.bf.distance.stopping.mean_absolute_error": (
            1.1219999999999999,
            0.1323631368621944,
        ),
        "distance_metrics.rms.test.test_ood.bfs.distance.classification.balanced_accuracy": (
            0.6905511811023622,
            0.015649296782954198,
        ),
    }
    for path, (expected_mean, expected_std) in expected.items():
        assert math.isclose(summary[path]["mean"], expected_mean, abs_tol=1e-12)
        assert math.isclose(summary[path]["std"], expected_std, abs_tol=1e-12)

    readme = (ROOT / "README.md").read_text()
    for displayed_value in (
        r"0.9133 \pm 0.0057",
        r"0.9629 \pm 0.0073",
        r"1.122 \pm 0.132",
        r"0.6906 \pm 0.0156",
    ):
        assert displayed_value in readme


def test_latent_trajectory_figure_contract() -> None:
    assert (ROOT / "figures" / "latent_trajectory_pca.png").is_file()
    source = json.loads(
        (RESULTS / "latent_trajectory_pca_seed11.json").read_text()
    )
    assert source["checkpoint_step"] == 100
    assert source["model_seed"] == 11
    assert source["split"] == "test_id"
    assert source["graphs"] == 200
    assert source["nodes"] == 20
    assert source["rollout"] == "autoregressive"
    assert source["pca_input"] == "mean-pooled processed state"
    assert len(source["displayed_graph_indices"]) == 12
    assert source["distance_by_step"][0]["n"] == 200
    assert source["distance_by_step"][0]["median"] > 2.5
    assert source["distance_by_step"][3]["median"] < 0.11
