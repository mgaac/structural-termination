import importlib

import numpy as np
import mlx.core as mx

from src.data import generated_dataset, load_dataset
from src.utils.repro import set_seed
from src.utils.termination import (
    compute_distance_termination_logits,
    compute_latent_distance,
)


def test_analysis_entry_points_import() -> None:
    for module_name in (
        "src.analysis.latent_convergence",
        "src.analysis.plot_latent_trajectory_pca",
        "src.analysis.termination_threshold_sweep",
    ):
        importlib.import_module(module_name)


def test_generated_dataset_is_seed_deterministic() -> None:
    set_seed(7)
    first = generated_dataset(2, 5, task="multitask")
    set_seed(7)
    second = generated_dataset(2, 5, task="multitask")

    for first_graph, second_graph in zip(first, second):
        assert first_graph.keys() == second_graph.keys()
        for key in first_graph:
            assert np.array_equal(np.asarray(first_graph[key]), np.asarray(second_graph[key]))


def test_legacy_loader_accepts_archives_without_prim_targets(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        num_graphs=1,
        num_nodes_0=2,
        edge_matrix_0=np.array([[0, 1], [1, 0], [1.0, 1.0]]),
        source_node_0=0,
        bf_distance_targets_0=np.zeros((1, 2)),
        bf_predecessor_targets_0=np.zeros((1, 2), dtype=np.int32),
        bfs_state_targets_0=np.zeros((1, 2)),
    )
    dataset = load_dataset(path)
    assert len(dataset) == 1
    assert "bf_distance_targets" in dataset[0]
    assert "bfs_state_targets" in dataset[0]
    assert "prim_state_targets" not in dataset[0]


def test_normalized_latent_distances() -> None:
    previous = mx.zeros((2, 2))
    current = mx.array([[3.0, 4.0], [0.0, 0.0]])
    assert float(compute_latent_distance(previous, current, "rms")) == 2.5
    assert float(compute_latent_distance(previous, current, "mean_nodewise_l2")) == 2.5


def test_algorithm_specific_distance_thresholds() -> None:
    settings = {
        "distance_type": "rms",
        "distance_threshold": 0.5,
        "distance_thresholds": {"bf": 1.0, "bfs": 2.0},
    }
    logits = compute_distance_termination_logits(
        settings,
        mx.zeros((1, 1)),
        mx.ones((1, 1)),
        algorithms=("bf", "bfs"),
    )
    assert float(logits["bf"]) == 0.0
    assert float(logits["bfs"]) == 1.0
