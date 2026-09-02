from __future__ import annotations

import pytest

from src.utils.config import ExperimentConfig, validate_config
from src.utils.termination_metrics import (
    binary_metrics,
    evaluate_always_continue,
    evaluate_distance_threshold,
    evaluate_fixed_step,
    select_distance_threshold,
    select_fixed_step,
)


TRACES = [
    {"distances": [0.9, 0.5, 0.1], "targets": [0, 0, 1]},
    {"distances": [0.8, 0.2], "targets": [0, 1]},
]


def test_binary_metrics_report_balanced_accuracy() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)


def test_always_continue_exposes_class_prior_failure() -> None:
    metrics = evaluate_always_continue(TRACES)
    assert metrics["classification"]["accuracy"] == pytest.approx(3 / 5)
    assert metrics["classification"]["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["stopping"]["mean_absolute_error"] == pytest.approx(1.0)
    assert metrics["stopping"]["exact_stop_accuracy"] == pytest.approx(0.0)


def test_distance_threshold_and_fixed_step_have_stopping_metrics() -> None:
    distance = evaluate_distance_threshold(TRACES, 0.3)
    assert distance["stopping"]["exact_stop_accuracy"] == pytest.approx(1.0)
    assert distance["classification"]["balanced_accuracy"] == pytest.approx(1.0)

    fixed = evaluate_fixed_step(TRACES, 1)
    assert fixed["stopping"]["mean_absolute_error"] == pytest.approx(0.5)


def test_validation_selection_is_algorithm_local() -> None:
    threshold, threshold_metrics = select_distance_threshold(TRACES)
    assert 0.2 < threshold < 0.5
    assert threshold_metrics["classification"]["balanced_accuracy"] == pytest.approx(1.0)

    step, step_metrics = select_fixed_step(TRACES)
    assert step == 2
    assert step_metrics["stopping"]["mean_absolute_error"] == pytest.approx(0.5)


def test_algorithm_specific_threshold_validation() -> None:
    config = ExperimentConfig()
    config.model.algorithms = ["bf", "bfs"]
    config.model.termination_distance = "rms"
    config.model.termination_distance_thresholds = {"bf": 0.2, "bfs": 0.3}
    config.data.train_path = "artifacts/reference/data/test_20n_100g.npz"
    config.data.val_path = "artifacts/reference/data/test_20n_100g.npz"
    config.data.test_path = "artifacts/reference/data/test_200n_20g.npz"
    validate_config(config)

    config.model.termination_distance_thresholds["prim"] = 0.4
    with pytest.raises(ValueError, match="not present"):
        validate_config(config)
