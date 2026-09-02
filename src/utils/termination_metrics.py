"""Pure termination-policy metrics and validation-only model selection."""

from __future__ import annotations

from math import inf
from statistics import mean
from typing import Callable, Iterable, Sequence


Trace = dict[str, object]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length.")
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_ratio(tp + tn, len(targets)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def _true_stop_index(targets: Sequence[int]) -> int:
    positives = [index for index, target in enumerate(targets) if target == 1]
    if len(positives) != 1:
        raise ValueError(
            "Every termination trace must contain exactly one terminal target; "
            f"found {len(positives)}."
        )
    return positives[0]


def evaluate_policy(
    traces: Sequence[Trace],
    predictor: Callable[[Trace], Sequence[int]],
) -> dict[str, object]:
    all_targets: list[int] = []
    all_predictions: list[int] = []
    signed_errors: list[int] = []
    absolute_errors: list[int] = []
    predicted_steps: list[int] = []
    true_steps: list[int] = []

    for trace in traces:
        targets = [int(value) for value in trace["targets"]]
        predictions = [int(value) for value in predictor(trace)]
        if len(predictions) != len(targets):
            raise ValueError("Policy prediction length does not match its trace.")
        true_step = _true_stop_index(targets)
        predicted_positive = [
            index for index, prediction in enumerate(predictions) if prediction == 1
        ]
        predicted_step = predicted_positive[0] if predicted_positive else len(targets)
        error = predicted_step - true_step

        all_targets.extend(targets)
        all_predictions.extend(predictions)
        signed_errors.append(error)
        absolute_errors.append(abs(error))
        predicted_steps.append(predicted_step)
        true_steps.append(true_step)

    classification = binary_metrics(all_targets, all_predictions)
    stopping = {
        "mean_signed_error": mean(signed_errors) if signed_errors else 0.0,
        "mean_absolute_error": mean(absolute_errors) if absolute_errors else 0.0,
        "exact_stop_accuracy": (
            mean(error == 0 for error in signed_errors) if signed_errors else 0.0
        ),
        "early_stop_rate": (
            mean(error < 0 for error in signed_errors) if signed_errors else 0.0
        ),
        "late_or_missing_stop_rate": (
            mean(error > 0 for error in signed_errors) if signed_errors else 0.0
        ),
        "predicted_steps": predicted_steps,
        "true_steps": true_steps,
        "signed_errors": signed_errors,
    }
    return {"classification": classification, "stopping": stopping}


def evaluate_distance_threshold(traces: Sequence[Trace], threshold: float) -> dict[str, object]:
    return evaluate_policy(
        traces,
        lambda trace: [
            int(float(distance) < threshold) for distance in trace["distances"]
        ],
    )


def evaluate_always_continue(traces: Sequence[Trace]) -> dict[str, object]:
    return evaluate_policy(traces, lambda trace: [0] * len(trace["targets"]))


def evaluate_fixed_step(traces: Sequence[Trace], step: int) -> dict[str, object]:
    if step < 0:
        raise ValueError("Fixed step must be non-negative.")
    return evaluate_policy(
        traces,
        lambda trace: [int(index == step) for index in range(len(trace["targets"]))],
    )


def threshold_candidates(traces: Sequence[Trace]) -> list[float]:
    values = sorted(
        {float(distance) for trace in traces for distance in trace["distances"]}
    )
    if not values:
        raise ValueError("Cannot select a threshold from empty traces.")
    candidates = [0.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + max(abs(values[-1]), 1.0) * 1e-6)
    return candidates


def select_distance_threshold(traces: Sequence[Trace]) -> tuple[float, dict[str, object]]:
    best_threshold = 0.0
    best_metrics: dict[str, object] | None = None
    best_key = (-inf, -inf, -inf)
    for threshold in threshold_candidates(traces):
        metrics = evaluate_distance_threshold(traces, threshold)
        key = (
            float(metrics["classification"]["balanced_accuracy"]),
            -float(metrics["stopping"]["mean_absolute_error"]),
            -threshold,
        )
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key
    assert best_metrics is not None
    return best_threshold, best_metrics


def select_fixed_step(traces: Sequence[Trace]) -> tuple[int, dict[str, object]]:
    max_length = max((len(trace["targets"]) for trace in traces), default=0)
    if max_length == 0:
        raise ValueError("Cannot select a fixed step from empty traces.")
    best_step = 0
    best_metrics: dict[str, object] | None = None
    best_key = (-inf, -inf, -inf)
    for step in range(max_length):
        metrics = evaluate_fixed_step(traces, step)
        key = (
            -float(metrics["stopping"]["mean_absolute_error"]),
            float(metrics["classification"]["balanced_accuracy"]),
            -step,
        )
        if key > best_key:
            best_step = step
            best_metrics = metrics
            best_key = key
    assert best_metrics is not None
    return best_step, best_metrics


def strip_per_graph_arrays(metrics: dict[str, object]) -> dict[str, object]:
    """Remove bulky arrays while preserving aggregate metrics."""
    stopping = dict(metrics["stopping"])
    for key in ("predicted_steps", "true_steps", "signed_errors"):
        stopping.pop(key, None)
    return {"classification": dict(metrics["classification"]), "stopping": stopping}
