"""Compare matched locked runs with and without termination supervision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPERVISED = ROOT / "results" / "locked_multiseed"
DEFAULT_UNSUPERVISED = ROOT / "results" / "locked_multiseed_no_supervision"
DISTANCE_METRICS = ("rms", "mean_nodewise_l2")
TEST_SPLITS = ("test_id", "test_ood")
ALGORITHMS = ("bf", "bfs")
STRUCTURAL_METRICS = {
    "balanced_accuracy": ("classification", "balanced_accuracy", True),
    "stopping_mae": ("stopping", "mean_absolute_error", False),
    "exact_stop_accuracy": ("stopping", "exact_stop_accuracy", True),
    "mean_signed_error": ("stopping", "mean_signed_error", None),
}
TASK_METRICS = (
    "acc/bf_distance",
    "acc/bf_predecessor",
    "acc/bfs_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-dir", type=Path, default=DEFAULT_SUPERVISED)
    parser.add_argument("--unsupervised-dir", type=Path, default=DEFAULT_UNSUPERVISED)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "protocol_manifest.json").read_text())


def _hashes_by_filename(manifest: dict) -> dict[str, str]:
    return {Path(path).name: digest for path, digest in manifest["sha256"].items()}


def _load_seed_results(run_dir: Path) -> dict[int, dict]:
    results = {}
    for path in sorted((run_dir / "seeds").glob("seed_*/locked_results.json")):
        payload = json.loads(path.read_text())
        seed = int(payload["seed"])
        if seed in results:
            raise ValueError(f"Duplicate seed {seed} under {run_dir}")
        results[seed] = payload
    if not results:
        raise ValueError(f"No per-seed results found under {run_dir}")
    return results


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Comparison values must be finite and non-empty.")
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _paired_summary(
    supervised: list[float],
    unsupervised: list[float],
    higher_is_better: bool | None,
) -> dict:
    deltas = [right - left for left, right in zip(supervised, unsupervised)]
    payload = {
        "supervised": _summary(supervised),
        "no_supervision": _summary(unsupervised),
        "paired_delta_no_supervision_minus_supervised": _summary(deltas),
        "per_seed_delta": deltas,
    }
    if higher_is_better is not None:
        improvements = [
            delta > 0 if higher_is_better else delta < 0 for delta in deltas
        ]
        payload["no_supervision_better_seed_count"] = sum(improvements)
        payload["supervision_better_seed_count"] = sum(
            delta < 0 if higher_is_better else delta > 0 for delta in deltas
        )
    return payload


def build_comparison(supervised_dir: Path, unsupervised_dir: Path) -> dict:
    supervised_manifest = _load_manifest(supervised_dir)
    unsupervised_manifest = _load_manifest(unsupervised_dir)
    checks = {
        "split_specs_equal": supervised_manifest["split_specs"]
        == unsupervised_manifest["split_specs"],
        "dataset_hashes_equal": _hashes_by_filename(supervised_manifest)
        == _hashes_by_filename(unsupervised_manifest),
        "all_overlap_counts_zero": all(
            value == 0
            for manifest in (supervised_manifest, unsupervised_manifest)
            for value in manifest["exact_graph_overlap_counts"].values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Locked-protocol mismatch: {checks}")

    supervised = _load_seed_results(supervised_dir)
    unsupervised = _load_seed_results(unsupervised_dir)
    if supervised.keys() != unsupervised.keys():
        raise ValueError("Supervised and unsupervised seed sets differ.")
    seeds = sorted(supervised)

    if any(
        not supervised[seed]["termination_supervision"]["enabled"] for seed in seeds
    ):
        raise ValueError("The supervised run contains a disabled-supervision seed.")
    if any(
        unsupervised[seed]["termination_supervision"]["enabled"] for seed in seeds
    ):
        raise ValueError("The no-supervision run contains an enabled-supervision seed.")

    structural = {}
    for distance in DISTANCE_METRICS:
        structural[distance] = {}
        for split in TEST_SPLITS:
            structural[distance][split] = {}
            for algorithm in ALGORITHMS:
                metrics = {}
                for label, (section, key, higher_is_better) in STRUCTURAL_METRICS.items():
                    supervised_values = [
                        supervised[seed]["distance_metrics"][distance]["test"][split][
                            algorithm
                        ]["distance"][section][key]
                        for seed in seeds
                    ]
                    unsupervised_values = [
                        unsupervised[seed]["distance_metrics"][distance]["test"][split][
                            algorithm
                        ]["distance"][section][key]
                        for seed in seeds
                    ]
                    metrics[label] = _paired_summary(
                        supervised_values, unsupervised_values, higher_is_better
                    )
                structural[distance][split][algorithm] = metrics

    task_metrics = {}
    for split in TEST_SPLITS:
        task_metrics[split] = {}
        for metric in TASK_METRICS:
            task_metrics[split][metric] = _paired_summary(
                [supervised[seed]["task_metrics"][split][metric] for seed in seeds],
                [unsupervised[seed]["task_metrics"][split][metric] for seed in seeds],
                True,
            )

    thresholds = {}
    for algorithm in ALGORITHMS:
        thresholds[algorithm] = _paired_summary(
            [
                supervised[seed]["distance_metrics"]["rms"]["selection"][algorithm][
                    "threshold"
                ]
                for seed in seeds
            ],
            [
                unsupervised[seed]["distance_metrics"]["rms"]["selection"][algorithm][
                    "threshold"
                ]
                for seed in seeds
            ],
            None,
        )

    return {
        "comparison": "no_supervision_minus_supervised",
        "seeds": seeds,
        "num_seeds": len(seeds),
        "protocol_checks": checks,
        "treatments": {
            "supervised": supervised_manifest["training"],
            "no_supervision": unsupervised_manifest["training"],
        },
        "rms_validation_thresholds": thresholds,
        "structural_policy": structural,
        "task_metrics": task_metrics,
    }


def main() -> None:
    args = parse_args()
    supervised_dir = args.supervised_dir.resolve()
    unsupervised_dir = args.unsupervised_dir.resolve()
    output = args.output or unsupervised_dir / "comparison_to_supervised.json"
    comparison = build_comparison(supervised_dir, unsupervised_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2))
    print(f"Wrote matched comparison to {output}")


if __name__ == "__main__":
    main()
