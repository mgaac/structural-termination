"""One-command reproduction and end-to-end smoke workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import load_model_from_checkpoint
from src.data import generated_dataset, load_dataset, save_dataset
from src.train import evaluate_model
from src.utils import load_config, save_config, validate_config
from src.utils.repro import set_seed
from src.utils.task_specs import metric_dict, resolve_selected_tasks


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "configs" / "reference.yaml"
REFERENCE_CHECKPOINT = (
    ROOT / "artifacts" / "reference" / "checkpoint" / "step_00000500"
)
REFERENCE_MANIFEST = ROOT / "artifacts" / "reference" / "manifest.json"
EXPECTED_RESULTS = ROOT / "results" / "reference" / "expected.json"


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> None:
    manifest = json.loads(REFERENCE_MANIFEST.read_text())
    failures = []
    for relative_path, expected_hash in manifest["sha256"].items():
        path = ROOT / relative_path
        if not path.exists():
            failures.append(f"missing {relative_path}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            failures.append(
                f"hash mismatch {relative_path}: expected {expected_hash}, got {actual_hash}"
            )
    if failures:
        raise ValueError("Reference bundle verification failed: " + "; ".join(failures))


def _assert_close(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected mapping, got {type(actual).__name__}")
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f"{path}: missing key {key}")
            _assert_close(value, actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: list length mismatch")
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            _assert_close(expected_value, actual_value, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(float(expected), float(actual), rtol=1e-6, atol=1e-7):
            raise AssertionError(f"{path}: expected {expected}, got {actual}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def reproduce_reference(output_dir: Path) -> None:
    verify_manifest()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(REFERENCE_CONFIG)
    validate_config(config)
    model, checkpoint_step = load_model_from_checkpoint(
        config,
        REFERENCE_CHECKPOINT,
        None,
    )
    selected_tasks = resolve_selected_tasks("all", model.algorithms)
    dataset = load_dataset(ROOT / config.data.test_path)
    losses, loss, accuracies = evaluate_model(
        model,
        dataset,
        config.model.embed_dim,
        config.model,
        selected_tasks,
    )
    evaluation = {
        "checkpoint_step": int(checkpoint_step),
        "dataset": config.data.test_path,
        "num_graphs": len(dataset),
        "rollout_mode": config.model.evaluation_rollout_mode,
        "loss": float(loss),
        **metric_dict("losses", losses, model.algorithms),
        **metric_dict("acc", accuracies, model.algorithms),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    threshold_dir = output_dir / "threshold_sweep"
    _run(
        [
            sys.executable,
            "-m",
            "src.analysis.termination_threshold_sweep",
            "--config",
            str(REFERENCE_CONFIG),
            "--checkpoint",
            str(REFERENCE_CHECKPOINT),
            "--dataset",
            str(ROOT / "artifacts/reference/data/test_200n_20g.npz"),
            "--threshold-min",
            "74",
            "--threshold-max",
            "80",
            "--threshold-step",
            "0.1",
            "--output-dir",
            str(threshold_dir),
        ]
    )

    convergence_dir = output_dir / "latent_convergence"
    _run(
        [
            sys.executable,
            "-m",
            "src.analysis.latent_convergence",
            "--config",
            str(REFERENCE_CONFIG),
            "--checkpoint",
            str(REFERENCE_CHECKPOINT),
            "--dataset",
            str(ROOT / "artifacts/reference/data/test_20n_100g.npz"),
            "--latent",
            "processed",
            "--distance",
            "l2",
            "--output-dir",
            str(convergence_dir),
        ]
    )

    threshold = json.loads((threshold_dir / "test_threshold_sweep.json").read_text())
    convergence = json.loads((convergence_dir / "dataset_stats.json").read_text())
    summary = {
        "evaluation": evaluation,
        "threshold_sweep": {
            "thresholds": threshold["thresholds"],
            "acc_bfs_termination": threshold["termination_accuracy"]["bfs"],
            "acc_bf_termination": threshold["termination_accuracy"]["bf"],
        },
        "latent_convergence": {
            "mean": convergence["mean"],
            "std": convergence["std"],
            "counts": convergence["counts"],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    expected = json.loads(EXPECTED_RESULTS.read_text())
    _assert_close(expected, summary)
    print(f"Reference results reproduced and verified: {output_dir}")


def smoke(output_dir: Path | None) -> None:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "reproduced" / f"smoke-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = output_dir / "data"
    data_dir.mkdir()

    seed = 42
    set_seed(seed)
    split_specs = {"train": 4, "val": 2, "test": 2}
    split_paths = {}
    for split, num_graphs in split_specs.items():
        path = data_dir / f"{split}.npz"
        save_dataset(
            generated_dataset(num_graphs, 4, task="multitask"),
            path,
            task="multitask",
        )
        split_paths[split] = path

    config = load_config(ROOT / "configs" / "smoke.yaml")
    config.data.train_path = str(split_paths["train"])
    config.data.val_path = str(split_paths["val"])
    config.data.test_path = str(split_paths["test"])
    config_path = output_dir / "config.yaml"
    save_config(config, config_path)
    run_dir = output_dir / "run"

    _run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--run-dir",
            str(run_dir),
            "--eval-only",
            "--accuracies-only",
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "src.analysis.termination_threshold_sweep",
            "--run-dir",
            str(run_dir),
            "--split",
            "val",
            "--thresholds",
            "0,12",
            "--output-dir",
            str(output_dir / "threshold_sweep"),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "src.analysis.latent_convergence",
            "--run-dir",
            str(run_dir),
            "--split",
            "test",
            "--max-graphs",
            "2",
            "--latent",
            "processed",
            "--distance",
            "mean_l2",
            "--output-dir",
            str(output_dir / "latent_convergence"),
        ]
    )
    print(f"End-to-end smoke reproduction completed: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reference = subparsers.add_parser("reference", help="Rebuild and verify displayed results.")
    reference.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reproduced" / "reference",
    )
    smoke_parser = subparsers.add_parser("smoke", help="Generate, train, and analyze a tiny run.")
    smoke_parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "reference":
        reproduce_reference(args.output_dir)
    else:
        smoke(args.output_dir)


if __name__ == "__main__":
    main()
