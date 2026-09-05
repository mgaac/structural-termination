"""One-command reproduction and end-to-end smoke workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.data import generated_dataset, save_dataset
from src.utils import load_config, save_config
from src.utils.repro import set_seed


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def reproduce_latest() -> None:
    commands = (
        [sys.executable, "-m", "src.analysis.locked_termination_experiment"],
        [
            sys.executable,
            "-m",
            "src.analysis.locked_termination_experiment",
            "--config",
            "configs/locked_termination_no_supervision.yaml",
            "--output-dir",
            "results/locked_multiseed_no_supervision",
        ],
        [sys.executable, "-m", "src.analysis.compare_termination_supervision"],
        [sys.executable, "-m", "src.analysis.plot_locked_results"],
        [sys.executable, "-m", "src.analysis.plot_latent_trajectory_pca"],
    )
    for command in commands:
        _run(command)
    print("Latest supervised and no-supervision results reproduced.")


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
    subparsers.add_parser(
        "latest",
        help="Rebuild the latest paired five-seed experiment and figures.",
    )
    smoke_parser = subparsers.add_parser("smoke", help="Generate, train, and analyze a tiny run.")
    smoke_parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        smoke(args.output_dir)
    else:
        reproduce_latest()


if __name__ == "__main__":
    main()
