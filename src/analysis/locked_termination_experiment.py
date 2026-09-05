"""Run a validation-locked, multi-seed structural-termination experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from src.analysis.common import load_model_from_checkpoint
from src.data import generated_dataset, load_dataset, save_dataset
from src.train import evaluate_model
from src.utils import (
    collect_termination_traces,
    load_config,
    save_config,
    validate_config,
)
from src.utils.repro import set_seed
from src.utils.task_specs import metric_dict, resolve_selected_tasks
from src.utils.termination_metrics import (
    evaluate_always_continue,
    evaluate_distance_threshold,
    evaluate_fixed_step,
    select_distance_threshold,
    select_fixed_step,
    strip_per_graph_arrays,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "locked_termination.yaml"
DISTANCES = ("rms", "mean_nodewise_l2")
SPLIT_SPECS = {
    "train": (1500, 20, 1101),
    "val": (200, 20, 2202),
    "test_id": (200, 20, 3303),
    "test_ood": (100, 200, 4404),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "locked_multiseed",
    )
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _dataset_path(output_dir: Path, split: str, graphs: int, nodes: int) -> Path:
    return output_dir / "data" / f"{split}_{nodes}n_{graphs}g.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_fingerprint(graph: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(graph["edge_matrix"]).tobytes())
    digest.update(str(int(graph["source_node"])).encode())
    return digest.hexdigest()


def prepare_datasets(config, output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    fingerprints: dict[str, set[str]] = {}
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, (graphs, nodes, seed) in SPLIT_SPECS.items():
        path = _dataset_path(output_dir, split, graphs, nodes)
        if not path.exists():
            set_seed(seed)
            save_dataset(
                generated_dataset(graphs, nodes, task="multitask"),
                path,
                task="multitask",
            )
        dataset = load_dataset(path)
        paths[split] = path
        fingerprints[split] = {_graph_fingerprint(graph) for graph in dataset}

    overlaps = {}
    split_names = list(paths)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                fingerprints[left] & fingerprints[right]
            )
    if any(overlaps.values()):
        raise ValueError(f"Exact graph overlap detected across locked splits: {overlaps}")

    manifest = {
        "protocol": "validation-locked-v1",
        "split_specs": {
            split: {"graphs": graphs, "nodes": nodes, "generator_seed": seed}
            for split, (graphs, nodes, seed) in SPLIT_SPECS.items()
        },
        "sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in paths.values()},
        "exact_graph_overlap_counts": overlaps,
        "model_seeds": [],
        "distance_metrics": list(DISTANCES),
        "training": {
            "epochs": int(config.training.epochs),
            "learning_rate": float(config.training.learning_rate),
            "batch_size": int(config.training.batch_size),
            "max_grad_norm": float(config.training.max_grad_norm),
            "termination_supervision_weight": float(
                config.model.termination_supervision_weight
            ),
            "termination_balance_loss": bool(
                config.model.termination_balance_loss
            ),
            "training_rollout": "teacher_forced",
            "evaluation_rollout": config.model.evaluation_rollout_mode,
        },
        "selection": {
            "distance_threshold": "maximum validation balanced accuracy; ties by MAE then smaller threshold",
            "fixed_step": "minimum validation stopping-step MAE; ties by balanced accuracy then earlier step",
        },
        "test_policy": "test_id and test_ood remain unopened until validation selections are fixed per seed",
    }
    (output_dir / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2))

    config.data.train_path = str(paths["train"])
    config.data.val_path = str(paths["val"])
    config.data.test_path = str(paths["test_id"])
    return paths


def _task_metrics(model, dataset, config, selected_tasks) -> dict[str, float]:
    losses, loss, accuracies = evaluate_model(
        model,
        dataset,
        config.model.embed_dim,
        config.model,
        selected_tasks,
    )
    payload = {"loss": float(loss)}
    payload.update(metric_dict("losses", losses, model.algorithms))
    payload.update(metric_dict("acc", accuracies, model.algorithms))
    return payload


def _evaluate_seed(seed: int, base_config, paths: dict[str, Path], output_dir: Path) -> dict:
    seed_dir = output_dir / "seeds" / f"seed_{seed}"
    run_dir = seed_dir / "run"
    result_path = seed_dir / "locked_results.json"
    seed_dir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(base_config)
    config.training.seed = seed
    config.data.train_path = str(paths["train"])
    config.data.val_path = str(paths["val"])
    config.data.test_path = str(paths["test_id"])
    config.model.evaluation_rollout_mode = "autoregressive"
    validate_config(config)
    config_path = seed_dir / "config_locked.yaml"
    save_config(config, config_path)

    final_checkpoint = (
        run_dir
        / "checkpoints"
        / f"step_{config.training.epochs:08d}"
        / "checkpoint.json"
    )
    if not final_checkpoint.exists():
        log_path = seed_dir / "training.log"
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--skip-final-test",
        ]
        if run_dir.exists():
            command.append("--resume")
        with log_path.open("a") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    model, checkpoint_step = load_model_from_checkpoint(
        config,
        run_dir / "checkpoints",
        run_dir,
    )
    model.eval()
    selected_tasks = resolve_selected_tasks("all", model.algorithms)
    validation_dataset = load_dataset(paths["val"])

    result = {
        "seed": seed,
        "checkpoint_step": int(checkpoint_step),
        "rollout_mode": config.model.evaluation_rollout_mode,
        "termination_supervision": {
            "enabled": config.model.termination_distance_signal,
            "weight": config.model.termination_supervision_weight,
            "balanced": config.model.termination_balance_loss,
        },
        "distance_metrics": {},
        "task_metrics": {},
    }

    for distance_type in DISTANCES:
        distance_config = copy.deepcopy(config)
        distance_config.model.termination_distance = distance_type
        validation_traces = collect_termination_traces(
            model, validation_dataset, distance_config.model, selected_tasks
        )
        selections = {}
        for algorithm in model.algorithms:
            threshold, threshold_val = select_distance_threshold(
                validation_traces[algorithm]
            )
            fixed_step, fixed_val = select_fixed_step(validation_traces[algorithm])
            selections[algorithm] = {
                "threshold": threshold,
                "fixed_step": fixed_step,
                "validation": {
                    "distance": strip_per_graph_arrays(threshold_val),
                    "always_continue": strip_per_graph_arrays(
                        evaluate_always_continue(validation_traces[algorithm])
                    ),
                    "fixed_step": strip_per_graph_arrays(fixed_val),
                },
            }

        result["distance_metrics"][distance_type] = {
            "selection": selections,
            "test": {},
        }

    result["task_metrics"]["val"] = _task_metrics(
        model, validation_dataset, config, selected_tasks
    )

    test_datasets = {
        split: load_dataset(paths[split]) for split in ("test_id", "test_ood")
    }
    for distance_type in DISTANCES:
        distance_config = copy.deepcopy(config)
        distance_config.model.termination_distance = distance_type
        distance_result = result["distance_metrics"][distance_type]
        selections = distance_result["selection"]
        for split in ("test_id", "test_ood"):
            test_traces = collect_termination_traces(
                model, test_datasets[split], distance_config.model, selected_tasks
            )
            distance_result["test"][split] = {}
            for algorithm in model.algorithms:
                threshold = float(selections[algorithm]["threshold"])
                fixed_step = int(selections[algorithm]["fixed_step"])
                distance_result["test"][split][algorithm] = {
                    "distance": strip_per_graph_arrays(
                        evaluate_distance_threshold(test_traces[algorithm], threshold)
                    ),
                    "always_continue": strip_per_graph_arrays(
                        evaluate_always_continue(test_traces[algorithm])
                    ),
                    "fixed_step": strip_per_graph_arrays(
                        evaluate_fixed_step(test_traces[algorithm], fixed_step)
                    ),
                }
    for split in ("test_id", "test_ood"):
        result["task_metrics"][split] = _task_metrics(
            model, test_datasets[split], config, selected_tasks
        )

    result_path.write_text(json.dumps(result, indent=2))
    return result


def _aggregate_scalar_paths(results: list[dict]) -> dict[str, dict[str, float]]:
    paths: dict[str, list[float]] = {}

    def visit(value, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            paths.setdefault(prefix, []).append(float(value))

    for result in results:
        visit(result, "")
    return {
        path: {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
        for path, values in sorted(paths.items())
        if len(values) == len(results)
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    config = load_config(args.config)
    if args.worker_seed is not None:
        paths = {
            split: _dataset_path(output_dir, split, graphs, nodes)
            for split, (graphs, nodes, _) in SPLIT_SPECS.items()
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Prepare the locked datasets before starting seed workers: "
                + ", ".join(missing)
            )
        _evaluate_seed(args.worker_seed, config, paths, output_dir)
        print(f"Locked seed complete: {args.worker_seed}")
        return

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 5 and not args.prepare_only:
        raise ValueError("The locked experiment requires at least five model seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Model seeds must be unique.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = prepare_datasets(config, output_dir)
    manifest_path = output_dir / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_seeds"] = seeds
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if args.prepare_only:
        print(f"Prepared locked datasets and manifest at {output_dir}")
        return

    results = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] seed={seed}")
        if args.skip_training:
            result_path = output_dir / "seeds" / f"seed_{seed}" / "locked_results.json"
            if not result_path.exists():
                raise FileNotFoundError(result_path)
            results.append(json.loads(result_path.read_text()))
        else:
            results.append(_evaluate_seed(seed, config, paths, output_dir))

    aggregate = {
        "protocol": "validation-locked-v1",
        "seeds": seeds,
        "num_seeds": len(seeds),
        "scalar_summary": _aggregate_scalar_paths(results),
    }
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"Locked experiment complete: {output_dir}")


if __name__ == "__main__":
    main()
