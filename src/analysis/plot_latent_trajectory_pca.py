"""Plot autoregressive processor trajectories and their true latent distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mlx.core as mx
import numpy as np

from src.analysis.common import load_model_from_checkpoint, portable_path
from src.data import load_dataset
from src.utils import load_config
from src.utils.eval import _forward_step
from src.utils.task_specs import execution_step_counts, feature_values_for_step
from src.utils.termination import resolve_termination_settings


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_DIR = ROOT / "results" / "locked_multiseed" / "seeds" / "seed_11"
DEFAULT_DATASET = (
    ROOT / "results" / "locked_multiseed" / "data" / "test_id_20n_200g.npz"
)
DEFAULT_OUTPUT = ROOT / "figures" / "latent_trajectory_pca.png"
DEFAULT_SOURCE = (
    ROOT / "results" / "locked_multiseed" / "latent_trajectory_pca_seed11.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--display-trajectories", type=int, default=12)
    return parser.parse_args()


def collect_trajectory(model, graph, config):
    algorithms = tuple(model.algorithms)
    step_counts = execution_step_counts(graph, algorithms)
    previous_hidden = mx.zeros(
        (int(graph["num_nodes"]), model.processor_embed_dim)
    )
    previous_distance_latent = None
    current_features = feature_values_for_step(graph, 0, algorithms)
    settings = resolve_termination_settings(config.model)
    pooled_states = [np.zeros(model.processor_embed_dim, dtype=np.float64)]
    distances = []

    for step_index in range(max(step_counts.values())):
        payload = _forward_step(
            model=model,
            graph_data=graph,
            step_index=step_index,
            previous_step_hidden_states=previous_hidden,
            previous_distance_latent=previous_distance_latent,
            termination_settings=settings,
            algorithm_order=algorithms,
            step_counts=step_counts,
            feature_values_override=current_features,
        )
        if payload is None:
            continue
        state = np.asarray(payload["processed_embeddings"], dtype=np.float64)
        pooled_states.append(state.mean(axis=0))
        distances.append(float(payload["termination_distance"].item()))
        previous_hidden = payload["processed_embeddings"]
        previous_distance_latent = payload["next_distance_latent"]
        current_features = payload["next_feature_values"]

    return np.stack(pooled_states), np.asarray(distances)


def pca_coordinates(trajectories):
    states = np.concatenate([states for states, _ in trajectories], axis=0)
    centered = states - states.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:2].T
    variance = singular_values**2
    explained = variance[:2] / variance.sum()

    coordinates = []
    cursor = 0
    for states, _ in trajectories:
        count = len(states)
        coordinates.append(centered[cursor : cursor + count] @ basis)
        cursor += count
    return coordinates, explained


def distance_summary(trajectories):
    max_steps = max(len(distances) for _, distances in trajectories)
    matrix = np.full((len(trajectories), max_steps), np.nan)
    for index, (_, distances) in enumerate(trajectories):
        matrix[index, : len(distances)] = distances
    return {
        "median": np.nanmedian(matrix, axis=0),
        "q25": np.nanpercentile(matrix, 25, axis=0),
        "q75": np.nanpercentile(matrix, 75, axis=0),
        "counts": np.sum(np.isfinite(matrix), axis=0),
    }


def render_figure(
    coordinates,
    trajectories,
    displayed_indices,
    explained,
    summary,
    output_path,
):
    all_distances = np.concatenate([distances for _, distances in trajectories])
    color_min, color_max = np.percentile(all_distances, [5, 95])
    normalization = plt.Normalize(color_min, color_max)
    color_map = LinearSegmentedColormap.from_list(
        "distance_blue",
        ["#dbeafe", "#60a5fa", "#1d4ed8", "#172554"],
    )

    figure, (pca_axis, distance_axis) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.1),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    for graph_index in displayed_indices:
        xy = coordinates[graph_index]
        distances = trajectories[graph_index][1]
        pca_axis.plot(
            xy[:, 0],
            xy[:, 1],
            color="#94a3b8",
            linewidth=1,
            alpha=0.65,
        )
        pca_axis.scatter(
            xy[1:, 0],
            xy[1:, 1],
            c=distances,
            cmap=color_map,
            norm=normalization,
            s=30,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        pca_axis.scatter(
            xy[0, 0],
            xy[0, 1],
            facecolors="white",
            edgecolors="#334155",
            s=30,
            linewidths=0.8,
            zorder=4,
        )

    pca_axis.set_title("Processor-state trajectories", loc="left", fontweight="bold")
    pca_axis.set_xlabel(f"PC1 ({explained[0]:.0%} variance)")
    pca_axis.set_ylabel(f"PC2 ({explained[1]:.0%} variance)")
    pca_axis.grid(color="#e2e8f0", linewidth=0.7)
    pca_axis.set_axisbelow(True)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        ax=pca_axis,
        fraction=0.045,
        pad=0.03,
    )
    colorbar.set_label(r"RMS $d_t$")

    steps = np.arange(1, len(summary["median"]) + 1)
    distance_axis.fill_between(
        steps,
        summary["q25"],
        summary["q75"],
        color="#bfdbfe",
        alpha=0.75,
        label="Interquartile range",
    )
    distance_axis.plot(
        steps,
        summary["median"],
        color="#2563eb",
        marker="o",
        linewidth=2.2,
        label="Median",
    )
    distance_axis.set_title("Distance through execution", loc="left", fontweight="bold")
    distance_axis.set_xlabel("Recurrent step")
    distance_axis.set_ylabel(r"Actual RMS step distance $d_t$")
    distance_axis.set_xticks(steps)
    distance_axis.set_ylim(bottom=0)
    distance_axis.grid(color="#e2e8f0", linewidth=0.7)
    distance_axis.set_axisbelow(True)
    distance_axis.legend(loc="upper right", frameon=False)

    for axis in (pca_axis, distance_axis):
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Latent convergence during autoregressive execution",
        x=0.07,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.945,
        "Seed 11 · held-out 20-node graphs · PCA basis and distance curve use all 200 graphs; 12 trajectories shown",
        color="#475569",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.04, 0.03, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.display_trajectories <= 0:
        raise ValueError("--display-trajectories must be positive.")

    config = load_config(args.seed_dir / "config_locked.yaml")
    config.model.termination_distance = "rms"
    config.model.evaluation_rollout_mode = "autoregressive"
    model, checkpoint_step = load_model_from_checkpoint(
        config,
        args.seed_dir / "run" / "checkpoints",
        args.seed_dir / "run",
    )
    dataset = load_dataset(args.dataset)
    trajectories = [collect_trajectory(model, graph, config) for graph in dataset]
    coordinates, explained = pca_coordinates(trajectories)
    summary = distance_summary(trajectories)
    displayed_indices = np.linspace(
        0,
        len(dataset) - 1,
        min(args.display_trajectories, len(dataset)),
        dtype=int,
    )

    render_figure(
        coordinates,
        trajectories,
        displayed_indices,
        explained,
        summary,
        args.output,
    )

    steps = np.arange(1, len(summary["median"]) + 1)
    source = {
        "checkpoint_step": int(checkpoint_step),
        "model_seed": int(config.training.seed),
        "split": "test_id",
        "dataset": portable_path(args.dataset),
        "graphs": len(dataset),
        "nodes": int(dataset[0]["num_nodes"]),
        "rollout": "autoregressive",
        "pca_input": "mean-pooled processed state",
        "pca_explained_variance_ratio": explained.tolist(),
        "displayed_graph_indices": displayed_indices.tolist(),
        "distance_definition": "RMS over the full node-by-latent tensor",
        "distance_by_step": [
            {
                "step": int(step),
                "n": int(count),
                "median": float(median),
                "q25": float(q25),
                "q75": float(q75),
            }
            for step, count, median, q25, q75 in zip(
                steps,
                summary["counts"],
                summary["median"],
                summary["q25"],
                summary["q75"],
            )
        ],
    }
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_text(json.dumps(source, indent=2))
    print(f"Saved figure to: {args.output}")
    print(f"Saved source data to: {args.source_output}")


if __name__ == "__main__":
    main()
