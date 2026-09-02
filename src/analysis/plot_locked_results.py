"""Render the locked five-seed stopping summary from aggregate.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE = ROOT / "results" / "locked_multiseed" / "aggregate.json"
DEFAULT_OUTPUT = ROOT / "figures" / "locked_multiseed_summary.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric(summary: dict, split: str, algorithm: str, policy: str, name: str):
    key = f"distance_metrics.rms.test.{split}.{algorithm}.{policy}.{name}"
    return summary[key]["mean"], summary[key]["std"]


def main() -> None:
    args = parse_args()
    aggregate = json.loads(args.aggregate.read_text())
    summary = aggregate["scalar_summary"]
    rows = [
        ("test_id", "bf", "ID · Bellman–Ford"),
        ("test_id", "bfs", "ID · BFS"),
        ("test_ood", "bf", "OOD · Bellman–Ford"),
        ("test_ood", "bfs", "OOD · BFS"),
    ]
    policies = [
        ("distance", "Structural", "#2563eb"),
        ("fixed_step", "Fixed step", "#94a3b8"),
        ("always_continue", "Always continue", "#d1d5db"),
    ]
    plotted_metrics = [
        ("classification.balanced_accuracy", "Balanced accuracy"),
        ("stopping.exact_stop_accuracy", "Exact-stop rate"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    y = np.arange(len(rows))
    offsets = (-0.24, 0.0, 0.24)

    for axis, (metric_name, title) in zip(axes, plotted_metrics):
        for offset, (policy, label, color) in zip(offsets, policies):
            values = []
            errors = []
            for split, algorithm, _ in rows:
                value, error = metric(summary, split, algorithm, policy, metric_name)
                values.append(value)
                errors.append(error if policy == "distance" else 0.0)
            axis.barh(
                y + offset,
                values,
                height=0.21,
                xerr=errors,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                capsize=2,
                label=label,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlim(0, 1.02)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Mean across five model seeds")

    axes[0].set_yticks(y, [label for _, _, label in rows])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    figure.suptitle(
        "Validation-locked autoregressive stopping",
        x=0.08,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.945,
        "RMS latent-distance policy; error bars show sample SD across model seeds",
        color="#475569",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, -0.02),
    )
    figure.tight_layout(rect=(0.06, 0.08, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
