#!/usr/bin/env python3
"""Plot saved experiment results.

This script intentionally lives outside the package: experiments write results
first, then plotting reads those result directories offline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np

ARRAY_METRICS = {
    "accuracies",
    "reward_algorithm",
    "reward_oracle",
    "regret",
    "normalized_regret",
    "neighbor_disagreement",
    "consensus_drift",
}
REGRET_METRICS = {"regret", "normalized_regret"}
REWARD_METRICS = {"reward_algorithm", "reward_oracle"}
DISTANCE_METRICS = {"neighbor_disagreement", "consensus_drift"}
MAX_METRICS = REGRET_METRICS | DISTANCE_METRICS
PER_NODE_METRICS = REGRET_METRICS | REWARD_METRICS | DISTANCE_METRICS

NODE_LINESTYLE = {"primary": "-", "median": "--", "max": ":"}


def _read_eval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            step, value, *_ = line.split()
            steps.append(float(step))
            values.append(float(value))
    return np.asarray(steps), np.asarray(values)


def _load_raw_array(run_dir: Path, metric: str) -> np.ndarray:
    path = run_dir / f"{metric}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    values_by_worker = np.load(path)
    if values_by_worker.size == 0:
        raise ValueError(f"{path} is empty")
    return values_by_worker


def _load_series(
    run_dir: Path, metric: str, stat: str
) -> tuple[np.ndarray, np.ndarray]:
    if metric in ARRAY_METRICS:
        values_by_worker = _load_raw_array(run_dir, metric)
        if stat == "mean":
            values = values_by_worker.mean(axis=1)
        elif stat == "worst":
            reducer = np.max if metric in MAX_METRICS else np.min
            values = reducer(values_by_worker, axis=1)
        else:
            raise ValueError(f"unsupported stat for {metric}: {stat}")
        return np.arange(len(values)), values

    metric_path = run_dir / metric
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    steps, values = _read_eval(metric_path)
    if values.size == 0:
        raise ValueError(f"{metric_path} is empty")
    return steps, values


def _node_curves(
    raw: np.ndarray, metric: str, stat: str
) -> list[tuple[str, str, np.ndarray]]:
    """Per-node aggregations as (kind, suffix, values), deduped.

    `kind` is one of {"primary", "median", "max"} and selects line style.
    """
    median_vals = np.median(raw, axis=1)
    max_vals = raw.max(axis=1)

    curves: list[tuple[str, str, np.ndarray]] = []
    if stat == "mean":
        curves.append(("primary", "mean across nodes", raw.mean(axis=1)))
        curves.append(("median", "median across nodes", median_vals))
        curves.append(("max", "max node", max_vals))
    elif stat == "worst":
        if metric in MAX_METRICS:
            # Worst is max across nodes, so primary already covers max.
            curves.append(("primary", "worst node (max)", max_vals))
            curves.append(("median", "median across nodes", median_vals))
        else:
            curves.append(("primary", "worst node (min reward)", raw.min(axis=1)))
            curves.append(("median", "median across nodes", median_vals))
            curves.append(("max", "max node", max_vals))
    else:
        raise ValueError(f"unsupported stat for {metric}: {stat}")
    return curves


def _ylabel(metric: str) -> str:
    if metric in {"accuracies", "eval", "eval_worst"}:
        return "Accuracy"
    if metric in REGRET_METRICS:
        return "Regret"
    if metric == "neighbor_disagreement":
        return "Neighbor disagreement"
    if metric == "consensus_drift":
        return "Consensus drift"
    return "Reward"


def _node_title_suffix(metric: str, stat: str) -> str:
    base = _ylabel(metric)
    parts: list[str] = []
    if stat == "mean":
        parts.append("mean")
    elif metric in MAX_METRICS:
        parts.append("worst node (max)")
    else:
        parts.append("worst node (min)")
    parts.append("median")
    if not (stat == "worst" and metric in MAX_METRICS):
        parts.append("max")
    return f"{base} per node — showing {', '.join(parts)} across nodes"


def _color_for(index: int) -> str:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle[index % len(cycle)]


def _plot_curve(
    ax,
    steps: np.ndarray,
    values: np.ndarray,
    label: str,
    *,
    color: str | None = None,
    linestyle: str = "-",
    marker: bool = True,
) -> None:
    ax.plot(
        steps,
        values,
        marker="o" if marker else None,
        linewidth=1.7,
        color=color,
        linestyle=linestyle,
        label=label,
    )


def _default_label(run_dir: Path, max_length: int) -> str:
    tokens = run_dir.name.split("-")
    keep_prefixes = (
        "sampling_",
        "degree_",
        "sampler_",
        "eps_",
        "init_",
        "seed_",
    )
    label_parts = [
        token
        for token in tokens
        if token.startswith(keep_prefixes) or token in {"cs+", "cs_he", "gts"}
    ]
    label = ", ".join(label_parts) if label_parts else run_dir.name
    return shorten(label, width=max_length, placeholder="...")


def _aggregate_series(
    series: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not series:
        raise ValueError("no series to aggregate")
    length = min(len(values) for _, values in series)
    if length == 0:
        raise ValueError("cannot aggregate empty series")
    steps = series[0][0][:length]
    stacked = np.stack([values[:length] for _, values in series])
    return steps, stacked.mean(axis=0), stacked.std(axis=0)


def plot_runs(
    run_dirs: Sequence[Path],
    output: Path,
    metric: str,
    stat: str,
    title: str | None,
    labels: Sequence[str] | None,
    aggregate: bool,
    legend: str,
    max_label_length: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xlabel = "Evaluation index" if metric == "accuracies" else "Step"
    is_per_node = metric in PER_NODE_METRICS

    if aggregate:
        if is_per_node:
            raws = [_load_raw_array(run_dir, metric) for run_dir in run_dirs]
            length = min(len(r) for r in raws)
            if length == 0:
                raise ValueError("cannot aggregate empty per-node series")
            steps = np.arange(length)
            template = _node_curves(raws[0][:length], metric, stat)
            base_label = labels[0] if labels else f"{metric} {stat}"
            base_label = shorten(base_label, width=max_label_length, placeholder="...")
            color = _color_for(0)
            for kind_idx, (kind, suffix, _) in enumerate(template):
                stacked = np.stack(
                    [
                        _node_curves(r[:length], metric, stat)[kind_idx][2]
                        for r in raws
                    ]
                )
                mean = stacked.mean(axis=0)
                std = stacked.std(axis=0)
                _plot_curve(
                    ax,
                    steps,
                    mean,
                    f"{base_label} ({suffix})",
                    color=color,
                    linestyle=NODE_LINESTYLE[kind],
                    marker=(kind == "primary"),
                )
                ax.fill_between(
                    steps, mean - std, mean + std, alpha=0.15, color=color
                )
        else:
            series = [_load_series(run_dir, metric, stat) for run_dir in run_dirs]
            steps, mean, std = _aggregate_series(series)
            label = labels[0] if labels else f"{metric} {stat}"
            label = shorten(label, width=max_label_length, placeholder="...")
            _plot_curve(ax, steps, mean, label)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    else:
        if labels and len(labels) != len(run_dirs):
            raise SystemExit("--label must be passed once per run directory")
        for index, run_dir in enumerate(run_dirs):
            base_label = (
                labels[index] if labels else _default_label(run_dir, max_label_length)
            )
            base_label = shorten(base_label, width=max_label_length, placeholder="...")

            if is_per_node:
                raw = _load_raw_array(run_dir, metric)
                steps = np.arange(len(raw))
                color = _color_for(index)
                for kind, suffix, values in _node_curves(raw, metric, stat):
                    _plot_curve(
                        ax,
                        steps,
                        values,
                        f"{base_label} ({suffix})",
                        color=color,
                        linestyle=NODE_LINESTYLE[kind],
                        marker=(kind == "primary"),
                    )
            else:
                steps, values = _load_series(run_dir, metric, stat)
                _plot_curve(ax, steps, values, base_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_ylabel(metric))
    if metric in {"accuracies", "eval", "eval_worst"}:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    if is_per_node:
        node_suffix = _node_title_suffix(metric, stat)
        if title:
            plot_title = f"{title}\n{node_suffix}"
        elif aggregate:
            plot_title = f"Aggregate\n{node_suffix}"
        else:
            plot_title = node_suffix
    elif title:
        plot_title = title
    elif aggregate:
        plot_title = "Aggregate"
    else:
        plot_title = "Result comparison"
    ax.set_title(plot_title)

    if legend == "outside":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncols=min(3, max(1, len(ax.lines))),
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.28)
    elif legend == "best":
        ax.legend()
        fig.tight_layout()
    elif legend == "none":
        fig.tight_layout()
    else:
        raise ValueError(f"Unknown legend placement: {legend}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved banditdl result directory."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Directories containing eval/eval_worst/accuracies.npy",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("plot.png"), help="Output image path"
    )
    parser.add_argument(
        "--metric",
        choices=[
            "accuracies",
            "eval",
            "eval_worst",
            "reward_algorithm",
            "reward_oracle",
            "regret",
            "normalized_regret",
            "neighbor_disagreement",
            "consensus_drift",
        ],
        default="accuracies",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "worst"],
        default="mean",
        help="Worker statistic. For regret metrics, worst means highest regret.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Plot mean +/- std across the given run directories",
    )
    parser.add_argument(
        "--legend", choices=["outside", "best", "none"], default="outside"
    )
    parser.add_argument("--max-label-length", type=int, default=48)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Line label. Repeat once per run, or once with --aggregate",
    )
    args = parser.parse_args()

    plot_runs(
        args.run_dirs,
        args.output,
        args.metric,
        args.stat,
        args.title,
        args.label,
        args.aggregate,
        args.legend,
        args.max_label_length,
    )


if __name__ == "__main__":
    main()
