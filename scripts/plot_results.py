#!/usr/bin/env python3
"""Plot saved experiment results.

This script intentionally lives outside the package: experiments write results
first, then plotting reads those result directories offline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def _load_series(run_dir: Path, metric: str, stat: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "accuracies":
        path = run_dir / "accuracies.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        accuracies = np.load(path)
        if accuracies.size == 0:
            raise ValueError(f"{path} is empty")
        if stat == "mean":
            values = accuracies.mean(axis=1)
        elif stat == "worst":
            values = accuracies.min(axis=1)
        else:
            raise ValueError(f"unsupported stat for accuracies: {stat}")
        return np.arange(len(values)), values

    metric_path = run_dir / metric
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    steps, values = _read_eval(metric_path)
    if values.size == 0:
        raise ValueError(f"{metric_path} is empty")
    return steps, values


def _plot_series(ax, steps: np.ndarray, values: np.ndarray, label: str) -> None:
    ax.plot(steps, values, marker="o", linewidth=1.8, label=label)


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
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xlabel = "Evaluation index" if metric == "accuracies" else "Step"

    if aggregate:
        series = [_load_series(run_dir, metric, stat) for run_dir in run_dirs]
        steps, mean, std = _aggregate_series(series)
        label = labels[0] if labels else f"{metric} {stat}"
        _plot_series(ax, steps, mean, label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    else:
        if labels and len(labels) != len(run_dirs):
            raise SystemExit("--label must be passed once per run directory")
        for index, run_dir in enumerate(run_dirs):
            steps, values = _load_series(run_dir, metric, stat)
            label = labels[index] if labels else run_dir.name
            _plot_series(ax, steps, values, label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.set_title(title or ("Aggregate" if aggregate else "Result comparison"))
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved banditdl result directory.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Directories containing eval/eval_worst/accuracies.npy",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("plot.png"), help="Output image path")
    parser.add_argument("--metric", choices=["accuracies", "eval", "eval_worst"], default="accuracies")
    parser.add_argument("--stat", choices=["mean", "worst"], default="mean", help="Statistic used with accuracies.npy")
    parser.add_argument("--aggregate", action="store_true", help="Plot mean +/- std across the given run directories")
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
    )


if __name__ == "__main__":
    main()
