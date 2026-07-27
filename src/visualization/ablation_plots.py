"""Ablation figure generators.

Generates K_max sweep bar charts, rejection-rate ablation bars, and the
(RMSE, latency) Pareto plane for A10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def plot_rejection_vs_kmax(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_col: str = "M6_rejection_rate",
    sweep_col: str = "K_max",
    dpi: int = 300,
) -> Path:
    """Bar chart: rejection rate vs K_max (A2 ablation, Figure 2).

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG.
        metric_col: Rejection rate column.
        sweep_col: Sweep parameter column.
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key: dict[Any, list[float]] = {}
    for row in rows:
        key = row.get(sweep_col, "?")
        try:
            val = float(row.get(metric_col, 0.0) or 0.0)
        except (ValueError, TypeError):
            val = 0.0
        by_key.setdefault(key, []).append(val)

    if not by_key:
        logger.warning(f"No data for rejection vs {sweep_col} plot")
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.set_title(f"Rejection rate vs {sweep_col} (no data)", fontsize=8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return output_path

    keys = sorted(by_key.keys(), key=lambda x: (not str(x).lstrip("-").isdigit(), x))
    means = [float(np.mean(by_key[k])) for k in keys]
    stds = [float(np.std(by_key[k])) for k in keys]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys], fontsize=8)
    ax.set_xlabel(sweep_col, fontsize=8)
    ax.set_ylabel("Rejection rate", fontsize=8)
    ax.set_title(f"Rejection rate vs {sweep_col}", fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved rejection ablation plot: {output_path}")
    return output_path


def plot_ablation_bars(
    rows: list[dict[str, Any]],
    output_path: Path,
    metrics: list[str] | None = None,
    group_col: str = "experiment",
    dpi: int = 300,
) -> Path:
    """Grouped bar chart comparing multiple metrics across ablation variants.

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG.
        metrics: Metric columns to include (default: rejection + completion).
        group_col: Column to group bars by.
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        # Pick the first two available metric columns
        candidates = [
            "M6_rejection_rate",
            "M10_completion_rate",
            "M1_mean_latency_s",
            "M2_mean_cost_usd",
        ]
        metrics = [c for c in candidates if rows and c in rows[0]][:2]
        if not metrics:
            metrics = ["M6_rejection_rate"]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_col, "?"))
        groups.setdefault(key, []).append(row)

    group_names = sorted(groups.keys())
    n_groups = len(group_names)
    n_metrics = len(metrics)
    x = np.arange(n_groups)
    width = 0.8 / max(n_metrics, 1)

    fig, ax = plt.subplots(figsize=(max(3.5, n_groups * 1.0), 2.8))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for j, metric in enumerate(metrics):
        means: list[float] = []
        for gname in group_names:
            vals = [float(r.get(metric, 0.0) or 0.0) for r in groups[gname]]
            means.append(float(np.mean(vals)) if vals else 0.0)
        offset = (j - n_metrics / 2.0 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9, label=metric, color=colors[j % 10], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Metric value", fontsize=8)
    ax.set_title("Ablation comparison", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved ablation bars: {output_path}")
    return output_path


def plot_acceptance_vs_kmax(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_col: str = "M9_edge_service_fraction",
    sweep_col: str = "K_max",
    dpi: int = 300,
) -> Path:
    """Line plot: quote acceptance rate vs K_max.

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG.
        metric_col: Acceptance/completion rate column.
        sweep_col: Sweep column for x-axis.
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key: dict[Any, list[float]] = {}
    for row in rows:
        key = row.get(sweep_col, "?")
        try:
            val = float(row.get(metric_col, 0.0) or 0.0)
        except (ValueError, TypeError):
            val = 0.0
        by_key.setdefault(key, []).append(val)

    keys = sorted(by_key.keys(), key=lambda x: (not str(x).lstrip("-").isdigit(), x))
    means = [float(np.mean(by_key[k])) for k in keys]
    stds = [float(np.std(by_key[k])) for k in keys]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(keys))
    ax.plot(x, means, marker="o", linewidth=1.5, color="steelblue")
    ax.fill_between(
        x,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        color="steelblue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys], fontsize=8)
    ax.set_xlabel(sweep_col, fontsize=8)
    ax.set_ylabel("Acceptance rate", fontsize=8)
    ax.set_title(f"Acceptance rate vs {sweep_col}", fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved acceptance plot: {output_path}")
    return output_path
