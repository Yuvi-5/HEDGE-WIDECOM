"""Cost figure generators (M2 mean cost, M7 carbon, M15 energy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def _load_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    """Load metrics.csv as list of dicts."""
    import csv

    with metrics_csv.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_cost_vs_kmax(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_col: str = "M2_mean_cost_usd",
    sweep_col: str = "K_max",
    dpi: int = 300,
) -> Path:
    """Bar chart: mean cost vs K_max sweep.

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG.
        metric_col: Cost metric column.
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

    keys = sorted(by_key.keys(), key=lambda x: (str(x).isdigit() is False, x))
    means = [float(np.mean(by_key[k])) for k in keys]
    stds = [float(np.std(by_key[k])) for k in keys]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys], fontsize=8)
    ax.set_xlabel(sweep_col, fontsize=8)
    ax.set_ylabel("Mean cost (USD)", fontsize=8)
    ax.set_title(f"Cost vs {sweep_col}", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved cost plot: {output_path}")
    return output_path


def plot_carbon_vs_kmax(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_col: str = "M7_carbon_g_per_task",
    sweep_col: str = "K_max",
    dpi: int = 300,
) -> Path:
    """Bar chart: carbon-related metric vs K_max sweep.

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG.
        metric_col: Carbon metric column.
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

    keys = sorted(by_key.keys(), key=lambda x: (str(x).isdigit() is False, x))
    means = [float(np.mean(by_key[k])) for k in keys]
    stds = [float(np.std(by_key[k])) for k in keys]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=4, color="seagreen", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in keys], fontsize=8)
    ax.set_xlabel(sweep_col, fontsize=8)
    ax.set_ylabel(metric_col, fontsize=8)
    ax.set_title(f"Carbon metric vs {sweep_col}", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved carbon plot: {output_path}")
    return output_path
