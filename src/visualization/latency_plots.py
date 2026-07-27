"""Latency figure generators (M1 P50/P95 latency plots).

Generates:
  - P95 latency vs arrival rate (S1/S2 results)
  - K_max sweep latency bar chart (S3 results)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def _load_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    """Load metrics.csv as list of dicts."""
    import csv

    with metrics_csv.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_latency_vs_kmax(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_col: str = "M3_p90_latency_s",
    sweep_col: str = "K_max",
    dpi: int = 300,
) -> Path:
    """Bar chart: latency metric vs K_max sweep values.

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG path.
        metric_col: Latency column to plot (default: M3_p90_latency_s).
        sweep_col: Sweep column for x-axis (default: K_max).
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if metric_col not in (rows[0] if rows else {}):
        # Fall back to any latency metric
        for candidate in ["M1_mean_latency_s", "M3_p90_latency_s", "M13_edge_latency_s"]:
            if rows and candidate in rows[0]:
                metric_col = candidate
                break

    by_kmax: dict[Any, list[float]] = {}
    for row in rows:
        key = row.get(sweep_col, "?")
        try:
            val = float(row.get(metric_col, 0.0) or 0.0)
        except (ValueError, TypeError):
            val = 0.0
        by_kmax.setdefault(key, []).append(val)

    kmax_vals = sorted(by_kmax.keys(), key=lambda x: (str(x).isdigit() is False, x))
    means = [float(np.mean(by_kmax[k])) for k in kmax_vals]
    stds = [float(np.std(by_kmax[k])) for k in kmax_vals]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(kmax_vals))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in kmax_vals], fontsize=8)
    ax.set_xlabel(sweep_col, fontsize=8)
    ax.set_ylabel(metric_col, fontsize=8)
    ax.set_title(f"{metric_col} vs {sweep_col}", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved latency plot: {output_path}")
    return output_path


def plot_latency_cdf(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_col: str = "M1_mean_latency_s",
    group_col: str = "K_max",
    dpi: int = 300,
) -> Path:
    """CDF of latency for each group value.

    Args:
        rows: Metric CSV rows.
        output_path: Destination PNG path.
        metric_col: Latency column.
        group_col: Group-by column (lines on the CDF).
        dpi: Figure DPI.

    Returns:
        Path to saved PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_group: dict[Any, list[float]] = {}
    for row in rows:
        key = row.get(group_col, "?")
        try:
            val = float(row.get(metric_col, 0.0) or 0.0)
        except (ValueError, TypeError):
            val = 0.0
        by_group.setdefault(key, []).append(val)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    linestyles = ["-", "--", "-.", ":"]
    for i, key in enumerate(sorted(by_group.keys(), key=str)):
        vals = sorted(by_group[key])
        n = len(vals)
        p = np.arange(1, n + 1) / n
        ax.plot(vals, p, linestyle=linestyles[i % len(linestyles)], label=str(key), linewidth=1.2)

    ax.set_xlabel(metric_col, fontsize=8)
    ax.set_ylabel("CDF", fontsize=8)
    ax.set_title(f"Latency CDF by {group_col}", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved latency CDF: {output_path}")
    return output_path
