"""HEDGE paper figure package: the 24-30 figure evidence set.

Reads campaign output directories (metrics.csv, optionally events.jsonl /
node_snapshots.jsonl) and produces publication-style PNG figures. Column
names follow the M1-M17 metric schema emitted by
HEDGESimulationEngine._compute_final_metrics.

Usage (library, not CLI): call the family functions directly with a loaded
list of row dicts (csv.DictReader output) and an output path.
"""

from __future__ import annotations

import csv
import json
import sys as _sys
from pathlib import Path
from typing import Any

_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in _sys.path:
    _sys.path.insert(0, str(_src_dir))

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from metrics import bootstrap_ci
from visualization.style import (
    ALGO_ORDER,
    FIGSIZE_DOUBLE,
    FIGSIZE_SINGLE,
    GRID,
    INK,
    RED,
    SLATE,
    TEAL,
    algo_color,
    algo_label,
    apply_style,
    savefig,
)

apply_style()


def load_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    """Load metrics.csv as a list of row dicts (all values kept as strings)."""
    if not metrics_csv.exists():
        logger.warning(f"metrics.csv not found: {metrics_csv}")
        return []
    with metrics_csv.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, Any], col: str) -> float:
    try:
        return float(row[col])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def _group_values(
    rows: list[dict[str, Any]], metric_col: str, group_col: str
) -> dict[str, list[float]]:
    """Group per-seed metric values by group_col, dropping NaNs."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        key = r.get(group_col)
        if key is None:
            continue
        val = _f(r, metric_col)
        if np.isnan(val):
            continue
        groups.setdefault(key, []).append(val)
    return groups


def _ordered_keys(groups: dict[str, list[float]], preferred_order: list[str] | None) -> list[str]:
    if preferred_order:
        ordered = [k for k in preferred_order if k in groups]
        ordered += [k for k in groups if k not in ordered]
        return ordered
    return sorted(groups.keys())


# ---------------------------------------------------------------------------
# Family (a): main comparison bar+CI across algorithms
# ---------------------------------------------------------------------------


def plot_bar_with_ci(
    rows: list[dict[str, Any]],
    metric_col: str,
    group_col: str,
    output_path: Path,
    ylabel: str,
    title: str,
    color_fn: Any = None,
    label_fn: Any = None,
    preferred_order: list[str] | None = None,
    log_scale: bool = False,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Grouped bar chart with bootstrap 95% CI whiskers, one bar per group.

    Args:
        rows: Per-seed metric rows (e.g. from a 30-seed algorithm sweep).
        metric_col: Column to plot (e.g. "M1_p95_latency_s").
        group_col: Column defining bars (e.g. "algorithm").
        output_path: Destination PNG path.
        ylabel: Y-axis label.
        title: Figure title.
        color_fn: Optional fn(group_key) -> hex color; defaults to algo_color.
        label_fn: Optional fn(group_key) -> display label; defaults to algo_label.
        preferred_order: Fixed x-axis order; defaults to ALGO_ORDER-then-sorted.
        log_scale: Use a log y-axis (for metrics spanning orders of magnitude).
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    color_fn = color_fn or algo_color
    label_fn = label_fn or algo_label
    groups = _group_values(rows, metric_col, group_col)
    keys = _ordered_keys(groups, preferred_order or ALGO_ORDER)

    means = [float(np.mean(groups[k])) for k in keys]
    cis = [bootstrap_ci(groups[k], n_boot=2000) if len(groups[k]) > 1 else (means[i], means[i])
           for i, k in enumerate(keys)]
    lo = [means[i] - cis[i][0] for i in range(len(keys))]
    hi = [cis[i][1] - means[i] for i in range(len(keys))]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(keys))
    colors = [color_fn(k) for k in keys]
    ax.bar(x, means, yerr=[lo, hi], color=colors, width=0.65, capsize=3,
           error_kw={"linewidth": 1.0, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels([label_fn(k) for k in keys], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    if log_scale:
        ax.set_yscale("log")
    savefig(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Family (b)/(c): degradation curves (metric vs x, one line per series)
# ---------------------------------------------------------------------------


def plot_curve_by_series(
    rows: list[dict[str, Any]],
    metric_col: str,
    x_col: str,
    series_col: str,
    output_path: Path,
    ylabel: str,
    xlabel: str,
    title: str,
    color_fn: Any = None,
    label_fn: Any = None,
    series_order: list[str] | None = None,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Line plot of mean metric vs a numeric x-axis, one line per series.

    Args:
        rows: Per-seed metric rows spanning multiple x_col values and series.
        metric_col: Y-axis metric column.
        x_col: X-axis numeric column (e.g. "arrival_rate", "K_max").
        series_col: Column defining separate lines (e.g. "algorithm").
        output_path: Destination PNG path.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        title: Figure title.
        color_fn: Optional fn(series_key) -> hex color; defaults to algo_color.
        label_fn: Optional fn(series_key) -> display label; defaults to algo_label.
        series_order: Fixed legend/plot order.
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    color_fn = color_fn or algo_color
    label_fn = label_fn or algo_label

    series_keys = sorted({r.get(series_col) for r in rows if r.get(series_col) is not None})
    if series_order:
        series_keys = [k for k in series_order if k in series_keys] + [
            k for k in series_keys if k not in series_order
        ]

    fig, ax = plt.subplots(figsize=figsize)
    for skey in series_keys:
        srows = [r for r in rows if r.get(series_col) == skey]
        x_vals = sorted({_f(r, x_col) for r in srows if not np.isnan(_f(r, x_col))})
        means = []
        los = []
        his = []
        for xv in x_vals:
            vals = [_f(r, metric_col) for r in srows if _f(r, x_col) == xv]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                means.append(np.nan)
                los.append(0.0)
                his.append(0.0)
                continue
            m = float(np.mean(vals))
            means.append(m)
            if len(vals) > 1:
                lo_ci, hi_ci = bootstrap_ci(vals, n_boot=2000)
                los.append(m - lo_ci)
                his.append(hi_ci - m)
            else:
                los.append(0.0)
                his.append(0.0)
        ax.errorbar(
            x_vals, means, yerr=[los, his], label=label_fn(skey), color=color_fn(skey),
            marker="o", markersize=3, linewidth=1.4, capsize=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="best")
    savefig(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Family (e): pricing integrity (relative-change bar, for small-magnitude
# effects like SPA shading where an absolute 0-based bar would visually hide
# a real, statistically significant difference)
# ---------------------------------------------------------------------------


def plot_relative_change_bar(
    rows: list[dict[str, Any]],
    metric_col: str,
    group_col: str,
    baseline_key: str,
    output_path: Path,
    ylabel: str,
    title: str,
    color: str = TEAL,
    preferred_order: list[str] | None = None,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Bar chart of PERCENT CHANGE relative to a baseline group, for metrics
    whose absolute scale would hide a small-but-real effect on a 0-based axis
    (e.g. SPA shading's price reduction within a single controlled scenario).

    Args:
        rows: Per-seed metric rows.
        metric_col: Column to compare (e.g. "M2_mean_cost_usd").
        group_col: Column defining bars (e.g. "lambda_max").
        baseline_key: Group value treated as the 0% reference.
        output_path: Destination PNG path.
        ylabel: Y-axis label (should say "% change from <baseline>").
        title: Figure title.
        color: Bar color.
        preferred_order: Fixed x-axis order.
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    groups = _group_values(rows, metric_col, group_col)
    keys = _ordered_keys(groups, preferred_order)
    baseline_mean = float(np.mean(groups[baseline_key])) if baseline_key in groups else float("nan")

    pct = [
        100.0 * (float(np.mean(groups[k])) - baseline_mean) / baseline_mean
        if baseline_mean else 0.0
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(keys, pct, color=color, width=0.6)
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    savefig(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Family (e) continued: floor-clearance scatter (Bertrand floor / IR check)
# ---------------------------------------------------------------------------


def plot_floor_clearance_scatter(
    events: list[dict[str, Any]],
    output_path: Path,
    title: str,
    edge_only: bool = True,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Scatter of price_paid vs C1_total (Layer-1 cost), with the Bertrand
    floor (price == cost) reference line. Every point must lie ON OR ABOVE
    the line (Invariants I1/I2) -- this figure IS the visual invariant check.

    Args:
        events: Event dicts with price_paid, C1_total, is_resale, executor_id.
        output_path: Destination PNG path.
        title: Figure title.
        edge_only: If True, excludes cloud completions (cloud has no C1 floor
            constraint in the same sense; the interesting claim is about edge
            sellers never taking a loss).
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    pts = [
        e for e in events
        if e.get("event_type") == "completed"
        and float(e.get("C1_total", 0.0)) > 0
        and (not edge_only or e.get("executor_id") != "cloud")
    ]
    c1 = np.array([float(e["C1_total"]) for e in pts])
    paid = np.array([float(e["price_paid"]) for e in pts])

    fig, ax = plt.subplots(figsize=figsize)
    if len(c1) > 0:
        lo, hi = float(c1.min()), float(c1.max())
        ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1.0, linestyle="--",
                label="Bertrand floor (price = cost)")
        ax.scatter(c1, paid, s=6, color=TEAL, alpha=0.35, linewidths=0,
                   label="Completed tasks")
        below_floor = int(np.sum(paid < c1 - 1e-12))
        if below_floor:
            ax.set_title(f"{title}  [{below_floor} FLOOR VIOLATIONS]", fontsize=8, color=RED)
        else:
            ax.set_title(title, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("Layer-1 cost C1 (USD)")
    ax.set_ylabel("Price paid (USD)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=6, loc="best")
    savefig(fig, output_path)
    return output_path


def plot_markup_distribution(
    node_states: list[dict[str, Any]],
    output_path: Path,
    title: str,
    eta: float = 1.0,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Histogram of the Layer-2 congestion markup m_i = 1 + eta*l_hat/f_max,
    with the theoretical [1, 1+eta] bound (Invariant I6) marked.

    Args:
        node_states: Dicts with l_hat and f_max (e.g. node_snapshots records).
        output_path: Destination PNG path.
        title: Figure title.
        eta: Markup gain constant (default 1.0, from pricing.eta).
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    markups = [
        1.0 + eta * float(s["l_hat"]) / max(float(s["f_max"]), 1.0) for s in node_states
    ]
    fig, ax = plt.subplots(figsize=figsize)
    if markups:
        ax.hist(markups, bins=40, color=TEAL, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color=INK, linewidth=1.0, linestyle="--", label="m_i = 1 (floor)")
    ax.axvline(1.0 + eta, color=RED, linewidth=1.0, linestyle="--", label=f"m_i = 1+eta ({1.0+eta:.1f})")
    ax.set_xlabel("Markup m_i")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="best")
    savefig(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Family (g): ablation forest plot (delta vs full HEDGE, with CI)
# ---------------------------------------------------------------------------


def plot_ablation_forest(
    ablation_deltas: dict[str, tuple[float, float, float]],
    output_path: Path,
    xlabel: str,
    title: str,
    figsize: tuple[float, float] = FIGSIZE_DOUBLE,
) -> Path:
    """Horizontal forest plot of (mean_delta, ci_lo, ci_hi) per ablation.

    Args:
        ablation_deltas: {ablation_name: (mean_delta, ci_lo, ci_hi)}, where
            delta = ablation_value - full_hedge_value on the target metric.
        output_path: Destination PNG path.
        xlabel: X-axis label (name the metric and units of the delta).
        title: Figure title.
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    names = list(ablation_deltas.keys())
    means = [ablation_deltas[n][0] for n in names]
    los = [means[i] - ablation_deltas[n][1] for i, n in enumerate(names)]
    his = [ablation_deltas[n][2] - means[i] for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(names))
    ax.errorbar(means, y, xerr=[los, his], fmt="o", color=TEAL, ecolor=SLATE,
                capsize=3, markersize=5)
    ax.axvline(0.0, color=INK, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8)
    ax.invert_yaxis()
    savefig(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Family (j): fairness / per-node utilisation distribution
# ---------------------------------------------------------------------------


def plot_utilisation_violin(
    snapshots_by_algo: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Violin plot of per-node utilisation fraction (l_current/f_max), one
    violin per algorithm, from node_snapshots.jsonl-style records.

    Args:
        snapshots_by_algo: {algorithm: [snapshot dicts with l_current, f_max]}.
        output_path: Destination PNG path.
        title: Figure title.
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    keys = [k for k in ALGO_ORDER if k in snapshots_by_algo] + [
        k for k in snapshots_by_algo if k not in ALGO_ORDER
    ]
    data = []
    for k in keys:
        fracs = [
            float(s["l_current"]) / max(float(s["f_max"]), 1.0) for s in snapshots_by_algo[k]
        ]
        data.append(fracs if fracs else [0.0])

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(algo_color(keys[i]))
        body.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(keys) + 1))
    ax.set_xticklabels([algo_label(k) for k in keys], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Per-node utilisation fraction")
    ax.set_title(title, fontsize=8)
    savefig(fig, output_path)
    return output_path


def load_snapshots(snapshots_path: Path) -> list[dict[str, Any]]:
    """Load a node_snapshots.jsonl file as a list of dicts."""
    if not snapshots_path.exists():
        logger.warning(f"node_snapshots.jsonl not found: {snapshots_path}")
        return []
    with snapshots_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plot_burst_tracking(
    snapshots: list[dict[str, Any]],
    output_path: Path,
    burst_start: float,
    burst_duration: float,
    title: str,
    node_id: str | None = None,
    figsize: tuple[float, float] = FIGSIZE_DOUBLE,
) -> Path:
    """Kalman tracking trace: true load (l_current) vs forecast (l_hat) over
    time, centred on a Pareto burst window.

    Args:
        snapshots: Per-(node, tick) dicts with time, node_id, l_current, l_hat.
        output_path: Destination PNG path.
        burst_start: Simulation time the burst begins (seconds).
        burst_duration: Burst duration (seconds).
        title: Figure title.
        node_id: Specific node to trace; if None, averages l_current/l_hat
            across all nodes present at each timestamp.
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    if node_id is not None:
        rows = [s for s in snapshots if s.get("node_id") == node_id]
        rows.sort(key=lambda s: s["time"])
        times = [s["time"] for s in rows]
        actual = [s["l_current"] for s in rows]
        forecast = [s["l_hat"] for s in rows]
    else:
        by_time: dict[float, list[dict[str, Any]]] = {}
        for s in snapshots:
            by_time.setdefault(s["time"], []).append(s)
        times = sorted(by_time.keys())
        actual = [float(np.mean([s["l_current"] for s in by_time[t]])) for t in times]
        forecast = [float(np.mean([s["l_hat"] for s in by_time[t]])) for t in times]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, actual, color=INK, linewidth=1.0, label="True load (l_observed)")
    ax.plot(times, forecast, color=TEAL, linewidth=1.4, label="Kalman forecast (l_hat)")
    ax.axvspan(burst_start, burst_start + burst_duration, color=RED, alpha=0.12,
               label="Burst window")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Load (cycles/s)")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="best")
    # Zoom to the burst neighbourhood so the tracking dynamic is visible.
    ax.set_xlim(burst_start - 5.0, burst_start + burst_duration + 15.0)
    savefig(fig, output_path)
    return output_path


def plot_eua_topology_map(
    eua_csv_path: Path,
    output_path: Path,
    n_nodes: int,
    n_hotspots: int,
    seed: int,
    title: str,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
) -> Path:
    """Scatter map of the REAL EUA Melbourne base-station subset used by a run.

    Reads real GPS coordinates directly from the bundled CSV (not from node
    objects, which discard coordinates after building the delay matrix) and
    reproduces the same seeded subsample topology.py uses, so the plotted
    subset matches what the simulation actually ran on.

    Args:
        eua_csv_path: Path to data/eua_melbourne.csv.
        output_path: Destination PNG path.
        n_nodes: Number of stations subsampled (topology.N_nodes).
        n_hotspots: Number of hotspot-tier nodes (first n_hotspots of the
            subsample, matching topology.py's tier assignment convention).
        seed: Random seed (must match the simulation run for the subsample to
            correspond exactly).
        title: Figure title.
        figsize: Figure size in inches.

    Returns:
        Path to the saved figure.
    """
    import pandas as pd

    df = pd.read_csv(eua_csv_path)
    lat_col = next(c for c in df.columns if c.lower() in ("latitude", "lat"))
    lon_col = next(c for c in df.columns if c.lower() in ("longitude", "lon", "lng"))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_nodes, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    hotspot = sub.iloc[:n_hotspots]
    quiet_end = min(n_hotspots + 20, len(sub))
    quiet = sub.iloc[n_hotspots:quiet_end]
    normal = sub.iloc[quiet_end:]

    ax.scatter(normal[lon_col], normal[lat_col], s=18, color=SLATE, label="Normal", zorder=2)
    ax.scatter(quiet[lon_col], quiet[lat_col], s=18, color="#56B4E9", label="Quiet", zorder=2)
    ax.scatter(
        hotspot[lon_col], hotspot[lat_col], s=42, color=RED, label="Hotspot",
        edgecolor="white", linewidth=0.5, zorder=3,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    savefig(fig, output_path)
    return output_path


