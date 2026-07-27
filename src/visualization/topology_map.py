"""Render the HEDGE network topology (edge nodes + cloud fallback) for a config.

Builds the topology directly via hedge.core.topology.create_topology (no
simulation run required), then draws a presentation-quality map: node
position from real EUA-Melbourne GPS coordinates (or a circular layout for
synthetic topologies), node size scaled by compute capacity (f_max), node
color scaled by grid carbon intensity (beta), hotspot nodes ringed
separately, local mesh links drawn from each node's Phase-0 radio-coverage
peer set (C_u), and the logical cloud fallback node shown with its sampled
RTT.

Usage:
    python src/visualization/topology_map.py --config configs/default.yaml \\
        --output outputs/topology_map.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_dir = _Path(__file__).parent.parent
if str(_src_dir) not in _sys.path:
    _sys.path.insert(0, str(_src_dir))

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments.runner import load_config  # noqa: E402
from hedge.core.topology import create_topology  # noqa: E402

COL_BG = "#F8FAFC"
COL_TEXT = "#0F172A"
COL_MESH = "#94A3B8"
COL_CLOUD_FACE = "#EFF6FF"
COL_CLOUD_EDGE = "#2563EB"
COL_HOTSPOT_RING = "#DC2626"
CAPACITY_MIN_PT = 40.0
CAPACITY_MAX_PT = 260.0


def _node_positions(nodes: list[Any], source: str) -> dict[int, tuple[float, float]]:
    """Return {node.unique_id: (x, y)}.

    Real (longitude, latitude) for the EUA-Melbourne topology; a circular
    layout for synthetic topologies, which have no geography.
    """
    if source == "eua_melbourne" and all(n.coordinates is not None for n in nodes):
        return {n.unique_id: (n.coordinates[1], n.coordinates[0]) for n in nodes}
    n = len(nodes)
    return {
        node.unique_id: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }


def plot_topology(config: dict[str, Any], output_path: Path, dpi: int = 220) -> Path:
    """Build the topology from config and render it to output_path.

    Args:
        config: Resolved experiment config (as returned by load_config).
        output_path: Destination image path.
        dpi: Output resolution.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(int(config.get("simulation", {}).get("random_seed", 42)))
    nodes, _tau, cloud = create_topology(config, rng)
    source = config.get("topology", {}).get("source", "synthetic")
    pos = _node_positions(nodes, source)

    xs = [pos[n.unique_id][0] for n in nodes]
    ys = [pos[n.unique_id][1] for n in nodes]
    f_max_vals = np.array([n.f_max for n in nodes])
    beta_vals = np.array([n.beta for n in nodes])
    f_lo, f_hi = float(f_max_vals.min()), float(f_max_vals.max())
    sizes = CAPACITY_MIN_PT + (f_max_vals - f_lo) / max(f_hi - f_lo, 1.0) * (
        CAPACITY_MAX_PT - CAPACITY_MIN_PT
    )

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)

    # Local mesh links: each node's radio-coverage peer set (undirected, drawn once)
    drawn: set[tuple[int, int]] = set()
    for node in nodes:
        for peer in node.coverage_peers:
            if peer.unique_id == node.unique_id:
                continue
            key = (min(node.unique_id, peer.unique_id), max(node.unique_id, peer.unique_id))
            if key in drawn:
                continue
            drawn.add(key)
            x0, y0 = pos[node.unique_id]
            x1, y1 = pos[peer.unique_id]
            ax.plot([x0, x1], [y0, y1], color=COL_MESH, linewidth=0.8, alpha=0.5, zorder=1)

    # Cloud fallback: faint fan-out to every node (low alpha so it reads as a
    # halo, not a hairball), plus a labeled marker.
    cloud_x = (min(xs) + max(xs)) / 2.0
    cloud_y = max(ys) + 0.15 * (max(ys) - min(ys) + 1e-9)
    for x1, y1 in zip(xs, ys):
        ax.plot(
            [cloud_x, x1], [cloud_y, y1], color=COL_CLOUD_EDGE, linewidth=0.5, alpha=0.06, zorder=0
        )

    ax.scatter(
        xs,
        ys,
        s=sizes,
        c=beta_vals,
        cmap="RdYlGn_r",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    sc_for_cbar = ax.collections[-1]

    for node, x, y, size in zip(nodes, xs, ys, sizes):
        if getattr(node, "arrival_rate_multiplier", 1.0) > 1.0:
            ax.scatter(
                [x],
                [y],
                s=[size * 1.8],
                facecolors="none",
                edgecolors=COL_HOTSPOT_RING,
                linewidths=1.6,
                zorder=4,
            )

    ax.scatter(
        [cloud_x],
        [cloud_y],
        s=900,
        marker="s",
        facecolors=COL_CLOUD_FACE,
        edgecolors=COL_CLOUD_EDGE,
        linewidths=2,
        zorder=5,
    )
    ax.text(
        cloud_x,
        cloud_y,
        f"CLOUD\nRTT ~ {cloud.tau_c * 1000:.0f} ms",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COL_TEXT,
        zorder=6,
    )

    cbar = fig.colorbar(sc_for_cbar, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Grid carbon intensity beta (gCO2/kWh)", color=COL_TEXT)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="Edge node (size = compute capacity f_max)",
            markerfacecolor="#94A3B8",
            markeredgecolor="white",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="Hotspot node (elevated arrival rate)",
            markerfacecolor="none",
            markeredgecolor=COL_HOTSPOT_RING,
            markersize=12,
            markeredgewidth=1.6,
        ),
        Line2D([0], [0], color=COL_MESH, lw=1.2, label="Local mesh link (radio-coverage peer)"),
        Line2D(
            [0],
            [0],
            color=COL_CLOUD_EDGE,
            lw=1.2,
            alpha=0.4,
            label="Cloud WAN fallback (always reachable)",
        ),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        facecolor=COL_BG,
        edgecolor=COL_MESH,
        fontsize=9,
    )
    plt.setp(legend.get_texts(), color=COL_TEXT)

    n_mesh_edges = len(drawn)
    mean_degree = (2 * n_mesh_edges / len(nodes)) if nodes else 0.0
    ax.set_title(
        f"HEDGE topology -- {len(nodes)} edge nodes ({source}), 1 cloud fallback",
        color=COL_TEXT,
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    plt.figtext(
        0.5,
        0.01,
        f"{n_mesh_edges} local mesh links (mean coverage degree {mean_degree:.1f}) | "
        f"f_max in [{f_lo / 1e9:.1f}, {f_hi / 1e9:.1f}] GHz | "
        f"beta in [{beta_vals.min():.0f}, {beta_vals.max():.0f}] gCO2/kWh",
        ha="center",
        fontsize=9,
        color=COL_TEXT,
    )
    ax.set_xlabel("Longitude" if source == "eua_melbourne" else "x (layout)", color=COL_TEXT)
    ax.set_ylabel("Latitude" if source == "eua_melbourne" else "y (layout)", color=COL_TEXT)
    ax.tick_params(colors=COL_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COL_MESH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved topology map to {output_path}")
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render the HEDGE network topology")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/topology_map.png"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    config = load_config(args.config)
    plot_topology(config, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
