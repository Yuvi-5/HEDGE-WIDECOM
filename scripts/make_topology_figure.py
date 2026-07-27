"""System-model topology figure: real EUA-Melbourne deployment colored by
compute capacity (f_max), for the WIDECOM paper's System Model section.

A variant of src/visualization/topology_map.py with three changes:
color/size both encode f_max instead of carbon (beta), viridis instead
of RdYlGn_r, no carbon colorbar anywhere.

Uses the exact same config (hedge_c_base.yaml, N=125, coverage_radius_km
0.25) and the exact same RNG path as the main campaign's HEDGE_C seed 0
(engine.py: self.rng = np.random.default_rng(seed); create_topology(config,
self.rng) is the FIRST rng consumer) -- so this is byte-identical to the
topology that seed actually ran against, not a fresh independent draw.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hedge.core.topology import create_topology  # noqa: E402
from run_hedge_c_comparison import build_base_config  # noqa: E402

OUT_PATH = ROOT / "outputs" / "figures" / "system_topology.pdf"
SEED = 0

COL_BG = "#F8FAFC"
COL_TEXT = "#0F172A"
COL_MESH = "#94A3B8"
COL_CLOUD_FACE = "#EFF6FF"
COL_CLOUD_EDGE = "#2563EB"
COL_HOTSPOT_RING = "#DC2626"
CAPACITY_MIN_PT = 30.0
CAPACITY_MAX_PT = 190.0


def node_positions(nodes, source: str) -> dict[int, tuple[float, float]]:
    if source == "eua_melbourne" and all(n.coordinates is not None for n in nodes):
        return {n.unique_id: (n.coordinates[1], n.coordinates[0]) for n in nodes}
    import math
    n = len(nodes)
    return {
        node.unique_id: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }


def main() -> None:
    cfg = build_base_config()
    n_expected = int(cfg.get("topology", {}).get("N_nodes", 125))
    rng = np.random.default_rng(SEED)
    nodes, _tau, cloud = create_topology(cfg, rng)

    if len(nodes) != n_expected:
        logger.error(f"ABORTING: expected {n_expected} nodes, got {len(nodes)}.")
        sys.exit(1)

    source = cfg.get("topology", {}).get("source", "synthetic")
    pos = node_positions(nodes, source)
    xs = [pos[n.unique_id][0] for n in nodes]
    ys = [pos[n.unique_id][1] for n in nodes]

    f_max_vals = np.array([n.f_max for n in nodes])
    f_lo, f_hi = float(f_max_vals.min()), float(f_max_vals.max())
    sizes = CAPACITY_MIN_PT + (f_max_vals - f_lo) / max(f_hi - f_lo, 1.0) * (
        CAPACITY_MAX_PT - CAPACITY_MIN_PT
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)

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
            ax.plot([x0, x1], [y0, y1], color=COL_MESH, linewidth=0.6, alpha=0.45, zorder=1)

    cloud_x = (min(xs) + max(xs)) / 2.0
    cloud_y = max(ys) + 0.15 * (max(ys) - min(ys) + 1e-9)
    for x1, y1 in zip(xs, ys):
        ax.plot([cloud_x, x1], [cloud_y, y1], color=COL_CLOUD_EDGE, linewidth=0.4, alpha=0.05, zorder=0)

    sc = ax.scatter(
        xs, ys, s=sizes, c=f_max_vals / 1e9, cmap="viridis",
        edgecolors="white", linewidths=0.5, zorder=3,
    )

    n_hotspot = 0
    for node, x, y, size in zip(nodes, xs, ys, sizes):
        if getattr(node, "arrival_rate_multiplier", 1.0) > 1.0:
            n_hotspot += 1
            # Soft glow fill behind the node (larger, translucent, low zorder)
            # plus a bold double ring on top -- a thin ring alone got lost
            # against the busy mesh; this reads clearly even at column width.
            ax.scatter([x], [y], s=[size * 5.5], facecolors=COL_HOTSPOT_RING,
                       edgecolors="none", alpha=0.16, zorder=3.4)
            ax.scatter([x], [y], s=[size * 2.6], facecolors="none",
                       edgecolors=COL_HOTSPOT_RING, linewidths=2.4, zorder=4)
            ax.scatter([x], [y], s=[size * 3.6], facecolors="none",
                       edgecolors=COL_HOTSPOT_RING, linewidths=1.0, alpha=0.55, zorder=4)
            ax.annotate(
                f"HOTSPOT\n({int(getattr(node, 'arrival_rate_multiplier', 1.0))}× rate)",
                xy=(x, y), xytext=(14, -16), textcoords="offset points",
                fontsize=6.5, fontweight="bold", color=COL_HOTSPOT_RING,
                ha="left", va="top", zorder=7,
                arrowprops=dict(arrowstyle="-", color=COL_HOTSPOT_RING, linewidth=0.8, alpha=0.8),
            )

    # A text bbox sizes itself to the text content -- far more reliable than
    # guessing a scatter marker's "s=" (points^2 area) large enough to hold
    # two lines of bold text, which is what clipped the label before.
    ax.text(
        cloud_x, cloud_y, f"CLOUD\nRTT~{cloud.tau_c * 1000:.0f}ms",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color=COL_TEXT, zorder=6,
        bbox=dict(boxstyle="square,pad=0.55", facecolor=COL_CLOUD_FACE,
                  edgecolor=COL_CLOUD_EDGE, linewidth=2.2),
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label(r"Compute capacity $f_{\max}$ (GHz)", color=COL_TEXT, fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors=COL_TEXT)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", label="Edge node (size & color = $f_{\\max}$)",
               markerfacecolor="#94A3B8", markeredgecolor="white", markersize=8),
        Line2D([0], [0], marker="o", color="none", label="Hotspot node (elevated arrival rate)",
               markerfacecolor="none", markeredgecolor=COL_HOTSPOT_RING, markersize=9, markeredgewidth=1.4),
        Line2D([0], [0], color=COL_MESH, lw=1.0, label="Local mesh link (radio-coverage peer)"),
        Line2D([0], [0], color=COL_CLOUD_EDGE, lw=1.0, alpha=0.4, label="Cloud WAN fallback"),
    ]
    legend = ax.legend(handles=legend_elements, loc="lower left", frameon=True,
                        facecolor=COL_BG, edgecolor=COL_MESH, fontsize=6.5)
    plt.setp(legend.get_texts(), color=COL_TEXT)

    n_mesh_edges = len(drawn)
    mean_degree = (2 * n_mesh_edges / len(nodes)) if nodes else 0.0
    ax.set_title(
        f"HEDGE deployment: {len(nodes)} heterogeneous edge nodes (EUA-Melbourne), 1 cloud fallback",
        color=COL_TEXT, fontsize=10, fontweight="bold", pad=10,
    )
    plt.figtext(
        0.5, -0.02,
        f"{n_mesh_edges} local mesh links (mean coverage degree {mean_degree:.1f}) | "
        f"$f_{{\\max}}$ in [{f_lo / 1e9:.1f}, {f_hi / 1e9:.1f}] GHz | {n_hotspot} hotspot node(s)",
        ha="center", fontsize=7, color=COL_TEXT,
    )
    ax.set_xlabel("Longitude" if source == "eua_melbourne" else "x (layout)", color=COL_TEXT, fontsize=8)
    ax.set_ylabel("Latitude" if source == "eua_melbourne" else "y (layout)", color=COL_TEXT, fontsize=8)
    ax.tick_params(colors=COL_TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(COL_MESH)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(
        f"[system_topology] nodes={len(nodes)} f_max_range_ghz=[{f_lo/1e9:.2f},{f_hi/1e9:.2f}] "
        f"mesh_links={n_mesh_edges} mean_coverage_degree={mean_degree:.2f} hotspots={n_hotspot} "
        f"seed={SEED} source={source} -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
