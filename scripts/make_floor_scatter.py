"""Figure 5 data + plot: pricing-floor integrity scatter.

Re-runs ONE HEDGE_C seed (seed 0, full 3600s duration, identical config to
the main campaign) purely to dump per-task (price_paid, C1_total) pairs --
these are already-computed quantities read straight off
engine._mc.events, no pricing logic touched or re-derived. This is an
additive read-and-dump, no engine code is modified.

The "0 violations" annotation uses the REAL, already-validated full-campaign
count (74,250,640 tasks across all 30 HEDGE_C seeds, 0 floor violations)
rather than just this one seed's ~2.47M tasks, since that full-campaign
number is what the paper actually claims.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

from hedge.simulation.engine import HEDGESimulationEngine  # noqa: E402
from run_hedge_c_comparison import ARMS, arm_config, build_base_config  # noqa: E402

DUMP_CSV = ROOT / "outputs" / "floor_scatter_seed0.csv"
FIG_OUT = ROOT / "outputs" / "figures" / "fig5_floor_scatter.pdf"
SEED = 0
SAMPLE_N = 2000

# Full-campaign totals, already validated against the main campaign CSV.
FULL_CAMPAIGN_TASKS = 74_250_640
FULL_CAMPAIGN_VIOLATIONS = 0


def run_and_dump() -> int:
    """Re-run HEDGE_C seed 0, dump (price_paid, C1_total, w) for every
    completed task. Returns the number of completed tasks in this seed."""
    base_cfg = build_base_config()
    cfg = arm_config(base_cfg, ARMS["HEDGE_C"])
    logger.info(f"Re-running HEDGE_C seed={SEED} (full duration) for the floor-scatter dump...")
    engine = HEDGESimulationEngine(config=cfg, seed=SEED, output_dir=None)
    engine.run(n_tasks=None)

    completed = [e for e in engine._mc.events if e.get("event_type") == "completed"]
    logger.info(f"{len(completed)} completed tasks this seed.")

    DUMP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DUMP_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["price_paid", "C1_winner", "executor_id"])
        for e in completed:
            w.writerow([e["price_paid"], e["C1_total"], e.get("executor_id", "")])
    logger.info(f"Wrote {len(completed)} rows to {DUMP_CSV}")

    violations_this_seed = sum(
        1 for e in completed if float(e["price_paid"]) < float(e["C1_total"]) - 1e-9
    )
    logger.info(f"Floor violations this seed: {violations_this_seed} (must be 0)")
    if violations_this_seed != 0:
        logger.error("ABORTING: nonzero floor violations in this fresh re-run -- do not plot.")
        sys.exit(1)

    return len(completed)


def plot(n_completed_this_seed: int) -> None:
    edge_rows, cloud_rows = [], []
    with DUMP_CSV.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            pair = (float(r["price_paid"]), float(r["C1_winner"]))
            (cloud_rows if r.get("executor_id") == "cloud" else edge_rows).append(pair)

    # Edge (~88% of trades) and cloud (~12%) sit in two tight, well-separated
    # clusters roughly 4-5 orders of magnitude apart (edge floor ~1e-6 to
    # 3.5e-5 USD; cloud floor ~1.7 to 1.78 USD) -- confirmed by inspecting
    # the dump directly, not a rendering artifact. A single shared log-log
    # axis compresses the majority-share edge cluster into an illegible
    # corner, so this splits into two zoomed panels instead: same honest
    # data, each cluster's own internal structure now actually visible.
    random.seed(0)
    n_edge_sample = int(SAMPLE_N * len(edge_rows) / (len(edge_rows) + len(cloud_rows)))
    n_cloud_sample = SAMPLE_N - n_edge_sample
    edge_sample = random.sample(edge_rows, min(n_edge_sample, len(edge_rows)))
    cloud_sample = random.sample(cloud_rows, min(n_cloud_sample, len(cloud_rows)))

    below = sum(1 for p, c1 in edge_sample + cloud_sample if p < c1 - 1e-12)
    logger.info(
        f"Sampled {len(edge_sample)} edge + {len(cloud_sample)} cloud points; "
        f"{below} below the floor (must be 0)."
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.5), constrained_layout=True)
    panels = [
        (axes[0], edge_sample, "Edge trades", "#0072B2"),
        (axes[1], cloud_sample, "Cloud trades", "#555555"),
    ]
    for ax, sample, title, color in panels:
        xs = [c1 for _, c1 in sample]
        ys = [p for p, _ in sample]
        ax.scatter(xs, ys, s=6, color=color, alpha=0.5, edgecolor="none", zorder=3)
        lo = min(min(xs), min(ys)) * 0.85
        hi = max(max(xs), max(ys)) * 1.18
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=8, fontweight="normal")
        ax.set_xlabel("Cost floor (USD)", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("Price paid (USD)", fontsize=7)

    fig.suptitle(
        f"0 / {FULL_CAMPAIGN_TASKS:,} trades below floor (full campaign)",
        fontsize=6.8, y=1.1,
    )

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Wrote {FIG_OUT}")

    print(
        f"[fig5 floor_scatter] source=floor_scatter_seed0.csv ({n_completed_this_seed} tasks this "
        f"seed: {len(edge_rows)} edge / {len(cloud_rows)} cloud; {len(edge_sample)}+{len(cloud_sample)} "
        f"plotted) + full-campaign total ({FULL_CAMPAIGN_TASKS:,} tasks, "
        f"{FULL_CAMPAIGN_VIOLATIONS} violations)"
    )


def main() -> None:
    n = run_and_dump()
    plot(n)


if __name__ == "__main__":
    main()
