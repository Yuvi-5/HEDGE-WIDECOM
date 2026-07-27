"""Render the six WIDECOM paper figures from real campaign CSVs only.

Every value plotted comes from a named column in FINAL_SUMMARY_main_arms.csv
or FINAL_SUMMARY_phase0_ablation.csv (Figures 1-4, 6) or from a fresh,
additive event dump (Figure 5, see make_floor_scatter.py). No number in
this script is invented, estimated, or hand-typed from memory -- every
plotted quantity is read from a DataFrame cell and also printed to stdout
as a provenance log.

Honesty guards enforced structurally, not just by convention:
  - JFI/Myerson are never read for the main 6-arm comparison (Figures 1-4);
    only Figure 6's Phase-0 on/off panel touches JFI (a valid internal
    comparison, not a cross-arm claim -- both are the SAME mechanism).
  - Figure 3 only plots the 4 edge-serving arms; Vertical/Cloud-only are
    never given a fabricated zero bar for a trade type they don't have.

Idempotent: rerunning overwrites the same 5 filenames (Figure 5 is separate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
MAIN_CSV = ROOT / "outputs" / "FINAL_SUMMARY_main_arms.csv"
PHASE0_CSV = ROOT / "outputs" / "FINAL_SUMMARY_phase0_ablation.csv"

# --- House style: fixed arm -> label/color/hatch, per handoff doc Sec 2 ---
ARM_META = {
    "HEDGE_C":       {"label": "HEDGE",           "color": "#0072B2", "hatch": None},
    "HEDGE_C_Kmax0": {"label": "Vertical (K=0)",  "color": "#000000", "hatch": "//"},
    "B2_DDPS":       {"label": "DDPS",             "color": "#E69F00", "hatch": ".."},
    "B4_CostOPD":    {"label": "Cost-OPD",         "color": "#8C7A3F", "hatch": "xx"},
    "B6_GreedyNLF":  {"label": "Greedy-NLF",       "color": "#009E73", "hatch": "\\\\"},
    "B7_CloudOnly":  {"label": "Cloud-only",       "color": "#555555", "hatch": "//"},
}
ARM_ORDER = ["HEDGE_C", "HEDGE_C_Kmax0", "B2_DDPS", "B4_CostOPD", "B6_GreedyNLF", "B7_CloudOnly"]
EDGE_SERVING_ARMS = ["HEDGE_C", "B2_DDPS", "B4_CostOPD", "B6_GreedyNLF"]

RED_NEG = "#B0413E"
GRID_COLOR = "#D9D9D9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.5,
    "axes.grid.axis": "y",
    "pdf.fonttype": 42,  # embed as real text, not paths
})

provenance: list[str] = []


def log(fig_name: str, source: str, values: dict) -> None:
    line = f"[{fig_name}] source={source} values={values}"
    provenance.append(line)
    print(line)


def savefig(fig, name: str) -> None:
    out_path = OUT_DIR / name
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def bar_with_ci(ax, arms: list[str], means: list[float], los: list[float], his: list[float],
                 ylabel: str, annotate_fmt: str = "{:.2f}") -> None:
    x = range(len(arms))
    colors = [ARM_META[a]["color"] for a in arms]
    hatches = [ARM_META[a]["hatch"] for a in arms]
    yerr_lo = [max(0.0, m - lo) for m, lo in zip(means, los)]
    yerr_hi = [max(0.0, hi - m) for m, hi in zip(means, his)]
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6,
                   yerr=[yerr_lo, yerr_hi], capsize=2.5,
                   error_kw={"linewidth": 0.7, "ecolor": "#333333"})
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARM_META[a]["label"] for a in arms], rotation=28, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8)
    top = max(his) if his else max(means)
    for xi, (m, hi) in zip(x, zip(means, his)):
        ax.annotate(annotate_fmt.format(m), (xi, hi if hi == hi else m),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6.5)
    ax.margins(y=0.18)


def col(df: pd.DataFrame, arm: str, base: str, suffix: str = "_mean") -> float:
    return float(df.loc[df["arm"] == arm, base + suffix].iloc[0])


# ---------------------------------------------------------------------------
# Figure 1 -- Operational parity
# ---------------------------------------------------------------------------
def fig1_operational_parity(main: pd.DataFrame) -> None:
    arms = ARM_ORDER
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 6.4), constrained_layout=True)

    lat_m = [col(main, a, "M1_mean_latency_s") for a in arms]
    lat_lo = [col(main, a, "M1_mean_latency_s", "_ci_lo") for a in arms]
    lat_hi = [col(main, a, "M1_mean_latency_s", "_ci_hi") for a in arms]
    bar_with_ci(axes[0], arms, lat_m, lat_lo, lat_hi, "Mean latency (s)", "{:.3f}")
    log("fig1.A mean_latency", "FINAL_SUMMARY_main_arms.csv:M1_mean_latency_s_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 4) for v in lat_m])))

    cost_m = [col(main, a, "M2_mean_cost_usd") for a in arms]
    cost_lo = [col(main, a, "M2_mean_cost_usd", "_ci_lo") for a in arms]
    cost_hi = [col(main, a, "M2_mean_cost_usd", "_ci_hi") for a in arms]
    bar_with_ci(axes[1], arms, cost_m, cost_lo, cost_hi, "Mean cost (USD)", "{:.3f}")
    log("fig1.B mean_cost", "FINAL_SUMMARY_main_arms.csv:M2_mean_cost_usd_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 4) for v in cost_m])))

    rej_m = [col(main, a, "M6_rejection_rate") * 100 for a in arms]
    rej_lo = [col(main, a, "M6_rejection_rate", "_ci_lo") * 100 for a in arms]
    rej_hi = [col(main, a, "M6_rejection_rate", "_ci_hi") * 100 for a in arms]
    bar_with_ci(axes[2], arms, rej_m, rej_lo, rej_hi, "Rejection rate (%)", "{:.1f}")
    log("fig1.C rejection", "FINAL_SUMMARY_main_arms.csv:M6_rejection_rate_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 2) for v in rej_m])))

    savefig(fig, "fig1_operational_parity.pdf")


# ---------------------------------------------------------------------------
# Figure 2 -- Market revenue (log scale, hatched zero-stub)
# ---------------------------------------------------------------------------
def fig2_market_revenue(main: pd.DataFrame) -> None:
    arms = ARM_ORDER
    rev_m = [col(main, a, "M5_market_revenue_usd") for a in arms]
    rev_lo = [col(main, a, "M5_market_revenue_usd", "_ci_lo") for a in arms]
    rev_hi = [col(main, a, "M5_market_revenue_usd", "_ci_hi") for a in arms]
    log("fig2 market_revenue", "FINAL_SUMMARY_main_arms.csv:M5_market_revenue_usd_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 3) for v in rev_m])))

    fig, ax = plt.subplots(figsize=(3.3, 2.5), constrained_layout=True)
    x = range(len(arms))
    floor = 0.05  # visual floor for log scale; zero-revenue arms drawn as a stub here
    plotted = [max(v, floor) for v in rev_m]
    colors = [ARM_META[a]["color"] for a in arms]
    hatches = [ARM_META[a]["hatch"] for a in arms]
    bars = ax.bar(x, plotted, color=colors, edgecolor="black", linewidth=0.6)
    for bar, h, real_v in zip(bars, hatches, rev_m):
        if h or real_v == 0:
            bar.set_hatch(h or "//")
    ax.set_yscale("log")
    ax.set_ylabel("Edge market revenue (USD, log scale)", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARM_META[a]["label"] for a in arms], rotation=28, ha="right", fontsize=7.5)
    for xi, real_v in zip(x, rev_m):
        label = "0" if real_v == 0 else f"{real_v:.1f}"
        ax.annotate(label, (xi, max(real_v, floor)), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_ylim(floor * 0.6, max(rev_hi) * 2.2)
    savefig(fig, "fig2_market_revenue.pdf")


# ---------------------------------------------------------------------------
# Figure 3 -- Economic sustainability (edge-serving arms only)
# ---------------------------------------------------------------------------
def fig3_sustainability(main: pd.DataFrame) -> None:
    arms = EDGE_SERVING_ARMS
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.6), constrained_layout=True)

    profit_m = [col(main, a, "econ_edge_profit") for a in arms]
    profit_lo = [col(main, a, "econ_edge_profit", "_ci_lo") for a in arms]
    profit_hi = [col(main, a, "econ_edge_profit", "_ci_hi") for a in arms]
    x = range(len(arms))
    colors = [RED_NEG if v < 0 else ARM_META[a]["color"] for a, v in zip(arms, profit_m)]
    hatches = [ARM_META[a]["hatch"] for a in arms]
    yerr_lo = [max(0.0, m - lo) for m, lo in zip(profit_m, profit_lo)]
    yerr_hi = [max(0.0, hi - m) for m, hi in zip(profit_m, profit_hi)]
    bars = axes[0].bar(x, profit_m, color=colors, edgecolor="black", linewidth=0.6,
                        yerr=[yerr_lo, yerr_hi], capsize=2.5,
                        error_kw={"linewidth": 0.7, "ecolor": "#333333"})
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Edge profit per task (USD)", fontsize=8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([ARM_META[a]["label"] for a in arms], rotation=28, ha="right", fontsize=7.5)
    top_a = max(profit_hi) * 1.18
    bottom_a = min(min(profit_lo), 0) - 0.22 * max(profit_hi)  # generous headroom below 0 for negative annotations
    axes[0].set_ylim(bottom_a, top_a)
    for xi, (m, hi, lo) in zip(x, zip(profit_m, profit_hi, profit_lo)):
        y = hi if m >= 0 else lo
        va = "bottom" if m >= 0 else "top"
        offset = 4 if m >= 0 else -14
        axes[0].annotate(f"{m:+.2f}", (xi, y), xytext=(0, offset), textcoords="offset points",
                          ha="center", va=va, fontsize=6.5)
    log("fig3.A edge_profit", "FINAL_SUMMARY_main_arms.csv:econ_edge_profit_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 4) for v in profit_m])))

    margin_m = [col(main, a, "econ_edge_margin_pct") for a in arms]
    margin_lo = [col(main, a, "econ_edge_margin_pct", "_ci_lo") for a in arms]
    margin_hi = [col(main, a, "econ_edge_margin_pct", "_ci_hi") for a in arms]
    log("fig3.B edge_margin_pct", "FINAL_SUMMARY_main_arms.csv:econ_edge_margin_pct_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 2) for v in margin_m])))

    # DDPS's margin (~-1570%) would crush the others on a linear axis; use
    # symlog so both the near-zero (Cost-OPD/Greedy ~0%, HEDGE ~98%) and the
    # deeply negative (DDPS) bars stay legible, with the true value annotated.
    colors2 = [RED_NEG if v < 0 else ARM_META[a]["color"] for a, v in zip(arms, margin_m)]
    yerr_lo2 = [max(0.0, m - lo) for m, lo in zip(margin_m, margin_lo)]
    yerr_hi2 = [max(0.0, hi - m) for m, hi in zip(margin_m, margin_hi)]
    bars2 = axes[1].bar(x, margin_m, color=colors2, edgecolor="black", linewidth=0.6,
                         yerr=[yerr_lo2, yerr_hi2], capsize=2.5,
                         error_kw={"linewidth": 0.7, "ecolor": "#333333"})
    for bar, h in zip(bars2, hatches):
        if h:
            bar.set_hatch(h)
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Edge margin (%, symlog)", fontsize=8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([ARM_META[a]["label"] for a in arms], rotation=28, ha="right", fontsize=7.5)
    # Symlog compresses large magnitudes visually, but the axes' bottom edge
    # still sits right at the data minimum by default -- push it well past
    # the most negative bar (DDPS, ~-1570%) so annotations don't collide
    # with the rotated x-tick labels directly below.
    axes[1].set_ylim(min(margin_lo) * 3, max(margin_hi) * 3)
    for xi, m in zip(x, margin_m):
        va = "bottom" if m >= 0 else "top"
        offset = 4 if m >= 0 else -8
        axes[1].annotate(f"{m:+.1f}%", (xi, m), xytext=(0, offset), textcoords="offset points",
                          ha="center", va=va, fontsize=6.5)

    savefig(fig, "fig3_sustainability.pdf")


# ---------------------------------------------------------------------------
# Figure 4 -- K_max dial (Vertical K=0 vs deployed K=15)
# ---------------------------------------------------------------------------
def fig4_kmax_dial(main: pd.DataFrame) -> None:
    arms = ["HEDGE_C_Kmax0", "HEDGE_C"]  # K=0 first, K=15 (deployed) second
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.3), constrained_layout=True)

    rej_m = [col(main, a, "M6_rejection_rate") * 100 for a in arms]
    rej_lo = [col(main, a, "M6_rejection_rate", "_ci_lo") * 100 for a in arms]
    rej_hi = [col(main, a, "M6_rejection_rate", "_ci_hi") * 100 for a in arms]
    bar_with_ci(axes[0], arms, rej_m, rej_lo, rej_hi, "Rejection rate (%)", "{:.1f}")
    log("fig4.A rejection", "FINAL_SUMMARY_main_arms.csv:M6_rejection_rate_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 2) for v in rej_m])))

    cost_m = [col(main, a, "M2_mean_cost_usd") for a in arms]
    cost_lo = [col(main, a, "M2_mean_cost_usd", "_ci_lo") for a in arms]
    cost_hi = [col(main, a, "M2_mean_cost_usd", "_ci_hi") for a in arms]
    bar_with_ci(axes[1], arms, cost_m, cost_lo, cost_hi, "Mean cost (USD)", "{:.3f}")
    log("fig4.B mean_cost", "FINAL_SUMMARY_main_arms.csv:M2_mean_cost_usd_mean",
        dict(zip([ARM_META[a]["label"] for a in arms], [round(v, 4) for v in cost_m])))

    savefig(fig, "fig4_kmax_dial.pdf")


# ---------------------------------------------------------------------------
# Figure 6 -- Phase-0 election contribution (internal on/off, JFI valid here)
# ---------------------------------------------------------------------------
def fig6_phase0_ablation(phase0: pd.DataFrame) -> None:
    arms = ["Phase0_On", "Phase0_Off"]
    labels = {"Phase0_On": "Phase-0 on", "Phase0_Off": "Phase-0 off"}
    colors = {"Phase0_On": "#0072B2", "Phase0_Off": "#555555"}
    hatches = {"Phase0_On": None, "Phase0_Off": "//"}

    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.3), constrained_layout=True)

    jfi_m = [col(phase0, a, "M11_jfi") for a in arms]
    jfi_lo = [col(phase0, a, "M11_jfi", "_ci_lo") for a in arms]
    jfi_hi = [col(phase0, a, "M11_jfi", "_ci_hi") for a in arms]
    x = range(len(arms))
    yerr_lo = [max(0.0, m - lo) for m, lo in zip(jfi_m, jfi_lo)]
    yerr_hi = [max(0.0, hi - m) for m, hi in zip(jfi_m, jfi_hi)]
    bars = axes[0].bar(x, jfi_m, color=[colors[a] for a in arms], edgecolor="black",
                        linewidth=0.6, yerr=[yerr_lo, yerr_hi], capsize=2.5,
                        error_kw={"linewidth": 0.7, "ecolor": "#333333"})
    for bar, a in zip(bars, arms):
        if hatches[a]:
            bar.set_hatch(hatches[a])
    axes[0].set_ylabel("Jain's Fairness Index", fontsize=8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([labels[a] for a in arms], rotation=20, ha="right", fontsize=7.5)
    for xi, (m, hi) in zip(x, zip(jfi_m, jfi_hi)):
        axes[0].annotate(f"{m:.3f}", (xi, hi), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=6.5)
    axes[0].margins(y=0.22)
    log("fig6.A jfi", "FINAL_SUMMARY_phase0_ablation.csv:M11_jfi_mean",
        dict(zip([labels[a] for a in arms], [round(v, 4) for v in jfi_m])))

    rej_m = [col(phase0, a, "M6_rejection_rate") * 100 for a in arms]
    rej_lo = [col(phase0, a, "M6_rejection_rate", "_ci_lo") * 100 for a in arms]
    rej_hi = [col(phase0, a, "M6_rejection_rate", "_ci_hi") * 100 for a in arms]
    yerr_lo2 = [max(0.0, m - lo) for m, lo in zip(rej_m, rej_lo)]
    yerr_hi2 = [max(0.0, hi - m) for m, hi in zip(rej_m, rej_hi)]
    bars2 = axes[1].bar(x, rej_m, color=[colors[a] for a in arms], edgecolor="black",
                         linewidth=0.6, yerr=[yerr_lo2, yerr_hi2], capsize=2.5,
                         error_kw={"linewidth": 0.7, "ecolor": "#333333"})
    for bar, a in zip(bars2, arms):
        if hatches[a]:
            bar.set_hatch(hatches[a])
    axes[1].set_ylabel("Rejection rate (%)", fontsize=8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([labels[a] for a in arms], rotation=20, ha="right", fontsize=7.5)
    for xi, (m, hi) in zip(x, zip(rej_m, rej_hi)):
        axes[1].annotate(f"{m:.2f}", (xi, hi), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=6.5)
    axes[1].margins(y=0.22)
    log("fig6.B rejection", "FINAL_SUMMARY_phase0_ablation.csv:M6_rejection_rate_mean",
        dict(zip([labels[a] for a in arms], [round(v, 3) for v in rej_m])))

    savefig(fig, "fig6_phase0_ablation.pdf")


def main() -> None:
    if not MAIN_CSV.exists() or not PHASE0_CSV.exists():
        print(f"STOP: missing source CSV(s). Checked:\n  {MAIN_CSV}\n  {PHASE0_CSV}", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(MAIN_CSV)
    phase0_df = pd.read_csv(PHASE0_CSV)

    print("=== Rendering Figures 1, 2, 3, 4, 6 (Figure 5 is a separate script) ===")
    fig1_operational_parity(main_df)
    fig2_market_revenue(main_df)
    fig3_sustainability(main_df)
    fig4_kmax_dial(main_df)
    fig6_phase0_ablation(phase0_df)

    print("\n=== Provenance log (also printed inline above) ===")
    for line in provenance:
        print(line)


if __name__ == "__main__":
    main()
