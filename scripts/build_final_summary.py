"""Build a single consolidated per-arm summary CSV (mean + 95% bootstrap CI)
from the three raw per-seed CSVs, for easy handoff -- the raw per-seed CSVs
remain the reproducibility record, this is the "just show me the numbers"
artifact.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import bootstrap_ci  # noqa: E402

MAIN_CSV = ROOT / "outputs" / "hedge_c_comparison.csv"
PHASE0_CSV = ROOT / "outputs" / "phase0_ablation.csv"
ECON_CSV = ROOT / "outputs" / "economics_supplement.csv"
OUT_MAIN = ROOT / "outputs" / "FINAL_SUMMARY_main_arms.csv"
OUT_PHASE0 = ROOT / "outputs" / "FINAL_SUMMARY_phase0_ablation.csv"

CORE_METRICS = [
    "M1_p95_latency_s", "M1_mean_latency_s", "M2_mean_cost_usd",
    "M5_market_revenue_usd", "M6_rejection_rate", "M7_carbon_g_per_task",
    "M9_edge_service_fraction", "M11_jfi", "M16_myerson_efficiency",
]
ECON_METRICS = [
    "edge_revenue", "edge_true_cost", "edge_profit", "edge_margin_pct",
    "cloud_revenue", "cloud_true_cost", "cloud_profit",
    "total_revenue", "total_user_spend",
    "edge_mean_a_buyer", "edge_mean_surplus", "edge_surplus_pct_of_valuation",
    "cloud_mean_a_buyer", "cloud_mean_surplus", "cloud_surplus_pct_of_valuation",
]

ARMS = ["HEDGE_C", "HEDGE_C_Kmax0", "B2_DDPS", "B4_CostOPD", "B6_GreedyNLF", "B7_CloudOnly"]
PHASE0_ARMS = ["Phase0_On", "Phase0_Off"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def arm_values(rows: list[dict[str, str]], arm: str, metric: str) -> list[float]:
    out = []
    for r in rows:
        if r.get("arm") != arm:
            continue
        v = r.get(metric)
        if v in (None, "", "nan", "N/A"):
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def summarize(rows: list[dict[str, str]], arms: list[str], metrics: list[str], n_field: str) -> list[dict]:
    out_rows = []
    for arm in arms:
        row = {"arm": arm}
        n_seeds = len({r["seed"] for r in rows if r.get("arm") == arm})
        row["n_seeds"] = n_seeds
        for metric in metrics:
            vals = arm_values(rows, arm, metric)
            if not vals:
                row[f"{metric}_mean"] = "N/A"
                row[f"{metric}_ci_lo"] = "N/A"
                row[f"{metric}_ci_hi"] = "N/A"
                continue
            mean = sum(vals) / len(vals)
            if len(vals) >= 2:
                lo, hi = bootstrap_ci(vals, seed=0)
            else:
                lo, hi = mean, mean
            row[f"{metric}_mean"] = f"{mean:.6g}"
            row[f"{metric}_ci_lo"] = f"{lo:.6g}"
            row[f"{metric}_ci_hi"] = f"{hi:.6g}"
        out_rows.append(row)
    return out_rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    main_rows = load_csv(MAIN_CSV)
    econ_rows = load_csv(ECON_CSV)
    phase0_rows = load_csv(PHASE0_CSV)

    core_summary = summarize(main_rows, ARMS, CORE_METRICS, "seed")
    econ_summary = summarize(econ_rows, ARMS, ECON_METRICS, "seed")

    # Merge core + econ summaries by arm into one row per arm.
    econ_by_arm = {r["arm"]: r for r in econ_summary}
    merged = []
    for row in core_summary:
        arm = row["arm"]
        econ_row = econ_by_arm.get(arm, {})
        merged_row = dict(row)
        for k, v in econ_row.items():
            if k in ("arm", "n_seeds"):
                continue
            merged_row[f"econ_{k}"] = v
        merged_row["econ_n_seeds"] = econ_row.get("n_seeds", "N/A")
        merged.append(merged_row)

    write_csv(merged, OUT_MAIN)

    phase0_summary = summarize(phase0_rows, PHASE0_ARMS, CORE_METRICS, "seed")
    write_csv(phase0_summary, OUT_PHASE0)


if __name__ == "__main__":
    main()
