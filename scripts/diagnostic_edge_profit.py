"""One-off diagnostic (not part of the paper deliverables): compute actual
edge-seller PROFIT (revenue - true physical cost, i.e. price_paid - C1_total)
AND buyer surplus (a_buyer - price_paid, both already logged per task) per
arm, not just revenue. Revenue alone doesn't show whether a baseline is
pricing below its own true cost (B2/DDPS intentionally prices off a
narrower cost basis than C1_total, so its reported revenue can look higher
than its true profit), and price alone doesn't show
whether buyers are left comfortable slack vs. their private valuation or
priced right up against it.

Runs a light, capped-n_tasks probe per arm (not full 3600s duration),
sequentially (no parallel workers), in its own output directory -- so it
doesn't collide with or meaningfully compete with the main 30-seed
campaign's 10 workers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_hedge_c_comparison import ARMS, arm_config, build_base_config  # noqa: E402
from hedge.simulation.engine import HEDGESimulationEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "diagnostic_profit"
N_TASKS = 320000  # enough to clear the 300s warmup with real post-warmup runway
SEED = 0


def run_one(arm_label: str, overrides: dict) -> dict:
    base_cfg = build_base_config()
    cfg = arm_config(base_cfg, overrides)
    out_dir = OUTPUT_ROOT / arm_label
    engine = HEDGESimulationEngine(config=cfg, seed=SEED, output_dir=out_dir)
    engine.run(n_tasks=N_TASKS)

    completed = [e for e in engine._mc.events if e.get("event_type") == "completed"]
    edge = [e for e in completed if str(e.get("executor_id", "")) != "cloud"]
    cloud = [e for e in completed if str(e.get("executor_id", "")) == "cloud"]

    edge_revenue = sum(float(e["price_paid"]) for e in edge)
    edge_true_cost = sum(float(e["C1_total"]) for e in edge)
    edge_profit = edge_revenue - edge_true_cost

    cloud_revenue = sum(float(e["price_paid"]) for e in cloud)
    cloud_true_cost = sum(float(e["C1_total"]) for e in cloud)
    cloud_profit = cloud_revenue - cloud_true_cost

    n_loss_making = sum(
        1 for e in edge if float(e["price_paid"]) < float(e["C1_total"]) - 1e-9
    )

    # Buyer surplus: a_buyer (private valuation, already logged per task) minus
    # what they actually paid. Positive by construction (IR), but the MAGNITUDE
    # tells us whether a mechanism is pricing close to the buyer's ceiling
    # (thin surplus) or leaving them lots of slack (thick surplus) -- this is
    # what actually answers "is HEDGE's bigger markup still comfortable for
    # buyers, relative to what the baselines leave them."
    edge_a_buyer_sum = sum(float(e["a_buyer"]) for e in edge)
    edge_surplus_sum = sum(float(e["a_buyer"]) - float(e["price_paid"]) for e in edge)
    edge_n_ir_violations = sum(
        1 for e in edge if float(e["price_paid"]) > float(e["a_buyer"]) + 1e-9
    )

    cloud_a_buyer_sum = sum(float(e["a_buyer"]) for e in cloud)
    cloud_surplus_sum = sum(float(e["a_buyer"]) - float(e["price_paid"]) for e in cloud)

    return dict(
        arm=arm_label,
        n_completed=len(completed),
        n_edge=len(edge),
        n_cloud=len(cloud),
        edge_revenue=edge_revenue,
        edge_true_cost=edge_true_cost,
        edge_profit=edge_profit,
        edge_margin_pct=(100.0 * edge_profit / edge_revenue) if edge_revenue else float("nan"),
        n_edge_tasks_priced_below_true_cost=n_loss_making,
        pct_edge_tasks_priced_below_true_cost=(100.0 * n_loss_making / len(edge)) if edge else 0.0,
        cloud_revenue=cloud_revenue,
        cloud_true_cost=cloud_true_cost,
        cloud_profit=cloud_profit,
        edge_mean_a_buyer=(edge_a_buyer_sum / len(edge)) if edge else float("nan"),
        edge_mean_surplus=(edge_surplus_sum / len(edge)) if edge else float("nan"),
        edge_surplus_pct_of_valuation=(100.0 * edge_surplus_sum / edge_a_buyer_sum) if edge_a_buyer_sum else float("nan"),
        edge_n_ir_violations=edge_n_ir_violations,
        cloud_mean_a_buyer=(cloud_a_buyer_sum / len(cloud)) if cloud else float("nan"),
        cloud_mean_surplus=(cloud_surplus_sum / len(cloud)) if cloud else float("nan"),
        cloud_surplus_pct_of_valuation=(100.0 * cloud_surplus_sum / cloud_a_buyer_sum) if cloud_a_buyer_sum else float("nan"),
    )


def main() -> None:
    from loguru import logger

    results = []
    for arm_label, overrides in ARMS.items():
        logger.info(f"Running diagnostic for {arm_label} (n_tasks={N_TASKS})...")
        r = run_one(arm_label, overrides)
        results.append(r)
        logger.info(
            f"{arm_label}: edge_revenue=${r['edge_revenue']:.4f} "
            f"edge_true_cost=${r['edge_true_cost']:.4f} "
            f"edge_PROFIT=${r['edge_profit']:.4f} "
            f"margin={r['edge_margin_pct']:.1f}% "
            f"loss_making_tasks={r['n_edge_tasks_priced_below_true_cost']}/{r['n_edge']} "
            f"({r['pct_edge_tasks_priced_below_true_cost']:.1f}%)"
        )

    print()
    print("=== EDGE SELLER PROFIT (revenue vs. true physical cost C1_total) ===")
    print(
        f"{'arm':16}{'n_edge':>10}{'edge_revenue':>16}{'edge_true_cost':>18}"
        f"{'edge_PROFIT':>16}{'margin%':>10}{'loss-making%':>14}"
    )
    for r in results:
        print(
            f"{r['arm']:16}{r['n_edge']:>10,}{r['edge_revenue']:>16.4f}"
            f"{r['edge_true_cost']:>18.6f}{r['edge_profit']:>16.4f}"
            f"{r['edge_margin_pct']:>10.1f}{r['pct_edge_tasks_priced_below_true_cost']:>14.1f}"
        )

    print()
    print("=== BUYER SURPLUS on edge-served tasks (a_buyer - price_paid) ===")
    print(
        f"{'arm':16}{'mean a_buyer':>16}{'mean price_paid':>18}"
        f"{'mean surplus':>16}{'surplus % of a_buyer':>22}{'IR violations':>16}"
    )
    for r in results:
        mean_price = r["edge_revenue"] / r["n_edge"] if r["n_edge"] else float("nan")
        print(
            f"{r['arm']:16}{r['edge_mean_a_buyer']:>16.6f}{mean_price:>18.6f}"
            f"{r['edge_mean_surplus']:>16.6f}{r['edge_surplus_pct_of_valuation']:>22.1f}"
            f"{r['edge_n_ir_violations']:>16,}"
        )

    print()
    print("=== BUYER SURPLUS on cloud-served tasks (same across arms with equal cloud access) ===")
    print(f"{'arm':16}{'mean a_buyer':>16}{'mean surplus':>16}{'surplus % of a_buyer':>22}")
    for r in results:
        if r["n_cloud"]:
            print(
                f"{r['arm']:16}{r['cloud_mean_a_buyer']:>16.6f}"
                f"{r['cloud_mean_surplus']:>16.6f}{r['cloud_surplus_pct_of_valuation']:>22.1f}"
            )


if __name__ == "__main__":
    main()
