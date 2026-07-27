"""Supplementary 30-seed campaign: edge profit, cloud revenue/profit, total
user spend, and buyer surplus, per arm -- fields NOT captured by the main
21-metric campaign (run_hedge_c_comparison.py), requested explicitly for
the final CSVs/numbers.tex beyond the original kickoff scope.

Deliberately uses a capped n_tasks (not the full 3600s duration) so this
can complete in minutes rather than hours: N_TASKS=320000 reaches roughly
t=427s of simulated time (well past the 300s warmup, ~127s of real
post-warmup sampling), which is more than enough for a stable per-seed
mean/sum given ~85k post-warmup events per seed. This trades some duration
for feasibility within the time budget -- it is a smaller-duration
companion sample, not a full-duration rerun of the main campaign.

Resumable the same way as the main campaign (per-seed JSON existence
check). Uses a smaller worker count by default so it can run alongside
the still-in-progress main campaign without starving it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from experiments.runner import write_metrics_csv  # noqa: E402
from hedge.simulation.engine import HEDGESimulationEngine  # noqa: E402
from run_hedge_c_comparison import ARMS, arm_config, build_base_config  # noqa: E402

N_SEEDS = 30
N_WORKERS = 6  # deliberately lighter than the main campaign's 10, to coexist
N_TASKS = 320000  # ~427s simulated, well past the 300s warmup
OUTPUT_ROOT = ROOT / "outputs" / "economics_supplement"
CSV_PATH = ROOT / "outputs" / "economics_supplement.csv"


def run_one_seed(arm_label: str, cfg: dict, seed: int, seed_dir: Path, force: bool = False) -> dict:
    summary_path = seed_dir / "econ_summary.json"
    if not force and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    engine = HEDGESimulationEngine(config=cfg, seed=seed, output_dir=seed_dir)
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

    total_revenue = edge_revenue + cloud_revenue  # == total user spend, no platform skim

    edge_a_buyer_sum = sum(float(e["a_buyer"]) for e in edge)
    edge_surplus_sum = sum(float(e["a_buyer"]) - float(e["price_paid"]) for e in edge)
    cloud_a_buyer_sum = sum(float(e["a_buyer"]) for e in cloud)
    cloud_surplus_sum = sum(float(e["a_buyer"]) - float(e["price_paid"]) for e in cloud)

    n_loss_making = sum(1 for e in edge if float(e["price_paid"]) < float(e["C1_total"]) - 1e-9)
    n_ir_violations = sum(
        1 for e in completed if float(e["price_paid"]) > float(e["a_buyer"]) + 1e-9
    )

    row = {
        "arm": arm_label,
        "seed": seed,
        "n_completed": len(completed),
        "n_edge": len(edge),
        "n_cloud": len(cloud),
        "edge_revenue": edge_revenue,
        "edge_true_cost": edge_true_cost,
        "edge_profit": edge_profit,
        "edge_margin_pct": (100.0 * edge_profit / edge_revenue) if edge_revenue else float("nan"),
        "n_edge_loss_making": n_loss_making,
        "cloud_revenue": cloud_revenue,
        "cloud_true_cost": cloud_true_cost,
        "cloud_profit": cloud_profit,
        "total_revenue": total_revenue,
        "total_user_spend": total_revenue,
        "edge_mean_a_buyer": (edge_a_buyer_sum / len(edge)) if edge else float("nan"),
        "edge_mean_surplus": (edge_surplus_sum / len(edge)) if edge else float("nan"),
        "edge_surplus_pct_of_valuation": (100.0 * edge_surplus_sum / edge_a_buyer_sum) if edge_a_buyer_sum else float("nan"),
        "cloud_mean_a_buyer": (cloud_a_buyer_sum / len(cloud)) if cloud else float("nan"),
        "cloud_mean_surplus": (cloud_surplus_sum / len(cloud)) if cloud else float("nan"),
        "cloud_surplus_pct_of_valuation": (100.0 * cloud_surplus_sum / cloud_a_buyer_sum) if cloud_a_buyer_sum else float("nan"),
        "n_ir_violations": n_ir_violations,
        "n_tasks_cap": N_TASKS,
    }

    seed_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)

    return row


def _load_existing_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Economics supplement campaign (profit/surplus/cloud-split).")
    parser.add_argument("--arm", choices=list(ARMS.keys()), default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS, dest="n_seeds")
    args = parser.parse_args(argv)

    arms_to_run = {args.arm: ARMS[args.arm]} if args.arm else ARMS
    base_cfg = build_base_config()

    jobs: list[tuple[str, dict, int, Path]] = []
    already_done = 0
    for arm_label, overrides in arms_to_run.items():
        cfg = arm_config(base_cfg, overrides)
        for seed in range(args.n_seeds):
            seed_dir = OUTPUT_ROOT / arm_label / f"seed_{seed:02d}"
            if not args.force and (seed_dir / "econ_summary.json").exists():
                already_done += 1
            jobs.append((arm_label, cfg, seed, seed_dir))

    total_runs = len(jobs)
    logger.info(
        f"Economics supplement: {total_runs} runs, {already_done} already done, "
        f"{total_runs - already_done} to run, {N_WORKERS} workers, n_tasks={N_TASKS}"
    )

    start = time.time()
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=N_WORKERS, verbose=10)(
            delayed(run_one_seed)(arm_label, cfg, seed, seed_dir, args.force)
            for arm_label, cfg, seed, seed_dir in jobs
        )
    except ImportError:
        logger.warning("joblib not available, running single-threaded")
        results = [
            run_one_seed(arm_label, cfg, seed, seed_dir, args.force)
            for arm_label, cfg, seed, seed_dir in jobs
        ]

    elapsed = time.time() - start
    logger.info(f"All {total_runs} runs finished in {elapsed:.1f}s")

    rows = list(results)
    ir_total = sum(int(r.get("n_ir_violations", 0)) for r in rows)
    logger.info(f"Total IR violations across these {len(rows)} rows: {ir_total} (must be 0)")

    if args.arm:
        existing = _load_existing_csv_rows(CSV_PATH)
        kept = [r for r in existing if r.get("arm") != args.arm]
        merged = kept + [{k: str(v) for k, v in row.items()} for row in rows]
        write_metrics_csv(merged, CSV_PATH)
    else:
        write_metrics_csv(rows, CSV_PATH)

    for arm_label in arms_to_run:
        arm_rows = [r for r in rows if r["arm"] == arm_label]
        if not arm_rows:
            continue
        n = len(arm_rows)

        def _mean(key: str) -> float:
            vals = [r[key] for r in arm_rows if key in r and r[key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        logger.info(
            f"{arm_label} (n={n}): edge_profit=${_mean('edge_profit'):.4f} "
            f"cloud_profit=${_mean('cloud_profit'):.4f} "
            f"total_spend=${_mean('total_user_spend'):.2f} "
            f"edge_surplus%={_mean('edge_surplus_pct_of_valuation'):.1f}"
        )


if __name__ == "__main__":
    main()
