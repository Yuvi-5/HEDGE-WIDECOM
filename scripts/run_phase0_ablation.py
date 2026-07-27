"""HEDGE-C Phase-0 on/off ablation: 2 arms x N seeds, same operating point
and base config as the main comparison, but only toggling market.enable_phase0.

Reuses run_hedge_c_comparison.py's helpers (same directory import) rather
than duplicating the campaign scaffolding.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

from run_hedge_c_comparison import (  # noqa: E402
    N_SEEDS,
    N_WORKERS,
    ROOT,
    arm_config,
    build_base_config,
    run_one_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.runner import write_metrics_csv  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "phase0_ablation"
CSV_PATH = ROOT / "outputs" / "phase0_ablation.csv"

PHASE0_ARMS: dict[str, dict] = {
    "Phase0_On": {"market": {"enable_phase0": True}},
    "Phase0_Off": {"market": {"enable_phase0": False}},
}


def _load_existing_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HEDGE-C Phase-0 on/off ablation.")
    parser.add_argument("--arm", choices=list(PHASE0_ARMS.keys()), default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS, dest="n_seeds")
    parser.add_argument("--n-tasks", type=int, default=None, dest="n_tasks")
    parser.add_argument("--workers", type=int, default=N_WORKERS, dest="workers")
    args = parser.parse_args(argv)

    arms_to_run = {args.arm: PHASE0_ARMS[args.arm]} if args.arm else PHASE0_ARMS
    base_cfg = build_base_config()

    jobs: list[tuple[str, dict, int, Path]] = []
    already_done = 0
    for arm_label, overrides in arms_to_run.items():
        cfg = arm_config(base_cfg, overrides)
        for seed in range(args.n_seeds):
            seed_dir = OUTPUT_ROOT / arm_label / f"seed_{seed:02d}"
            if not args.force and (seed_dir / "hedge_c_summary.json").exists():
                already_done += 1
            jobs.append((arm_label, cfg, seed, seed_dir))

    total_runs = len(jobs)
    logger.info(
        f"Phase-0 ablation: {total_runs} runs, {already_done} already done, "
        f"{total_runs - already_done} to run, {args.workers} workers, n_tasks={args.n_tasks}"
    )

    start = time.time()
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.workers, verbose=10)(
            delayed(run_one_seed)(arm_label, cfg, seed, seed_dir, args.force, args.n_tasks)
            for arm_label, cfg, seed, seed_dir in jobs
        )
    except ImportError:
        logger.warning("joblib not available, running single-threaded")
        results = [
            run_one_seed(arm_label, cfg, seed, seed_dir, args.force, args.n_tasks)
            for arm_label, cfg, seed, seed_dir in jobs
        ]

    elapsed = time.time() - start
    logger.info(f"All {total_runs} runs finished in {elapsed:.1f}s")

    rows = list(results)
    floor_total = sum(int(r.get("floor_violations", 0)) for r in rows)
    logger.info(f"Floor violations across these {len(rows)} rows: {floor_total} (must be 0)")

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
        rej = sum(r["M6_rejection_rate"] for r in arm_rows) / n
        logger.info(f"{arm_label} (n={n}): rejection={rej:.4f}")


if __name__ == "__main__":
    main()
