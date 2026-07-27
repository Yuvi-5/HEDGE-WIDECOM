"""Single entry point for the Phase 3 detached launch: runs the main 6-arm
comparison, then the 2-arm Phase-0 ablation, sequentially (not concurrently
-- two simultaneous joblib pools of 10 workers each would oversubscribe
CPU). Both scripts are independently resumable, so a relaunch after this
process is killed (e.g. by session-level cycling) just continues from
whatever seeds already have a hedge_c_summary.json.

Usage (matches the two underlying scripts' CLI):
    python run_all_campaigns.py                    # full 30-seed run, both campaigns
    python run_all_campaigns.py --n-seeds 10        # seed-cut, applies to both
    python run_all_campaigns.py --n-tasks 100 --n-seeds 1   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402

import run_hedge_c_comparison  # noqa: E402
import run_phase0_ablation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both HEDGE-C campaigns sequentially.")
    parser.add_argument("--n-seeds", type=int, default=None, dest="n_seeds")
    parser.add_argument("--n-tasks", type=int, default=None, dest="n_tasks")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    passthrough: list[str] = []
    if args.n_seeds is not None:
        passthrough += ["--n-seeds", str(args.n_seeds)]
    if args.n_tasks is not None:
        passthrough += ["--n-tasks", str(args.n_tasks)]
    if args.force:
        passthrough += ["--force"]

    t0 = time.time()
    logger.info("=== Starting main 6-arm HEDGE-C comparison ===")
    run_hedge_c_comparison.main(passthrough)
    t1 = time.time()
    logger.info(f"=== Main comparison finished in {t1 - t0:.1f}s ===")

    logger.info("=== Starting Phase-0 on/off ablation ===")
    run_phase0_ablation.main(passthrough)
    t2 = time.time()
    logger.info(f"=== Phase-0 ablation finished in {t2 - t1:.1f}s ===")

    logger.info(f"=== All campaigns finished. Total wall time: {t2 - t0:.1f}s ===")


if __name__ == "__main__":
    main()
