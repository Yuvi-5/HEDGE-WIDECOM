"""HEDGE-C main comparison: 6 arms x N seeds at the calibrated real-data
operating point, for the WIDECOM 2026 submission.

Arms: HEDGE_C (K_max=15, full mechanism minus carbon/SPA/Kalman/resale),
HEDGE_C_Kmax0 (vertical-only ablation, cloud-fallback-only comparison
anchor -- NOT a stand-in for TORS/B1, never labelled as such), B2_DDPS,
B4_CostOPD, B6_GreedyNLF, B7_CloudOnly.

Deviations from the generic experiments.runner path, both driven by this
script's own validation needs:
  - Base config is configs/hedge_c_base.yaml (not real_heavy.yaml
    directly), which carries the four HEDGE-C trim flags.
  - _run_one constructs HEDGESimulationEngine directly (rather than calling
    experiments.runner.run_single_seed) so the engine object -- and its
    in-memory event log, engine._mc.events, which is populated regardless
    of output.save_events -- stays reachable after engine.run() returns.
    This lets floor-violation / resale-leak counts be computed across the
    full campaign with zero extra runs.
  - Resumability checks a separate hedge_c_summary.json (not the engine's
    own summary.json, which only ever holds the 21 native metric keys and
    would silently drop the extra floor-check fields on a resumed run).
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

from loguru import logger  # noqa: E402

from experiments.runner import load_config, write_metrics_csv  # noqa: E402
from hedge.simulation.engine import HEDGESimulationEngine  # noqa: E402

N_SEEDS = 30
N_WORKERS = 10
OUTPUT_ROOT = ROOT / "outputs" / "hedge_c_comparison"
CSV_PATH = ROOT / "outputs" / "hedge_c_comparison.csv"
HEDGE_C_CONFIG = ROOT / "configs" / "hedge_c_base.yaml"

# Calibrated real-data operating point (NOT real_heavy.yaml's raw
# trace_target_lambda=16.0 default, which is stale/unvalidated -- see
# real_heavy.yaml's own header comment).
OPERATING_POINT = {
    "cloud_eligible_fraction": 0.12,
    "trace_target_lambda": 6.0,
    "n_hotspots": 1,
    "hotspot_rate_multiplier": 100.0,
}

ARMS: dict[str, dict] = {
    "HEDGE_C": {"experiment": {"algorithm": "HEDGE"}},
    "HEDGE_C_Kmax0": {"experiment": {"algorithm": "HEDGE"}, "hedge": {"K_max": 0}},
    "B2_DDPS": {"experiment": {"algorithm": "B2"}},
    "B4_CostOPD": {"experiment": {"algorithm": "B4"}},
    "B6_GreedyNLF": {"experiment": {"algorithm": "B6"}},
    "B7_CloudOnly": {"experiment": {"algorithm": "B7"}},
}


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def build_base_config() -> dict:
    """Load hedge_c_base.yaml, deep-merge the calibrated operating point,
    and disable per-event/snapshot file I/O for performance (the floor
    check reads engine._mc.events in memory, not from disk, so this
    doesn't cost us anything)."""
    cfg = load_config(HEDGE_C_CONFIG)
    cfg.setdefault("arrivals", {})
    _deep_update(cfg["arrivals"], OPERATING_POINT)
    cfg.setdefault("output", {})["save_events"] = False
    cfg["output"]["save_snapshots"] = False
    return cfg


def arm_config(base_cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    _deep_update(cfg, overrides)
    return cfg


def run_one_seed(
    arm_label: str,
    cfg: dict,
    seed: int,
    seed_dir: Path,
    force: bool = False,
    n_tasks: int | None = None,
) -> dict:
    """Run one seed for one arm, returning a flat row with metadata, the
    engine's 21 native metrics, and floor/resale-leak validation counts.

    Resumable off seed_dir/hedge_c_summary.json (own file, not the
    engine's native summary.json).
    """
    summary_path = seed_dir / "hedge_c_summary.json"
    if not force and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    engine = HEDGESimulationEngine(config=cfg, seed=seed, output_dir=seed_dir)
    metrics = engine.run(n_tasks=n_tasks)

    completed = [e for e in engine._mc.events if e.get("event_type") == "completed"]
    rejected = [e for e in engine._mc.events if e.get("event_type") == "rejected"]
    floor_violations = sum(
        1 for e in completed if float(e["price_paid"]) < float(e["C1_total"]) - 1e-9
    )
    is_resale_violations = sum(1 for e in completed if e.get("is_resale"))

    row: dict = {
        "experiment": cfg.get("label", "experiment"),
        "algorithm": str(cfg.get("experiment", {}).get("algorithm", "HEDGE")),
        "seed": seed,
        "N_nodes": int(cfg.get("topology", {}).get("N_nodes", 125)),
        "K_max": int(cfg.get("hedge", {}).get("K_max", 15)),
        "arrival_rate": float(cfg.get("arrivals", {}).get("trace_target_lambda", 0.0)),
        "arm": arm_label,
        "floor_violations": floor_violations,
        "is_resale_violations": is_resale_violations,
        "n_tasks_evaluated": len(completed) + len(rejected),
        "n_completed": len(completed),
        "n_rejected": len(rejected),
        # Persisted so Phase 4 can confirm the trim flags actually took effect
        # in the merged config, without re-reading YAML at aggregation time.
        "cfg_pi_co2": float(cfg.get("pricing", {}).get("pi_co2", -1.0)),
        "cfg_lambda_max": float(cfg.get("pricing", {}).get("lambda_max", -1.0)),
        "cfg_predictor_mode": str(cfg.get("predictor", {}).get("mode", "")),
    }
    row.update(metrics)

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
    parser = argparse.ArgumentParser(description="HEDGE-C 6-arm comparison campaign.")
    parser.add_argument(
        "--arm",
        choices=list(ARMS.keys()),
        default=None,
        help="Rerun only this arm and merge into the existing CSV. Omit to run all 6.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore resumability.")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS, dest="n_seeds")
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        dest="n_tasks",
        help="Cap tasks per seed (smoke tests / timing probes).",
    )
    args = parser.parse_args(argv)

    arms_to_run = {args.arm: ARMS[args.arm]} if args.arm else ARMS
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
    to_run = total_runs - already_done
    logger.info(
        f"HEDGE-C comparison: {total_runs} runs ({len(arms_to_run)} arms x "
        f"{args.n_seeds} seeds), {already_done} already done, {to_run} to run, "
        f"{N_WORKERS} workers, n_tasks={args.n_tasks}"
    )

    start = time.time()
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=N_WORKERS, verbose=10)(
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
    logger.info(f"All {total_runs} runs finished in {elapsed:.1f}s ({elapsed / max(to_run,1):.1f}s/new-run)")

    rows = list(results)
    floor_total = sum(int(r.get("floor_violations", 0)) for r in rows)
    resale_total = sum(int(r.get("is_resale_violations", 0)) for r in rows)
    logger.info(f"Floor violations across these {len(rows)} rows: {floor_total} (must be 0)")
    logger.info(f"Resale-leak violations across these {len(rows)} rows: {resale_total} (must be 0)")

    if args.arm:
        existing = _load_existing_csv_rows(CSV_PATH)
        kept = [r for r in existing if r.get("arm") != args.arm]
        merged = kept + [{k: str(v) for k, v in row.items()} for row in rows]
        write_metrics_csv(merged, CSV_PATH)
        logger.info(f"Merged {len(rows)} '{args.arm}' rows into {CSV_PATH} ({len(kept)} preserved).")
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
            f"{arm_label} (n={n}): "
            f"M1p95={_mean('M1_p95_latency_s'):.3f} "
            f"M2={_mean('M2_mean_cost_usd'):.6f} "
            f"M6={_mean('M6_rejection_rate'):.4f} "
            f"M9={_mean('M9_edge_service_fraction'):.4f} "
            f"floor_viol={sum(int(r.get('floor_violations', 0)) for r in arm_rows)}"
        )


if __name__ == "__main__":
    main()
