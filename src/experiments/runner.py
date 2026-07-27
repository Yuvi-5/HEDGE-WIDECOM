"""Experiment runner: single-config run across multiple seeds with CSV output.

Usage:
    python src/experiments/runner.py --config configs/ablations/a2_kmax_sweep.yaml --seeds 2
    python src/experiments/runner.py --config configs/default.yaml --seeds 30 --output outputs/main/
"""

from __future__ import annotations

# Path bootstrap: ensure src/ is importable when this script is run directly.
import sys as _sys
from pathlib import Path as _Path

_src_dir = _Path(__file__).parent.parent
if str(_src_dir) not in _sys.path:
    _sys.path.insert(0, str(_src_dir))

import argparse
import copy
import csv
import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from hedge.simulation.engine import HEDGESimulationEngine

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config with recursive _base_ inheritance.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Merged config dict (base merged with overrides).
    """
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    base_key = raw.pop("_base_", None)
    if base_key is not None:
        base_path = (config_path.parent / base_key).resolve()
        base_cfg = load_config(base_path)
        raw = _deep_merge(base_cfg, raw)
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; override wins on scalar conflicts."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def expand_sweep(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Expand the 'sweep' section into per-variant configs.

    Args:
        config: Config dict possibly containing a 'sweep' key.

    Returns:
        Tuple of (variant_configs, sweep_values_per_variant, sweep_param_names).
        If no sweep key, returns ([config], [{}], []).
    """
    config = copy.deepcopy(config)
    sweep = config.pop("sweep", None)
    if sweep is None:
        return [config], [{}], []

    param_names = list(sweep.keys())
    variant_configs: list[dict[str, Any]] = [copy.deepcopy(config)]
    variant_sweep_vals: list[dict[str, Any]] = [{}]

    for param_name, values in sweep.items():
        expanded_cfgs: list[dict[str, Any]] = []
        expanded_vals: list[dict[str, Any]] = []
        for cfg, sv in zip(variant_configs, variant_sweep_vals):
            for val in values:
                c = copy.deepcopy(cfg)
                if not _set_nested_param(c, param_name, val):
                    raise ValueError(
                        f"Sweep axis '{param_name}' does not resolve to any config key. "
                        f"Use a dotted path (e.g. 'predictor.mode') to disambiguate a leaf "
                        f"key that appears in multiple sections, or check for a typo "
                        f"against the leaf key actually used in default.yaml / engine.py."
                    )
                expanded_cfgs.append(c)
                new_sv = copy.copy(sv)
                new_sv[param_name] = val
                expanded_vals.append(new_sv)
        variant_configs = expanded_cfgs
        variant_sweep_vals = expanded_vals

    return variant_configs, variant_sweep_vals, param_names


def _set_nested_param(config: dict[str, Any], param_name: str, value: Any) -> bool:
    """Set param_name in config to value; supports dotted paths and bare leaf keys.

    A dotted param_name (e.g. "predictor.mode") resolves an exact path from the
    config root, disambiguating leaf keys that appear under multiple sections
    (e.g. "mode" exists under both arrivals.mode and predictor.mode). A bare
    param_name is resolved via depth-first search for a key of that exact name,
    matching the first one found.

    Args:
        config: Config dict to search/modify in place.
        param_name: Key name (dotted path or bare leaf key) to find and set.
        value: Value to assign.

    Returns:
        True if the key was found and set; False otherwise.
    """
    if "." in param_name:
        parts = param_name.split(".")
        node: Any = config
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            return False
        node[parts[-1]] = value
        return True

    if param_name in config:
        config[param_name] = value
        return True
    for k, v in config.items():
        if isinstance(v, dict):
            if _set_nested_param(v, param_name, value):
                return True
    return False


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------


def run_single_seed(
    config: dict[str, Any],
    seed: int,
    sweep_vals: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    n_tasks: int | None = None,
) -> dict[str, Any]:
    """Run one simulation seed and return a flat metric row dict.

    Args:
        config: Experiment config dict.
        seed: Random seed for reproducibility.
        sweep_vals: Sweep parameter values for this variant (added to CSV row).
        output_dir: Per-seed output directory; None skips file I/O.
        n_tasks: Stop after this many tasks if set.

    Returns:
        Dict with metadata + all M1-M16 metric values.
    """
    if sweep_vals is None:
        sweep_vals = {}

    engine = HEDGESimulationEngine(config=config, seed=seed, output_dir=output_dir)
    metrics = engine.run(n_tasks=n_tasks)

    row: dict[str, Any] = {
        "experiment": config.get("label", "experiment"),
        "algorithm": str(config.get("experiment", {}).get("algorithm", "HEDGE")),
        "seed": seed,
        "N_nodes": int(config.get("topology", {}).get("N_nodes", 50)),
        "K_max": int(config.get("hedge", {}).get("K_max", 4)),
        "arrival_rate": float(config.get("arrivals", {}).get("lambda_bar", 1.0)),
    }
    row.update(sweep_vals)
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_metrics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write per-seed metric rows to CSV.

    Args:
        rows: List of flat metric row dicts.
        output_path: Destination CSV path.
    """
    if not rows:
        logger.warning("No rows to write to metrics CSV")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {output_path}")


def compute_and_write_wilcoxon(
    rows: list[dict[str, Any]],
    output_path: Path,
    sweep_param: str = "K_max",
    reference_value: Any = 0,
    compare_value: Any = 4,
    metric_key: str = "M6_rejection_rate",
) -> None:
    """Compute paired Wilcoxon test between two sweep values and write results.

    Args:
        rows: All metric rows from the sweep.
        output_path: Destination CSV for Wilcoxon results.
        sweep_param: Column name for the sweep parameter (e.g., 'K_max').
        reference_value: Baseline sweep value (e.g., 0 for K_max=0).
        compare_value: Comparison sweep value (e.g., 4 for K_max=4).
        metric_key: Metric column to test (e.g., 'M6_rejection_rate').
    """
    from metrics import wilcoxon_test

    ref_vals = [
        float(r[metric_key])
        for r in rows
        if str(r.get(sweep_param)) == str(reference_value) and metric_key in r
    ]
    cmp_vals = [
        float(r[metric_key])
        for r in rows
        if str(r.get(sweep_param)) == str(compare_value) and metric_key in r
    ]

    wtest_result: dict[str, Any]
    if len(ref_vals) < 2 or len(cmp_vals) < 2 or len(ref_vals) != len(cmp_vals):
        logger.warning(
            f"Wilcoxon skipped: n_ref={len(ref_vals)}, n_cmp={len(cmp_vals)} "
            f"(need equal, >= 2 samples)"
        )
        wtest_result = {"statistic": 0.0, "p_value": 1.0, "significant": False}
    else:
        try:
            wtest_result = wilcoxon_test(ref_vals, cmp_vals)
        except Exception as exc:
            logger.warning(f"Wilcoxon test raised: {exc}")
            wtest_result = {"statistic": 0.0, "p_value": 1.0, "significant": False}

    wilcoxon_rows = [
        {
            "sweep_param": sweep_param,
            "reference": reference_value,
            "comparison": compare_value,
            "metric": metric_key,
            "statistic": wtest_result["statistic"],
            "p_value": wtest_result["p_value"],
            "significant": wtest_result["significant"],
        }
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(wilcoxon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wilcoxon_rows)
    logger.info(f"Wrote Wilcoxon results to {output_path}")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_experiment(
    config_path: Path,
    n_seeds: int = 30,
    seed_start: int = 0,
    output_dir: Path | None = None,
    n_tasks: int | None = None,
) -> Path:
    """Run full experiment: load config, expand sweep, run all seeds, write outputs.

    Args:
        config_path: Path to YAML config (may have _base_ and sweep keys).
        n_seeds: Number of seeds to run per variant.
        seed_start: First seed value.
        output_dir: Root output directory; auto-named from label+timestamp if None.
        n_tasks: If set, limit each seed to this many tasks (for fast testing).

    Returns:
        Path to the written metrics.csv file.
    """
    config = load_config(config_path)
    label = str(config.get("label", "experiment"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = Path("outputs") / f"{label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved config for reproducibility
    clean_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.dump(clean_cfg, fh, default_flow_style=False)

    variant_configs, variant_sweep_vals, sweep_params = expand_sweep(config)
    logger.info(
        f"Running {len(variant_configs)} variant(s), {n_seeds} seed(s) each "
        f"(sweep params: {sweep_params})"
    )

    all_rows: list[dict[str, Any]] = []
    seeds = list(range(seed_start, seed_start + n_seeds))

    for i, (variant_cfg, sweep_vals) in enumerate(zip(variant_configs, variant_sweep_vals)):
        logger.info(f"Variant {i + 1}/{len(variant_configs)}: {sweep_vals}")
        variant_out = output_dir / f"variant_{i:02d}" if output_dir else None
        for seed in seeds:
            logger.info(f"  seed={seed} ...")
            seed_out = variant_out / f"seed_{seed:02d}" if variant_out else None
            row = run_single_seed(variant_cfg, seed, sweep_vals, seed_out, n_tasks)
            all_rows.append(row)

    metrics_path = output_dir / "metrics.csv"
    write_metrics_csv(all_rows, metrics_path)

    # Wilcoxon test between first and last sweep values on rejection rate
    if sweep_params and len(all_rows) > 0:
        param = sweep_params[0]
        unique_vals = sorted(
            {r[param] for r in all_rows if param in r},
            key=lambda x: (isinstance(x, str), x),
        )
        metric_key = "M6_rejection_rate"
        if all_rows and metric_key not in all_rows[0]:
            candidates = [k for k in all_rows[0] if k.startswith("M")]
            metric_key = candidates[0] if candidates else list(all_rows[0].keys())[-1]
        if len(unique_vals) >= 2:
            compute_and_write_wilcoxon(
                all_rows,
                output_dir / "wilcoxon.csv",
                sweep_param=param,
                reference_value=unique_vals[0],
                compare_value=unique_vals[-1],
                metric_key=metric_key,
            )

    # Generate figures (non-fatal if visualization dependencies missing)
    try:
        from visualization.generate_all import generate_all_figures

        generate_all_figures(output_dir, output_dir / "figures")
    except Exception as exc:
        logger.warning(f"Figure generation skipped: {exc}")

    return metrics_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: run experiment and write outputs."""
    parser = argparse.ArgumentParser(description="HEDGE experiment runner")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output directory (auto-named if omitted)"
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        dest="n_tasks",
        help="Limit tasks per seed (for fast testing)",
    )
    args = parser.parse_args()

    metrics_path = run_experiment(
        config_path=args.config,
        n_seeds=args.seeds,
        seed_start=args.seed_start,
        output_dir=args.output,
        n_tasks=args.n_tasks,
    )
    logger.info(f"Done. Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
