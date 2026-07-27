"""Compute the mean coverage-set size |C_u| for the N=125 real EUA Melbourne
topology at coverage_radius_km=0.25 (the value used throughout HEDGE-C).

No simulation loop needed -- topology construction alone determines
coverage_peers, and at N_nodes=125 (the full EUA dataset) no rng-dependent
subsampling occurs, so this is seed-invariant (verified: eua_loader only
subsamples when len(coords) > n_nodes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loguru import logger  # noqa: E402

from experiments.runner import load_config  # noqa: E402
from hedge.core.topology import create_topology  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "hedge_c_base.yaml")
    rng = np.random.default_rng(0)
    nodes, tau, cloud = create_topology(cfg, rng)

    expected_n = int(cfg.get("topology", {}).get("N_nodes", 125))
    if len(nodes) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} nodes (full EUA dataset, no subsampling), got "
            f"{len(nodes)} -- subsampling occurred, coverage-set size would not be "
            f"seed-invariant. Investigate before trusting this statistic."
        )

    sizes = [len(n.coverage_peers) for n in nodes]
    result = {
        "n_nodes": len(nodes),
        "coverage_radius_km": cfg.get("topology", {}).get("coverage_radius_km"),
        "mean_coverage_set_size": float(np.mean(sizes)),
        "min_coverage_set_size": int(np.min(sizes)),
        "max_coverage_set_size": int(np.max(sizes)),
        "median_coverage_set_size": float(np.median(sizes)),
    }

    out_path = ROOT / "outputs" / "coverage_stat.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    logger.info(f"Coverage stat: {result}")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
