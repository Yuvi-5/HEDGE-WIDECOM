"""Phase 1 gate: verify each of HEDGE + the evaluated baselines runs end-to-end without error.

Coverage:
- Every algorithm in BASELINE_REGISTRY (plus HEDGE) produces a finite metrics dict.
- B7 (CloudOnly) produces zero edge wins.
- Non-degenerate metrics: at least some tasks complete for most algorithms.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from hedge.simulation.engine import HEDGESimulationEngine

# ---------------------------------------------------------------------------
# Shared config: 8 nodes, fast arrival, short tasks
# ---------------------------------------------------------------------------

_BASE_CFG: dict[str, Any] = {
    "label": "algo_test",
    "simulation": {
        "duration": 3600.0,
        "random_seed": 0,
        "delta_tp": 0.1,
        "delta_q": 0.1,
        "warmup_duration": 0.0,
    },
    "topology": {
        "source": "synthetic",
        "N_nodes": 8,
        "tau_c_mean": 0.035,
        "tau_c_std": 0.005,
    },
    "arrivals": {
        "mode": "poisson",
        "lambda_bar": 3.0,
        "n_hotspots": 1,
        "hotspot_rate_multiplier": 3.0,
        "quiet_rate_multiplier": 0.3,
    },
    "task_model": {
        "w_min": 2.0e8,
        "w_max": 2.0e9,  # small tasks: at 1GHz, max execution = 2s
        "s_min": 5.0e5,
        "s_max": 1.0e7,
        "d_min": 1.0,  # wide deadline to allow edge service
        "d_max": 10.0,
        "a_buyer_min": 1.0,
        "a_buyer_max": 20.0,
    },
    "hedge": {
        "K_max": 4,
        "H_max": 2,
        "tau_lock": 0.02,
        "theta_commit": 0.02,
        "W_fail": 20,
        "theta_risk": 0.12,
        "epsilon_trans": 0.05,
        "x_weight": 0.5,
        "y_weight": 0.5,
        "alpha_u": 0.5,
        "gamma_u": 0.5,
    },
    "pricing": {
        "a_ref": 5.0e-10,
        "b_shape": 1.0,
        "eta": 1.0,
        "lambda_max": 0.5,
        "R_ref": 1.0,
        "alpha_R": 0.2,
        "pi_co2": 5.0e-5,
        "mu_c": 1.5,
    },
    "cloud": {
        "f_max": 1.0e12,
        "kappa": 1.0e-27,
        "rho": 1.0e-4,
        "beta": 80.0,
        "pi_E": 0.06,
        "P_idle": 0.0,
    },
    "predictor": {
        "mode": "instantaneous",
    },
    "market": {
        "enable_admission_gate": True,
    },
    "output": {
        "base_dir": "outputs",
        "log_level": "WARNING",
    },
}

_N_TASKS = 200


def _cfg_for(algorithm: str) -> dict[str, Any]:
    cfg = {k: v for k, v in _BASE_CFG.items()}
    cfg["experiment"] = {"algorithm": algorithm}
    return cfg


def _make_engine(algorithm: str, seed: int = 0) -> HEDGESimulationEngine:
    return HEDGESimulationEngine(config=_cfg_for(algorithm), seed=seed, output_dir=None)


# ---------------------------------------------------------------------------
# Parametrized smoke: all algorithms finish without exception
# ---------------------------------------------------------------------------

_ALL_ALGORITHMS = ["HEDGE", "B2", "B4", "B6", "B7"]


@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_algorithm_runs_without_exception(algorithm: str) -> None:
    """Each algorithm completes a 200-task run without exception."""
    engine = _make_engine(algorithm, seed=1)
    metrics = engine.run(n_tasks=_N_TASKS)
    assert isinstance(metrics, dict)
    assert len(engine._mc.events) == _N_TASKS


@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_algorithm_metrics_all_finite(algorithm: str) -> None:
    """All metric values are finite floats for every algorithm."""
    engine = _make_engine(algorithm, seed=2)
    metrics = engine.run(n_tasks=_N_TASKS)
    non_finite = {k: v for k, v in metrics.items() if not math.isfinite(v)}
    assert not non_finite, f"Algorithm {algorithm}: non-finite metrics {non_finite}"


# ---------------------------------------------------------------------------
# Algorithm-specific semantic checks
# ---------------------------------------------------------------------------


def test_b7_cloud_only_no_edge_wins() -> None:
    """B7 (CloudOnly) never assigns any task to an edge node."""
    engine = _make_engine("B7", seed=3)
    engine.run(n_tasks=_N_TASKS)
    edge_events = [
        e for e in engine._mc.events if e["event_type"] == "completed" and e["executor_id"] != "cloud"
    ]
    assert len(edge_events) == 0, f"B7 produced {len(edge_events)} edge wins (should be 0)"


def test_hedge_produces_some_edge_wins() -> None:
    """HEDGE should route at least some tasks to edge nodes (not all cloud or rejected)."""
    engine = _make_engine("HEDGE", seed=0)
    engine.run(n_tasks=_N_TASKS)
    edge_events = [
        e for e in engine._mc.events if e["event_type"] == "completed" and e["executor_id"] != "cloud"
    ]
    # With 8 nodes and moderate load, we expect several edge wins
    assert len(edge_events) > 0, "HEDGE produced no edge wins at all"


@pytest.mark.parametrize("algorithm", _ALL_ALGORITHMS)
def test_event_count_exact(algorithm: str) -> None:
    """Exactly N events are recorded for N-task runs."""
    engine = _make_engine(algorithm, seed=6)
    engine.run(n_tasks=_N_TASKS)
    assert len(engine._mc.events) == _N_TASKS


def test_hedge_kmax_zero_with_phase0_still_cloud_only() -> None:
    """I9 at the full-engine level: with Phase 0 wired in (market.enable_phase0
    defaults True), HEDGE(K_max=0) must still produce zero edge wins. The final
    peer pool is ([orchestrator] + P0)[:k_max], so k_max=0 must yield an empty
    pool regardless of Phase 0 curation, exactly as the pre-Phase0 placeholder
    (peer_pool[:k_max] on the full node list) did."""
    cfg = _cfg_for("HEDGE")
    cfg["hedge"] = {**cfg["hedge"], "K_max": 0}
    engine = HEDGESimulationEngine(config=cfg, seed=7, output_dir=None)
    engine.run(n_tasks=_N_TASKS)
    edge_events = [
        e for e in engine._mc.events if e["event_type"] == "completed" and e["executor_id"] != "cloud"
    ]
    assert len(edge_events) == 0, (
        f"HEDGE(K_max=0) with Phase 0 enabled produced {len(edge_events)} edge wins "
        "(Proposition 2 / I9 violated: HEDGE(K_max=0) should degenerate to cloud-only)"
    )
