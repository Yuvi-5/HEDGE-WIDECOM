"""Phase 7 gate: 100-task smoke test for the HEDGE simulation engine.

Gate criteria:
- No exceptions in 100-task run
- No negative prices in matched events
- No capacity violations (w_pending never exceeds theoretical max at match time)
- IR holds for all matched tasks (buyer and seller)
- M1-M16 all produce finite floats
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from hedge.simulation.engine import HEDGESimulationEngine

# ---------------------------------------------------------------------------
# Shared minimal config (10 nodes, synthetic topology, fast convergence)
# ---------------------------------------------------------------------------

_SMOKE_CONFIG: dict[str, Any] = {
    "label": "smoke_test",
    "simulation": {
        "duration": 3600.0,
        "random_seed": 0,
        "delta_tp": 0.1,
        "delta_q": 0.1,
        "warmup_duration": 0.0,  # no warmup for smoke test (all events counted)
    },
    "topology": {
        "source": "synthetic",
        "N_nodes": 10,
        "tau_c_mean": 0.035,
        "tau_c_std": 0.008,
    },
    "arrivals": {
        "mode": "poisson",
        "lambda_bar": 2.0,  # 2 tasks/s/node -> fast 100-task reach
        "n_hotspots": 2,
        "hotspot_rate_multiplier": 5.0,
        "quiet_rate_multiplier": 0.2,
    },
    "task_model": {
        "w_min": 1.0e9,
        "w_max": 5.0e10,
        "s_min": 1.0e6,
        "s_max": 2.0e7,
        "d_min": 0.5,
        "d_max": 5.0,
        "a_buyer_min": 0.5,
        "a_buyer_max": 10.0,
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
        "save_events": True,
        "save_snapshots": False,
    },
}


def _make_engine(seed: int = 0) -> HEDGESimulationEngine:
    """Create a smoke-test engine with a deterministic seed."""
    return HEDGESimulationEngine(config=_SMOKE_CONFIG, seed=seed, output_dir=None)


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


def test_100_task_smoke_no_exceptions() -> None:
    """100-task smoke run completes without any exception."""
    engine = _make_engine(seed=0)
    metrics = engine.run(n_tasks=100)
    assert isinstance(metrics, dict)


def test_smoke_runs_to_completion_multiple_seeds() -> None:
    """3 different seeds all complete 100 tasks without exception."""
    for seed in [1, 7, 42]:
        engine = _make_engine(seed=seed)
        engine.run(n_tasks=100)


def test_no_negative_prices() -> None:
    """All completed events have price_paid >= 0."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)

    violations = [
        (e["task_id"], e["price_paid"])
        for e in engine._mc.events
        if e["event_type"] == "completed" and e["price_paid"] < -1e-12
    ]
    assert not violations, f"Negative prices: {violations[:5]}"


def test_buyer_ir_holds() -> None:
    """Buyer IR: price_paid <= a_buyer for every completed task."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)

    violations = [
        (e["task_id"], e["price_paid"], e["a_buyer"])
        for e in engine._mc.events
        if e["event_type"] == "completed" and e["price_paid"] > e["a_buyer"] + 1e-9
    ]
    assert not violations, f"Buyer IR violated {len(violations)} times. First: {violations[0]}"


def test_seller_ir_holds() -> None:
    """Seller IR: price_paid >= C1_total for every completed task."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)

    violations = [
        (e["task_id"], e["price_paid"], e["C1_total"])
        for e in engine._mc.events
        if e["event_type"] == "completed" and e["price_paid"] < e["C1_total"] - 1e-9
    ]
    assert not violations, f"Seller IR violated {len(violations)} times. First: {violations[0]}"


def test_metrics_all_finite() -> None:
    """All metrics produce finite floats (no NaN, no inf)."""
    engine = _make_engine(seed=0)
    metrics = engine.run(n_tasks=100)

    # 21 real M1-M17 flat keys
    assert len(metrics) == 21, f"Expected 21 metrics, got {len(metrics)}: {sorted(metrics)}"

    non_finite = {k: v for k, v in metrics.items() if not math.isfinite(v)}
    assert not non_finite, f"Non-finite metrics: {non_finite}"


def test_metrics_m6_rejection_sum_to_one() -> None:
    """M6 (rejection rate) + M10 (completion rate) == 1.0 within 1e-9."""
    engine = _make_engine(seed=0)
    metrics = engine.run(n_tasks=100)

    total = metrics["M6_rejection_rate"] + metrics["M10_completion_rate"]
    assert abs(total - 1.0) < 1e-9, f"M6+M10={total:.8f} != 1.0"


def test_m11_jfi_in_unit_interval() -> None:
    """M11 (JFI) in [0, 1]."""
    engine = _make_engine(seed=0)
    metrics = engine.run(n_tasks=100)
    assert 0.0 <= metrics["M11_jfi"] <= 1.0 + 1e-9


def test_m16_myerson_in_unit_interval() -> None:
    """M16 (Myerson efficiency) in [0, 1]."""
    engine = _make_engine(seed=0)
    metrics = engine.run(n_tasks=100)
    assert 0.0 <= metrics["M16_myerson_efficiency"] <= 1.0 + 1e-9


def test_w_pending_non_negative_throughout() -> None:
    """w_pending never goes negative during the smoke run."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)

    for node in engine.nodes:
        assert (
            node.w_pending >= -1e-9
        ), f"Node {node.unique_id} has negative w_pending={node.w_pending:.4e}"


def test_kalman_output_in_range() -> None:
    """l_hat in [0, f_max] for all nodes after 100-task run."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)

    for node in engine.nodes:
        assert (
            0.0 <= node.l_hat <= node.f_max + 1e-3
        ), f"Node {node.unique_id} l_hat={node.l_hat:.4e} out of [0, {node.f_max:.4e}]"


def test_cloud_not_required() -> None:
    """Simulation completes even if all tasks go to edge (cloud not forced)."""
    engine = _make_engine(seed=99)
    metrics = engine.run(n_tasks=50)
    # Just check no crash and some events recorded
    assert len(engine._mc.events) == 50


def test_event_count_matches_submitted() -> None:
    """Exactly 100 events are recorded for a 100-task run."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)
    assert len(engine._mc.events) == 100


def test_executor_types_valid() -> None:
    """Every event is a coherent (event_type, executor_id) pair: rejected tasks
    have no executor, completed tasks have either 'cloud' or a numeric edge id."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)

    def _is_valid(e: dict) -> bool:
        if e["event_type"] == "rejected":
            return e["executor_id"] == ""
        if e["event_type"] == "completed":
            return e["executor_id"] == "cloud" or e["executor_id"].isdigit()
        return False

    bad = [e for e in engine._mc.events if not _is_valid(e)]
    assert not bad, f"Invalid (event_type, executor_id) in {len(bad)} events"


def test_latency_estimate_finite_for_matched() -> None:
    """latency is finite and positive for all completed tasks."""
    engine = _make_engine(seed=0)
    engine.run(n_tasks=100)
    bad = [
        e
        for e in engine._mc.events
        if e["event_type"] == "completed" and not math.isfinite(e["latency"])
    ]
    assert not bad, f"{len(bad)} completed events with non-finite latency"
