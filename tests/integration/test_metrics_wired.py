"""Phase 1 gate: verify the real MetricsCollector is wired into the engine.

Coverage:
- Engine output keys match the MetricsCollector M1-M16 documented keys.
- M7 carbon > 0 when pi_co2 > 0 and tasks complete.
- M5 market revenue > 0 when edge tasks complete.
- M2 mean cost > 0 when tasks complete.
- Metric keys use METRICS.md naming (not legacy collector stub names).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from hedge.simulation.engine import HEDGESimulationEngine

_CFG: dict[str, Any] = {
    "label": "metrics_wired_test",
    "simulation": {
        "duration": 3600.0,
        "random_seed": 0,
        "delta_tp": 0.1,
        "delta_q": 0.1,
        "warmup_duration": 0.0,
    },
    "topology": {
        "source": "synthetic",
        "N_nodes": 6,
        "tau_c_mean": 0.03,
        "tau_c_std": 0.005,
    },
    "arrivals": {
        "mode": "poisson",
        "lambda_bar": 2.0,
        "n_hotspots": 1,
        "hotspot_rate_multiplier": 4.0,
        "quiet_rate_multiplier": 0.3,
    },
    "task_model": {
        "w_min": 5.0e8,
        "w_max": 1.0e10,
        "s_min": 5.0e5,
        "s_max": 5.0e6,
        "d_min": 0.5,
        "d_max": 4.0,
        "a_buyer_min": 0.5,
        "a_buyer_max": 15.0,
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
        "pi_co2": 5.0e-5,  # nonzero to ensure M7 > 0
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
    "experiment": {"algorithm": "HEDGE"},
    "output": {
        "base_dir": "outputs",
        "log_level": "WARNING",
    },
}

_EXPECTED_REAL_KEYS = {
    "M1_p50_latency_s",
    "M1_p95_latency_s",
    "M1_p99_latency_s",
    "M1_mean_latency_s",
    "M2_mean_cost_usd",
    "M3_system_utilisation",
    "M4_utilisation_variance",
    "M5_market_revenue_usd",
    "M6_rejection_rate",
    "M7_carbon_g_per_task",
    "M8_quote_variance",
    "M9_edge_service_fraction",
    "M10_completion_rate",
    "M11_jfi",
    "M12_tsfr",
    "M13_err",
    "M14_rmse_load",
    "M14_rmse_revenue",
    "M15_energy_joules_per_task",
    "M16_myerson_efficiency",
    "M17_primary_offload_fraction",
}


def _run(n_tasks: int = 300, seed: int = 0) -> dict[str, float]:
    engine = HEDGESimulationEngine(config=_CFG, seed=seed, output_dir=None)
    return engine.run(n_tasks=n_tasks)


def test_real_metric_keys_present() -> None:
    """All 21 real MetricsCollector keys are present in engine output."""
    metrics = _run()
    missing = _EXPECTED_REAL_KEYS - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"


def test_no_metrics_md_naming_mismatch() -> None:
    """Engine does NOT expose stub M5_cloud_rate or M7_cloud_rate (wrong METRICS.md numbering)."""
    metrics = _run()
    assert "M7_cloud_rate" not in metrics, "M7_cloud_rate is stub naming, should not appear"
    assert "M13_edge_latency_s" not in metrics, "M13_edge_latency_s is stub naming"


def test_m7_carbon_positive_when_tasks_complete() -> None:
    """M7 (carbon per completed task) is > 0 when pi_co2 > 0 and tasks complete."""
    metrics = _run(n_tasks=300)
    completed = metrics.get("M10_completion_rate", 0.0)
    if completed > 0:
        assert metrics["M7_carbon_g_per_task"] > 0.0, (
            f"M7 carbon should be >0 when pi_co2=5e-5 and tasks complete, "
            f"got {metrics['M7_carbon_g_per_task']}"
        )


def test_m5_market_revenue_positive_when_edge_wins() -> None:
    """M5 (market revenue) is > 0 when edge tasks complete and generate revenue."""
    metrics = _run(n_tasks=300)
    edge_fraction = metrics.get("M9_edge_service_fraction", 0.0)
    if edge_fraction > 0 and metrics.get("M10_completion_rate", 0.0) > 0:
        assert metrics["M5_market_revenue_usd"] > 0.0, (
            f"M5 market revenue should be positive when edge_fraction={edge_fraction:.3f}, "
            f"got {metrics['M5_market_revenue_usd']}"
        )


def test_m2_mean_cost_positive_when_tasks_complete() -> None:
    """M2 (mean cost per task) is > 0 when tasks complete."""
    metrics = _run(n_tasks=300)
    if metrics.get("M10_completion_rate", 0.0) > 0:
        assert (
            metrics["M2_mean_cost_usd"] > 0.0
        ), f"M2 mean cost should be positive, got {metrics['M2_mean_cost_usd']}"


def test_rejection_plus_completion_rate_sum_to_one() -> None:
    """M6 (rejection) + M10 (completion) = 1.0 within tolerance."""
    metrics = _run(n_tasks=300)
    total = metrics["M6_rejection_rate"] + metrics["M10_completion_rate"]
    assert abs(total - 1.0) < 1e-6, f"M6+M10={total:.8f} != 1.0"


def test_m11_jfi_range() -> None:
    """M11 (Jain's Fairness Index) is in [0, 1]."""
    metrics = _run(n_tasks=300)
    assert 0.0 <= metrics["M11_jfi"] <= 1.0 + 1e-9


def test_m16_myerson_range() -> None:
    """M16 (Myerson efficiency) is in [0, 1]."""
    metrics = _run(n_tasks=300)
    assert 0.0 <= metrics["M16_myerson_efficiency"] <= 1.0 + 1e-9


def test_energy_positive_when_tasks_complete() -> None:
    """M15 (fleet energy per task) is > 0 when tasks complete."""
    metrics = _run(n_tasks=300)
    if metrics.get("M10_completion_rate", 0.0) > 0:
        assert (
            metrics["M15_energy_joules_per_task"] > 0.0
        ), f"M15 energy should be positive, got {metrics['M15_energy_joules_per_task']}"


