"""Gate tests for Phase 10: Metrics M1-M16.

Gate criteria:
- All 16 metrics produce finite floats on 100-task synthetic run.
- M11 JFI in [0, 1].
- M16 Myerson efficiency in [0, 1].
- M6 rejection rate + M10 completion rate == 1.0 (within 1e-10).
- M12 TSFR == 0 outside burst window for Poisson steady-state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metrics import (
    MetricsCollector,
    bootstrap_ci,
    compute_carbon_per_task,
    compute_completion_rate,
    compute_err,
    compute_fleet_energy_per_task,
    compute_jains_fairness,
    compute_latency_metrics,
    compute_market_revenue,
    compute_mean_cost,
    compute_myerson_efficiency,
    compute_prediction_rmse,
    compute_primary_offload_fraction,
    compute_quote_acceptance_rate,
    compute_rejection_rate,
    compute_stackelberg_convergence,
    compute_tsfr,
    compute_utilisation_variance,
    wilcoxon_test,
)

# ---------------------------------------------------------------------------
# Synthetic event/snapshot generators
# ---------------------------------------------------------------------------


def _make_events(
    rng: np.random.Generator,
    n_total: int = 100,
    rejection_frac: float = 0.2,
    cloud_frac: float = 0.3,
    resale_frac: float = 0.1,
) -> list[dict[str, Any]]:
    """Generate synthetic task events with all required fields."""
    events: list[dict[str, Any]] = []
    n_rejected = int(n_total * rejection_frac)
    n_completed = n_total - n_rejected
    n_cloud = int(n_completed * cloud_frac)
    n_resale = int(n_completed * resale_frac)

    for i in range(n_completed):
        is_cloud = i < n_cloud
        is_resale = (not is_cloud) and (i < n_resale + n_cloud) and (i >= n_cloud)
        executor_id = "cloud" if is_cloud else f"node_{i % 5}"
        latency = float(rng.uniform(0.01, 0.09))
        price = float(rng.uniform(1e-3, 5e-3))
        events.append(
            {
                "event_type": "completed",
                "latency": latency,
                "price_paid": price,
                "executor_id": executor_id,
                "carbon_g": float(rng.uniform(0.01, 0.5)),
                "energy_joules": float(rng.uniform(0.1, 2.0)),
                "is_resale": is_resale,
                "arrival_time": float(i) * 10.0,
                "deadline": 0.1,
            }
        )

    for j in range(n_rejected):
        events.append(
            {
                "event_type": "rejected",
                "latency": float("inf"),
                "price_paid": 0.0,
                "executor_id": "",
                "carbon_g": 0.0,
                "energy_joules": 0.0,
                "is_resale": False,
                "arrival_time": float(j + n_completed) * 10.0,
                "deadline": 0.1,
            }
        )
    return events


def _make_node_snapshots(
    rng: np.random.Generator,
    n_nodes: int = 5,
    n_ticks: int = 100,
) -> list[dict[str, Any]]:
    """Generate synthetic per-node state snapshots."""
    f_maxes = [float(rng.uniform(1e9, 4e9)) for _ in range(n_nodes)]
    snapshots: list[dict[str, Any]] = []
    for tick in range(n_ticks):
        for ni, f_max in enumerate(f_maxes):
            l_current = float(rng.uniform(0.1, 0.8) * f_max)
            l_hat = l_current * float(rng.uniform(0.9, 1.1))
            r_hat = float(rng.uniform(0.5, 2.0))
            snapshots.append(
                {
                    "time": float(tick) * 0.1,
                    "node_id": f"node_{ni}",
                    "l_current": l_current,
                    "f_max": f_max,
                    "l_hat": min(l_hat, f_max),
                    "R_hat": r_hat,
                    "R_realised": r_hat * float(rng.uniform(0.8, 1.2)),
                }
            )
    return snapshots


def _make_quote_history(
    rng: np.random.Generator,
    n_nodes: int = 5,
    n_rounds: int = 200,
) -> list[dict[str, Any]]:
    """Generate synthetic quote broadcast history."""
    history: list[dict[str, Any]] = []
    for t_idx in range(n_rounds):
        for ni in range(n_nodes):
            history.append(
                {
                    "time": float(t_idx) * 0.1,
                    "node_id": f"node_{ni}",
                    "p_dagger_ref": float(rng.uniform(1e-12, 2e-11)),
                }
            )
    return history


@pytest.fixture(scope="module")
def synthetic_data() -> dict[str, Any]:
    """100-task synthetic dataset for metric gate tests."""
    rng = np.random.default_rng(42)
    events = _make_events(rng, n_total=100, rejection_frac=0.2)
    node_snapshots = _make_node_snapshots(rng, n_nodes=5, n_ticks=100)
    quote_history = _make_quote_history(rng, n_nodes=5, n_rounds=200)
    return {
        "events": events,
        "node_snapshots": node_snapshots,
        "quote_history": quote_history,
    }


# ---------------------------------------------------------------------------
# M1: End-to-End Latency
# ---------------------------------------------------------------------------


def test_m1_latency_all_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_latency_metrics(synthetic_data["events"])
    for key in ("p50", "p95", "p99", "mean"):
        assert key in result
        assert math.isfinite(result[key]), f"M1 {key} is not finite"
        assert result[key] >= 0.0


def test_m1_latency_empty_returns_zeros() -> None:
    result = compute_latency_metrics([])
    assert result == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}


def test_m1_latency_percentile_order(synthetic_data: dict[str, Any]) -> None:
    result = compute_latency_metrics(synthetic_data["events"])
    assert result["p50"] <= result["p95"] <= result["p99"]


# ---------------------------------------------------------------------------
# M2: Mean Cost
# ---------------------------------------------------------------------------


def test_m2_cost_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_mean_cost(synthetic_data["events"])
    assert math.isfinite(result)
    assert result > 0.0


def test_m2_cost_empty_returns_zero() -> None:
    assert compute_mean_cost([]) == 0.0


# ---------------------------------------------------------------------------
# M3: System Utilisation (via MetricsCollector._compute_m3)
# ---------------------------------------------------------------------------


def test_m3_utilisation_finite(synthetic_data: dict[str, Any]) -> None:
    collector = MetricsCollector()
    for snap in synthetic_data["node_snapshots"]:
        collector.record_snapshot(snap)
    m3 = collector._compute_m3()
    assert math.isfinite(m3)
    assert 0.0 <= m3 <= 1.0


def test_m3_utilisation_empty_is_zero() -> None:
    collector = MetricsCollector()
    assert collector._compute_m3() == 0.0


# ---------------------------------------------------------------------------
# M4: Utilisation Variance
# ---------------------------------------------------------------------------


def test_m4_variance_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_utilisation_variance(synthetic_data["node_snapshots"])
    assert math.isfinite(result)
    assert result >= 0.0


def test_m4_variance_empty_is_zero() -> None:
    assert compute_utilisation_variance([]) == 0.0


# ---------------------------------------------------------------------------
# M5: Market Revenue
# ---------------------------------------------------------------------------


def test_m5_revenue_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_market_revenue(synthetic_data["events"])
    assert math.isfinite(result)
    assert result >= 0.0


def test_m5_revenue_excludes_cloud(synthetic_data: dict[str, Any]) -> None:
    cloud_only = [{"event_type": "completed", "price_paid": 5.0, "executor_id": "cloud"}]
    assert compute_market_revenue(cloud_only) == 0.0


# ---------------------------------------------------------------------------
# M6 + M10: Rejection Rate + Completion Rate == 1.0 (KEY GATE)
# ---------------------------------------------------------------------------


def test_m6_plus_m10_equals_one(synthetic_data: dict[str, Any]) -> None:
    rejection = compute_rejection_rate(synthetic_data["events"])
    completion = compute_completion_rate(synthetic_data["events"])
    assert (
        abs(rejection + completion - 1.0) < 1e-10
    ), f"M6 ({rejection:.6f}) + M10 ({completion:.6f}) != 1.0"


def test_m6_rejection_in_range(synthetic_data: dict[str, Any]) -> None:
    result = compute_rejection_rate(synthetic_data["events"])
    assert 0.0 <= result <= 1.0


def test_m6_rejection_empty_is_zero() -> None:
    assert compute_rejection_rate([]) == 0.0


def test_m10_completion_in_range(synthetic_data: dict[str, Any]) -> None:
    result = compute_completion_rate(synthetic_data["events"])
    assert 0.0 <= result <= 1.0


def test_m6_all_rejected() -> None:
    events = [{"event_type": "rejected"} for _ in range(10)]
    assert compute_rejection_rate(events) == 1.0
    assert compute_completion_rate(events) == 0.0


def test_m6_all_completed() -> None:
    events = [
        {"event_type": "completed", "price_paid": 1e-3, "executor_id": "node_0"} for _ in range(10)
    ]
    assert compute_rejection_rate(events) == 0.0
    assert compute_completion_rate(events) == 1.0


# ---------------------------------------------------------------------------
# M7: Carbon
# ---------------------------------------------------------------------------


def test_m7_carbon_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_carbon_per_task(synthetic_data["events"])
    assert math.isfinite(result)
    assert result >= 0.0


def test_m7_carbon_empty_is_zero() -> None:
    assert compute_carbon_per_task([]) == 0.0


# ---------------------------------------------------------------------------
# M8: Stackelberg Convergence
# ---------------------------------------------------------------------------


def test_m8_convergence_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_stackelberg_convergence(synthetic_data["quote_history"])
    assert "quote_variance" in result
    assert "converged" in result
    assert math.isfinite(float(result["quote_variance"]))
    assert result["quote_variance"] >= 0.0


def test_m8_convergence_empty_is_converged() -> None:
    result = compute_stackelberg_convergence([])
    assert result["quote_variance"] == 0.0
    assert result["converged"] is True


def test_m8_converged_on_constant_quotes() -> None:
    history = [{"time": float(i) * 0.1, "p_dagger_ref": 1e-11} for i in range(100)]
    result = compute_stackelberg_convergence(history)
    # Floating-point variance of equal values may be a tiny non-zero epsilon
    assert result["quote_variance"] < 1e-30
    assert result["converged"] is True


# ---------------------------------------------------------------------------
# M9: Quote Acceptance Rate
# ---------------------------------------------------------------------------


def test_m9_acceptance_rate_in_range(synthetic_data: dict[str, Any]) -> None:
    result = compute_quote_acceptance_rate(synthetic_data["events"])
    assert 0.0 <= result <= 1.0


def test_m9_all_cloud_is_zero() -> None:
    events = [{"event_type": "completed", "executor_id": "cloud"} for _ in range(5)]
    assert compute_quote_acceptance_rate(events) == 0.0


def test_m9_no_cloud_is_one() -> None:
    events = [{"event_type": "completed", "executor_id": "node_0"} for _ in range(5)]
    assert compute_quote_acceptance_rate(events) == 1.0


def test_m17_all_same_node_is_zero() -> None:
    """M17: every task served at its own home node -> 0 offload."""
    events = [
        {"event_type": "completed", "executor_id": "node_0", "home_node_id": "node_0"}
        for _ in range(5)
    ]
    assert compute_primary_offload_fraction(events) == 0.0


def test_m17_all_offloaded_is_one() -> None:
    """M17: every task served at a different edge node -> 1.0 offload."""
    events = [
        {"event_type": "completed", "executor_id": "node_1", "home_node_id": "node_0"}
        for _ in range(5)
    ]
    assert compute_primary_offload_fraction(events) == 1.0


def test_m17_cloud_execution_not_counted_as_offload() -> None:
    """M17: cloud execution is excluded (not a peer edge node)."""
    events = [
        {"event_type": "completed", "executor_id": "cloud", "home_node_id": "node_0"}
        for _ in range(5)
    ]
    assert compute_primary_offload_fraction(events) == 0.0


def test_m17_rejected_tasks_excluded() -> None:
    """M17: rejected tasks (no execution) don't count toward the denominator."""
    events = [{"event_type": "rejected", "executor_id": "", "home_node_id": "node_0"}]
    assert compute_primary_offload_fraction(events) == 0.0


def test_m17_mixed_fraction() -> None:
    """M17: mixed same-node/offloaded/cloud/rejected events give the right ratio."""
    events = [
        {"event_type": "completed", "executor_id": "node_0", "home_node_id": "node_0"},
        {"event_type": "completed", "executor_id": "node_1", "home_node_id": "node_0"},
        {"event_type": "completed", "executor_id": "cloud", "home_node_id": "node_0"},
        {"event_type": "rejected", "executor_id": "", "home_node_id": "node_0"},
    ]
    # 3 completed tasks total; 1 of them offloaded to a peer edge node
    assert abs(compute_primary_offload_fraction(events) - (1.0 / 3.0)) < 1e-12


# ---------------------------------------------------------------------------
# M11: Jain's Fairness Index (KEY GATE: must be in [0, 1])
# ---------------------------------------------------------------------------


def test_m11_jfi_in_range(synthetic_data: dict[str, Any]) -> None:
    result = compute_jains_fairness(synthetic_data["node_snapshots"])
    assert math.isfinite(result), "M11 JFI is not finite"
    assert 0.0 <= result <= 1.0, f"M11 JFI {result} outside [0, 1]"


def test_m11_jfi_empty_is_zero() -> None:
    assert compute_jains_fairness([]) == 0.0


def test_m11_jfi_equal_load_is_one() -> None:
    snaps = [{"l_current": 1e9, "f_max": 2e9} for _ in range(10)]
    result = compute_jains_fairness(snaps)
    assert abs(result - 1.0) < 1e-10


def test_m11_jfi_all_zero_load() -> None:
    snaps = [{"l_current": 0.0, "f_max": 2e9} for _ in range(10)]
    result = compute_jains_fairness(snaps)
    assert result == 1.0  # sum_f2 = 0 -> returns 1.0


# ---------------------------------------------------------------------------
# M12: TSFR (KEY GATE: == 0 outside burst window for Poisson steady-state)
# ---------------------------------------------------------------------------


def test_m12_tsfr_zero_steady_state(synthetic_data: dict[str, Any]) -> None:
    result = compute_tsfr(
        synthetic_data["events"],
        burst_start=-1.0,
        burst_duration=0.0,
    )
    assert result == 0.0, f"M12 TSFR={result} should be 0 in steady state"


def test_m12_tsfr_captures_burst_violations() -> None:
    events = [
        {
            "event_type": "completed",
            "arrival_time": 5.0,
            "latency": 0.2,
            "deadline": 0.1,
        },
        {
            "event_type": "completed",
            "arrival_time": 5.5,
            "latency": 0.05,
            "deadline": 0.1,
        },
    ]
    result = compute_tsfr(events, burst_start=4.0, burst_duration=2.0)
    assert result == 0.5  # 1 out of 2 violated


def test_m12_tsfr_empty_events_is_zero() -> None:
    assert compute_tsfr([], burst_start=0.0, burst_duration=1.0) == 0.0


# ---------------------------------------------------------------------------
# M13: Edge-Resale Ratio
# ---------------------------------------------------------------------------


def test_m13_err_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_err(synthetic_data["events"])
    assert math.isfinite(result)
    assert 0.0 <= result <= 1.0


def test_m13_no_resale_is_zero() -> None:
    events = [{"event_type": "completed", "is_resale": False} for _ in range(5)]
    assert compute_err(events) == 0.0


def test_m13_all_resale_is_one() -> None:
    events = [{"event_type": "completed", "is_resale": True} for _ in range(5)]
    assert compute_err(events) == 1.0


# ---------------------------------------------------------------------------
# M14: Prediction RMSE
# ---------------------------------------------------------------------------


def test_m14_rmse_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_prediction_rmse(synthetic_data["node_snapshots"])
    assert "RMSE_load" in result
    assert "RMSE_revenue" in result
    assert math.isfinite(result["RMSE_load"])
    assert math.isfinite(result["RMSE_revenue"])
    assert result["RMSE_load"] >= 0.0
    assert result["RMSE_revenue"] >= 0.0


def test_m14_rmse_empty_is_zero() -> None:
    result = compute_prediction_rmse([])
    assert result == {"RMSE_load": 0.0, "RMSE_revenue": 0.0}


def test_m14_rmse_perfect_forecast_is_zero() -> None:
    snaps = [{"l_current": 1e9, "l_hat": 1e9, "R_hat": 1.0, "R_realised": 1.0} for _ in range(50)]
    result = compute_prediction_rmse(snaps)
    assert result["RMSE_load"] == 0.0
    assert result["RMSE_revenue"] == 0.0


# ---------------------------------------------------------------------------
# M15: Fleet Energy Per Task
# ---------------------------------------------------------------------------


def test_m15_energy_finite(synthetic_data: dict[str, Any]) -> None:
    result = compute_fleet_energy_per_task(synthetic_data["events"])
    assert math.isfinite(result)
    assert result >= 0.0


def test_m15_energy_empty_is_zero() -> None:
    assert compute_fleet_energy_per_task([]) == 0.0


# ---------------------------------------------------------------------------
# M16: Myerson Efficiency (KEY GATE: must be in [0, 1])
# ---------------------------------------------------------------------------


def test_m16_myerson_in_range(synthetic_data: dict[str, Any]) -> None:
    result = compute_myerson_efficiency(synthetic_data["events"], synthetic_data["node_snapshots"])
    assert math.isfinite(result), "M16 Myerson efficiency is not finite"
    assert 0.0 <= result <= 1.0, f"M16 Myerson efficiency {result} outside [0, 1]"


def test_m16_myerson_empty_is_one() -> None:
    assert compute_myerson_efficiency([], []) == 1.0


def test_m16_myerson_no_rejection_is_high() -> None:
    rng = np.random.default_rng(0)
    events = [
        {"event_type": "completed", "price_paid": float(rng.uniform(1e-3, 3e-3))} for _ in range(50)
    ]
    result = compute_myerson_efficiency(events, [])
    assert 0.0 <= result <= 1.0


def test_m16_myerson_uses_logged_a_buyer_not_circular_price_proxy() -> None:
    """Regression: M16 must use the true logged a_buyer, not a self-referential
    price_paid*1.1 proxy.

    Setup: HEDGE charges a uniform low competitive price (0.01) to every buyer,
    but half the buyers privately valued the task far higher (1.0 vs 0.02). A
    monopolist (Myerson-optimal) would charge 1.0 and serve only the high-value
    half, netting expected profit 0.5 per arrival. HEDGE's uniform low price
    nets only 0.01 per arrival -- a genuinely low efficiency ratio (0.02), which
    the old price-derived proxy could never detect because it always inferred
    valuations from the very prices it was evaluating (collapsing every case to
    ratio ~ 1/1.1 regardless of actual mispricing).
    """
    events = [
        {"event_type": "completed", "price_paid": 0.01, "a_buyer": 0.02} for _ in range(5)
    ] + [{"event_type": "completed", "price_paid": 0.01, "a_buyer": 1.0} for _ in range(5)]
    result = compute_myerson_efficiency(events, [])
    assert abs(result - 0.02) < 1e-9, f"expected ~0.02 (0.01/0.5), got {result}"


def _myerson_reference_o_n2(events: list[dict[str, Any]]) -> float:
    """Naive O(n^2) reference implementation, used only to cross-check the
    optimized O(n log n) implementation for exact numeric agreement."""
    completed = [e for e in events if e["event_type"] == "completed"]
    n_total = sum(1 for e in events if e["event_type"] in ("completed", "rejected"))
    if not completed or n_total == 0:
        return 1.0
    observed_valuations = [float(e.get("a_buyer", e["price_paid"] * 1.1)) for e in completed]
    total_revenue = sum(float(e["price_paid"]) for e in completed)
    pi_hedge = total_revenue / n_total
    pi_myerson = 0.0
    for p in observed_valuations:
        frac_above = sum(1.0 for v in observed_valuations if v >= p) / n_total
        pi_candidate = p * frac_above
        if pi_candidate > pi_myerson:
            pi_myerson = pi_candidate
    if pi_myerson <= 0.0:
        return 1.0
    return float(max(0.0, min(1.0, pi_hedge / pi_myerson)))


def test_m16_myerson_matches_naive_o_n2_reference() -> None:
    """The O(n log n) bisect-based implementation must agree exactly with the
    naive O(n^2) double-loop definition on randomized inputs, including ties."""
    rng = np.random.default_rng(1)
    for _ in range(8):
        n = int(rng.integers(20, 500))
        events: list[dict[str, Any]] = []
        for _ in range(n):
            a = float(rng.uniform(0.5, 10.0))
            price = float(rng.uniform(0.001, a))
            events.append({"event_type": "completed", "price_paid": price, "a_buyer": a})
        # Inject exact ties to stress the bisect boundary handling.
        if n > 4:
            tie_val = events[0]["a_buyer"]
            events[1]["a_buyer"] = tie_val
            events[2]["a_buyer"] = tie_val
        n_rej = int(rng.integers(0, 10))
        events += [{"event_type": "rejected"} for _ in range(n_rej)]

        expected = _myerson_reference_o_n2(events)
        actual = compute_myerson_efficiency(events, [])
        assert abs(expected - actual) < 1e-9, f"n={n}: expected={expected}, actual={actual}"


def test_m16_myerson_perf_large_n() -> None:
    """Regression: M16 must stay fast at campaign scale (~1.7e5 completed tasks
    per seed). The naive O(n^2) implementation took ~24 minutes at this n; the
    O(n log n) implementation must finish in well under a second."""
    import time

    rng = np.random.default_rng(2)
    n = 170_000
    events = [
        {
            "event_type": "completed",
            "price_paid": float(rng.uniform(0.001, 5.0)),
            "a_buyer": float(rng.uniform(0.5, 10.0)),
        }
        for _ in range(n)
    ]
    t0 = time.perf_counter()
    result = compute_myerson_efficiency(events, [])
    elapsed = time.perf_counter() - t0
    assert 0.0 <= result <= 1.0
    assert elapsed < 5.0, f"M16 took {elapsed:.2f}s for n={n}; expected O(n log n) (<5s)"


# ---------------------------------------------------------------------------
# MetricsCollector integration: compute_all on 100-task synthetic run
# ---------------------------------------------------------------------------


def test_collector_compute_all_all_finite(synthetic_data: dict[str, Any]) -> None:
    """Gate: all 17 metrics produce finite values on 100-task synthetic run."""
    collector = MetricsCollector()
    for event in synthetic_data["events"]:
        collector.record_event(event)
    for snap in synthetic_data["node_snapshots"]:
        collector.record_snapshot(snap)
    for quote in synthetic_data["quote_history"]:
        collector.record_quote(quote)

    results = collector.compute_all()

    def check_finite(val: Any, path: str) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                check_finite(v, f"{path}.{k}")
        elif isinstance(val, bool):
            pass  # converged flag is bool, not float
        else:
            assert math.isfinite(float(val)), f"{path} is not finite: {val}"

    for key in (
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
        "M13",
        "M14",
        "M15",
        "M16",
        "M17",
    ):
        assert key in results, f"{key} missing from compute_all output"
        check_finite(results[key], key)


def test_collector_m6_plus_m10_gate(synthetic_data: dict[str, Any]) -> None:
    """Gate: M6 + M10 == 1.0 via collector."""
    collector = MetricsCollector()
    for event in synthetic_data["events"]:
        collector.record_event(event)
    results = collector.compute_all()
    assert abs(results["M6"] + results["M10"] - 1.0) < 1e-10


def test_collector_m11_jfi_gate(synthetic_data: dict[str, Any]) -> None:
    """Gate: M11 JFI in [0, 1] via collector."""
    collector = MetricsCollector()
    for snap in synthetic_data["node_snapshots"]:
        collector.record_snapshot(snap)
    results = collector.compute_all()
    assert 0.0 <= results["M11"] <= 1.0


def test_collector_m16_myerson_gate(synthetic_data: dict[str, Any]) -> None:
    """Gate: M16 Myerson efficiency in [0, 1] via collector."""
    collector = MetricsCollector()
    for event in synthetic_data["events"]:
        collector.record_event(event)
    results = collector.compute_all()
    assert 0.0 <= results["M16"] <= 1.0


def test_collector_m12_tsfr_steady_state_gate(synthetic_data: dict[str, Any]) -> None:
    """Gate: M12 TSFR == 0 in steady state via collector (default burst_start=-1.0)."""
    collector = MetricsCollector()  # defaults: burst_start=-1.0, burst_duration=0.0
    for event in synthetic_data["events"]:
        collector.record_event(event)
    results = collector.compute_all()
    assert results["M12"] == 0.0, f"M12 TSFR={results['M12']} in steady state"


# ---------------------------------------------------------------------------
# MetricsCollector: export_csv
# ---------------------------------------------------------------------------


def test_collector_export_csv(tmp_path: Path, synthetic_data: dict[str, Any]) -> None:
    collector = MetricsCollector()
    for event in synthetic_data["events"]:
        collector.record_event(event)
    for snap in synthetic_data["node_snapshots"]:
        collector.record_snapshot(snap)

    out = tmp_path / "metrics.csv"
    collector.export_csv(out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "M2" in content
    assert "M6" in content
    assert "M11" in content


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_sample_mean() -> None:
    rng = np.random.default_rng(7)
    values = list(rng.normal(loc=5.0, scale=1.0, size=200).tolist())
    sample_mean = float(np.mean(values))
    lo, hi = bootstrap_ci(values, confidence=0.95, n_boot=2000, rng=rng)
    assert lo < sample_mean < hi, f"Sample mean {sample_mean:.3f} not in CI [{lo:.3f}, {hi:.3f}]"


def test_bootstrap_ci_lower_lt_upper() -> None:
    rng = np.random.default_rng(0)
    values = list(rng.uniform(0, 10, size=50).tolist())
    lo, hi = bootstrap_ci(values, rng=rng)
    assert lo < hi


def test_wilcoxon_significant_for_different_distributions() -> None:
    rng = np.random.default_rng(13)
    a = list(rng.normal(loc=1.0, scale=0.1, size=30).tolist())
    b = list(rng.normal(loc=2.0, scale=0.1, size=30).tolist())
    result = wilcoxon_test(a, b)
    assert result["significant"] is True
    assert "p_value" in result
    assert result["p_value"] < 0.05


def test_wilcoxon_not_significant_for_same_distribution() -> None:
    rng = np.random.default_rng(99)
    a = list(rng.normal(loc=1.0, scale=0.01, size=30).tolist())
    b = [x + rng.normal(0, 1e-6) for x in a]  # almost identical
    result = wilcoxon_test(a, b)
    assert not result["significant"]
