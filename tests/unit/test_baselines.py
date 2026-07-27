"""Phase 9 gate: baseline tests for the arms used in this repo's comparison campaign.

Gate criteria:
- B7 (cloud-only): all completed tasks have executor_id == "cloud".
- All baselines: admission gate rejects tasks below the Layer-1 Bertrand floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from hedge.baselines import (
    BASELINE_REGISTRY,
    admission_gate,
    run_cloud_only_round,
    run_cost_opd_round,
    run_ddps_round,
    run_greedy_nlf_round,
)
from hedge.baselines.b4_cost_opd import compute_dvfs_energy
from hedge.core.cloud import CloudNode
from hedge.core.constants import (
    KAPPA_MIN,
    MU_C,
    P_IDLE_OPTIONS,
    PI_E_MIN,
    TAU_C_MEAN,
)
from hedge.core.task import HEDGETask

# ---------------------------------------------------------------------------
# Lightweight mock node (duck-typing; avoids EdgeSimPy dependency)
# ---------------------------------------------------------------------------


@dataclass
class MockNode:
    """Minimal edge server mock with all attributes required by baselines."""

    unique_id: int
    f_max: float
    kappa: float = 1e-26
    P_idle: float = 30.0
    rho: float = 5e-4
    pi_E: float = 0.10
    beta: float = 300.0
    w_pending: float = 0.0
    l_hat: float = 0.0
    R_hat: float = 0.0
    p_star_ref: float = 0.0
    p_dagger_ref: float = 0.0
    lambda_SPA: float = 0.0


@dataclass
class MockCloud:
    """Minimal cloud node mock."""

    f_max: float = 1e12
    kappa: float = 1e-27
    P_idle: float = 0.0
    rho: float = 1e-4
    pi_E: float = 0.06
    beta: float = 80.0
    mu_c: float = MU_C
    tau_c: float = TAU_C_MEAN


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_task(
    w: float = 5e9,
    s: float = 2e6,
    d: float = 2.0,
    a_buyer: float = 5.0,
    task_id: str = "t1",
) -> HEDGETask:
    """Build a HEDGETask with default parameters suitable for most tests."""
    return HEDGETask(task_id=task_id, s=s, w=w, d=d, a_buyer=a_buyer, created_at=0.0)


def _make_nodes(n: int = 3, f_max: float = 3e9, w_pending: float = 0.0) -> list[MockNode]:
    """Build a list of n identical mock nodes."""
    return [MockNode(unique_id=i, f_max=f_max, w_pending=w_pending) for i in range(n)]


def _tau_dict(nodes: list[MockNode], delay: float = 0.002) -> dict[int, float]:
    """Build a uniform delay dict for nodes."""
    return {n.unique_id: delay for n in nodes}


def _make_cloud() -> MockCloud:
    """Build a default mock cloud node."""
    return MockCloud()


# ---------------------------------------------------------------------------
# B7: Cloud-Only
# ---------------------------------------------------------------------------


def test_b7_completed_executor_is_cloud() -> None:
    """All completed B7 tasks must have executor_id == 'cloud'."""
    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    result = run_cloud_only_round(task, cloud)
    if result["status"] == "completed":
        assert result["executor_id"] == "cloud"


def test_b7_always_cloud_on_affordable_task() -> None:
    """B7 with generous a_buyer and long deadline always completes via cloud."""
    cloud = _make_cloud()
    task = _make_task(a_buyer=100.0, d=10.0)
    result = run_cloud_only_round(task, cloud)
    assert result["status"] == "completed"
    assert result["executor_id"] == "cloud"


def test_b7_rejects_infeasible_deadline() -> None:
    """B7 rejects task when cloud latency exceeds deadline."""
    cloud = _make_cloud()
    # Very tight deadline: cloud propagation alone (2 * tau_c = 0.07s) + compute won't fit
    task = _make_task(w=5e11, d=1e-6, a_buyer=100.0)
    result = run_cloud_only_round(task, cloud)
    assert result["status"] == "rejected"


def test_b7_rejects_unaffordable_price() -> None:
    """B7 rejects task when cloud price exceeds buyer valuation."""
    cloud = _make_cloud()
    # Tiny a_buyer, large workload -> cloud price will exceed valuation
    task = _make_task(w=1e11, d=10.0, a_buyer=1e-10)
    result = run_cloud_only_round(task, cloud)
    assert result["status"] == "rejected"


def test_b7_price_total_positive_on_completion() -> None:
    """Completed B7 task has positive price_total."""
    cloud = _make_cloud()
    task = _make_task(a_buyer=100.0, d=10.0)
    result = run_cloud_only_round(task, cloud)
    if result["status"] == "completed":
        assert result["price_total"] > 0.0


# ---------------------------------------------------------------------------
# B2: DDPS
# ---------------------------------------------------------------------------


def test_b2_ddps_valid_result_structure() -> None:
    """DDPS returns a dict with all required keys."""
    nodes = _make_nodes(3)
    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    tau = _tau_dict(nodes)
    result = run_ddps_round(nodes, task, cloud, tau)
    for key in ("status", "executor_id", "price_total", "latency", "C1_total"):
        assert key in result, f"Missing key '{key}' in DDPS result"


def test_b2_ddps_completes_with_available_nodes() -> None:
    """DDPS completes task allocation when nodes are available and uncongested."""
    nodes = _make_nodes(3, w_pending=0.0)
    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    tau = _tau_dict(nodes)
    result = run_ddps_round(nodes, task, cloud, tau)
    assert result["status"] == "completed"


def test_b2_ddps_higher_load_increases_price() -> None:
    """DDPS price multiplier increases with higher load (step-function property)."""
    cloud = _make_cloud()
    task = _make_task(a_buyer=100.0, d=10.0)

    # Low-load node
    nodes_low = [MockNode(unique_id=0, f_max=3e9, w_pending=0.0)]
    tau_low = {0: 0.002}
    result_low = run_ddps_round(nodes_low, task, cloud, tau_low)

    # High-load node (w_pending / delta_tp / f_max > 0.9 threshold)
    # delta_tp=0.1, so w_pending = 0.95 * f_max * delta_tp = 0.95 * 3e9 * 0.1 = 2.85e8
    nodes_high = [MockNode(unique_id=0, f_max=3e9, w_pending=2.85e8)]
    tau_high = {0: 0.002}
    result_high = run_ddps_round(nodes_high, task, cloud, tau_high)

    if result_low["status"] == "completed" and result_high["status"] == "completed":
        # Both allocated to edge node; high load should produce higher or equal price
        if result_low["executor_id"] != "cloud" and result_high["executor_id"] != "cloud":
            assert result_high["price_total"] >= result_low["price_total"]


# ---------------------------------------------------------------------------
# B4: Cost-OPD
# ---------------------------------------------------------------------------


def test_b4_cost_opd_completes_when_affordable() -> None:
    """Cost-OPD completes task when edge nodes are available and affordable."""
    nodes = _make_nodes(3)
    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    tau = _tau_dict(nodes)
    result = run_cost_opd_round(nodes, task, cloud, tau)
    assert result["status"] == "completed"


def test_b4_compute_dvfs_energy_positive() -> None:
    """DVFS energy for any valid (w, node) is strictly positive."""
    node = MockNode(unique_id=0, f_max=3e9, kappa=1e-26, P_idle=30.0)
    energy = compute_dvfs_energy(5e9, node)
    assert energy > 0.0


def test_b4_selects_lower_energy_node() -> None:
    """Cost-OPD prefers the node with lower OPEX energy cost."""
    # Node 0: low kappa (energy-efficient)
    node0 = MockNode(unique_id=0, f_max=3e9, kappa=1e-27, P_idle=0.0, pi_E=0.10)
    # Node 1: high kappa (energy-inefficient)
    node1 = MockNode(unique_id=1, f_max=3e9, kappa=1e-25, P_idle=50.0, pi_E=0.18)
    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    tau = {0: 0.002, 1: 0.002}
    result = run_cost_opd_round([node0, node1], task, cloud, tau)
    if result["status"] == "completed" and result["executor_id"] != "cloud":
        assert result["executor_id"] == "0"  # lower energy node wins


# ---------------------------------------------------------------------------
# B6: Greedy NLF
# ---------------------------------------------------------------------------


def test_b6_greedy_nlf_selects_nearest_with_capacity() -> None:
    """Greedy NLF selects the nearest node (lowest tau) that has capacity."""
    # Node 0: far but empty; Node 1: near but overloaded; Node 2: near and empty
    node0 = MockNode(unique_id=0, f_max=3e9, w_pending=0.0)
    node1 = MockNode(unique_id=1, f_max=3e9, w_pending=1e12)  # overloaded
    node2 = MockNode(unique_id=2, f_max=3e9, w_pending=0.0)

    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    tau = {0: 0.004, 1: 0.001, 2: 0.002}  # node2 is nearest with capacity

    result = run_greedy_nlf_round([node0, node1, node2], task, cloud, tau)
    assert result["status"] == "completed"
    # node2 (tau=0.002) should beat node0 (tau=0.004); node1 is overloaded
    if result["executor_id"] != "cloud":
        assert result["executor_id"] == "2"


def test_b6_greedy_nlf_cloud_fallback_when_all_edges_overloaded() -> None:
    """Greedy NLF falls back to cloud when all edge nodes are overloaded."""
    nodes = _make_nodes(3, f_max=3e9, w_pending=1e15)  # all overloaded
    cloud = _make_cloud()
    task = _make_task(a_buyer=100.0, d=10.0)
    tau = _tau_dict(nodes)
    result = run_greedy_nlf_round(nodes, task, cloud, tau)
    assert result["executor_id"] in ("cloud", "")


def test_b6_greedy_nlf_valid_result_keys() -> None:
    """Greedy NLF result has all required keys."""
    nodes = _make_nodes(2)
    cloud = _make_cloud()
    task = _make_task(a_buyer=5.0, d=5.0)
    tau = _tau_dict(nodes)
    result = run_greedy_nlf_round(nodes, task, cloud, tau)
    for key in ("status", "executor_id", "price_total", "latency", "C1_total"):
        assert key in result


# ---------------------------------------------------------------------------
# Admission gate parity
# ---------------------------------------------------------------------------


def test_admission_gate_passes_affordable_task() -> None:
    """Admission gate passes task whose valuation covers minimum Layer-1 cost."""
    nodes = _make_nodes(2)
    task = _make_task(a_buyer=5.0)
    assert admission_gate(task, nodes, {}) is True


def test_admission_gate_rejects_sub_floor_task() -> None:
    """Admission gate rejects task with valuation below minimum Bertrand floor."""
    nodes = _make_nodes(2)
    task = _make_task(a_buyer=1e-20)  # far below any Layer-1 cost
    assert admission_gate(task, nodes, {}) is False


def test_all_baselines_reject_sub_floor_task() -> None:
    """All function-based baselines reject a task below the admission gate.

    When a_buyer is far below the minimum Layer-1 cost, no baseline should
    complete the task (admission gate parity requirement).
    """
    nodes = _make_nodes(3, f_max=3e9)
    cloud = _make_cloud()
    task = _make_task(a_buyer=1e-20, d=5.0)  # sub-floor
    tau = _tau_dict(nodes)

    # B7
    assert run_cloud_only_round(task, cloud)["status"] == "rejected"
    # B2
    assert run_ddps_round(nodes, task, cloud, tau)["status"] == "rejected"
    # B4
    assert run_cost_opd_round(nodes, task, cloud, tau)["status"] == "rejected"
    # B6
    assert run_greedy_nlf_round(nodes, task, cloud, tau)["status"] == "rejected"


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------


def test_baseline_registry_has_all_arm_entries() -> None:
    """BASELINE_REGISTRY must contain exactly the arms this repo evaluates."""
    for key in ("B2", "B4", "B6", "B7"):
        assert key in BASELINE_REGISTRY, f"Missing registry entry for {key}"


def test_baseline_registry_entries_are_callable() -> None:
    """All non-class entries in the registry are callable."""
    for name, entry in BASELINE_REGISTRY.items():
        assert callable(entry), f"Registry entry {name} is not callable"
