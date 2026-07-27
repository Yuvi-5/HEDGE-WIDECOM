"""B6: Greedy NLF (Nearest-Loaded-First) heuristic baseline.

Selects the nearest node (lowest propagation delay tau_ui) that has available
capacity. No pricing mechanism; tasks are charged at Layer-1 cost.
Falls back to cloud if no edge node has capacity.

Simple heuristic with no economic properties (IR, WBB not guaranteed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hedge.core.constants import DELTA_TP
from hedge.market.afgm import compute_cloud_latency, compute_node_latency
from hedge.pricing.layer1 import compute_layer1_cost

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


def _node_has_capacity(node: Any, task: "HEDGETask", delta_tp: float = DELTA_TP) -> bool:
    """True if node can serve the task without overloading.

    Capacity check: instantaneous arrival rate + task demand fits within f_max.

    Args:
        node: Edge server with f_max and w_pending.
        task: Task with w (cycles) and d (deadline, seconds).
        delta_tp: Cadence window for rate estimate (seconds).

    Returns:
        True if node has remaining capacity.
    """
    l_instant = float(node.w_pending) / max(delta_tp, 1e-12)
    task_rate = task.w / max(task.d, 1e-9)
    return (l_instant + task_rate) <= float(node.f_max)


def run_greedy_nlf_round(
    peer_pool: list[Any],
    task: "HEDGETask",
    cloud: Any,
    tau_dict: dict[int, float],
    delta_tp: float = DELTA_TP,
    bandwidth_metro: float = 1e9,
    bandwidth_cloud: float = 1e8,
) -> dict[str, Any]:
    """Greedy NLF allocation: nearest edge with capacity, else cloud fallback.

    Among all edge nodes with available capacity and feasible deadline,
    selects the one with the smallest propagation delay. Falls back to cloud
    if no edge qualifies.

    Args:
        peer_pool: All candidate edge nodes.
        task: Task to allocate.
        cloud: Cloud fallback node.
        tau_dict: Propagation delays keyed by node unique_id (seconds).
        delta_tp: Cadence period for capacity estimation (seconds).
        bandwidth_metro: Metro link bandwidth (bits/s).
        bandwidth_cloud: Cloud uplink bandwidth (bits/s).

    Returns:
        Dict with keys: status, executor_id, price_total, latency, C1_total.
    """
    # Filter edge nodes that have capacity and affordable cost
    edge_candidates = [n for n in peer_pool if _node_has_capacity(n, task, delta_tp)]

    for node in sorted(edge_candidates, key=lambda n: tau_dict.get(n.unique_id, 0.005)):
        tau = tau_dict.get(node.unique_id, 0.005)
        latency = compute_node_latency(task, node, tau, bandwidth_metro)
        C1_w, _, _, _ = compute_layer1_cost(
            task.w, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
        )
        if latency <= task.d and C1_w <= task.a_buyer:
            return {
                "status": "completed",
                "executor_id": str(node.unique_id),
                "price_total": C1_w,
                "latency": latency,
                "C1_total": C1_w,
            }

    # Cloud fallback
    C1_cloud, _, _, _ = compute_layer1_cost(
        task.w, cloud.f_max, cloud.kappa, cloud.P_idle, cloud.rho, cloud.pi_E, cloud.beta
    )
    L_cloud = compute_cloud_latency(task, cloud, bandwidth_cloud)
    price_cloud = float(cloud.mu_c) * C1_cloud
    if L_cloud <= task.d and price_cloud <= task.a_buyer:
        return {
            "status": "completed",
            "executor_id": "cloud",
            "price_total": price_cloud,
            "latency": L_cloud,
            "C1_total": C1_cloud,
        }

    return {
        "status": "rejected",
        "executor_id": "",
        "price_total": 0.0,
        "latency": float("inf"),
        "C1_total": 0.0,
    }
