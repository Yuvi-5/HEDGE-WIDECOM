"""B7: Cloud-Only baseline.

All tasks dispatched to cloud unconditionally. Establishes the latency floor
and cost ceiling for the HEDGE comparison suite.

Cloud has infinite capacity and never rejects tasks due to overload.
Tasks are rejected only when:
- Cloud RTT + compute time exceeds the task deadline, OR
- Cloud total price exceeds the buyer's valuation (a_buyer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hedge.market.afgm import compute_cloud_latency
from hedge.pricing.layer1 import compute_layer1_cost

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


def run_cloud_only_round(
    task: "HEDGETask",
    cloud: Any,
    bandwidth_cloud: float = 1e8,
) -> dict[str, Any]:
    """Cloud-only allocation: unconditionally dispatch to cloud.

    Does not consult any edge nodes. Rejects only if cloud deadline is
    infeasible or price exceeds buyer valuation.

    Args:
        task: Task to allocate.
        cloud: Cloud node with f_max, kappa, P_idle, rho, pi_E, beta, tau_c, mu_c.
        bandwidth_cloud: Cloud uplink bandwidth (bits/s). Default 100 Mbps.

    Returns:
        Dict with keys: status, executor_id, price_total, latency, C1_total.
        executor_id is always "cloud" when status is "completed".
    """
    C1_cloud, _, _, _ = compute_layer1_cost(
        task.w, cloud.f_max, cloud.kappa, cloud.P_idle, cloud.rho, cloud.pi_E, cloud.beta
    )
    latency = compute_cloud_latency(task, cloud, bandwidth_cloud)
    price_total = float(cloud.mu_c) * C1_cloud

    if latency > task.d:
        return {
            "status": "rejected",
            "executor_id": "",
            "price_total": 0.0,
            "latency": latency,
            "C1_total": 0.0,
        }

    if price_total > task.a_buyer:
        return {
            "status": "rejected",
            "executor_id": "",
            "price_total": 0.0,
            "latency": latency,
            "C1_total": 0.0,
        }

    return {
        "status": "completed",
        "executor_id": "cloud",
        "price_total": price_total,
        "latency": latency,
        "C1_total": C1_cloud,
    }
