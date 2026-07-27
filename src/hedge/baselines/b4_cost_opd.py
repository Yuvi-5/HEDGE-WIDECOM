"""B4: Cost-OPD - Cost-Optimised edge offloading baseline.

Reference: Zhang et al., IEEE TMC 2024. DAG-structured task offloading.
Adaptation for monolithic tasks: use only the cost-optimisation component.
Minimises OPEX energy cost only; no CAPEX amortisation, no carbon, no latency weighting.
No predictive pricing module.

Expected comparison: B4 vs HEDGE shows the value of CAPEX + carbon integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hedge.core.constants import JOULES_PER_KWH
from hedge.market.afgm import compute_cloud_latency, compute_node_latency
from hedge.pricing.layer1 import compute_layer1_cost

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


def compute_dvfs_energy(w: float, node: Any) -> float:
    """Compute DVFS execution energy for w CPU cycles at a node (Joules).

    Args:
        w: Task CPU workload (cycles).
        node: Node with f_max (Hz), kappa (F), P_idle (W).

    Returns:
        Total execution energy in Joules.
    """
    return float(node.kappa) * float(node.f_max) ** 2 * w + float(node.P_idle) * w / float(
        node.f_max
    )


def _opex_energy_cost(w: float, node: Any) -> float:
    """OPEX-only energy cost: electricity cost without CAPEX or carbon (USD).

    Args:
        w: Task CPU workload (cycles).
        node: Node with pi_E (USD/kWh).

    Returns:
        Energy cost in USD.
    """
    energy_j = compute_dvfs_energy(w, node)
    return energy_j * float(node.pi_E) / JOULES_PER_KWH


def run_cost_opd_round(
    peer_pool: list[Any],
    task: "HEDGETask",
    cloud: Any,
    tau_dict: dict[int, float],
    bandwidth_metro: float = 1e9,
    bandwidth_cloud: float = 1e8,
) -> dict[str, Any]:
    """Cost-OPD allocation: select minimum-OPEX node that is deadline feasible.

    Charges the Layer-1 total cost as the price (minimum feasible; no markup).
    Does not weight latency in selection; pure cost minimisation.

    Args:
        peer_pool: Candidate edge nodes.
        task: Task to allocate.
        cloud: Cloud fallback node.
        tau_dict: Propagation delays keyed by node unique_id (seconds).
        bandwidth_metro: Metro link bandwidth (bits/s).
        bandwidth_cloud: Cloud uplink bandwidth (bits/s).

    Returns:
        Dict with keys: status, executor_id, price_total, latency, C1_total.
    """
    candidates: list[tuple[Any, float, float, float]] = []  # node, opex, latency, C1

    for node in peer_pool:
        opex = _opex_energy_cost(task.w, node)
        C1_w, _, _, _ = compute_layer1_cost(
            task.w, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
        )
        tau = tau_dict.get(node.unique_id, 0.005)
        latency = compute_node_latency(task, node, tau, bandwidth_metro)
        if C1_w <= task.a_buyer and latency <= task.d:
            candidates.append((node, opex, latency, C1_w))

    C1_cloud, _, _, _ = compute_layer1_cost(
        task.w, cloud.f_max, cloud.kappa, cloud.P_idle, cloud.rho, cloud.pi_E, cloud.beta
    )
    opex_cloud = _opex_energy_cost(task.w, cloud)
    L_cloud = compute_cloud_latency(task, cloud, bandwidth_cloud)
    price_cloud = cloud.mu_c * C1_cloud
    if price_cloud <= task.a_buyer and L_cloud <= task.d:
        candidates.append((cloud, opex_cloud, L_cloud, C1_cloud))

    if not candidates:
        return {
            "status": "rejected",
            "executor_id": "",
            "price_total": 0.0,
            "latency": float("inf"),
            "C1_total": 0.0,
        }

    winner, _opex_w, L_win, C1_win = min(candidates, key=lambda x: x[1])
    is_cloud = hasattr(winner, "mu_c")
    if is_cloud:
        executor_id = "cloud"
        price_total = cloud.mu_c * C1_win
    else:
        executor_id = str(winner.unique_id)
        price_total = C1_win  # charge at cost (no markup in Cost-OPD)

    return {
        "status": "completed",
        "executor_id": executor_id,
        "price_total": price_total,
        "latency": L_win,
        "C1_total": C1_win,
    }
