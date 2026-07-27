"""B2: DDPS - Dynamic Differential Pricing-Based Edge Offloading System.

Reference: IEEE TNSN 2025. Adapted for dense-edge (energy harvesting layer removed).
Uses step-function load-dependent pricing instead of continuous Stackelberg.
Vertical structure only: no horizontal edge-to-edge wholesale market.
No CAPEX cost floor, no SPA competitive shading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hedge.core.constants import DELTA_TP
from hedge.market.afgm import compute_cloud_latency, compute_node_latency
from hedge.pricing.layer1 import compute_layer1_cost

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask

_LOAD_THRESHOLDS: list[float] = [0.3, 0.6, 0.9]
_PRICE_MULTIPLIERS: list[float] = [1.0, 1.5, 2.5, 5.0]


def _ddps_node_quote(
    node: Any,
    task: "HEDGETask",
    tau: float,
    delta_tp: float = DELTA_TP,
    bandwidth: float = 1e9,
) -> tuple[float, float]:
    """Compute DDPS step-function price and latency for one edge node.

    Args:
        node: Edge server with f_max, kappa, P_idle, rho, pi_E, beta, w_pending.
        task: Task to price.
        tau: One-way propagation delay from buyer to node (seconds).
        delta_tp: Cadence period for instantaneous load estimate (seconds).
        bandwidth: Link bandwidth (bits/s).

    Returns:
        (price_per_cycle, latency) where price_per_cycle is in USD/cycle.
    """
    _, C1_dvfs, _, _ = compute_layer1_cost(
        task.w, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
    )
    base_per_cycle = C1_dvfs / task.w  # OPEX-only floor, no CAPEX/carbon, by design

    # Instantaneous load fraction using w_pending / delta_tp as rate estimate
    l_instant = node.w_pending / max(delta_tp, 1e-12)
    load_frac = l_instant / max(float(node.f_max), 1.0)
    idx = int(np.searchsorted(_LOAD_THRESHOLDS, load_frac))
    price_per_cycle = base_per_cycle * _PRICE_MULTIPLIERS[idx]

    latency = compute_node_latency(task, node, tau, bandwidth)
    return price_per_cycle, latency


def run_ddps_round(
    peer_pool: list[Any],
    task: "HEDGETask",
    cloud: Any,
    tau_dict: dict[int, float],
    alpha_u: float = 0.5,
    gamma_u: float = 0.5,
    bandwidth_metro: float = 1e9,
    bandwidth_cloud: float = 1e8,
    delta_tp: float = DELTA_TP,
) -> dict[str, Any]:
    """DDPS allocation: step-function pricing with AFGM selection.

    Contacts all peers in the pool (vertical structure; no K_max fanout limit).
    Selects the winner by minimising alpha_u * L + gamma_u * p over affordable set.

    Args:
        peer_pool: All candidate edge nodes.
        task: Task to allocate.
        cloud: Cloud fallback node.
        tau_dict: Propagation delays keyed by node unique_id (seconds).
        alpha_u: Latency weight in winner selection.
        gamma_u: Price weight in winner selection.
        bandwidth_metro: Metro link bandwidth (bits/s).
        bandwidth_cloud: Cloud uplink bandwidth (bits/s).
        delta_tp: Cadence period for instantaneous load estimate (seconds).

    Returns:
        Dict with keys: status, executor_id, price_total, latency, C1_total.
    """
    candidates: list[tuple[Any, float, float]] = []

    for node in peer_pool:
        tau = tau_dict.get(node.unique_id, 0.005)
        p_per_cycle, latency = _ddps_node_quote(node, task, tau, delta_tp, bandwidth_metro)
        if p_per_cycle * task.w <= task.a_buyer and latency <= task.d:
            candidates.append((node, p_per_cycle, latency))

    C1_cloud, _, _, _ = compute_layer1_cost(
        task.w, cloud.f_max, cloud.kappa, cloud.P_idle, cloud.rho, cloud.pi_E, cloud.beta
    )
    p_cloud = cloud.mu_c * C1_cloud / task.w
    L_cloud = compute_cloud_latency(task, cloud, bandwidth_cloud)
    if p_cloud * task.w <= task.a_buyer and L_cloud <= task.d:
        candidates.append((cloud, p_cloud, L_cloud))

    if not candidates:
        return {
            "status": "rejected",
            "executor_id": "",
            "price_total": 0.0,
            "latency": float("inf"),
            "C1_total": 0.0,
        }

    winner, p_win, L_win = min(candidates, key=lambda x: alpha_u * x[2] + gamma_u * x[1])
    is_cloud = hasattr(winner, "mu_c")
    if is_cloud:
        C1_win = C1_cloud
        executor_id = "cloud"
    else:
        C1_win, _, _, _ = compute_layer1_cost(
            task.w, winner.f_max, winner.kappa, winner.P_idle, winner.rho, winner.pi_E, winner.beta
        )
        executor_id = str(winner.unique_id)

    return {
        "status": "completed",
        "executor_id": executor_id,
        "price_total": p_win * task.w,
        "latency": L_win,
        "C1_total": C1_win,
    }
