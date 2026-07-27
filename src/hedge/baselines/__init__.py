"""HEDGE baselines: registry and shared admission gate.

All baselines share the same IR-based admission gate as HEDGE to ensure
utility gains reflect pricing/matching quality, not task selection differences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hedge.core.constants import B_SHAPE
from hedge.pricing.layer1 import compute_layer1_cost

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


def admission_gate(
    task: "HEDGETask",
    nodes: list[Any],
    config: dict[str, Any],
    cloud: Any = None,
) -> bool:
    """Shared pre-filter: reject task if buyer cannot cover the cheapest Layer-1 cost.

    All baselines apply this gate before their allocation logic to ensure
    comparison utility differences are not caused by differential task selection.

    Per the paper's Definition (Economic Admission Control Gate): a buyer
    whose reservation falls below every EDGE floor is
    NOT immediately rejected -- it is routed to cloud, and only unmatched if
    cloud's own price also exceeds the reservation ("buyers whose reservation
    falls below all Bertrand floors route to cloud"). Omitting cloud from
    this check (as an earlier version of this function did) wrongly rejects
    tasks that cloud would have served profitably, before they ever reach
    the RFQ/AFGM stage where cloud is properly considered.

    Args:
        task: Task with private valuation a_buyer (USD).
        nodes: All candidate edge nodes (must expose f_max, kappa, P_idle, rho, pi_E, beta).
        config: Experiment config dict (reads "b_shape", defaults to B_SHAPE = 1.0).
        cloud: Optional CloudNode (same required attributes); included in the
            cost-floor comparison per the paper's two-stage admission gate.
            None preserves the edge-only check (e.g. for baselines/tests that
            deliberately evaluate edge-only feasibility).

    Returns:
        True if task passes pre-filter (buyer can cover the cheapest Bertrand
        floor across edges and, if provided, cloud).
        False if immediately rejected without running allocation.
    """
    candidates = list(nodes)
    if cloud is not None:
        candidates.append(cloud)
    if not candidates:
        return False
    b_shape = float(config.get("pricing", {}).get("b_shape", B_SHAPE))
    min_c1 = min(
        compute_layer1_cost(task.w, n.f_max, n.kappa, n.P_idle, n.rho, n.pi_E, n.beta)[0]
        for n in candidates
    )
    return task.a_buyer >= min_c1 / b_shape


from hedge.baselines.b2_ddps import run_ddps_round
from hedge.baselines.b4_cost_opd import run_cost_opd_round
from hedge.baselines.b6_greedy_nlf import run_greedy_nlf_round
from hedge.baselines.b7_cloud_only import run_cloud_only_round

BASELINE_REGISTRY: dict[str, Any] = {
    "B2": run_ddps_round,
    "B4": run_cost_opd_round,
    "B6": run_greedy_nlf_round,
    "B7": run_cloud_only_round,
}

__all__ = [
    "admission_gate",
    "BASELINE_REGISTRY",
    "run_ddps_round",
    "run_cost_opd_round",
    "run_greedy_nlf_round",
    "run_cloud_only_round",
]
