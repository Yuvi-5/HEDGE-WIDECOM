"""AFGM matching: affordable set construction and Gale-Shapley-inspired winner selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hedge.core.constants import S_RET_FRACTION

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


def compute_node_latency(
    task: "HEDGETask",
    node: Any,
    tau: float,
    bandwidth: float = 1e9,
) -> float:
    """Compute end-to-end latency L_i(T) for task T at node i (Equation 10).

    Args:
        task: Task with fields s (bits), w (cycles), d (seconds).
        node: Node with f_max (Hz) and w_pending (cycles) attributes.
        tau: One-way propagation delay from buyer to node i (seconds).
        bandwidth: Upload/download bandwidth (bits/s). Default 1 Gbps.

    Returns:
        L_i(T) in seconds.
    """
    transfer = (task.s + task.s * S_RET_FRACTION) / max(bandwidth, 1.0)
    propagation = 2.0 * tau
    queue_wait = float(node.w_pending) / max(float(node.f_max), 1.0)
    execution = task.w / max(float(node.f_max), 1.0)
    return float(transfer + propagation + queue_wait + execution)


def compute_cloud_latency(
    task: "HEDGETask",
    cloud: Any,
    bandwidth: float = 1e8,
) -> float:
    """Compute end-to-end latency L_c(T) for task T at cloud node.

    Args:
        task: Task with fields s (bits), w (cycles).
        cloud: Cloud node with f_max (Hz) and tau_c (seconds, RTT/2) attributes.
        bandwidth: Cloud uplink bandwidth (bits/s). Default 100 Mbps.

    Returns:
        L_c(T) in seconds.
    """
    transfer = (task.s + task.s * S_RET_FRACTION) / max(bandwidth, 1.0)
    propagation = 2.0 * float(cloud.tau_c)
    execution = task.w / max(float(cloud.f_max), 1.0)
    return float(transfer + propagation + execution)


def build_affordable_set(
    peers_with_quotes: list[tuple[Any, float, float]],
    cloud_with_quote: tuple[Any, float, float] | None,
    task: "HEDGETask",
    a_buyer: float,
) -> list[tuple[Any, float, float]]:
    """Build F_b: the set of affordable and deadline-feasible sellers (Step 4, Algorithm 1).

    Seller i enters F_b iff: p_dagger_i * w <= a_buyer AND L_i(T) <= d.
    Buyer IR (Invariant I3) is structurally guaranteed by this gate.

    Args:
        peers_with_quotes: List of (node, p_dagger_per_cycle, L_i) tuples for edge peers.
        cloud_with_quote: (cloud_node, p_c_per_cycle, L_c) or None if cloud excluded.
        task: Task with a_buyer (private), w (cycles), d (seconds).
        a_buyer: Private buyer valuation (USD). NEVER transmitted; used only here.

    Returns:
        List of (node, p_dagger, L_i) tuples, edge peers first, cloud last if included.
    """
    affordable: list[tuple[Any, float, float]] = []
    for node, p_dagger, L_i in peers_with_quotes:
        if p_dagger * task.w <= a_buyer and L_i <= task.d:
            affordable.append((node, p_dagger, L_i))
    if cloud_with_quote is not None:
        cloud, p_c, L_c = cloud_with_quote
        if p_c * task.w <= a_buyer and L_c <= task.d:
            affordable.append((cloud, p_c, L_c))
    return affordable


def afgm_select(
    affordable_set: list[tuple[Any, float, float]],
    alpha_u: float = 0.5,
    gamma_u: float = 0.5,
    tau_dict: dict[int, float] | None = None,
) -> tuple[Any, float, float] | None:
    """Select winning seller using AFGM buyer preference function (Section 3c).

    Minimises Phi_u(i) = alpha_u * L_i(T) + gamma_u * p_dagger_i over F_b.
    Ties on Phi_u(i) break by (1) lower tau_ij, (2) lower beta_i / greener node.
    Nodes without a unique_id (e.g. cloud)
    or without a beta attribute fall back to +inf on the relevant tie-break key,
    so they never win a tie purely by missing an attribute.

    Args:
        affordable_set: List of (node, p_dagger_per_cycle, L_i) tuples.
        alpha_u: Latency weight (default 0.5). Should satisfy alpha_u + gamma_u = 1.
        gamma_u: Price weight (default 0.5).
        tau_dict: {node.unique_id: tau} one-way propagation delays (seconds), used
            only for the first tie-break level. None skips straight to the beta
            tie-break (e.g. when the caller has no per-node tau available).

    Returns:
        (winner_node, winner_p_dagger, winner_L_i) or None if set is empty.
    """
    if not affordable_set:
        return None

    def _tau_key(node: Any) -> float:
        if tau_dict is None:
            return 0.0
        uid = getattr(node, "unique_id", None)
        return tau_dict.get(uid, float("inf")) if uid is not None else float("inf")

    def _beta_key(node: Any) -> float:
        return float(getattr(node, "beta", float("inf")))

    return min(
        affordable_set,
        key=lambda x: (alpha_u * x[2] + gamma_u * x[1], _tau_key(x[0]), _beta_key(x[0])),
    )
