"""Phase 0: Orchestrator selection via A_eco competitiveness score (Equation 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hedge.core.constants import K_MAX, THETA_RISK
from hedge.pricing.layer1 import compute_layer1_cost

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


def compute_aeco_score(
    p_dagger_ref: float,
    tau_ui: float,
    all_p_dagger_refs: list[float],
    all_tau_uis: list[float],
    x_weight: float = 0.5,
    y_weight: float = 0.5,
) -> float:
    """Compute normalised A_eco(u, i) competitiveness score (Equation 4).

    Args:
        p_dagger_ref: Node i's shaded standing quote (USD/cycle). Must be > 0.
        tau_ui: Propagation delay from user to node i (seconds). Must be > 0.
        all_p_dagger_refs: Shaded standing quotes for all candidates in C_u.
        all_tau_uis: Propagation delays for all candidates in C_u.
        x_weight: Price weight (default 0.5). Must satisfy x_weight + y_weight = 1.
        y_weight: Latency weight (default 0.5).

    Returns:
        Normalised score in [0, 1]. Higher is better for orchestrator role.
    """
    if p_dagger_ref <= 0.0 or tau_ui <= 0.0:
        return 0.0
    total = sum(
        x_weight / max(p, 1e-30) + y_weight / max(t, 1e-30)
        for p, t in zip(all_p_dagger_refs, all_tau_uis)
        if p > 0.0 and t > 0.0
    )
    if total == 0.0:
        return 0.0
    return (x_weight / p_dagger_ref + y_weight / tau_ui) / total


def is_capacity_feasible(l_hat: float, f_max: float, w: float, d: float) -> bool:
    """Check capacity feasibility: l_hat + w/d <= f_max.

    Args:
        l_hat: Current load estimate at node i (cycles/s).
        f_max: Maximum CPU frequency of node i (Hz).
        w: Task CPU workload (cycles).
        d: Task deadline (seconds). Must be > 0.

    Returns:
        True if node i can admit the task without overloading.
    """
    return l_hat + w / max(d, 1e-12) <= f_max


def compute_gc_entry_bid(
    node: Any,
    task: "HEDGETask",
    p_hat_dagger_j: float,
    theta_risk: float = THETA_RISK,
) -> float | None:
    """Check whether an overloaded node may enter Phase 0 with a GC-play bid (Eq 6-8).

    Entry condition (Eq 6):
        C1_i/w <= p_bid_i < p_star_i   AND   p_bid_i - p_hat_dagger_j >= theta_risk * C1_i/w

    p_bid_i is the node's own cached Layer-2.5 shaded ask (p_dagger_ref); this
    reuses the same "no fresh computation required" cached-quote design as the
    rest of Phase 0 (Section 1a, step 2).

    Args:
        node: Candidate node with p_dagger_ref, p_star_ref, f_max, kappa, P_idle,
              rho, pi_E, beta attributes.
        task: Task (w cycles) being offered.
        p_hat_dagger_j: Best (lowest) quote already accepted into the pool that
            this node's bid must beat by the risk-adjusted margin.
        theta_risk: Risk-premium coefficient (default THETA_RISK = 0.12).

    Returns:
        p_bid_i (the node's own p_dagger_ref) if the entry condition holds, else None.
    """
    C1_i, _, _, _ = compute_layer1_cost(
        task.w, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
    )
    floor = C1_i / max(task.w, 1e-30)
    p_bid = float(node.p_dagger_ref)

    if not (floor <= p_bid < float(node.p_star_ref)):
        return None
    if p_bid - p_hat_dagger_j >= theta_risk * floor:
        return p_bid
    return None


def run_phase0(
    candidate_nodes: list[Any],
    task: "HEDGETask",
    tau_dict: dict[int, float],
    x_weight: float = 0.5,
    y_weight: float = 0.5,
    k_max: int = K_MAX,
    enable_gc_risk_premium: bool = False,
    theta_risk: float = THETA_RISK,
    all_nodes: list[Any] | None = None,
) -> tuple[Any, list[Any]]:
    """Run Phase 0 orchestrator selection for a single task.

    Args:
        candidate_nodes: Nodes in user's radio coverage set C_u. Each must expose
            unique_id (int), p_dagger_ref (float), p_star_ref (float), f_max (float),
            l_hat (float).
        task: Task (s, w, d) being offered.
        tau_dict: {node.unique_id: tau_ui} propagation delays (seconds). Must cover
            every node in all_nodes, not just candidate_nodes, when all_nodes is set.
        x_weight: Price weight in A_eco (default 0.5).
        y_weight: Latency weight in A_eco (default 0.5).
        k_max: Maximum peer pool size (default K_MAX).
        enable_gc_risk_premium: If True, overloaded losers that fail the standard
            capacity-feasibility check may still enter the peer pool via the GC
            risk-premium bid (Section 1e, Eq 6-8), filling any remaining slots up
            to k_max. Default False preserves the original feasible-losers-only
            pool for backward compatibility.
        theta_risk: Risk-premium coefficient for GC-play entry (default THETA_RISK).
        all_nodes: Full topology node list. When given and the pool is still short
            of k_max after feasible losers and GC-play entrants, remaining slots are
            filled by proximity from nodes outside C_u, nearest-by-tau first,
            feasibility-checked. None (default)
            skips this step -- e.g. when coverage gating is disabled, candidate_nodes
            already equals the full node list and there is nothing outside C_u to draw
            from.

    Returns:
        (orchestrator, peer_pool_P0): Winner node and feasible-loser pool (len <= k_max),
        optionally extended with GC-play entrants and/or outside-C_u proximity fill.

    Raises:
        ValueError: If candidate_nodes is empty.
    """
    if not candidate_nodes:
        raise ValueError("candidate_nodes must be non-empty for Phase 0")

    all_p_dagger = [max(n.p_dagger_ref, 1e-30) for n in candidate_nodes]
    all_tau = [max(tau_dict.get(n.unique_id, 0.001), 1e-30) for n in candidate_nodes]

    # Equivalent to calling compute_aeco_score(p, t, all_p_dagger, all_tau, ...) once
    # per candidate: both lists are already clamped strictly positive above, so the
    # normalisation sum inside compute_aeco_score (and its p>0/t>0 filter) is the same
    # total for every candidate. Computing it once here avoids recomputing an O(N) sum
    # N times (O(N^2) per RFQ round), which dominated runtime at larger N_nodes.
    total = sum(x_weight / p + y_weight / t for p, t in zip(all_p_dagger, all_tau))
    scores = (
        [0.0] * len(candidate_nodes)
        if total == 0.0
        else [(x_weight / p + y_weight / t) / total for p, t in zip(all_p_dagger, all_tau)]
    )

    winner_idx = max(range(len(scores)), key=lambda i: scores[i])
    orchestrator = candidate_nodes[winner_idx]

    losers = [(i, n) for i, n in enumerate(candidate_nodes) if i != winner_idx]
    feasible_losers = [
        n for i, n in losers if is_capacity_feasible(n.l_hat, n.f_max, task.w, task.d)
    ]
    peer_pool = feasible_losers[:k_max]

    if enable_gc_risk_premium and len(peer_pool) < k_max:
        feasible_ids = {n.unique_id for n in peer_pool}
        overloaded = [n for i, n in losers if n.unique_id not in feasible_ids]
        # Best price already in the pool is what a GC-play entrant must beat by
        # the risk-adjusted margin; fall back to the orchestrator's own price.
        pool_prices = [orchestrator.p_dagger_ref] + [n.p_dagger_ref for n in peer_pool]
        p_hat_dagger_j = min(pool_prices)
        # Consider the most competitive (lowest-quote) overloaded candidates first.
        for n in sorted(overloaded, key=lambda n: n.p_dagger_ref):
            if len(peer_pool) >= k_max:
                break
            if compute_gc_entry_bid(n, task, p_hat_dagger_j, theta_risk) is not None:
                peer_pool.append(n)
                p_hat_dagger_j = min(p_hat_dagger_j, n.p_dagger_ref)

    if all_nodes is not None and len(peer_pool) < k_max:
        in_pool_ids = {n.unique_id for n in peer_pool}
        in_pool_ids.add(orchestrator.unique_id)
        candidate_ids = {n.unique_id for n in candidate_nodes}
        outside_c_u = [
            n
            for n in all_nodes
            if n.unique_id not in candidate_ids
            and n.unique_id not in in_pool_ids
            and is_capacity_feasible(n.l_hat, n.f_max, task.w, task.d)
        ]
        outside_c_u.sort(key=lambda n: tau_dict.get(n.unique_id, float("inf")))
        for n in outside_c_u:
            if len(peer_pool) >= k_max:
                break
            peer_pool.append(n)

    return orchestrator, peer_pool[:k_max]
