"""RFQ round (Algorithm 1): full single-round market protocol from broadcast to settlement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hedge.core.constants import K_MAX, LAMBDA_MAX, S_RET_FRACTION
from hedge.market.afgm import (
    afgm_select,
    build_affordable_set,
    compute_cloud_latency,
    compute_node_latency,
)
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price
from hedge.pricing.layer2_5 import (
    compute_aggressiveness,
    compute_competitiveness_signal,
    compute_spa_price,
)

if TYPE_CHECKING:
    from hedge.core.task import HEDGETask


@dataclass
class RFQResult:
    """Result of a single RFQ round."""

    winner: Any  # node or CloudNode; None if unmatched
    price_per_cycle: float  # p_dagger_i (USD/cycle); inf if unmatched
    price_total: float  # p_dagger_i * w (USD); inf if unmatched
    executor_type: str  # "edge" | "cloud" | "unmatched"
    message_count: int
    affordable_set_size: int
    C1_winner: float  # Layer-1 cost for winning task (USD); 0.0 if unmatched
    seller_receipt: float  # equals price_total (no subsidy); used for WBB check


def quote_node(
    node: Any,
    task: "HEDGETask",
    peer_standing_quotes: dict[int, float],
    tau: float,
    bandwidth: float = 1e9,
    lambda_max: float = LAMBDA_MAX,
) -> tuple[float, float, float, float]:
    """Compute Layer-2.5 quote for node i in response to RFQ (Steps 2a-2h, Algorithm 1).

    Args:
        node: Edge server with hardware and predictor attributes.
        task: Task (s, w, d, ...).
        peer_standing_quotes: Cached p_star_ref values from other peers, keyed by unique_id.
            Used for SPA delta_comp. This simulates the Tier-1 broadcast cache.
        tau: One-way propagation delay from buyer to node i (seconds).
        bandwidth: Link bandwidth (bits/s). Default 1 Gbps.
        lambda_max: Protocol aggressiveness cap for SPA shading. Default LAMBDA_MAX.

    Returns:
        (p_dagger, p_star, L_i, C1_w): Shaded price per cycle, unshaded price per cycle,
        latency (seconds), and total Layer-1 cost (USD).
    """
    C1_w, _, _, _ = compute_layer1_cost(
        task.w, node.f_max, node.kappa, node.P_idle, node.rho, node.pi_E, node.beta
    )
    m_i = compute_markup(node.l_hat, node.f_max)
    p_star = compute_stackelberg_price(C1_w, task.w, m_i)

    peer_quotes = [v for k, v in peer_standing_quotes.items() if k != node.unique_id]
    delta_comp = compute_competitiveness_signal(p_star, peer_quotes)
    lambda_SPA = compute_aggressiveness(node.l_hat, node.f_max, node.R_hat, lambda_max=lambda_max)
    p_dagger = compute_spa_price(C1_w, task.w, p_star, lambda_SPA, delta_comp)

    L_i = compute_node_latency(task, node, tau, bandwidth)
    return p_dagger, p_star, L_i, C1_w


def run_rfq_round(
    peer_pool: list[Any],
    task: "HEDGETask",
    cloud: Any,
    tau_dict: dict[int, float],
    peer_standing_quotes: dict[int, float] | None = None,
    k_max: int = K_MAX,
    alpha_u: float = 0.5,
    gamma_u: float = 0.5,
    bandwidth_metro: float = 1e9,
    bandwidth_cloud: float = 1e8,
    lambda_max: float = LAMBDA_MAX,
) -> RFQResult:
    """Execute Algorithm 1 RFQ round from broadcast to settlement.

    Args:
        peer_pool: Candidate peer nodes (capped at k_max). Each must expose
            unique_id, f_max, kappa, P_idle, rho, pi_E, beta, l_hat, R_hat, w_pending.
        task: Task being offered. task.a_buyer is private; never passed to peers.
        cloud: CloudNode with mu_c, f_max, kappa, P_idle, rho, pi_E, beta, tau_c.
        tau_dict: {node.unique_id: tau} one-way delays from buyer to each node (seconds).
        peer_standing_quotes: Cached p_star_ref by node unique_id for SPA delta_comp.
            If None, SPA uses no peer context (delta_comp=0 for all, no shading).
        k_max: Max peers contacted (default K_MAX = 4).
        alpha_u: Latency weight in AFGM (default 0.5).
        gamma_u: Price weight in AFGM (default 0.5).
        bandwidth_metro: Metro link bandwidth (bits/s). Default 1 Gbps.
        bandwidth_cloud: Cloud uplink bandwidth (bits/s). Default 100 Mbps.
        lambda_max: Protocol aggressiveness cap for SPA shading. Default LAMBDA_MAX.

    Returns:
        RFQResult with winner, price, executor_type, message_count, and WBB fields.
    """
    if peer_standing_quotes is None:
        peer_standing_quotes = {}

    peers = peer_pool[:k_max]
    k_actual = len(peers)
    messages = k_actual  # Step 1: k RFQ broadcasts

    # Step 2: collect quotes from peers
    peers_with_quotes: list[tuple[Any, float, float]] = []
    for peer in peers:
        tau = tau_dict.get(peer.unique_id, 0.005)
        p_dagger, _p_star, L_i, _C1 = quote_node(
            peer, task, peer_standing_quotes, tau, bandwidth_metro, lambda_max=lambda_max
        )
        peers_with_quotes.append((peer, p_dagger, L_i))
        messages += 1  # quote reply

    # Step 3: cloud quote (posted price, no fresh computation)
    C1_cloud, _, _, _ = compute_layer1_cost(
        task.w, cloud.f_max, cloud.kappa, cloud.P_idle, cloud.rho, cloud.pi_E, cloud.beta
    )
    p_c = cloud.mu_c * C1_cloud / task.w
    L_c = compute_cloud_latency(task, cloud, bandwidth_cloud)
    cloud_with_quote: tuple[Any, float, float] = (cloud, p_c, L_c)

    # Step 4: build affordable set F_b (buyer IR gate)
    affordable = build_affordable_set(peers_with_quotes, cloud_with_quote, task, task.a_buyer)
    affordable_size = len(affordable)

    # Step 5: empty F_b - no match
    if not affordable:
        messages += k_actual  # RELEASE all (no winner to accept)
        return RFQResult(
            winner=None,
            price_per_cycle=float("inf"),
            price_total=float("inf"),
            executor_type="unmatched",
            message_count=messages,
            affordable_set_size=0,
            C1_winner=0.0,
            seller_receipt=0.0,
        )

    # Step 6: AFGM selection
    messages += affordable_size  # AFGM notifications to affordable sellers

    winner_tuple = afgm_select(affordable, alpha_u, gamma_u, tau_dict=tau_dict)
    assert winner_tuple is not None  # affordable is non-empty
    winner, winner_price, _winner_L = winner_tuple

    # Step 7: ACCEPT + RELEASE
    messages += 1  # ACCEPT to winner
    messages += max(0, affordable_size - 1)  # RELEASE to non-winners

    # Determine executor type (cloud has no unique_id; use mu_c attribute as marker)
    is_cloud = hasattr(winner, "mu_c")
    executor_type = "cloud" if is_cloud else "edge"

    # Compute winner C1 for seller IR audit
    if is_cloud:
        C1_w = C1_cloud
    else:
        C1_w, _, _, _ = compute_layer1_cost(
            task.w, winner.f_max, winner.kappa, winner.P_idle, winner.rho, winner.pi_E, winner.beta
        )

    price_total = winner_price * task.w

    return RFQResult(
        winner=winner,
        price_per_cycle=winner_price,
        price_total=price_total,
        executor_type=executor_type,
        message_count=messages,
        affordable_set_size=affordable_size,
        C1_winner=C1_w,
        seller_receipt=price_total,  # no subsidy: seller receives exactly what buyer pays
    )
