"""Layer-2.5 pricing: Strategic Price Adjustment (SPA) with Bertrand floor (Equations 40-42).

The SPA rule shades the Stackelberg price toward the cheapest peer's quote:
    p_dagger_i = max(C1_i/w, p_star_i * (1 - lambda_SPA_i * delta_comp_i))   [USD/cycle]

Structural guarantees (Proposition 4 / Invariant I5):
    C1_i/w <= p_dagger_i <= p_star_i

These hold unconditionally because:
  - max(C1_i/w, ...) enforces the lower Bertrand floor.
  - lambda_SPA_i in [0, lambda_max] and delta_comp_i in [0, 1) imply the shading factor
    (1 - lambda_SPA_i * delta_comp_i) is in (0, 1], so the shaded term <= p_star_i.
"""

from __future__ import annotations

from hedge.core.constants import LAMBDA_MAX, R_REF


def compute_competitiveness_signal(
    p_star_ref_self: float,
    peer_reference_quotes: list[float],
) -> float:
    """Compute competitiveness signal delta_comp_i in [0, 1) (Equation 40).

    Args:
        p_star_ref_self: Node i's own standing quote in reference basis (USD/cycle). Must be > 0.
        peer_reference_quotes: Cached standing quotes from all other peers (USD/cycle).
            Empty list means no competition; returns 0.

    Returns:
        delta_comp_i in [0, 1). Zero when node i is already the cheapest peer.
    """
    if not peer_reference_quotes or p_star_ref_self <= 0.0:
        return 0.0
    min_peer_quote = min(peer_reference_quotes)
    raw = (p_star_ref_self - min_peer_quote) / p_star_ref_self
    return max(0.0, raw)


def compute_aggressiveness(
    l_hat: float,
    f_max: float,
    R_hat: float,
    R_ref: float = R_REF,
    lambda_max: float = LAMBDA_MAX,
) -> float:
    """Compute SPA aggressiveness coefficient lambda_SPA_i in [0, lambda_max] (Equation 42).

    Args:
        l_hat: Kalman load forecast (Hz = cycles/s), clamped by filter to [0, f_max].
        f_max: Maximum node CPU frequency (Hz).
        R_hat: EMA revenue forecast (USD/RFQ).
        R_ref: Reference revenue calibration constant (USD/RFQ). Default R_REF = 1.0.
        lambda_max: Protocol aggressiveness cap. Default LAMBDA_MAX = 0.5.

    Returns:
        lambda_SPA_i in [0, lambda_max]. Zero when node is saturated or revenue >= R_ref.
    """
    load_factor = 1.0 - l_hat / f_max
    revenue_factor = 1.0 - R_hat / R_ref
    raw = lambda_max * load_factor * revenue_factor
    return max(0.0, min(lambda_max, raw))


def compute_spa_price(
    C1_w: float,
    w: float,
    p_star: float,
    lambda_SPA: float,
    delta_comp: float,
) -> float:
    """Compute Layer-2.5 shaded per-cycle price p_dagger_i (Equation 41, USD/cycle).

    Structural guarantee (Bertrand floor, Invariant I1):
        p_dagger >= C1_w / w   for all valid inputs.
    Cascade monotonicity (Invariant I5):
        C1_w / w <= p_dagger <= p_star.

    Args:
        C1_w: Layer-1 total cost for the task (USD). Must be > 0.
        w: Task CPU workload (cycles). Must be > 0.
        p_star: Layer-2 Stackelberg per-cycle price (USD/cycle). Must be >= C1_w/w.
        lambda_SPA: Aggressiveness in [0, lambda_max].
        delta_comp: Competitiveness signal in [0, 1).

    Returns:
        p_dagger_i: Shaded per-cycle price (USD/cycle).
    """
    bertrand_floor = C1_w / w
    shaded = p_star * (1.0 - lambda_SPA * delta_comp)
    return max(bertrand_floor, shaded)
