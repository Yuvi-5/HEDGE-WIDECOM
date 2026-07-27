"""Tier-3 load inversion: infer node load state from an observed standing quote.

The Stackelberg price formula p_star = sqrt(a_ref * b * m * C1/w) can be inverted
to estimate m (and thus l_hat) from an observed p_star. This is used for market
intelligence (passive demand sensing) and is NOT on the RFQ hot path.
"""

from __future__ import annotations

import math

from hedge.core.constants import A_REF, B_SHAPE, ETA


def invert_load_from_price(
    p_star: float,
    C1_w: float,
    w: float,
    f_max: float,
    a_ref: float = A_REF,
    b: float = B_SHAPE,
    eta: float = ETA,
) -> float:
    """Estimate l_hat by inverting the Stackelberg pricing formula (Tier-3 inversion).

    Inverts: p_star = sqrt(a_ref * b * m * C1_w / w)
    to obtain m, then recovers l_hat from m = 1 + eta * l_hat / f_max.

    Args:
        p_star: Observed standing Stackelberg price (USD/cycle). Must be > 0.
        C1_w: Layer-1 total cost for reference task workload w (USD). Must be > 0.
        w: Reference task workload (cycles). Must be > 0.
        f_max: Maximum CPU frequency of the observed node (Hz).
        a_ref: Reference per-cycle valuation (USD/cycle). Default A_REF.
        b: Demand curve shape parameter. Default B_SHAPE.
        eta: Markup gain parameter. Default ETA.

    Returns:
        Estimated l_hat (cycles/s), clamped to [0, f_max]. Returns 0 if inputs invalid.
    """
    if p_star <= 0.0 or C1_w <= 0.0 or w <= 0.0 or f_max <= 0.0:
        return 0.0
    denom = a_ref * b * C1_w / w
    if denom <= 0.0:
        return 0.0
    m_inferred = (p_star**2) / denom
    l_hat_raw = (m_inferred - 1.0) / max(eta, 1e-30) * f_max
    return max(0.0, min(f_max, l_hat_raw))


def markup_from_price(
    p_star: float,
    C1_w: float,
    w: float,
    a_ref: float = A_REF,
    b: float = B_SHAPE,
) -> float:
    """Infer congestion markup m from an observed Stackelberg price.

    Args:
        p_star: Observed per-cycle price (USD/cycle).
        C1_w: Layer-1 total cost for task workload w (USD).
        w: Task workload (cycles).
        a_ref: Reference per-cycle valuation. Default A_REF.
        b: Demand shape. Default B_SHAPE.

    Returns:
        Inferred markup m in [1, 1+eta]. Clamped to [1, inf) and returned raw.
    """
    if p_star <= 0.0 or C1_w <= 0.0 or w <= 0.0:
        return 1.0
    c0 = C1_w / w
    denom = a_ref * b * c0
    if denom <= 0.0:
        return 1.0
    return max(1.0, (p_star**2) / denom)
