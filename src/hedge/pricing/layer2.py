"""Layer-2 pricing: congestion markup + closed-form Stackelberg price (Equations 33, 38).

The monopoly Stackelberg price is:
    p_star_i = sqrt(a_ref * b * m_i * C1_i(w) / w)   [USD/cycle]

Because C1_i(w)/w = c0_i is constant in w (Layer-1 linearity), p_star_i is also constant
in w. Total transaction price p_star_i * w is therefore exactly linear in w (no batching
arbitrage).

The price is strictly above the Bertrand floor C1_i/w = c0_i for all parameter values
used in this simulator, since a_ref * b * m_i >= a_ref = 5e-10 >> c0_max ~ 1e-12.
"""

from __future__ import annotations

import math

from hedge.core.constants import A_REF, B_SHAPE, ETA


def compute_markup(
    l_hat_next: float,
    f_max: float,
    eta: float = ETA,
) -> float:
    """Compute congestion markup multiplier m_i in [1, 1+eta] (Equation 33).

    Args:
        l_hat_next: Kalman load forecast one step ahead (Hz = cycles/s).
            Clamped by the Kalman filter, but the clip here is explicit per Invariant I6.
        f_max: Maximum node CPU frequency (Hz).
        eta: Markup gain parameter (dimensionless). Default ETA = 1.0.

    Returns:
        m_i in [1.0, 1.0 + eta]. Value is 1.0 when node is empty, 1+eta at full load.
    """
    m = 1.0 + eta * l_hat_next / f_max
    # Explicit clip enforces Invariant I6 even if l_hat_next drifts slightly outside [0, f_max]
    return max(1.0, min(1.0 + eta, m))


def compute_stackelberg_price(
    C1_w: float,
    w: float,
    m_i: float,
    a_ref: float = A_REF,
    b: float = B_SHAPE,
) -> float:
    """Compute the Layer-2 Stackelberg per-cycle price p_star_i (Equation 38, USD/cycle).

    Args:
        C1_w: Layer-1 total cost for the task (USD). Must be > 0.
        w: Task CPU workload (cycles). Must be > 0.
        m_i: Congestion markup multiplier in [1, 1+eta].
        a_ref: Deployment-wide reference per-cycle valuation (USD/cycle). Default A_REF = 5e-10.
        b: Demand curve shape parameter. Default B_SHAPE = 1.0.

    Returns:
        p_star_i: Stackelberg per-cycle price (USD/cycle).
            Total price = p_star_i * w is linear in w.
            Always > C1_w / w for parameter ranges in this simulator (see module docstring).
    """
    c0: float = C1_w / w  # per-cycle cost (USD/cycle), constant in w due to L1 linearity
    return math.sqrt(a_ref * b * m_i * c0)
