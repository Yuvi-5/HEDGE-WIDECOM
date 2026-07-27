"""M16: Reference-Price Myerson Efficiency."""

from __future__ import annotations

import bisect
from typing import Any


def compute_myerson_efficiency(
    events: list[dict[str, Any]],
    node_snapshots: list[dict[str, Any]],
) -> float:
    """Ratio of HEDGE-realised expected per-task profit to Myerson-optimal profit (M16).

    Procedure:
    1. Take buyer valuations from the logged "a_buyer" field (the true, private
       valuation the simulator generated for each task). Sellers never see this
       value during matching (pricing uses a_ref only, per PRICING_PIPELINE.md);
       using it here is offline analysis, not a live-mechanism input, so it does
       not violate the seller-privacy design. Falls back to the price_paid * 1.1
       proxy from earlier METRICS.md drafts only when "a_buyer" is absent (e.g.
       synthetic unit-test fixtures), to stay backward compatible.
    2. Find Myerson-optimal price p_M that maximises p * Pr(a_buyer >= p) over the
       empirical valuation distribution.
    3. Compute pi_Myerson = p_M * Pr(a_buyer >= p_M).
    4. Compute pi_HEDGE = total_revenue / n_arrivals (expected revenue per arrival).
    5. Return pi_HEDGE / pi_Myerson clamped to [0, 1].

    Per Remark 14 of PRICING_PIPELINE.md, this ratio should be >= 0.90.

    Args:
        events: Event log dicts; completed events need "price_paid" and ideally
                "a_buyer".
        node_snapshots: Unused in this computation; accepted for API consistency.

    Returns:
        Myerson efficiency ratio in [0, 1]; 1.0 if insufficient data.
    """
    completed = [e for e in events if e["event_type"] == "completed"]
    n_total = sum(1 for e in events if e["event_type"] in ("completed", "rejected"))

    if not completed or n_total == 0:
        return 1.0

    # True buyer valuations when logged; proxy fallback for older event schemas
    observed_valuations = [
        float(e["a_buyer"]) if "a_buyer" in e else float(e["price_paid"]) * 1.1 for e in completed
    ]

    # HEDGE realized expected revenue per arrival (includes rejected tasks at 0 revenue)
    total_revenue = sum(float(e["price_paid"]) for e in completed)
    pi_hedge = total_revenue / n_total

    # Myerson-optimal: maximize p * Pr(a_buyer >= p) over empirical distribution.
    # Search over all observed valuations as candidate prices. O(n log n) via a
    # sorted array + bisect (count of elements >= p), not the naive O(n^2)
    # double loop -- this matters at scale: 1e5+ completed tasks per seed
    # makes O(n^2) take tens of minutes per seed.
    sorted_asc = sorted(observed_valuations)
    n_obs = len(sorted_asc)
    pi_myerson = 0.0
    for p in observed_valuations:
        # Count of elements >= p: n_obs - (index of first element >= p)
        frac_above = (n_obs - bisect.bisect_left(sorted_asc, p)) / n_total
        pi_candidate = p * frac_above
        if pi_candidate > pi_myerson:
            pi_myerson = pi_candidate

    if pi_myerson <= 0.0:
        return 1.0

    return float(max(0.0, min(1.0, pi_hedge / pi_myerson)))
