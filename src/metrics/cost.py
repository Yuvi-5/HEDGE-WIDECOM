"""M2: Mean Cost Per Accepted Task."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_mean_cost(events: list[dict[str, Any]]) -> float:
    """Mean settlement payment (USD) over all accepted tasks (M2).

    Args:
        events: Event log dicts; each completed event must have a "price_paid" key.

    Returns:
        Mean payment in USD; 0.0 if no completed tasks.
    """
    payments = [float(e["price_paid"]) for e in events if e["event_type"] == "completed"]
    return float(np.mean(payments)) if payments else 0.0
