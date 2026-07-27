"""M7: Carbon Emissions Per Task."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_carbon_per_task(events: list[dict[str, Any]]) -> float:
    """Mean gCO2 emitted per completed task (M7).

    carbon_g per task = beta_i * E_i / JOULES_PER_KWH (Equation 13 in SYSTEM_MODEL.md).
    The carbon_g field must be pre-computed and stored in each completed event.

    Args:
        events: Event log dicts; each completed event must have a "carbon_g" key.

    Returns:
        Mean gCO2 per task; 0.0 if no completed tasks.
    """
    carbon = [float(e["carbon_g"]) for e in events if e["event_type"] == "completed"]
    return float(np.mean(carbon)) if carbon else 0.0
