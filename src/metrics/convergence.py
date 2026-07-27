"""M8: Stackelberg Equilibrium Convergence."""

from __future__ import annotations

from typing import Any

import numpy as np

# Relative threshold: converged when the rolling coefficient of variation
# (std / mean) of p_dagger_ref falls below 1%. An absolute variance threshold
# is not meaningful here because p_dagger_ref spans many orders of magnitude
# across configs (USD/cycle prices range roughly 1e-11 to 1e-8), so a fixed
# absolute variance cutoff is either unreachable (too tight) or vacuous (too
# loose) depending on the price scale in force.
CONVERGENCE_CV_THRESHOLD: float = 0.01


def compute_stackelberg_convergence(
    quote_history: list[dict[str, Any]],
    window: float = 60.0,
) -> dict[str, Any]:
    """Track convergence of standing quotes to SPA equilibrium (M8).

    Computes rolling variance of p_dagger_ref over the last `window` seconds.
    Convergence is judged on the coefficient of variation (std / mean) rather
    than raw variance, since raw variance is not scale-invariant across the
    price magnitudes seen in different configs.

    Args:
        quote_history: List of dicts with "time" (seconds) and "p_dagger_ref" (USD/cycle).
        window: Rolling window length in seconds.

    Returns:
        Dict with keys: quote_variance (float, (USD/cycle)^2), quote_cv (float,
        dimensionless), and converged (bool).
    """
    if not quote_history:
        return {"quote_variance": 0.0, "quote_cv": 0.0, "converged": True}
    t_last = max(float(h["time"]) for h in quote_history)
    recent = [float(h["p_dagger_ref"]) for h in quote_history if float(h["time"]) > t_last - window]
    if not recent:
        return {"quote_variance": 0.0, "quote_cv": 0.0, "converged": True}
    variance = float(np.var(recent))
    mean = float(np.mean(recent))
    cv = float(np.sqrt(variance) / mean) if mean > 0.0 else 0.0
    return {
        "quote_variance": variance,
        "quote_cv": cv,
        "converged": bool(cv < CONVERGENCE_CV_THRESHOLD),
    }
