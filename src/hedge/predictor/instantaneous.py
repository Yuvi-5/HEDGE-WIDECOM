"""Instantaneous (no-forecast) load estimator (ablation A4: No Kalman).

l_hat_i(t) = y_i(t), the raw observed load with no smoothing or prediction.
Same update-per-tick interface (update() / get_latest_forecast()) as other
load predictors, so the engine can swap forecasters via config without
touching the cadence loop.
"""

from __future__ import annotations


class InstantaneousLoadFilter:
    """Pass-through load estimator: l_hat = clip(y_observed, 0, f_max)."""

    def __init__(self, f_max: float) -> None:
        """Initialise the pass-through estimator.

        Args:
            f_max: Maximum node frequency (Hz); used for output clamp.
        """
        if f_max <= 0.0:
            raise ValueError(f"f_max must be > 0, got {f_max}")
        self.f_max: float = f_max
        self._l_hat: float = 0.0

    def update(self, y_observed: float) -> float:
        """Return the clamped raw observation with no filtering.

        Args:
            y_observed: Measured load at this cadence tick (Hz = cycles/s).

        Returns:
            l_hat: clip(y_observed, 0, f_max).
        """
        self._l_hat = float(min(max(y_observed, 0.0), self.f_max))
        return self._l_hat

    def get_latest_forecast(self) -> float:
        """Return the most recently computed l_hat (O(1) cache read)."""
        return self._l_hat
