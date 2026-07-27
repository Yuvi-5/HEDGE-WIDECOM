"""EMA revenue tracker for HEDGE edge nodes (Equation 28)."""

from __future__ import annotations

from hedge.core.constants import ALPHA_R, R_REF


class EMARevenueTracker:
    """Exponential moving average of realised per-task revenue.

    Updated on task completion events (event-driven, not cadence-driven).
    Initialised at R_ref so pricing is well-defined before any task settles.
    """

    def __init__(self, R_ref: float = R_REF, alpha_R: float = ALPHA_R) -> None:
        """Initialise at R_ref.

        Args:
            R_ref: Reference revenue used for initialisation (USD/RFQ).
            alpha_R: Smoothing coefficient in (0, 1). Default 0.2 gives 5-step time constant.
        """
        if not (0.0 < alpha_R < 1.0):
            raise ValueError(f"alpha_R must be in (0, 1), got {alpha_R}")
        self._R_hat: float = R_ref
        self.alpha_R: float = alpha_R

    def on_task_settlement(self, R_realised: float) -> None:
        """Update EMA on task completion event (Equation 28).

        Args:
            R_realised: Realised settlement revenue for the completed task (USD).
        """
        R_realised = max(0.0, R_realised)  # clamp: negative revenue breaks layer2_5
        self._R_hat = (1.0 - self.alpha_R) * self._R_hat + self.alpha_R * R_realised

    def get_latest_forecast(self) -> float:
        """Return current R_hat (O(1) cache read).

        Returns:
            Smoothed revenue estimate (USD/RFQ). Non-negative when R_realised >= 0.
        """
        return self._R_hat
