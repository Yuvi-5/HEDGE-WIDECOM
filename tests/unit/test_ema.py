"""Phase 3 gate tests: EMA revenue tracker."""

from __future__ import annotations

import math

from hedge.core.constants import ALPHA_R, R_REF
from hedge.predictor.ema import EMARevenueTracker

# ---------------------------------------------------------------------------
# EMA revenue tracker
# ---------------------------------------------------------------------------


def test_ema_init_at_R_ref() -> None:
    """EMARevenueTracker initialises at R_ref."""
    ema = EMARevenueTracker(R_ref=R_REF, alpha_R=ALPHA_R)
    assert math.isclose(ema.get_latest_forecast(), R_REF, rel_tol=1e-12)


def test_ema_update_increases_on_high_settlement() -> None:
    """R_hat increases when settlement > R_hat (Eq. 28)."""
    ema = EMARevenueTracker(R_ref=1.0, alpha_R=0.2)
    ema.on_task_settlement(5.0)  # much higher than R_ref = 1.0
    assert ema.get_latest_forecast() > 1.0


def test_ema_update_decreases_on_low_settlement() -> None:
    """R_hat decreases when settlement < R_hat."""
    ema = EMARevenueTracker(R_ref=5.0, alpha_R=0.2)
    ema.on_task_settlement(0.0)
    assert ema.get_latest_forecast() < 5.0


def test_ema_decays_toward_settlement() -> None:
    """After many settlements at a constant value, R_hat converges to that value."""
    target = 3.0
    ema = EMARevenueTracker(R_ref=1.0, alpha_R=0.2)
    for _ in range(100):
        ema.on_task_settlement(target)
    assert math.isclose(ema.get_latest_forecast(), target, rel_tol=0.01)


def test_ema_unchanged_without_settlement() -> None:
    """R_hat stays constant when no settlements are reported."""
    ema = EMARevenueTracker(R_ref=2.0, alpha_R=0.2)
    r1 = ema.get_latest_forecast()
    r2 = ema.get_latest_forecast()
    assert r1 == r2 == 2.0


def test_ema_alpha_controls_speed() -> None:
    """Higher alpha_R means faster convergence to realised value."""
    target = 5.0
    ema_fast = EMARevenueTracker(R_ref=1.0, alpha_R=0.5)
    ema_slow = EMARevenueTracker(R_ref=1.0, alpha_R=0.1)
    for _ in range(20):
        ema_fast.on_task_settlement(target)
        ema_slow.on_task_settlement(target)
    # Faster alpha should be closer to target
    assert abs(ema_fast.get_latest_forecast() - target) < abs(
        ema_slow.get_latest_forecast() - target
    )


def test_ema_exact_formula() -> None:
    """EMA update matches Eq. 28: R_hat = (1-a)*R_hat + a*R_realised exactly."""
    alpha = 0.3
    R0 = 2.0
    R_real = 5.0
    ema = EMARevenueTracker(R_ref=R0, alpha_R=alpha)
    ema.on_task_settlement(R_real)
    expected = (1 - alpha) * R0 + alpha * R_real
    assert math.isclose(ema.get_latest_forecast(), expected, rel_tol=1e-12)
