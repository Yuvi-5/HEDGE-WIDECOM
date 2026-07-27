"""Gate tests: InstantaneousLoadFilter.

Coverage:
- Output stays clamped to [0, f_max].
- Construction-time validation.
- Pure passthrough behaviour (l_hat == raw observation, once clamped).
"""

from __future__ import annotations

import pytest

from hedge.predictor.instantaneous import InstantaneousLoadFilter

F_MAX = 3e9


def test_instantaneous_rejects_invalid_f_max() -> None:
    with pytest.raises(ValueError):
        InstantaneousLoadFilter(f_max=-1.0)


def test_instantaneous_is_pure_passthrough() -> None:
    f = InstantaneousLoadFilter(f_max=F_MAX)
    assert f.update(0.3 * F_MAX) == pytest.approx(0.3 * F_MAX)
    assert f.update(0.9 * F_MAX) == pytest.approx(0.9 * F_MAX)
    assert f.update(0.0) == pytest.approx(0.0)


def test_instantaneous_clamps_to_f_max() -> None:
    f = InstantaneousLoadFilter(f_max=F_MAX)
    assert f.update(10.0 * F_MAX) == pytest.approx(F_MAX)
    assert f.update(-1.0) == pytest.approx(0.0)
