"""Invariant I1: Bertrand floor - p_dagger >= C1/w for all valid inputs.

These tests must never be broken regardless of configuration changes.
See INVARIANTS.md I1 and I5 for the mathematical guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from hedge.core.constants import LAMBDA_MAX, R_REF
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.pricing.layer2 import compute_markup, compute_stackelberg_price
from hedge.pricing.layer2_5 import (
    compute_aggressiveness,
    compute_competitiveness_signal,
    compute_spa_price,
)

# ---------------------------------------------------------------------------
# Gate test: Bertrand floor holds for 10,000 random (node, task) pairs (I1)
# ---------------------------------------------------------------------------


def test_bertrand_floor_10k() -> None:
    """p_dagger >= C1/w for 10,000 random (node, task, load, revenue) inputs.

    This is the primary invariant test for Proposition 4 (Invariant I1).
    Covers the full valid parameter space for nodes and tasks in this simulator.
    """
    rng = np.random.default_rng(42)
    violations: list[str] = []

    for _ in range(10_000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))
        l_hat = float(rng.uniform(0.0, f_max))
        R_hat = float(rng.uniform(0.0, 5.0))
        # Random peer reference quotes (0 to 5 peers)
        n_peers = int(rng.integers(0, 6))
        peers = [float(rng.uniform(1e-13, 1e-9)) for _ in range(n_peers)]

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(l_hat, f_max)
        p_star = compute_stackelberg_price(C1_w, w, m_i)

        p_star_self = p_star  # reference basis equals actual (per-cycle price)
        delta_comp = compute_competitiveness_signal(p_star_self, peers)
        lambda_SPA = compute_aggressiveness(l_hat, f_max, R_hat)
        p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA, delta_comp)
        floor = C1_w / w

        if p_dagger < floor - 1e-20:
            violations.append(
                f"p_dagger={p_dagger:.4e} < floor={floor:.4e} "
                f"(f_max={f_max:.2e}, w={w:.2e}, l_hat={l_hat:.2e})"
            )

    assert not violations, (
        f"Bertrand floor violated {len(violations)}/10000 times. " f"First: {violations[0]}"
    )


# ---------------------------------------------------------------------------
# Adversarial: maximum shading parameters should still respect the floor
# ---------------------------------------------------------------------------


def test_bertrand_floor_adversarial_max_shading() -> None:
    """Floor holds even at maximum lambda_SPA and delta_comp near 1 (worst-case shading)."""
    rng = np.random.default_rng(7)
    for _ in range(1000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(0.0, f_max)  # idle node: maximum markup headroom
        p_star = compute_stackelberg_price(C1_w, w, m_i)

        # Maximum aggressiveness possible
        lambda_SPA = LAMBDA_MAX
        # delta_comp near 1 (not exactly 1 - that would require min_peer = 0)
        delta_comp = 0.9999

        p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA, delta_comp)
        floor = C1_w / w

        assert (
            p_dagger >= floor - 1e-20
        ), f"Adversarial floor violated: p_dagger={p_dagger:.4e}, floor={floor:.4e}"


# ---------------------------------------------------------------------------
# Cascade monotonicity: C1/w <= p_dagger <= p_star (Invariant I5)
# ---------------------------------------------------------------------------


def test_cascade_monotonicity_10k() -> None:
    """C1/w <= p_dagger <= p_star for 10,000 random inputs (Invariant I5)."""
    rng = np.random.default_rng(55)
    violations: list[str] = []

    for _ in range(10_000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        rho = float(rng.uniform(2e-4, 2e-3))
        pi_E = float(rng.uniform(0.06, 0.18))
        beta = float(rng.uniform(40.0, 700.0))
        w = float(rng.uniform(1e9, 1e11))
        l_hat = float(rng.uniform(0.0, f_max))
        R_hat = float(rng.uniform(0.0, 5.0))
        n_peers = int(rng.integers(0, 6))
        peers = [float(rng.uniform(1e-13, 1e-9)) for _ in range(n_peers)]

        C1_w, _, _, _ = compute_layer1_cost(w, f_max, kappa, P_idle, rho, pi_E, beta)
        m_i = compute_markup(l_hat, f_max)
        p_star = compute_stackelberg_price(C1_w, w, m_i)
        delta_comp = compute_competitiveness_signal(p_star, peers)
        lambda_SPA = compute_aggressiveness(l_hat, f_max, R_hat)
        p_dagger = compute_spa_price(C1_w, w, p_star, lambda_SPA, delta_comp)
        floor = C1_w / w

        if p_dagger < floor - 1e-20:
            violations.append(f"LOWER: p_dagger={p_dagger:.4e} < C1/w={floor:.4e}")
        elif p_dagger > p_star + 1e-20:
            violations.append(f"UPPER: p_dagger={p_dagger:.4e} > p_star={p_star:.4e}")

    assert not violations, (
        f"Cascade monotonicity violated {len(violations)}/10000 times. " f"First: {violations[0]}"
    )


# ---------------------------------------------------------------------------
# Aggressiveness bounds: lambda_SPA in [0, lambda_max] (Invariant I7)
# ---------------------------------------------------------------------------


def test_aggressiveness_bounds_10k() -> None:
    """lambda_SPA in [0, lambda_max] for 10,000 random inputs (Invariant I7)."""
    f_max = 3e9
    rng = np.random.default_rng(99)
    for _ in range(10_000):
        l_hat = float(rng.uniform(0.0, 2.0 * f_max))  # allow beyond f_max
        R_hat = float(rng.uniform(-2.0, 10.0))  # allow negative R_hat
        lam = compute_aggressiveness(l_hat, f_max, R_hat)
        assert (
            0.0 <= lam <= LAMBDA_MAX + 1e-12
        ), f"I7 violated: lambda_SPA={lam:.4e} outside [0, {LAMBDA_MAX}]"
