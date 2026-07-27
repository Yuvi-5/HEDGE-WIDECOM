"""Invariant tests: I2 (seller IR) and I3 (buyer IR) for all matched tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hedge.core.constants import MU_C, R_REF
from hedge.core.task import HEDGETask
from hedge.market.rfq import run_rfq_round
from hedge.pricing.layer1 import compute_layer1_cost


@dataclass
class _Node:
    unique_id: int
    f_max: float
    kappa: float
    rho: float
    P_idle: float
    beta: float
    pi_E: float
    w_pending: float = 0.0
    l_hat: float = 0.0
    R_hat: float = 0.0
    p_star_ref: float = 0.0
    p_dagger_ref: float = 0.0

    @property
    def W_q(self) -> float:
        """Queue wait scalar (seconds)."""
        return self.w_pending / self.f_max


@dataclass
class _Cloud:
    f_max: float = 1e12
    kappa: float = 1e-27
    rho: float = 1e-4
    P_idle: float = 0.0
    beta: float = 80.0
    pi_E: float = 0.06
    mu_c: float = MU_C
    tau_c: float = 0.035


CLOUD = _Cloud()


def test_seller_ir_1000_rounds() -> None:
    """I2: price_total >= C1_winner for all matched tasks (seller never loses).

    Invariant I2 follows from Invariant I1 (Bertrand floor):
    p_dagger * w >= (C1/w) * w = C1.
    """
    rng = np.random.default_rng(42)
    violations: list[str] = []

    for i in range(1000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        kappa = float(10 ** rng.uniform(-27, -25))
        rho = float(rng.uniform(2e-4, 1e-3))
        P_idle = float(rng.choice([20.0, 30.0, 50.0]))
        beta = float(rng.uniform(40.0, 700.0))
        pi_E = float(rng.uniform(0.06, 0.18))
        l_hat_frac = float(rng.uniform(0.0, 0.9))

        node = _Node(
            unique_id=i,
            f_max=f_max,
            kappa=kappa,
            rho=rho,
            P_idle=P_idle,
            beta=beta,
            pi_E=pi_E,
            l_hat=l_hat_frac * f_max,
            R_hat=float(rng.uniform(0.0, 2.0)),
        )

        w = float(rng.uniform(1e9, 1e10))
        a_buyer = float(rng.uniform(0.1, 10.0))
        d = float(rng.uniform(0.2, 2.0))
        task = HEDGETask(f"T{i}", s=2e6, w=w, d=d, a_buyer=a_buyer, created_at=0.0)
        tau_dict = {i: 0.002}

        result = run_rfq_round(
            peer_pool=[node],
            task=task,
            cloud=CLOUD,
            tau_dict=tau_dict,
            k_max=1,
        )

        if result.executor_type != "unmatched":
            if result.price_total < result.C1_winner - 1e-10:
                violations.append(
                    f"Round {i}: seller IR violated "
                    f"price_total={result.price_total:.4e} < C1={result.C1_winner:.4e}"
                )

    assert (
        not violations
    ), f"Seller IR (I2) violated {len(violations)}/1000 times. First: {violations[0]}"


def test_buyer_ir_1000_rounds() -> None:
    """I3: price_total <= a_buyer for all matched tasks (buyer never overpays).

    Guaranteed by affordable-set gate in build_affordable_set (AFGM Step 4).
    """
    rng = np.random.default_rng(99)
    violations: list[str] = []

    for i in range(1000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        node = _Node(
            unique_id=i,
            f_max=f_max,
            kappa=float(10 ** rng.uniform(-27, -25)),
            rho=float(rng.uniform(2e-4, 1e-3)),
            P_idle=float(rng.choice([20.0, 30.0, 50.0])),
            beta=float(rng.uniform(40.0, 700.0)),
            pi_E=float(rng.uniform(0.06, 0.18)),
            l_hat=float(rng.uniform(0.0, 0.9)) * f_max,
            R_hat=float(rng.uniform(0.0, 2.0)),
        )

        w = float(rng.uniform(1e9, 1e10))
        a_buyer = float(rng.uniform(0.01, 10.0))
        d = float(rng.uniform(0.2, 2.0))
        task = HEDGETask(f"T{i}", s=2e6, w=w, d=d, a_buyer=a_buyer, created_at=0.0)

        result = run_rfq_round(
            peer_pool=[node],
            task=task,
            cloud=CLOUD,
            tau_dict={i: 0.002},
            k_max=1,
        )

        if result.executor_type != "unmatched":
            if result.price_total > a_buyer + 1e-10:
                violations.append(
                    f"Round {i}: buyer IR violated "
                    f"price_total={result.price_total:.4e} > a_buyer={a_buyer:.4e}"
                )

    assert (
        not violations
    ), f"Buyer IR (I3) violated {len(violations)}/1000 times. First: {violations[0]}"


def test_seller_ir_cloud_route() -> None:
    """Seller IR holds when cloud is the winner."""
    rng = np.random.default_rng(7)
    violations: list[str] = []

    for i in range(200):
        w = float(rng.uniform(1e9, 1e10))
        a_buyer = float(rng.uniform(0.5, 10.0))
        d = float(rng.uniform(0.5, 5.0))
        task = HEDGETask(f"T{i}", s=1e6, w=w, d=d, a_buyer=a_buyer, created_at=0.0)

        result = run_rfq_round(
            peer_pool=[],
            task=task,
            cloud=CLOUD,
            tau_dict={},
            k_max=0,
        )

        if result.executor_type == "cloud":
            if result.price_total < result.C1_winner - 1e-10:
                violations.append(
                    f"Cloud IR violated: price={result.price_total:.4e} < C1={result.C1_winner:.4e}"
                )

    assert (
        not violations
    ), f"Cloud seller IR violated {len(violations)} times. First: {violations[0]}"
