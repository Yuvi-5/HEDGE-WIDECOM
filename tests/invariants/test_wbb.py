"""Invariant I4: Weak Budget Balance - total buyer payments equal total seller receipts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hedge.core.constants import MU_C
from hedge.core.task import HEDGETask
from hedge.market.rfq import run_rfq_round


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


def test_wbb_10k_rounds() -> None:
    """I4: seller_receipt == price_total for 10,000 random single-round markets.

    In HEDGE without resale, the buyer pays exactly what the seller receives
    (no external subsidy, no fee extraction by the platform).
    """
    rng = np.random.default_rng(42)
    violations: list[str] = []

    for i in range(10_000):
        f_max = float(10 ** rng.uniform(9.0, 9.7))
        node = _Node(
            unique_id=i % 1000,
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
        d = float(rng.uniform(0.1, 2.0))
        task = HEDGETask(f"T{i}", s=2e6, w=w, d=d, a_buyer=a_buyer, created_at=0.0)

        result = run_rfq_round(
            peer_pool=[node],
            task=task,
            cloud=CLOUD,
            tau_dict={node.unique_id: 0.002},
            k_max=1,
        )

        if result.executor_type != "unmatched":
            diff = abs(result.seller_receipt - result.price_total)
            if diff > 1e-10:
                violations.append(
                    f"Round {i}: WBB diff={diff:.2e} "
                    f"(payment={result.price_total:.4e}, receipt={result.seller_receipt:.4e})"
                )

    assert (
        not violations
    ), f"WBB (I4) violated {len(violations)}/10000 times. First: {violations[0]}"


def test_wbb_multi_peer_round() -> None:
    """I4 holds when multiple peers are in F_b (only winner transacts; others RELEASE)."""
    rng = np.random.default_rng(55)

    for i in range(500):
        n_peers = int(rng.integers(2, 5))
        nodes = [
            _Node(
                unique_id=j,
                f_max=float(10 ** rng.uniform(9.0, 9.7)),
                kappa=float(10 ** rng.uniform(-27, -25)),
                rho=float(rng.uniform(2e-4, 1e-3)),
                P_idle=float(rng.choice([20.0, 30.0, 50.0])),
                beta=float(rng.uniform(40.0, 700.0)),
                pi_E=float(rng.uniform(0.06, 0.18)),
                l_hat=float(rng.uniform(0.0, 0.7)) * float(10 ** rng.uniform(9.0, 9.7)),
                R_hat=float(rng.uniform(0.0, 1.0)),
            )
            for j in range(n_peers)
        ]
        task = HEDGETask(
            f"T{i}",
            s=2e6,
            w=float(rng.uniform(1e9, 1e10)),
            d=float(rng.uniform(0.2, 2.0)),
            a_buyer=float(rng.uniform(0.1, 10.0)),
            created_at=0.0,
        )
        tau_dict = {j: 0.002 for j in range(n_peers)}

        result = run_rfq_round(
            peer_pool=nodes,
            task=task,
            cloud=CLOUD,
            tau_dict=tau_dict,
            k_max=4,
        )

        if result.executor_type != "unmatched":
            diff = abs(result.seller_receipt - result.price_total)
            assert diff < 1e-10, f"Multi-peer WBB violated at round {i}: diff={diff:.2e}"
