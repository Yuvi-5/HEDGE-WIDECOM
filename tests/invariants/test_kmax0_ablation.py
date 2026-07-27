"""Invariant I9: HEDGE(K_max=0) Proposition-2 ablation - degenerates to cloud-only.

Tests a HEDGE-internal property only: setting K_max=0 in HEDGE's own
run_rfq_round collapses the peer pool to empty and therefore recovers
cloud-only routing (Proposition 2).
"""

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


def test_kmax_zero_recovery_1000_tasks() -> None:
    """I9: K_max=0 produces cloud-only routing for all 1000 random tasks.

    Proposition 2: HEDGE(K_max=0) == cloud-only routing (same code path B7
    uses). No edge peer may be selected when K_max=0 because P_b is empty
    by construction.
    """
    rng = np.random.default_rng(42)
    violations: list[str] = []

    for i in range(1000):
        task = HEDGETask(
            f"T{i}",
            s=float(rng.uniform(1e6, 2e7)),
            w=float(rng.uniform(1e9, 1e10)),
            d=float(rng.uniform(0.1, 5.0)),
            a_buyer=float(rng.uniform(0.1, 10.0)),
            created_at=0.0,
        )

        result = run_rfq_round(
            peer_pool=[],
            task=task,
            cloud=CLOUD,
            tau_dict={},
            k_max=0,
        )

        if result.executor_type == "edge":
            violations.append(
                f"Task {i}: edge selected at K_max=0 (executor_type={result.executor_type})"
            )

    assert (
        not violations
    ), f"K_max=0 cloud-only recovery (I9) violated {len(violations)}/1000 times. First: {violations[0]}"


def test_kmax_zero_no_edge_even_with_pool() -> None:
    """K_max=0 prevents edge selection even when peer_pool is non-empty.

    The k_max cap in run_rfq_round slices peer_pool[:0] = [], ensuring
    no peer can quote regardless of what is passed in peer_pool.
    """
    f_max = 4e9
    node = _Node(99, f_max=f_max, kappa=1e-26, rho=1.25e-3, P_idle=30.0, beta=200.0, pi_E=0.12)
    node.l_hat = 0.0
    node.R_hat = 0.0

    task = HEDGETask("T_test", s=2e6, w=1e9, d=5.0, a_buyer=100.0, created_at=0.0)

    result = run_rfq_round(
        peer_pool=[node],
        task=task,
        cloud=CLOUD,
        tau_dict={99: 0.001},
        k_max=0,
    )

    assert result.executor_type != "edge", f"Edge peer selected at K_max=0 despite non-empty pool"


def test_kmax_zero_cloud_wins_when_affordable() -> None:
    """With K_max=0 and affordable cloud, cloud is selected."""
    task = HEDGETask("T_cloud", s=2e6, w=1e9, d=5.0, a_buyer=100.0, created_at=0.0)

    result = run_rfq_round(
        peer_pool=[],
        task=task,
        cloud=CLOUD,
        tau_dict={},
        k_max=0,
    )

    assert (
        result.executor_type == "cloud"
    ), f"Expected cloud with generous budget, got {result.executor_type}"


def test_kmax_zero_unmatched_when_cloud_infeasible() -> None:
    """With K_max=0 and tight deadline, task is unmatched (cloud too slow)."""
    slow_cloud = _Cloud(tau_c=10.0)  # 10s RTT >> any deadline
    task = HEDGETask("T_tight", s=2e6, w=1e9, d=0.05, a_buyer=100.0, created_at=0.0)

    result = run_rfq_round(
        peer_pool=[],
        task=task,
        cloud=slow_cloud,
        tau_dict={},
        k_max=0,
    )

    assert (
        result.executor_type == "unmatched"
    ), f"Expected unmatched with slow cloud and tight deadline, got {result.executor_type}"
