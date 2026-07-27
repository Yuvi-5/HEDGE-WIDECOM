"""Adv3 gate tests: low-valuation parasite injection.

Coverage:
- parasite_fraction=0.0 (default) draws a_buyer identically to the pre-Adv3
  scheduler, in the same RNG call order (regression: no behaviour change for
  every existing config, which all default to parasite_fraction=0.0).
- parasite_fraction=1.0 draws every a_buyer in [0.5, 1.0] * C1_min/b_shape.
- Draws are reproducible given a fixed seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hedge.core.node import HEDGEEdgeServer
from hedge.pricing.layer1 import compute_layer1_cost
from hedge.simulation.scheduler import PoissonScheduler


def _make_nodes(n: int = 4) -> list[HEDGEEdgeServer]:
    nodes = []
    for i in range(n):
        node = HEDGEEdgeServer(
            obj_id=i + 1,
            f_max=1e9 * (i + 1),
            kappa=1e-26,
            rho=5e-4,
            P_idle=20.0,
            beta=200.0,
            pi_E=0.1,
        )
        node.arrival_rate_multiplier = 1.0
        nodes.append(node)
    return nodes


_CFG: dict[str, Any] = {
    "arrivals": {"lambda_bar": 1.0},
    "task_model": {
        "w_min": 1.0e8,
        "w_max": 5.0e8,
        "s_min": 1.0e6,
        "s_max": 5.0e6,
        "d_min": 0.5,
        "d_max": 5.0,
        "a_buyer_min": 0.5,
        "a_buyer_max": 10.0,
    },
    "pricing": {"b_shape": 1.0},
}


def test_no_parasites_matches_plain_uniform_draw_order() -> None:
    """With parasite_fraction=0 (default), a_buyer is drawn as the 4th uniform
    call per task, in the exact same per-node/per-task order as before Adv3 was
    added (regression: rng.poisson(...) per node, then s/w/d/a_buyer per task)."""
    nodes = _make_nodes()
    rng_scheduler = np.random.default_rng(123)
    scheduler = PoissonScheduler(nodes, _CFG, rng_scheduler)

    rng_reference = np.random.default_rng(123)
    task_cfg = _CFG["task_model"]
    lambda_bar = _CFG["arrivals"]["lambda_bar"]
    delta_tp = 1.0

    arrivals = scheduler.generate_tick(sim_time=0.0, delta_tp=delta_tp)

    expected: list[tuple[float, float, float, float]] = []
    for node in nodes:
        lam = lambda_bar * node.arrival_rate_multiplier
        n = int(rng_reference.poisson(lam * delta_tp))
        for _ in range(n):
            s_ref = rng_reference.uniform(task_cfg["s_min"], task_cfg["s_max"])
            w_ref = rng_reference.uniform(task_cfg["w_min"], task_cfg["w_max"])
            d_ref = rng_reference.uniform(task_cfg["d_min"], task_cfg["d_max"])
            a_ref = rng_reference.uniform(task_cfg["a_buyer_min"], task_cfg["a_buyer_max"])
            expected.append((s_ref, w_ref, d_ref, a_ref))

    assert len(arrivals) == len(expected)
    for arrival, (s_ref, w_ref, d_ref, a_ref) in zip(arrivals, expected):
        assert arrival.task.s == pytest.approx(s_ref)
        assert arrival.task.w == pytest.approx(w_ref)
        assert arrival.task.d == pytest.approx(d_ref)
        assert arrival.task.a_buyer == pytest.approx(a_ref)


def test_all_parasites_fall_within_bertrand_floor_band() -> None:
    """With parasite_fraction=1.0, every a_buyer is in [0.5, 1.0] * C1_min/b_shape."""
    nodes = _make_nodes()
    cfg: dict[str, Any] = {
        **_CFG,
        "arrivals": {**_CFG["arrivals"], "parasite_fraction": 1.0},
    }
    scheduler = PoissonScheduler(nodes, cfg, np.random.default_rng(7))
    arrivals = scheduler.generate_tick(sim_time=0.0, delta_tp=5.0)
    assert len(arrivals) > 0

    for arrival in arrivals:
        task = arrival.task
        c1_min = min(
            compute_layer1_cost(task.w, n.f_max, n.kappa, n.P_idle, n.rho, n.pi_E, n.beta)[0]
            for n in nodes
        )
        floor = c1_min / cfg["pricing"]["b_shape"]
        assert (
            0.5 * floor <= task.a_buyer <= 1.0 * floor + 1e-9
        ), f"parasite a_buyer={task.a_buyer} outside [0.5,1.0]*floor={floor}"


def test_parasite_fraction_reproducible_with_same_seed() -> None:
    nodes1, nodes2 = _make_nodes(), _make_nodes()
    cfg: dict[str, Any] = {
        **_CFG,
        "arrivals": {**_CFG["arrivals"], "parasite_fraction": 0.5},
    }
    s1 = PoissonScheduler(nodes1, cfg, np.random.default_rng(99))
    s2 = PoissonScheduler(nodes2, cfg, np.random.default_rng(99))
    a1 = s1.generate_tick(0.0, 5.0)
    a2 = s2.generate_tick(0.0, 5.0)
    assert [t.task.a_buyer for t in a1] == [t.task.a_buyer for t in a2]
