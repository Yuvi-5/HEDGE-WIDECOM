"""Tests for TraceScheduler (Alibaba trace-driven arrivals) and
remap_to_heavy_envelope, gated on the bundled data file being present."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.loaders.alibaba_loader import load_alibaba_trace, remap_to_heavy_envelope
from hedge.simulation.scheduler import TraceScheduler

_ALIBABA_PATH = Path("data/alibaba_1h_subset.parquet")
pytestmark = pytest.mark.skipif(
    not _ALIBABA_PATH.exists(), reason="Alibaba subset not generated"
)


@pytest.fixture()
def _nodes() -> list[Any]:
    class _Node:
        def __init__(self, uid: int, mult: float) -> None:
            self.unique_id = uid
            self.arrival_rate_multiplier = mult
            self.f_max = 3e9

    return [_Node(i, 1.0) for i in range(10)]


def _cfg(**arrivals_overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "simulation": {"duration": 60.0, "delta_tp": 0.1, "random_seed": 0},
        "arrivals": {
            "mode": "alibaba",
            "trace_path": str(_ALIBABA_PATH),
            "trace_target_lambda": 2.0,
            "max_tasks": 2000,
        },
    }
    cfg["arrivals"].update(arrivals_overrides)
    return cfg


def test_reproducible_same_seed(_nodes: list[Any]) -> None:
    """Same seed produces byte-identical first arrivals."""
    cfg = _cfg()
    cfg["simulation"]["random_seed"] = 5

    sched1 = TraceScheduler(_nodes, cfg, np.random.default_rng(5))
    sched2 = TraceScheduler(_nodes, cfg, np.random.default_rng(5))

    a1 = sched1.generate_tick(0.0, 5.0)
    a2 = sched2.generate_tick(0.0, 5.0)
    assert [(a.task.created_at, a.task.w, a.home_node_idx) for a in a1] == [
        (a.task.created_at, a.task.w, a.home_node_idx) for a in a2
    ]


def test_differs_across_seeds(_nodes: list[Any]) -> None:
    """Different seeds produce different arrival patterns."""
    cfg0 = _cfg()
    cfg0["simulation"]["random_seed"] = 0
    cfg1 = _cfg()
    cfg1["simulation"]["random_seed"] = 1

    sched0 = TraceScheduler(_nodes, cfg0, np.random.default_rng(0))
    sched1 = TraceScheduler(_nodes, cfg1, np.random.default_rng(1))

    a0 = sched0.generate_tick(0.0, 5.0)
    a1 = sched1.generate_tick(0.0, 5.0)
    assert [(a.task.created_at, a.home_node_idx) for a in a0] != [
        (a.task.created_at, a.home_node_idx) for a in a1
    ]


def test_long_duration_high_lambda_does_not_precompute_all_tiles(_nodes: list[Any]) -> None:
    """Lazy tiling: a config that would need thousands of tile repetitions
    (high trace_target_lambda relative to the natural trace span) must not
    raise at construction and must not hang on sequential ticks --
    regression test for the old eager-tiling design that materialised every
    repetition upfront (would need ~400,000 tiles here)."""
    cfg = _cfg(trace_target_lambda=500.0)
    cfg["simulation"]["duration"] = 3600.0
    sched = TraceScheduler(_nodes, cfg, np.random.default_rng(0))
    # Construction itself must not have precomputed 400k tiles; a bounded
    # sequential run (the real calling convention: consecutive, contiguous
    # ticks) must complete quickly and yield sane, monotonic arrivals.
    total = 0
    t = 0.0
    delta = 0.1
    for _ in range(200):
        arrivals = sched.generate_tick(t, delta)
        total += len(arrivals)
        for a in arrivals:
            assert t <= a.task.created_at < t + delta + 1e-9
        t += delta
    assert total > 0


def test_generate_tick_arrivals_strictly_advance(_nodes: list[Any]) -> None:
    """Sequential ticks never return an arrival with created_at outside the
    requested [sim_time, sim_time+delta_tp) window, across a tile boundary."""
    cfg = _cfg(trace_target_lambda=50.0)
    cfg["simulation"]["duration"] = 30.0
    sched = TraceScheduler(_nodes, cfg, np.random.default_rng(0))
    t = 0.0
    delta = 0.5
    seen_ids: set[str] = set()
    while t < 30.0:
        arrivals = sched.generate_tick(t, delta)
        for a in arrivals:
            assert t <= a.task.created_at < t + delta + 1e-9
            assert a.task.task_id not in seen_ids, "duplicate task_id across tiles"
            seen_ids.add(a.task.task_id)
        t += delta
    assert len(seen_ids) > 0


def test_remap_to_heavy_envelope_bounds_and_heterogeneity() -> None:
    """Remapped w/s/d land inside the target ranges and retain spread (not
    all saturated to one boundary)."""
    tasks = load_alibaba_trace(_ALIBABA_PATH, max_tasks=2000, seed=0)
    w_range = (1.0e7, 5.0e7)
    d_range = (0.15, 0.5)
    s_range = (4.5e7, 8.0e7)
    remapped = remap_to_heavy_envelope(tasks, w_range, d_range, s_range)

    ws = np.array([t.w for t in remapped])
    ds = np.array([t.d for t in remapped])
    ss = np.array([t.s for t in remapped])

    assert ws.min() >= w_range[0] - 1e-6 and ws.max() <= w_range[1] + 1e-6
    assert ds.min() >= d_range[0] - 1e-6 and ds.max() <= d_range[1] + 1e-6
    assert ss.min() >= s_range[0] - 1e-6 and ss.max() <= s_range[1] + 1e-6

    # Heterogeneity preserved: values must actually spread across the range,
    # not saturate to a single boundary value.
    assert ws.std() > 0.1 * (w_range[1] - w_range[0])
    assert ds.std() > 0.1 * (d_range[1] - d_range[0])
    assert ss.std() > 0.1 * (s_range[1] - s_range[0])

    # a_buyer and created_at are untouched by the remap.
    originals = {t.task_id: (t.a_buyer, t.created_at) for t in tasks}
    for t in remapped:
        assert t.a_buyer == originals[t.task_id][0]
        assert t.created_at == originals[t.task_id][1]


def test_cloud_eligible_fraction_zero_is_byte_identical(_nodes: list[Any]) -> None:
    """cloud_eligible_fraction=0.0 (default) produces byte-identical arrivals
    to not setting it at all -- existing configs are unaffected."""
    cfg_default = _cfg(remap_to_heavy_envelope=True, trace_target_lambda=20.0)
    cfg_explicit_zero = _cfg(
        remap_to_heavy_envelope=True, trace_target_lambda=20.0, cloud_eligible_fraction=0.0
    )
    sched_default = TraceScheduler(_nodes, cfg_default, np.random.default_rng(0))
    sched_zero = TraceScheduler(_nodes, cfg_explicit_zero, np.random.default_rng(0))

    a_default = sched_default.generate_tick(0.0, 5.0)
    a_zero = sched_zero.generate_tick(0.0, 5.0)
    assert [(a.task.created_at, a.task.w, a.task.d, a.task.s) for a in a_default] == [
        (a.task.created_at, a.task.w, a.task.d, a.task.s) for a in a_zero
    ]


def test_cloud_eligible_fraction_produces_bimodal_mix(_nodes: list[Any]) -> None:
    """cloud_eligible_fraction > 0: some tasks land outside the tight
    edge-native envelope (left at natural light-scale w), producing a
    genuinely bimodal w distribution instead of the uniform tight-envelope
    distribution seen at fraction=0.0."""
    cfg = _cfg(
        remap_to_heavy_envelope=True,
        trace_target_lambda=200.0,
        cloud_eligible_fraction=0.3,
        heavy_w_range=[1.0e7, 5.0e7],
        heavy_d_range=[0.15, 0.5],
        heavy_s_range=[4.5e7, 8.0e7],
    )
    cfg["simulation"]["duration"] = 30.0
    sched = TraceScheduler(_nodes, cfg, np.random.default_rng(0))
    all_tasks = []
    t = 0.0
    while t < 30.0:
        arrivals = sched.generate_tick(t, 1.0)
        all_tasks.extend(a.task for a in arrivals)
        t += 1.0

    ws = np.array([task.w for task in all_tasks])
    assert len(ws) > 50, "need enough tasks for a meaningful distribution check"
    outside_envelope = np.sum((ws < 1.0e7 - 1e-6) | (ws > 5.0e7 + 1e-6))
    assert outside_envelope > 0, (
        "expected some tasks left outside the tight edge-native envelope "
        "(natural light-scale, cloud-eligible), found none"
    )
    # Roughly in the right ballpark of the configured fraction (loose bound;
    # this is a sanity check on the split existing, not an exact-match test).
    frac_outside = outside_envelope / len(ws)
    assert 0.05 < frac_outside < 0.6


def test_unknown_arrivals_mode_raises() -> None:
    """Engine fails loud on an unrecognised arrivals.mode (D8/Part 1.5)."""
    import yaml

    from hedge.simulation.engine import HEDGESimulationEngine

    cfg = {
        "simulation": {"duration": 10.0, "random_seed": 0},
        "topology": {"source": "synthetic", "N_nodes": 4},
        "arrivals": {"mode": "azure"},
    }
    with pytest.raises(ValueError, match="Unknown arrivals.mode"):
        HEDGESimulationEngine(config=cfg, seed=0, output_dir=None)
