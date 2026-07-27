"""Phase 8 gate: data pipeline tests for all data sources used by this repo.

Gate criteria:
- EUA: Haversine and tau matrix functions work correctly (file-dependent tests skipped
  when EUA CSV is absent).
- Alibaba: FileNotFoundError raised cleanly when data file absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.loaders.alibaba_loader import load_alibaba_trace
from data.loaders.eua_loader import (
    compute_tau_matrix,
    haversine_distance_km,
    load_eua_topology,
)
from hedge.core.constants import (
    A_BUYER_MAX,
    A_BUYER_MIN,
    D_MAX,
    D_MIN,
    S_MAX,
    S_MIN,
    W_MAX,
    W_MIN,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EUA_CSV = Path("data/eua_melbourne.csv")
_ALIBABA_PATH = Path("data/alibaba_1h_subset.parquet")


# ---------------------------------------------------------------------------
# Haversine and tau matrix (no file dependency)
# ---------------------------------------------------------------------------


def test_haversine_same_point_is_zero() -> None:
    """Distance from a point to itself is 0."""
    assert haversine_distance_km(37.0, 144.0, 37.0, 144.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance() -> None:
    """Melbourne CBD to Sydney CBD ~ 714 km (within 5 km)."""
    melbourne = (-37.8136, 144.9631)
    sydney = (-33.8688, 151.2093)
    d = haversine_distance_km(*melbourne, *sydney)
    assert 700.0 <= d <= 730.0, f"Expected ~714 km, got {d:.1f} km"


def test_haversine_symmetry() -> None:
    """Distance is symmetric."""
    d1 = haversine_distance_km(-37.8, 144.9, -33.9, 151.2)
    d2 = haversine_distance_km(-33.9, 151.2, -37.8, 144.9)
    assert d1 == pytest.approx(d2, rel=1e-10)


def test_compute_tau_matrix_shape_and_diagonal() -> None:
    """tau matrix has correct shape and zero diagonal."""
    coords = np.array(
        [
            [-37.81, 144.96],
            [-37.82, 144.97],
            [-37.80, 144.95],
        ]
    )
    tau = compute_tau_matrix(coords)
    assert tau.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(tau), np.zeros(3))


def test_compute_tau_matrix_symmetric() -> None:
    """tau matrix is symmetric."""
    coords = np.array(
        [
            [-37.81, 144.96],
            [-37.82, 144.97],
            [-37.80, 144.95],
            [-37.79, 144.98],
        ]
    )
    tau = compute_tau_matrix(coords)
    np.testing.assert_allclose(tau, tau.T, atol=1e-15)


def test_compute_tau_matrix_non_negative() -> None:
    """All delay values are non-negative."""
    coords = np.array(
        [
            [-37.81, 144.96],
            [-37.82, 144.97],
            [-37.80, 144.95],
        ]
    )
    tau = compute_tau_matrix(coords)
    assert np.all(tau >= 0.0)


def test_compute_tau_matrix_capped_at_5ms() -> None:
    """Delays are capped at 5 ms (TAU_MESH_CAP)."""
    # Melbourne to Sydney -> would be > 5ms without cap
    coords = np.array(
        [
            [-37.8136, 144.9631],  # Melbourne
            [-33.8688, 151.2093],  # Sydney
        ]
    )
    tau = compute_tau_matrix(coords)
    assert tau[0, 1] <= 0.005 + 1e-12
    assert tau[1, 0] <= 0.005 + 1e-12


def test_close_nodes_have_small_delay() -> None:
    """Adjacent base stations (1 km apart) have sub-ms delay."""
    # 1 km apart in latitude ~ 0.009 degrees
    coords = np.array(
        [
            [-37.8136, 144.9631],
            [-37.8226, 144.9631],  # ~1 km south
        ]
    )
    tau = compute_tau_matrix(coords)
    assert tau[0, 1] < 0.001  # < 1 ms for 1 km


# ---------------------------------------------------------------------------
# EUA file-dependent tests (skipped when data file absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_eua_loader_125_nodes() -> None:
    """EUA loader produces 125 nodes from full dataset."""
    rng = np.random.default_rng(42)
    config: dict = {
        "arrivals": {"n_hotspots": 5, "hotspot_rate_multiplier": 5.0, "quiet_rate_multiplier": 0.2},
        "topology": {"tau_c_mean": 0.035, "tau_c_std": 0.008},
    }
    nodes, tau, cloud = load_eua_topology(_EUA_CSV, n_nodes=125, config=config, rng=rng)
    assert len(nodes) == 125


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_eua_loader_correlate_hotspot_weak_hardware() -> None:
    """correlate_hotspot_weak_hardware assigns hotspot role to the weakest-f_max nodes."""
    rng = np.random.default_rng(0)
    config: dict = {
        "arrivals": {
            "n_hotspots": 5,
            "hotspot_rate_multiplier": 8.0,
            "quiet_rate_multiplier": 0.2,
            "correlate_hotspot_weak_hardware": True,
        },
        "topology": {},
    }
    nodes, _, _ = load_eua_topology(_EUA_CSV, n_nodes=50, config=config, rng=rng)
    hotspot_f_max = [n.f_max for n in nodes if n.arrival_rate_multiplier == 8.0]
    other_f_max = [n.f_max for n in nodes if n.arrival_rate_multiplier != 8.0]
    assert len(hotspot_f_max) == 5
    assert max(hotspot_f_max) <= min(other_f_max), (
        "hotspot nodes must be exactly the weakest-f_max nodes"
    )


@pytest.mark.skipif(not _EUA_CSV.exists(), reason="EUA Melbourne CSV not downloaded")
def test_eua_loader_delay_matrix_bounds() -> None:
    """EUA delay matrix: non-negative, capped at 5ms, symmetric."""
    rng = np.random.default_rng(0)
    config: dict = {"arrivals": {}, "topology": {}}
    nodes, tau, cloud = load_eua_topology(_EUA_CSV, n_nodes=50, config=config, rng=rng)
    assert np.all(tau >= 0.0), "Negative delays found"
    assert np.all(np.diag(tau) == 0.0), "Diagonal must be zero"
    assert np.all(tau <= 0.005 + 1e-12), "Delay cap exceeded"
    np.testing.assert_allclose(tau, tau.T, atol=1e-15)


def test_eua_loader_raises_on_missing_file() -> None:
    """load_eua_topology raises FileNotFoundError for non-existent path."""
    rng = np.random.default_rng(0)
    with pytest.raises(FileNotFoundError):
        load_eua_topology(
            data_path=Path("data/does_not_exist_eua.csv"),
            n_nodes=10,
            config={},
            rng=rng,
        )


# ---------------------------------------------------------------------------
# Alibaba loader (stub: raise on missing file)
# ---------------------------------------------------------------------------


def test_alibaba_raises_on_missing_file() -> None:
    """load_alibaba_trace raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError, match="Alibaba"):
        load_alibaba_trace(Path("data/does_not_exist_alibaba.csv"))


@pytest.mark.skipif(not _ALIBABA_PATH.exists(), reason="Alibaba subset not generated")
def test_alibaba_task_ranges() -> None:
    """Alibaba-derived tasks have valid field ranges."""
    tasks = load_alibaba_trace(_ALIBABA_PATH, max_tasks=1000, seed=0)
    assert len(tasks) > 0
    for t in tasks:
        assert W_MIN <= t.w <= W_MAX, f"w={t.w:.3e}"
        assert D_MIN <= t.d <= D_MAX, f"d={t.d}"
        assert S_MIN <= t.s <= S_MAX, f"s={t.s:.3e}"
        assert A_BUYER_MIN <= t.a_buyer <= A_BUYER_MAX
