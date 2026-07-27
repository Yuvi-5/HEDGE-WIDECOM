"""EUA Melbourne topology loader.

Loads real GPS base-station coordinates and converts to propagation-delay matrix
for the HEDGE simulation topology.

Download the raw dataset before use:
    https://github.com/swinedge/eua-dataset
Expected file: data/eua_melbourne.csv  (columns: latitude, longitude)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hedge.core.cloud import CloudNode
from hedge.core.constants import (
    BETA_MAX,
    BETA_MIN,
    F_MAX_MAX,
    F_MAX_MIN,
    KAPPA_MAX,
    KAPPA_MIN,
    P_IDLE_OPTIONS,
    PI_E_MAX,
    PI_E_MIN,
    RHO_MAX,
    RHO_MIN,
    TAU_C_MEAN,
    TAU_C_STD,
)
from hedge.core.node import HEDGEEdgeServer

# Speed of light in fiber (km/s)
_FIBER_SPEED_KM_S: float = 2e5

# Max mesh delay cap per DATA_PIPELINE.md Section 1c (seconds)
_TAU_MESH_CAP: float = 0.005


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km using the Haversine formula.

    Args:
        lat1: Latitude of point 1 (degrees).
        lon1: Longitude of point 1 (degrees).
        lat2: Latitude of point 2 (degrees).
        lon2: Longitude of point 2 (degrees).

    Returns:
        Distance in kilometres.
    """
    R: float = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2.0 * R * np.arcsin(np.sqrt(a)))


def compute_tau_matrix(station_coords: np.ndarray) -> np.ndarray:
    """Compute NxN propagation delay matrix in seconds from GPS coordinates.

    tau_ij = haversine_km(i, j) / FIBER_SPEED.
    Delays exceeding _TAU_MESH_CAP are capped (nodes outside neighbourhood).

    Args:
        station_coords: Float array of shape (N, 2) with (latitude, longitude) rows.

    Returns:
        Symmetric float64 array of shape (N, N) with diagonal = 0.
    """
    n = len(station_coords)
    tau = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d_km = haversine_distance_km(
                station_coords[i, 0],
                station_coords[i, 1],
                station_coords[j, 0],
                station_coords[j, 1],
            )
            delay = min(d_km / _FIBER_SPEED_KM_S, _TAU_MESH_CAP)
            tau[i, j] = delay
            tau[j, i] = delay
    return tau


def _assign_heterogeneous_params(
    nodes: list[HEDGEEdgeServer],
    rng: np.random.Generator,
) -> None:
    """Assign per-node hardware parameters drawn from log-uniform distributions.

    Follows DATA_PIPELINE.md Section 1d.
    """
    for node in nodes:
        node.f_max = float(
            np.clip(rng.lognormal(mean=np.log(3e9), sigma=0.5), F_MAX_MIN, F_MAX_MAX)
        )
        node.kappa = float(10 ** rng.uniform(np.log10(KAPPA_MIN), np.log10(KAPPA_MAX)))
        node.rho = float(rng.uniform(RHO_MIN, RHO_MAX))
        node.pi_E = float(rng.uniform(PI_E_MIN, PI_E_MAX))
        node.beta = float(rng.uniform(BETA_MIN, BETA_MAX))
        p_idle_idx = int(rng.integers(0, len(P_IDLE_OPTIONS)))
        node.P_idle = P_IDLE_OPTIONS[p_idle_idx]


def _assign_load_tiers(
    nodes: list[HEDGEEdgeServer],
    n_hotspots: int,
    n_quiet: int,
    hotspot_mult: float,
    quiet_mult: float,
    rng: np.random.Generator,
    correlate_hotspot_weak_hardware: bool = False,
    cluster_hotspots: bool = False,
) -> None:
    """Assign hotspot / quiet / normal arrival-rate roles.

    Hardware (f_max, kappa, ...) and load tier are drawn independently by
    design; correlate_hotspot_weak_hardware is a disclosed, opt-in modeling extension
    (default off, preserves the independent draw everywhere else) that assigns
    the hotspot role to the n_hotspots weakest-f_max nodes instead of a random
    subset -- e.g. small/cheap edge boxes in busy urban micro-cells, offloading
    to nearby under-utilised, more powerful nodes rather than falling back to
    cloud.

    cluster_hotspots is a second, orthogonal, opt-in extension (default off):
    instead of scattering the hotspot role over n_hotspots independently-random
    nodes, it picks ONE seed node and assigns the hotspot role to the seed plus
    its n_hotspots-1 physically nearest neighbours (by haversine distance on
    node.coordinates), producing a spatially coherent cluster of saturated
    cells with genuinely idle neighbours elsewhere in the topology -- "a
    handful of cells are saturated while their neighbours sit idle" (paper
    Section subsec:landscape). Composes with correlate_hotspot_weak_hardware:
    when both are set, the cluster seed is the weakest-f_max node rather than
    a random one, so the saturated cluster is also the under-provisioned one.
    No-op (falls back to the scattered assignment below) when any node lacks
    coordinates (synthetic topology has no geography to cluster over).
    """
    n = len(nodes)
    n_hotspots = min(n_hotspots, n)
    n_quiet = min(n_quiet, n - n_hotspots)

    if cluster_hotspots and all(node.coordinates is not None for node in nodes):
        seed_idx = (
            min(range(n), key=lambda i: nodes[i].f_max)
            if correlate_hotspot_weak_hardware
            else int(rng.integers(0, n))
        )
        seed_lat, seed_lon = nodes[seed_idx].coordinates
        dists = sorted(
            range(n),
            key=lambda i: haversine_distance_km(
                seed_lat, seed_lon, nodes[i].coordinates[0], nodes[i].coordinates[1]
            ),
        )
        hotspot_indices = set(dists[:n_hotspots])
        remaining = [i for i in range(n) if i not in hotspot_indices]
        rng.shuffle(remaining)
        quiet_indices = set(remaining[:n_quiet])
        for i, node in enumerate(nodes):
            if i in hotspot_indices:
                node.arrival_rate_multiplier = hotspot_mult
            elif i in quiet_indices:
                node.arrival_rate_multiplier = quiet_mult
            else:
                node.arrival_rate_multiplier = 1.0
        return

    if correlate_hotspot_weak_hardware:
        order = sorted(range(n), key=lambda i: nodes[i].f_max)
        hotspot_indices = set(order[:n_hotspots])
        remaining = order[n_hotspots:]
        rng.shuffle(remaining)
        quiet_indices = set(remaining[:n_quiet])
        for i, node in enumerate(nodes):
            if i in hotspot_indices:
                node.arrival_rate_multiplier = hotspot_mult
            elif i in quiet_indices:
                node.arrival_rate_multiplier = quiet_mult
            else:
                node.arrival_rate_multiplier = 1.0
        return

    roles: list[str] = (
        ["hotspot"] * n_hotspots + ["quiet"] * n_quiet + ["normal"] * (n - n_hotspots - n_quiet)
    )
    rng.shuffle(roles)

    for node, role in zip(nodes, roles):
        if role == "hotspot":
            node.arrival_rate_multiplier = hotspot_mult
        elif role == "quiet":
            node.arrival_rate_multiplier = quiet_mult
        else:
            node.arrival_rate_multiplier = 1.0


def load_eua_topology(
    data_path: str | Path,
    n_nodes: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[HEDGEEdgeServer], np.ndarray, CloudNode]:
    """Load EUA Melbourne topology from CSV and return nodes, delay matrix, cloud.

    Args:
        data_path: Path to the EUA Melbourne CSV (columns: latitude, longitude).
        n_nodes: Number of base stations to use (max 125; subsampled if needed).
        config: Experiment config dict (reads arrivals and topology sections).
        rng: Seeded random number generator for heterogeneity and load tiers.

    Returns:
        nodes: List of HEDGEEdgeServer with GPS-derived delay matrix.
        tau: Symmetric NxN propagation delay matrix in seconds.
        cloud: CloudNode with sampled RTT.

    Raises:
        FileNotFoundError: If data_path does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"EUA Melbourne data not found at '{data_path}'. "
            "Download from https://github.com/swinedge/eua-dataset and place at that path "
            "(see data/README.md)."
        )

    df = pd.read_csv(data_path)

    # Normalise column names (dataset may use lat/lon or latitude/longitude)
    df.columns = [c.strip().lower() for c in df.columns]
    lat_col = next((c for c in df.columns if c in ("latitude", "lat")), None)
    lon_col = next((c for c in df.columns if c in ("longitude", "lon", "lng")), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"EUA CSV must have latitude/longitude columns. Found: {list(df.columns)}")

    coords = df[[lat_col, lon_col]].dropna().to_numpy(dtype=np.float64)

    # Subsample if needed (reproducible via rng)
    if len(coords) > n_nodes:
        idx = rng.choice(len(coords), size=n_nodes, replace=False)
        idx.sort()
        coords = coords[idx]
    elif len(coords) < n_nodes:
        n_nodes = len(coords)

    tau = compute_tau_matrix(coords)

    nodes: list[HEDGEEdgeServer] = []
    for i in range(n_nodes):
        server = HEDGEEdgeServer(
            obj_id=i + 1,
            f_max=3e9,  # placeholder; overwritten by _assign_heterogeneous_params
            kappa=1e-26,
            rho=5e-4,
            P_idle=30.0,
            beta=200.0,
            pi_E=0.12,
            coordinates=(float(coords[i, 0]), float(coords[i, 1])),
        )
        nodes.append(server)

    _assign_heterogeneous_params(nodes, rng)

    arrivals_cfg = config.get("arrivals", {})
    _assign_load_tiers(
        nodes=nodes,
        n_hotspots=int(arrivals_cfg.get("n_hotspots", 5)),
        n_quiet=20,
        hotspot_mult=float(arrivals_cfg.get("hotspot_rate_multiplier", 5.0)),
        quiet_mult=float(arrivals_cfg.get("quiet_rate_multiplier", 0.2)),
        rng=rng,
        correlate_hotspot_weak_hardware=bool(
            arrivals_cfg.get("correlate_hotspot_weak_hardware", False)
        ),
        cluster_hotspots=bool(arrivals_cfg.get("cluster_hotspots", False)),
    )

    topo_cfg = config.get("topology", {})
    tau_c_mean = float(topo_cfg.get("tau_c_mean", TAU_C_MEAN))
    tau_c_std = float(topo_cfg.get("tau_c_std", TAU_C_STD))
    tau_c = max(0.001, float(rng.normal(tau_c_mean, tau_c_std)))
    cloud = CloudNode(tau_c=tau_c)

    return nodes, tau, cloud
