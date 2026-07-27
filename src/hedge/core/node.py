"""HEDGEEdgeServer - edge node acting as both buyer and seller in the HEDGE market."""

from __future__ import annotations

from edge_sim_py import EdgeServer

from hedge.core.constants import DELTA_Q


class HEDGEEdgeServer(EdgeServer):  # type: ignore[misc]
    """Edge server node participating in the HEDGE cooperative-competitive market.

    Subclasses EdgeSimPy EdgeServer and adds HEDGE-specific pricing, queue, and
    connectivity state. Pricing (Layer 1/2/2.5) is computed by the free functions
    in hedge.pricing against this node's own state, not by methods on this class.
    """

    _instances: list[HEDGEEdgeServer] = []
    _object_count: int = 0

    def __init__(
        self,
        obj_id: int,
        f_max: float,
        kappa: float,
        rho: float,
        P_idle: float,
        beta: float,
        pi_E: float,
        coordinates: tuple[float, float] | None = None,
    ) -> None:
        """Create a HEDGEEdgeServer with hardware and market parameters.

        Args:
            obj_id: Unique integer identifier within the simulation.
            f_max: Maximum CPU frequency (Hz).
            kappa: DVFS capacitance constant (farads).
            rho: CAPEX depreciation rate (USD/s).
            P_idle: Idle baseline power draw (W).
            beta: Carbon intensity of local grid (gCO2/kWh).
            pi_E: Local electricity price (USD/kWh).
            coordinates: Optional (x, y) position for EUA-topology placement.
        """
        super().__init__(obj_id=obj_id, coordinates=coordinates)

        # Ensure unique_id is accessible (EdgeSimPy stores it as .id; rfq.py expects .unique_id)
        self.unique_id: int = obj_id

        # Tier-2 private: hardware parameters (never transmitted to buyers)
        self.f_max: float = f_max
        self.kappa: float = kappa
        self.rho: float = rho
        self.P_idle: float = P_idle
        self.beta: float = beta
        self.pi_E: float = pi_E

        # Queue state - maintained by the simulation scheduler
        self.w_pending: float = 0.0  # running sum of pending CPU cycles

        # Tier-1 public state (recomputed and broadcast every DELTA_Q seconds)
        self.p_star_ref: float = 0.0  # Layer-2 reference quote (USD/cycle)
        self.p_dagger_ref: float = 0.0  # Layer-2.5 shaded reference quote (USD/cycle)

        # Tier-2 private: predictor state (updated every DELTA_TP seconds)
        self.l_observed: float = 0.0  # true instantaneous load l_i(t) (cycles/s)
        self.l_hat: float = 0.0  # Kalman load estimate (cycles/s)
        self.R_hat: float = 0.0  # EMA revenue estimate (USD/RFQ)
        self.lambda_SPA: float = 0.0  # current SPA aggressiveness coefficient
        self.p_fail_hat: float = 0.0  # online P_fail(i) sliding-window estimate (Eq 9)

        # Connectivity
        self.peer_delays: dict[int, float] = {}  # peer_id -> propagation delay (seconds)
        self.coverage_users: list[int] = []  # user IDs that can reach this node
        # C_i: other nodes (including self) within coverage_radius_km of this
        # node's own coordinates (paper Section subsec:phase0, C_u). Populated
        # by topology._assign_coverage_sets. Falls back to the full node list
        # when coordinates is None (synthetic topology, no geography).
        self.coverage_peers: list["HEDGEEdgeServer"] = []

        # Load tier assignment (set by topology builder)
        self.arrival_rate_multiplier: float = 1.0

    @property
    def W_q(self) -> float:
        """Advertised queue-wait scalar W_i(t) = w_pending / f_max (seconds)."""
        return self.w_pending / self.f_max
