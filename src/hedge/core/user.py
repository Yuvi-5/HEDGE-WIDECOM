"""HEDGEUser - user device that generates tasks and orchestrates RFQ submission."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from edge_sim_py import User

from hedge.core.task import HEDGETask


class HEDGEUser(User):  # type: ignore[misc]
    """User device submitting computation tasks to the HEDGE market.

    Subclasses EdgeSimPy User. Task generation and RFQ dispatch are driven by
    hedge.simulation.scheduler and hedge.simulation.engine against this user's
    own state (coverage_set, task_queue), not by methods on this class.
    """

    _instances: list[HEDGEUser] = []
    _object_count: int = 0

    def __init__(
        self,
        obj_id: int,
        coordinates: tuple[float, float] | None = None,
    ) -> None:
        """Create a HEDGEUser at an optional spatial position.

        Args:
            obj_id: Unique integer identifier within the simulation.
            coordinates: Optional (x, y) position (used by EUA topology).
        """
        super().__init__(obj_id=obj_id)
        self.coordinates = coordinates

        # Edge servers reachable by this user (populated by topology builder)
        self.coverage_set: list[int] = []  # HEDGEEdgeServer obj_id values

        # Tasks awaiting submission or currently in-flight
        self.task_queue: list[HEDGETask] = []

        # Seeded RNG; injected by the experiment runner for reproducibility
        self.rng: np.random.Generator | None = None
