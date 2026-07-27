"""HEDGETask - the indivisible computation unit traded in the HEDGE market."""

from __future__ import annotations

from dataclasses import dataclass

from hedge.core.constants import S_RET_FRACTION

_VALID_STATUSES = frozenset({"pending", "submitted", "executing", "completed", "rejected"})


@dataclass
class HEDGETask:
    """Single indivisible computation task submitted by a user device.

    Tasks are binary: executed in full at exactly one node or rejected.
    The buyer's private valuation (a_buyer) is never transmitted to sellers.
    """

    task_id: str
    s: float  # input data size (bits)
    w: float  # CPU workload (cycles)
    d: float  # deadline (seconds)
    a_buyer: float  # private valuation (USD) - NEVER transmitted to sellers
    created_at: float  # simulation arrival time (seconds)
    user_id: str = ""
    status: str = "pending"  # pending | submitted | executing | completed | rejected

    def __post_init__(self) -> None:
        """Validate fields at construction boundary."""
        if self.s <= 0.0:
            raise ValueError(f"s (data size) must be > 0, got {self.s}")
        if self.w <= 0.0:
            raise ValueError(f"w (workload) must be > 0, got {self.w}")
        if self.d <= 0.0:
            raise ValueError(f"d (deadline) must be > 0, got {self.d}")
        if self.a_buyer <= 0.0:
            raise ValueError(f"a_buyer (valuation) must be > 0, got {self.a_buyer}")
        if self.created_at < 0.0:
            raise ValueError(f"created_at must be >= 0, got {self.created_at}")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}, got '{self.status}'")

    @property
    def s_ret(self) -> float:
        """Return data size (bits) = s * S_RET_FRACTION."""
        return self.s * S_RET_FRACTION
