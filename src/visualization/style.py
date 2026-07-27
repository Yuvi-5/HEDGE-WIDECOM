"""Shared style constants for the HEDGE paper figure package.

Centralises palette, fonts, and figure-size conventions so every figure in
the 24-30 figure evidence package reads as one consistent system. Colors are
assigned to algorithm identity in a FIXED order (never re-cycled per figure),
so a reader can track HEDGE (or any baseline) by color alone across the
entire report.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Categorical palette: Okabe-Ito (colorblind-safe, standard for scientific
# figures). Fixed algorithm -> color assignment used across EVERY figure.
# ---------------------------------------------------------------------------
ALGO_COLORS: dict[str, str] = {
    "HEDGE": "#0072B2",  # blue -- the system under test, always this color
    "B0": "#56B4E9",  # sky blue -- HEDGE-NoForecast
    "B1": "#D55E00",  # vermillion -- TORS (primary baseline)
    "B2": "#E69F00",  # orange -- DDPS
    "B3": "#CC79A7",  # reddish purple -- BARGAIN-MATCH
    "B4": "#8C7A3F",  # dark gold -- Cost-OPD (Okabe-Ito yellow is too low-contrast as a fill)
    "B5": "#999999",  # neutral gray -- FDRL-V (excluded from main runs)
    "B6": "#009E73",  # bluish green -- Greedy-NLF
    "B7": "#000000",  # black -- Cloud-only
}

ALGO_LABELS: dict[str, str] = {
    "HEDGE": "HEDGE",
    "B0": "B0 (No-Forecast)",
    "B1": "B1 (TORS)",
    "B2": "B2 (DDPS)",
    "B3": "B3 (BARGAIN-MATCH)",
    "B4": "B4 (Cost-OPD)",
    "B5": "B5 (FDRL-V)",
    "B6": "B6 (Greedy-NLF)",
    "B7": "B7 (Cloud-only)",
}

# Fixed display order (not alphabetical) -- HEDGE first, then baselines
# in numeric order.
ALGO_ORDER: list[str] = ["HEDGE", "B0", "B1", "B2", "B3", "B4", "B6", "B7"]

# Secondary palette for two/three-way comparisons and non-algorithm series
# (kept from the meeting-deck figures for cross-deliverable consistency).
TEAL = "#0B6E6D"
SLATE = "#5B7285"
RED = "#B0413E"
GOLD = "#8C7A3F"
GREEN = "#2E6F40"

INK = "#1A1A1A"
GRID = "#D9D9D9"

DPI = 300
FIGSIZE_SINGLE = (3.5, 2.6)  # inches, IEEE single-column
FIGSIZE_DOUBLE = (7.16, 3.2)  # inches, IEEE double-column
FONT_SIZE = 8


def apply_style() -> None:
    """Apply shared matplotlib rcParams for the whole figure package."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": DPI,
            "figure.dpi": DPI,
        }
    )


def algo_color(algorithm: str) -> str:
    """Return the fixed color for an algorithm name; gray fallback if unknown."""
    return ALGO_COLORS.get(algorithm, "#999999")


def algo_label(algorithm: str) -> str:
    """Return the display label for an algorithm name."""
    return ALGO_LABELS.get(algorithm, algorithm)


def savefig(fig: plt.Figure, path: object) -> None:
    """Save a figure at the package DPI and close it."""
    fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
