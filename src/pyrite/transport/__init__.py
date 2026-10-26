"""Advanced transport and saved-trajectory analysis.

All listed exports are advanced-stable. Case remains a supported lowered record;
prefer Scene and pyrite.simulate for full-scene calculations. Segment positions
and lengths use angstroms, electron energies keV; summary units are in column
names. Saved-capture reductions perform file I/O only when called.

Lazy identity aliases; importing this namespace loads no numerical engines or
material datasets. Resolving an export loads its existing implementation.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.montecarlo.case import Case as Case
    from pyrite.montecarlo.trajectory_balance import (
        capture_secondary_energy_balance as capture_secondary_energy_balance,
    )
    from pyrite.montecarlo.trajectory_summary import depth_loss_profile as depth_loss_profile
    from pyrite.montecarlo.trajectory_summary import (
        iter_primary_summaries as iter_primary_summaries,
    )
    from pyrite.montecarlo.trajectory_summary import (
        summarize_trajectories as summarize_trajectories,
    )
    from pyrite.montecarlo.transport import simulate_trajectories as simulate_trajectories

_EXPORTS = {
    "Case": ("pyrite.montecarlo.case", "Case"),
    "simulate_trajectories": ("pyrite.montecarlo.transport", "simulate_trajectories"),
    "summarize_trajectories": ("pyrite.montecarlo.trajectory_summary", "summarize_trajectories"),
    "iter_primary_summaries": ("pyrite.montecarlo.trajectory_summary", "iter_primary_summaries"),
    "depth_loss_profile": ("pyrite.montecarlo.trajectory_summary", "depth_loss_profile"),
    "capture_secondary_energy_balance": (
        "pyrite.montecarlo.trajectory_balance",
        "capture_secondary_energy_balance",
    ),
}

__all__ = [
    "Case",
    "simulate_trajectories",
    "summarize_trajectories",
    "iter_primary_summaries",
    "depth_loss_profile",
    "capture_secondary_energy_balance",
]


def __getattr__(name: str) -> Any:
    """Resolve a supported name without changing its implementation or metadata."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
