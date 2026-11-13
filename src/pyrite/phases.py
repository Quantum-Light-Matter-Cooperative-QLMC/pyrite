"""Deprecated experimental transport/spectrum phase split (#776, #1162).

The two-step workflow is now stable: :func:`pyrite.transport.simulate_trajectories`
returns a :class:`pyrite.transport.Trajectories`, and :func:`pyrite.score_spectra`
scores it under any detector without transporting again. The names below
delegate to those and warn until their removal release.
:class:`SimulationProgress` stays here unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._simulation_progress import SimulationProgress

if TYPE_CHECKING:
    from ._numerics_groups import KernelTuning
    from .campaign.model import Numerics, Scene
    from .campaign.trajectory_product import Trajectories
    from .campaign.trajectory_product import Trajectories as TransportedScene  # noqa: F401
    from .results.model import Result

__all__ = ["SimulationProgress", "TransportedScene", "score_spectra", "simulate_transport"]

Progress = Callable[[SimulationProgress], None]


def simulate_transport(
    scene: Scene, *, numerics: Numerics | None = None, progress: Progress | None = None
) -> Trajectories:
    """Deprecated: use :func:`pyrite.transport.simulate_trajectories`.

    Transports ``scene``'s scalar-detector population exactly as
    :func:`pyrite.simulate` does and returns the typed
    :class:`~pyrite.transport.Trajectories`.
    """
    from ._api_deprecations import warn_api
    from .campaign.trajectories import scene_trajectories

    warn_api("pyrite.phases.simulate_transport")
    return scene_trajectories(scene, numerics=numerics, progress=progress)


def score_spectra(
    transported: Trajectories,
    *,
    kernel_tuning: KernelTuning | None = None,
    progress: Progress | None = None,
) -> Result:
    """Deprecated: use :func:`pyrite.score_spectra`.

    ``kernel_tuning`` becomes ``numerics=transported.numerics.with_kernel_tuning(...)``;
    its transport fields must equal the transport's.
    """
    from ._api_deprecations import warn_api
    from .campaign.trajectory_product import Trajectories
    from .campaign.trajectory_product import Trajectories as TransportedScene  # noqa: F401
    from .scoring import score_spectra as stable

    warn_api("pyrite.phases.score_spectra")
    if not isinstance(transported, Trajectories):
        raise TypeError("transported must come from simulate_transport")
    numerics = transported.numerics
    if kernel_tuning is not None and numerics is not None:
        numerics = numerics.with_kernel_tuning(kernel_tuning)
    return stable(transported, numerics=numerics, progress=progress)


def __getattr__(name: str) -> Any:
    if name == "TransportedScene":
        from ._api_deprecations import warn_api
        from .campaign.trajectory_product import Trajectories

        warn_api("pyrite.phases.TransportedScene")
        return Trajectories
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
