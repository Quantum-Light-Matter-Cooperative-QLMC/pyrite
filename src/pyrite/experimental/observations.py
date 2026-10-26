"""Bounded experimental photon transport, aperture tallies and persisted artifacts.

Importable, documented, provisional contract; existing domain imports remain valid.
Resolving an attribute imports the original implementation without wrapping it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.photon_event_observation import PhotonEventTally as PhotonEventTally
    from pyrite.photon_event_observation import PhotonRelaxationBudget as PhotonRelaxationBudget
    from pyrite.photon_event_observation import load_photon_events as load_photon_events
    from pyrite.photon_event_observation import save_photon_events as save_photon_events
    from pyrite.photon_events import PhotonEventObservation as PhotonEventObservation
    from pyrite.photon_events import simulate_photon_events as simulate_photon_events
    from pyrite.photon_observation import PhotonContinuumTally as PhotonContinuumTally
    from pyrite.photon_observation import load_photon_continuum as load_photon_continuum
    from pyrite.photon_observation import save_photon_continuum as save_photon_continuum
    from pyrite.photon_transport import PhotonContinuumObservation as PhotonContinuumObservation
    from pyrite.photon_transport import simulate_photon_continuum as simulate_photon_continuum

_EXPORTS = {
    "PhotonContinuumObservation": ("pyrite.photon_transport", "PhotonContinuumObservation"),
    "simulate_photon_continuum": ("pyrite.photon_transport", "simulate_photon_continuum"),
    "PhotonContinuumTally": ("pyrite.photon_observation", "PhotonContinuumTally"),
    "save_photon_continuum": ("pyrite.photon_observation", "save_photon_continuum"),
    "load_photon_continuum": ("pyrite.photon_observation", "load_photon_continuum"),
    "PhotonEventObservation": ("pyrite.photon_events", "PhotonEventObservation"),
    "simulate_photon_events": ("pyrite.photon_events", "simulate_photon_events"),
    "PhotonEventTally": ("pyrite.photon_event_observation", "PhotonEventTally"),
    "PhotonRelaxationBudget": ("pyrite.photon_event_observation", "PhotonRelaxationBudget"),
    "save_photon_events": ("pyrite.photon_event_observation", "save_photon_events"),
    "load_photon_events": ("pyrite.photon_event_observation", "load_photon_events"),
}

__all__ = [
    "PhotonContinuumObservation",
    "simulate_photon_continuum",
    "PhotonContinuumTally",
    "save_photon_continuum",
    "load_photon_continuum",
    "PhotonEventObservation",
    "simulate_photon_events",
    "PhotonEventTally",
    "PhotonRelaxationBudget",
    "save_photon_events",
    "load_photon_events",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
