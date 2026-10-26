"""Source-level radiation calculations, independent of detector acquisition.

photon_energies and the three mc_* spectrum functions are advanced-stable.
directional_source and directional_sources remain experimental and return
pyrite.results.directional.DirectionalSpectrum. Directional densities are
photons/incident-electron/eV/sr; solid-angle-integrated spectra omit /sr.
No detector response or acquisition counts are added by this facade.

Lazy identity aliases; importing this namespace loads no numerical engines or
material datasets. Resolving an export loads its existing implementation.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.directional import directional_source as directional_source
    from pyrite.directional import directional_sources as directional_sources
    from pyrite.materials.emission import photon_energies as photon_energies
    from pyrite.montecarlo.spectrum import mc_brem_spectrum as mc_brem_spectrum
    from pyrite.montecarlo.spectrum import mc_cxr_spectrum as mc_cxr_spectrum
    from pyrite.montecarlo.spectrum import (
        mc_cxr_spectrum_solid_angle as mc_cxr_spectrum_solid_angle,
    )

_EXPORTS = {
    "photon_energies": ("pyrite.materials.emission", "photon_energies"),
    "directional_source": ("pyrite.directional", "directional_source"),
    "directional_sources": ("pyrite.directional", "directional_sources"),
    "mc_cxr_spectrum": ("pyrite.montecarlo.spectrum", "mc_cxr_spectrum"),
    "mc_cxr_spectrum_solid_angle": ("pyrite.montecarlo.spectrum", "mc_cxr_spectrum_solid_angle"),
    "mc_brem_spectrum": ("pyrite.montecarlo.spectrum", "mc_brem_spectrum"),
}

__all__ = [
    "photon_energies",
    "directional_source",
    "directional_sources",
    "mc_cxr_spectrum",
    "mc_cxr_spectrum_solid_angle",
    "mc_brem_spectrum",
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
