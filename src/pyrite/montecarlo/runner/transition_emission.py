"""Transition radiation of a case's transported electrons on its line grid (#103).

Scores ``spec_transition`` [photons/eV/sr per incident electron] at the case's
observation direction from the planar boundary crossings of the same
transport the other components use. TR is the ``g = 0`` field of the
homogeneous layers: it adds to PXR/CBS as a separate component except for
reflections with ``g`` along the surface normal, whose cross term needs
diffracted TR and is not included (``tr-line-cross-term``).

Population: the electrons below ``Ne_brem``, which transport follows down to
the continuum cutoff ``E_cut_brem_keV``; crossings after a history ends at the
cutoff are not recorded. Permittivities are the independent-atom forward
scattering of each transport layer's composition (Chantler ``f'``, ``f''``).

On a CuPy (CUDA/ROCm) backend the crossing amplitudes, the stack solve and
the history and bunch sums run on the device: each chunk of whole histories
uploads its crossing records once, the fields accumulate on the device, and
only the finished spectrum returns to the host. Other backends score on the
host with NumPy.

Validation: tr-crossing-superposition, tr-coherent-bunch
"""

from __future__ import annotations

import numpy as np

from ..._backend import cp
from ..spectrum.transition_radiation import composition_dielectric
from ..spectrum.transition_radiation_crossings import (
    TRANSITION_MODEL,
    _take,
    boundary_crossings,
    crossing_transition_radiation,
)

__all__ = [
    "TRANSITION_MODEL",
    "block_transition_spectrum",
    "case_layer_eps",
    "case_layers",
    "case_transition_spectrum",
]


def case_layers(case):
    """Transport layers ``[(z_top, z_bottom, composition)]`` of a case."""
    layers = case.get("abs_layers")
    if layers is not None:
        return list(layers)
    return [(0.0, float(case["thickness_ang"]), case["composition"])]


def case_layer_eps(case, energy_eV) -> list[np.ndarray]:
    """One permittivity per transport layer at ``energy_eV``; refuses table gaps.

    Validation: tr-detected-omission-bound
    """
    eps = [composition_dielectric(layer[2], energy_eV) for layer in case_layers(case)]
    if not all(np.all(np.isfinite(value)) for value in eps):
        raise ValueError("transition radiation needs Chantler dielectric data over the line grid")
    return eps


def case_transition_spectrum(case, transport, *, xp=None) -> np.ndarray:
    """TR on ``transport['E_grid']`` at ``transport['n_hat']`` (sample frame).

    ``xp=None`` scores on the CuPy device when the backend has one, else with
    NumPy. The result is a host array.

    Validation: tr-crossing-superposition, tr-coherent-bunch
    """
    record = case["transition_radiation"]
    energy = np.asarray(transport["E_grid"], dtype=float)
    direction = np.asarray(transport["n_hat"], dtype=float)
    if direction[2] == 0.0:  # grazing the faces: no far-field TR leaves the stack
        return np.zeros(energy.shape)
    layers = case_layers(case)
    count = int(transport["Ne_brem"])
    crossings = boundary_crossings(transport["segs"], layers)
    crossings = _take(crossings, np.flatnonzero(crossings.electron_id < count))
    bunch = record["coherence"] == "bunch"
    population = None
    if bunch:
        from ..spectrum.coherent_population import physical_bunch_electrons

        population = physical_bunch_electrons(case)
    spectrum = crossing_transition_radiation(
        crossings,
        energy,
        direction,
        case_layer_eps(case, energy),
        n_electrons=count,
        coherence=record["coherence"],
        physical_electrons=population,
        xp=(np if cp is None else cp) if xp is None else xp,
    )
    return spectrum.get() if hasattr(spectrum, "get") else np.asarray(spectrum)


def block_transition_spectrum(case, segs, energy_eV, n_hat, ne_brem) -> np.ndarray:
    """Per-history TR of one whole-electron row block; zeros when TR is off.

    History-coherent TR is a sum over histories normalized by ``Ne_brem``, so
    disjoint whole-electron blocks add (``coherence="bunch"`` does not).

    Validation: tr-crossing-superposition
    """
    if case.get("transition_radiation") is None:
        return np.zeros(np.shape(energy_eV))
    transport = dict(segs=segs, E_grid=energy_eV, n_hat=n_hat, Ne_brem=ne_brem)
    return case_transition_spectrum(case, transport)
