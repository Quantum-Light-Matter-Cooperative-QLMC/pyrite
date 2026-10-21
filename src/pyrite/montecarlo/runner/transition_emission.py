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
from ...materials.optical import OpticalResponse
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


def case_layer_eps(
    case, energy_eV, *, layer_responses: tuple[OpticalResponse, ...] | None = None
) -> list[np.ndarray]:
    """One permittivity per transport layer at ``energy_eV``; refuses table gaps.

    ``layer_responses`` explicitly associates one scalar optical provider with
    each layer in order, for source-only diagnostics. No mixing with inferred
    atomic response, no extrapolation, and no persistent case integration.
    Without it the independent-atom X-ray path is retained, above the existing
    per-medium continuum floor. A sub-floor diagnostic needs explicit optics.

    Validation: tr-detected-omission-bound, optical-material-response
    """
    layers = case_layers(case)
    if layer_responses is not None:
        if len(layer_responses) != len(layers):
            raise ValueError("transition_radiation: supply one optical response per layer")
        if not all(isinstance(response, OpticalResponse) for response in layer_responses):
            raise ValueError(
                "transition_radiation: every layer needs an explicit scalar optical response"
            )
        eps = []
        for index, response in enumerate(layer_responses):
            try:
                eps.append(response.epsilon(energy_eV))
            except ValueError as error:
                raise ValueError(f"transition_radiation/layer[{index}]: {error}") from error
        return eps

    from ..._photon_continuum_floor import photon_continuum_floor_eV
    from ...materials import MediumSpec

    energy = np.asarray(energy_eV, dtype=float)
    for index, layer in enumerate(layers):
        medium = MediumSpec(key=f"layer-{index}", composition=tuple(layer[2]))
        floor = photon_continuum_floor_eV(medium)
        if not np.all(np.isfinite(energy)) or np.any(energy < floor):
            raise ValueError(
                f"transition_radiation/layer[{index}]: Chantler dielectric data alone "
                f"does not support this band below the X-ray floor {floor:g} eV; "
                "supply explicit optical responses for a source-only diagnostic"
            )
    eps = [composition_dielectric(layer[2], energy) for layer in layers]
    if not all(np.all(np.isfinite(value)) for value in eps):
        raise ValueError("transition radiation needs Chantler dielectric data over the line grid")
    return eps


def case_transition_spectrum(case, transport, *, xp=None, layer_responses=None) -> np.ndarray:
    """TR on ``transport['E_grid']`` at ``transport['n_hat']`` (sample frame).

    ``xp=None`` scores on the CuPy device when the backend has one, else with
    NumPy. The result is a host array. ``layer_responses`` opts a source-only
    diagnostic into explicit per-layer optical providers; it is not stored in
    the case or used by the production component dispatcher.

    Validation: tr-crossing-superposition, tr-coherent-bunch
    """
    record = case["transition_radiation"]
    energy = np.asarray(transport["E_grid"], dtype=float)
    direction = np.asarray(transport["n_hat"], dtype=float)
    explicit_eps = (
        None
        if layer_responses is None
        else case_layer_eps(case, energy, layer_responses=layer_responses)
    )
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
        case_layer_eps(case, energy) if explicit_eps is None else explicit_eps,
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
