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
    "case_optical_transition",
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


def case_optical_transition(
    case,
    transport,
    *,
    layer_responses: tuple[OpticalResponse, ...],
    source,
    direction=None,
    formation_tolerance: float | None = 0.05,
    energy_budget: float | None = 1e-3,
    xp=None,
):
    """Opt-in optical TR/CTR of a case's transported histories (#615).

    Scores the ``Ne_brem`` histories of ``transport`` on ``transport['E_grid']``
    with one explicit :class:`OpticalResponse` per layer and an
    :class:`~pyrite.montecarlo.spectrum.optical_transition.OpticalBunchSource`
    (physical charge and sampling law). ``direction`` defaults to
    ``transport['n_hat']``. Every node must lie in each response's supported
    bands; a case ``bunch_charge_pc`` must agree with the source charge. The
    returned result carries the per-layer material provenance and the source
    law and is persisted with its ``save``. The X-ray component dispatcher
    never calls this.

    Unbounded use is refused (``optical_transition_validity``): the
    formation-zone bound of the self power must not exceed
    ``formation_tolerance``, and the radiated energy per physical electron
    over both hemispheres and the band must stay below ``energy_budget`` of
    the kinetic energy with negligible pair emission at the band edges.
    ``None`` skips a check and records it as not evaluated.

    Validation: optical-planar-tr, optical-ctr-population, optical-material-response
    """
    from ...energy_grid.component_support import ComponentSupport, component_grid
    from ..spectrum.optical_transition import optical_transition_radiation

    if "bunch_charge_pc" in case and float(case["bunch_charge_pc"]) != float(source.charge_pC):
        raise ValueError("optical source charge disagrees with the case bunch_charge_pc")
    energy = np.asarray(transport["E_grid"], dtype=float)
    support = tuple(
        ComponentSupport(f"layer-{i}:{r.medium}", "optical_transition", r.bands_eV)
        for i, r in enumerate(layer_responses)
    )
    component_grid(
        energy, required=tuple((item.medium, item.component) for item in support), support=support
    )
    eps = case_layer_eps(case, energy, layer_responses=layer_responses)
    n_hat = np.asarray(transport["n_hat"] if direction is None else direction, dtype=float)
    count = int(transport["Ne_brem"])
    crossings = boundary_crossings(transport["segs"], case_layers(case))
    crossings = _take(crossings, np.flatnonzero(crossings.electron_id < count))
    materials = [
        {
            "layer": index,
            "medium": response.medium,
            "dataset_id": response.dataset_id,
            "source": response.source,
            "source_sha256": response.source_sha256,
            "response_digest": response.digest,
            "bands_eV": response.bands_eV,
            "license": response.license,
            "temperature_K": response.temperature_K,
            "uncertainty": response.uncertainty,
        }
        for index, response in enumerate(layer_responses)
    ]
    backend = (np if cp is None else cp) if xp is None else xp
    validity = _optical_validity(
        crossings, energy, n_hat, eps, source, count, formation_tolerance, energy_budget, backend
    )
    return optical_transition_radiation(
        crossings,
        energy,
        n_hat,
        eps,
        source=source,
        n_histories=count,
        provenance={
            "materials": materials,
            "transition_model": TRANSITION_MODEL,
            "validity": validity,
        },
        xp=backend,
    )


def _optical_validity(crossings, energy, n_hat, eps, source, count, formation, budget, xp):
    """Formation and energy-budget refusals of :func:`case_optical_transition`."""
    from ..spectrum.optical_transition import optical_transition_radiation
    from ..spectrum.optical_transition_validity import (
        formation_error_bound,
        kinetic_energy_eV,
        radiated_energy_fraction,
    )

    out: dict[str, object] = {"formation_bound": None, "energy_fraction": None}
    if formation is not None:
        bound = float(np.max(formation_error_bound(crossings, energy, n_hat, eps)))
        if not bound <= formation:
            raise ValueError(
                f"optical TR formation-zone bound {bound:.3g} exceeds {formation:g}: "
                "transported paths are not straight over the formation length"
            )
        out["formation_bound"] = bound
    if budget is not None:
        excess = max(source.physical_electrons - 1.0, 0.0)

        def score(e, d):
            eps_grid = [np.asarray(x)[:, None] for x in eps]  # (E,) -> (E, 1)
            r = optical_transition_radiation(
                crossings, e, d, eps_grid, source=source, n_histories=count, xp=xp
            )
            return r.yield_per_electron, r.self_yield, excess * r.form_factor * r.pair_yield

        out["energy_fraction"] = radiated_energy_fraction(
            score, energy, kinetic_energy_eV(crossings), max_fraction=budget
        )
    return out
