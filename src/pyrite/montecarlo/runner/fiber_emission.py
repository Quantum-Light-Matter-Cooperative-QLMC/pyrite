"""Campaign route of the opt-in full-fiber HOPG CXR domain (#1483).

A case carrying ``cxr_fiber`` (the resolved :class:`FiberCXRModel` record)
scores its lines over the model's full reciprocal domain, averaged as
intensities over the roll quadrature, and radiates its continuum from
BremsLib tables with exactly that domain's lattice share removed. Both
seams are the ones every basal case uses (``_cxr_for_segments`` and
:func:`.lattice_share.layer_emission_tables`), so a fiber case reaches the
same kernels as :func:`pyrite.montecarlo.spectrum.cxr.fiber_case.score_fiber_case`.

The case's own ``hkl_list`` (the catalog's basal families) still names the
reflections of Bragg-diffracted bremsstrahlung and Kossel terms, as the
explicit ``n_families`` set does for basal cases.

Validation: cxr-fiber-intensity-average
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

#: Case key of the resolved fiber model record (scientific identity).
FIBER_CASE_KEY = "cxr_fiber"
#: Execution-only case key: reflections per coupling-table batch.
FIBER_BATCH_CASE_KEY = "cxr_fiber_batch_size"

_DOMAINS: dict[str, Any] = {}


def is_fiber_case(case: Mapping[str, Any]) -> bool:
    """Whether ``case`` scores the full-fiber domain."""
    return case.get(FIBER_CASE_KEY) is not None


def fiber_domain(case: Mapping[str, Any]):
    """The prepared :class:`FiberReflectionSet` of a fiber ``case`` (cached)."""
    from ..spectrum.cxr.fiber import FiberCXRModel

    record = case[FIBER_CASE_KEY]
    batch = case.get(FIBER_BATCH_CASE_KEY)
    key = json.dumps([record, batch], sort_keys=True)
    domain = _DOMAINS.get(key)
    if domain is None:
        if len(_DOMAINS) >= 8:
            _DOMAINS.clear()
        domain = _DOMAINS[key] = FiberCXRModel.from_case_record(
            record, reflection_batch_size=batch
        ).prepare()
    return domain


def _refusal(case: Mapping[str, Any]) -> str | None:
    """Why a fiber ``case`` leaves the supported envelope, or ``None``."""
    checks = (
        (case.get("crystal") != "hopg", "requires the HOPG crystal"),
        (case.get("bremsstrahlung_model") != "bremslib", "requires the BremsLib continuum"),
        (case.get("coherent_emission"), "supports incoherent emission only"),
        (case.get("temporal_profile"), "does not support temporal profiles"),
        (case.get("reflection_selection") is not None, "owns its reflection domain"),
        (case.get("line_quadrature") != "bin-mean", "scores bin-mean lines"),
        ((case.get("line_grid_policy") or {}).get("windows"), "does not support feature windows"),
        (case.get("line_roulette") is not None, "does not support line roulette"),
        (case.get("cone_split") is not None, "does not support cone splitting"),
        (case.get("channeling") not in (None, "off"), "requires ordinary Born-line transport"),
        (case.get("transport_mode") == "mixed", "requires ordinary Born-line transport"),
        (case.get("transition_radiation") is not None, "does not support transition radiation"),
        (case.get("photon_transport") is not None, "does not support photon transport"),
        (case.get("layer_radiators") is not None, "requires a single slab"),
        (case.get("abs_layers") is not None, "requires a single slab"),
        (case.get("groove_spacing_ang") is not None, "requires a flat slab"),
        (case.get("mosaic_mc_sampling") is not None, "requires deterministic mosaic quadrature"),
        # The analytic width takes psi from the basal ``hkl_list`` (store_result).
        (case.get("mosaic_fwhm_rad") is not None, "requires mosaic_route='mc' for mosaicity"),
        (case.get("cxr_source") == "none", "needs CXR lines"),
        (case.get("adaptive_precision") is not None, "requires fixed electron counts"),
        (case.get("sinc_cutoff") is not None, "does not support a sinc cutoff"),
        (case.get("_brem_lattice_share", True) is False, "keeps the matched continuum"),
    )
    for refused, reason in checks:
        if refused:
            return reason
    for key in ("beam_uvw", "surface_hkl"):
        axis = case.get(key)
        if axis is not None and (list(axis[:2]) != [0, 0] or axis[2] <= 0):
            return "requires the positive c axis along the sample normal"
    return None


def validate_fiber_case(case: Mapping[str, Any]) -> None:
    """Refuse a fiber case outside the qualified envelope; normalize nothing.

    Validation: cxr-fiber-intensity-average
    """
    from ..spectrum.cxr.fiber import FiberCXRModel

    batch = case.get(FIBER_BATCH_CASE_KEY)
    if not is_fiber_case(case):
        if batch is not None:
            raise ValueError(f"{FIBER_BATCH_CASE_KEY} requires {FIBER_CASE_KEY}")
        return
    if (reason := _refusal(case)) is not None:
        raise ValueError(f"full-fiber CXR {reason} (#1483)")
    FiberCXRModel.from_case_record(case[FIBER_CASE_KEY], reflection_batch_size=batch)


#: Caller audit settings each roll call's own audit shares.
_AUDIT_SETTINGS = (
    "start_eV",
    "stop_eV",
    "band_eV",
    "ceiling_eV",
    "extension_stops_eV",
    "n_electrons",
    "mass_only",
)
#: Un-normalized audit sums, linear in line weight; folded at the roll weight.
_AUDIT_SUMS = (
    "line_mass",
    "mass_above",
    "mass_below",
    "band_mass",
    "mass_above_extension",
    "electron_mass",
)


def _roll_audit(caller):
    """A fresh audit for one roll call, sharing the caller's settings."""
    if caller is None:
        return None
    if caller.get("attribute") is not None:
        raise ValueError("full-fiber CXR does not support line attribution")
    audit = {key: caller[key] for key in _AUDIT_SETTINGS if key in caller}
    if "collect" in caller:
        # Host collection: the caller's chunks and every roll's stay on one side.
        caller["device_bytes"] = None
        audit["collect"] = []
    return audit


def _fold_roll_audit(caller, audit, weight):
    """Add one roll call's audit to ``caller`` at its quadrature ``weight``.

    Every sum, the per-electron masses and each collected line's coefficient
    are linear in the line weight, so the folded audit is that of the
    roll-averaged spectrum the same calls score (#1489).

    Validation: cxr-fiber-intensity-average
    """
    for key in _AUDIT_SUMS:
        if key in audit:
            existing = caller.get(key)
            value = weight * audit[key]
            caller[key] = value if existing is None else existing + value
    for energy, width, coefficient in audit.get("collect", ()):
        caller["collect"].append((energy, width, weight * coefficient))


def fiber_lines(segs, E_grid, case, n_hat, abs_layers, groove, Ne=None, extras=()):
    """Roll-averaged incoherent line density of a fiber ``case`` on ``E_grid``.

    Each roll node and reflection batch is one ordinary basal-route call with
    the domain's rows and per-reflection projected B; no table cache spans
    rolls (orientation enters the cached tables). ``extras`` are the caller's
    ``(coherent, truncation_audit, temporal, coefficient_capture)``. A
    truncation audit (line-grid resolution's collected population, or the
    spectrum phase's edge sums) is accumulated over every roll node and batch
    at the roll weight (1 basal, ``1/N`` otherwise), so it describes the
    roll-averaged spectrum (#1489). Temporal profiles and coefficient capture
    are refused.

    Validation: cxr-fiber-intensity-average
    Validation: line-grid-resonance-bandwidth
    """
    import numpy as np

    from .. import runner
    from ..spectrum.cxr.fiber import _validate_scoring

    coherent, audit, *unsupported = (*extras, None, None, None, None)[:4]
    if coherent or (coherent is None and case.get("coherent_emission")):
        raise ValueError("full-fiber CXR supports incoherent emission only")
    if any(value is not None for value in unsupported):
        raise ValueError("full-fiber CXR does not support temporal profiles or capture")
    phi0 = float(case.get("azimuth_rad", 0.0))
    _validate_scoring(
        phi0, dict(beam_uvw=case.get("beam_uvw"), surface_hkl=case.get("surface_hkl"))
    )
    domain = fiber_domain(case)
    lookup = domain._thermal_lookup()
    base = {k: v for k, v in case.items() if k not in (FIBER_CASE_KEY, FIBER_BATCH_CASE_KEY)}
    if base.get("beam_uvw") is None and base.get("surface_hkl") is None:
        base["surface_hkl"] = (0, 0, 1)
    grid = np.asarray(E_grid, dtype=float)

    def score(hkls, phi, weight):
        part = _roll_audit(audit)
        value = runner._cxr_for_segments(
            segs,
            grid,
            dict(base, hkl_list=list(hkls), B_ang2=lookup.__getitem__, azimuth_rad=phi),
            n_hat,
            abs_layers,
            groove,
            coherent=False,
            Ne=Ne,
            table_cache=None,
            truncation_audit=part,
        )
        if part is not None:
            _fold_roll_audit(audit, part, weight)
        return value

    return domain._average(score, grid.shape, phi0, weighted=True)


def fiber_emission_tables(case, layer, composition, tables):
    """BremsLib ``tables`` with the fiber domain's lattice share removed.

    Same contract as :func:`.lattice_share.layer_emission_tables`. Electrons
    below ``E_cut_lines_keV`` keep the full continuum.

    Validation: cxr-fiber-intensity-average
    """
    if tables is None:
        raise ValueError("full-fiber CXR requires BremsLib continuum tables")
    if layer not in (None, 0):
        raise ValueError("full-fiber CXR requires a single slab")
    from .lattice_share import _CACHE, _CACHE_LIMIT

    floor = float(case.get("E_cut_lines_keV", 5.0))
    ceiling = float(case["E0_keV"])  # range-checked against the kernel, so keyed
    key = (
        "fiber",
        json.dumps(case[FIBER_CASE_KEY], sort_keys=True),
        layer,
        tuple((str(el), float(n)) for el, n in composition),
        tuple(sorted((el, t.key, t.digest) for el, t in tables.items())),
        floor,
        ceiling,
    )
    cached = _CACHE.get(key)
    if cached is None:
        domain = fiber_domain(case)
        out, correction = domain.corrected_bremslib_tables(
            tables, composition, E_cut_keV=floor, max_energy_keV=ceiling
        )
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        cached = _CACHE[key] = (
            out,
            dict(
                layer=layer,
                corrected=True,
                crystal=case["crystal"],
                reflections=len(domain.hkls),
                line_cutoff_keV=floor,
                **correction,
            ),
        )
    return cached
