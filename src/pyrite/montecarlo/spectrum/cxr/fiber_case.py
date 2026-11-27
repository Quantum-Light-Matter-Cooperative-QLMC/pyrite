"""Paired full-fiber emission on an already transported single HOPG slab."""

import numpy as np

from ...._backend import _to_cpu
from .fiber import FiberCXRModel, _validate_scoring
from .fiber_selection import (
    check_selection_arguments,
    flag_clamped_share,
    row_scorer,
    select_fiber_reflections,
)


def _single_radiator(case, segments):
    """Reject unsupported physics instead of falling through to scalar scoring."""
    if case.get("coherent_emission") or case.get("cxr_source") == "none":
        raise ValueError("fiber scoring requires incoherent CXR emission")
    if case.get("channeling") not in (None, "off") or case.get("transport_mode") == "mixed":
        raise ValueError("fiber scoring requires ordinary Born-line transport")
    if (case.get("transition_radiation") or {}).get("model") == "two-beam":
        raise ValueError("fiber scoring does not support two-beam CXR")
    if int(segments.get("n_layers", 1)) != 1 or segments.get("cone_split") is not None:
        raise ValueError("fiber scoring requires a single slab without cone splitting")
    if case.get("bremsstrahlung_model") != "bremslib":
        raise ValueError("fiber scoring requires BremsLib, without isotropic fallback")
    # A lowered campaign fiber case (#1483) names the model this scorer applies.
    case = {k: v for k, v in case.items() if k not in ("cxr_fiber", "cxr_fiber_batch_size")}
    radiators = case.get("layer_radiators")
    if radiators is None:
        return case
    if len(radiators) != 1 or radiators[0] is None:
        raise ValueError("fiber scoring requires exactly one crystalline radiator")
    return {**case, **radiators[0], "layer_radiators": None}


def _max_energy(case, segments):
    maximum = float(case["E0_keV"])
    for key in ("E_keV", "E_repr_keV"):
        if segments.get(key) is not None:
            values = np.asarray(_to_cpu(segments[key]), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite trajectory energy")
            if values.size:
                maximum = max(maximum, float(values.max()))
    return maximum


def _checked_pilot(model, case, tp, E_grid_eV):
    """Return the radiator pilot and photon grid, or refuse before any scoring."""
    segments = tp["segs"]
    pilot = _single_radiator(case, segments)
    if pilot.get("crystal") != model.crystal:
        raise ValueError("model crystal and transported radiator must agree")
    if pilot.get("_brem_lattice_share", True) is False:
        raise ValueError("fiber scoring cannot disable the matched continuum correction")
    if pilot.get("sinc_cutoff") is not None:
        raise ValueError("fiber case scoring uses bin means, without a sinc cutoff")
    count = int(tp["Ne_lines"])
    if count <= 0 or count != int(tp["Ne_brem"]) or count != int(segments["Ne"]):
        raise ValueError("fiber scoring requires the same complete electron population")
    grid = np.asarray(E_grid_eV, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.isfinite(grid).all()
        or grid[0] < 10.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("expected a finite increasing photon grid starting at 10 eV or above")
    _validate_scoring(
        float(pilot.get("azimuth_rad", 0.0)),
        dict(
            beam_uvw=pilot.get("beam_uvw"),
            surface_hkl=pilot.get("surface_hkl"),
            mosaic_sampling=pilot.get("mosaic_mc_sampling") or "quadrature",
            composition=pilot["composition"],
        ),
    )
    return pilot, grid


def score_fiber_case(
    model, case, tp, E_grid_eV, *, line_yield_tolerance=None, selection_histories=None
):
    """Score a transported runner scene with the full-fiber production kernels.

    ``model`` is a :class:`FiberCXRModel`, or ``Numerics``/``CxrReflections``
    carrying ``fiber=FiberAverage(...)`` (tensor at its measured temperature).

    ``tp`` is the complete runner transport result (or reconstructed artifact
    inputs including ``segs``). Both channels use the same electrons, detector,
    escape geometry and photon nodes. The line density is a bin mean; the
    continuum is evaluated at the supplied nodes, as in the paired diagnostic.
    The result is a standalone emission record, not a normal sweep checkpoint.
    The automatic line-grid policy is not run here.
    ``total`` is CXR plus the BremsLib continuum only: characteristic,
    transition and diffracted radiation are not included.

    ``line_yield_tolerance`` (#1484) selects the reciprocal cutoff by measured
    yield instead of scoring the whole domain: ``model.cutoff_invang`` becomes
    the search cap, and
    :func:`~pyrite.montecarlo.spectrum.cxr.fiber_selection.select_fiber_reflections`
    stops on roll-averaged net shell rows of the electron prefix (at most
    ``selection_histories``, default 4096, and the row budget). The selected
    set is scored on every history, and the result gains
    ``reflection_selection`` (domain, roll count, cutoff, tail, shells).
    If the selected set's continuum correction clamps at any node, the record
    reports ``converged=False`` (``share_clamped_nodes``). ``selection_histories``
    without a tolerance is refused.

    Validation: cxr-fiber-intensity-average
    """
    from ... import runner
    from ...runner.case_tables import _case_bremslib_tables
    from ...runner.emission import _layer_brem_from_segments

    if not isinstance(model, FiberCXRModel):
        model = FiberCXRModel.from_numerics(model)
    pilot, grid = _checked_pilot(model, case, tp, E_grid_eV)
    if line_yield_tolerance is not None:
        check_selection_arguments(grid, line_yield_tolerance, selection_histories)
    elif selection_histories is not None:
        raise ValueError("selection_histories requires line_yield_tolerance")
    segments = tp["segs"]
    count = int(tp["Ne_lines"])
    phi0 = float(pilot.get("azimuth_rad", 0.0))
    pilot.update(
        reflection_selection=None,
        line_quadrature="bin-mean",
        coherent_emission=False,
        allow_physics_fallback=False,
    )
    if pilot.get("beam_uvw") is None and pilot.get("surface_hkl") is None:
        pilot["surface_hkl"] = (0, 0, 1)
    floor = float(pilot.get("E_cut_lines_keV", 5.0))
    max_energy = _max_energy(pilot, segments)
    cxr, selection, staged = None, None, None
    if line_yield_tolerance is None:
        domain = model.prepare()
    else:
        domain, cxr, selection, staged = select_fiber_reflections(
            model,
            pilot,
            tp,
            grid,
            line_yield_tolerance,
            floor_keV=floor,
            max_keV=max_energy,
            histories=selection_histories,
        )
    tables, correction = domain.corrected_bremslib_tables(
        _case_bremslib_tables(pilot),
        pilot["composition"],
        E_cut_keV=floor,
        max_energy_keV=max_energy,
    )
    if selection is not None:
        selection = flag_clamped_share(selection, correction)
    # Stage after validation and table construction; a selection whose prefix
    # was the whole population already staged it. The private pilot's callable
    # B is never serialized, hashed or sent to a worker.
    segments = runner._segments_on_device(segments) if staged is None else staged
    n_hat, groove = tp["n_hat"], tp.get("groove")
    abs_layers = pilot.get("abs_layers")
    if cxr is None:
        cxr = domain._average(
            row_scorer(segments, grid, pilot, n_hat, groove, count, domain), grid.shape, phi0
        )
    brem = np.asarray(
        _to_cpu(
            _layer_brem_from_segments(
                segments,
                grid,
                pilot,
                n_hat,
                abs_layers,
                layer=None,
                tables=tables,
                groove=groove,
                Ne=count,
            )
        ),
        dtype=float,
    )
    if brem.shape != grid.shape or not np.isfinite(brem).all():
        raise RuntimeError("invalid continuum spectrum from BremsLib scorer")
    result = dict(
        E_grid_eV=grid,
        cxr=cxr,
        brem=brem,
        total=cxr + brem,
        model=domain.record(),
        continuum_correction=correction,
        electrons=count,
        E_cut_lines_keV=floor,
        azimuth_rad=phi0,
        n_hat=np.asarray(_to_cpu(n_hat), dtype=float).tolist(),
        line_quadrature="bin-mean",
        continuum_quadrature="node",
    )
    if selection is not None:
        result["reflection_selection"] = selection
    return result
