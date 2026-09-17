"""_common

Backend-neutral shared figure plumbing: per-record line/brem split,
detector-response prep, and the trajectory/groove geometry helpers reused by the
mpl/altair/plotly backends. No matplotlib, Altair or Plotly imports here --
:mod:`pyrite.plots.mpl._common` holds the ONE matplotlib-only helper
(``_per_tilt_figs``) that doesn't belong in this neutral module.
"""

import numpy as np

from .._spectral_components import incident_spectrum, line_spectrum
from ..detectors import Detector, EagleXO, LegacyEDS, Timepix3
from ..detectors import eaglexo_response as eag
from ..detectors import timepix_response as tpx
from ..montecarlo.groove import surface_depth_ang
from ..results import (
    beam_current_na,
    detected_background,
    line_metrics,
)

# cache of the (expensive) Timepix efficiency-curve response, keyed by hardware +
# MC settings, so re-running the detector cell with unchanged settings doesn't
# rebuild it. (The per-grid detected-spectra response is already cached inside
# timepix_response.get_response.)
_EFF_CACHE = {}


def _mode(settings):
    return (
        "EDS-convolved" if getattr(settings, "convolve_with_det", False) else "response-free source"
    )


def _peak_line(r):
    """Response-free source line peak of one record -- the azimuth-selection key
    (strongest line wins)."""
    return float(np.max(line_spectrum(r)))


def _best_azimuth(grp, collapse_azimuth):
    """Collapse a same-energy record group to its single strongest-line azimuth
    when ``collapse_azimuth`` (and there's more than one to choose from); else
    return ``grp`` unchanged. The per-energy azimuth selection shared by every
    spectra/detector draw."""
    if collapse_azimuth and len(grp) > 1:
        return [max(grp, key=_peak_line)]
    return grp


def _case_title(case, tail="", *, latex=True, e0_keV=None, tilt_fmt="0.1f"):
    """Case-identifying title prefix -- material, thickness (um), polar tilt --
    shared by every spectra/detector title, with an optional inline beam energy
    (``e0_keV``) and a trailing renderer-specific clause (``tail``). ``latex=True``
    (matplotlib) renders `$\\mu$m` / `\\degree` math; ``latex=False`` (Vega-Lite)
    renders plain ASCII (`um` / `deg`) and joins ``tail`` with `--` instead of an
    em dash."""
    mat = case["name"].split()[0]
    thick = case["thickness_ang"] / 1e4
    tilt = format(case["tilt_deg"], tilt_fmt)
    if latex:
        e0 = f", {e0_keV:g} keV" if e0_keV is not None else ""
        head = rf"{mat}, {thick:.1f} $\mu$m{e0}, $\theta_\mathrm{{tilt}}={tilt}\degree$"
        return f"{head} — {tail}" if tail else head
    e0 = f", {e0_keV:g} keV" if e0_keV is not None else ""
    head = f"{mat}, {thick:.1f} um{e0}, theta_tilt={tilt} deg"
    return f"{head} -- {tail}" if tail else head


# Cross-call cache for the (expensive, per-record scipy peak-finding)
# `line_metrics` result -- `_metrics_map` already dedupes within one call via
# `id(rec)`, but every fresh call (e.g. a marimo tab re-rendering because an
# unrelated widget elsewhere changed) redid the whole O(records) pass from
# scratch. Measured on the densest checkpoint (mose2, 3720 records): this pass
# is why `heatmap_select_chart`/`scan_charts` cost ~0.9-1.1s per call.
#
# Keyed on CONTENT (the record's (name, E0_keV) pair), not `id(r)`/`id(settings)`:
# an identity key would be unsafe for a process-lifetime cache, since CPython
# reuses a freed object's address for the next allocation -- two unrelated
# records built at different times could collide on `id()` alone and silently
# return each other's metrics. `(case["name"], case["E0_keV"])` is already the
# results store's own primary key (`results[name][E0] = record`, see
# `results.store.store_result`), so it's guaranteed unique per record and --
# unlike the rest of `case` (which can carry unhashable `composition`/
# `hkl_list`/`abs_layers` entries) is always a plain hashable (str, float)
# pair. Derived source current is the only record/settings value
# `line_metrics` reads today -- extend this key if that grows.
_LINE_METRICS_CACHE = {}
_LINE_METRICS_CACHE_MAX = 100_000


def _cached_line_metrics(r, settings, rel_prominence, line_metric):
    case = r["case"]
    key = (
        case["name"],
        case["E0_keV"],
        beam_current_na(r, settings),
        rel_prominence,
        line_metric,
    )
    cached = _LINE_METRICS_CACHE.get(key)
    if cached is not None:
        return cached
    value = line_metrics(r, settings, rel_prominence, metric=line_metric)
    if len(_LINE_METRICS_CACHE) >= _LINE_METRICS_CACHE_MAX:
        _LINE_METRICS_CACHE.clear()
    _LINE_METRICS_CACHE[key] = value
    return value


def _metrics_map(recs, settings, rel_prominence, line_metric):
    """``id(rec) -> line_metrics(rec)`` for every record -- the same per-record
    metric dict every sweep draw (matplotlib and altair) builds. Computed once
    per record and cached ACROSS calls (see ``_cached_line_metrics``), so
    re-rendering after a change that doesn't touch these records (e.g. a
    marimo tab re-rendering because an unrelated widget changed) reuses the
    prior peak-finding results instead of redoing them."""
    return {id(r): _cached_line_metrics(r, settings, rel_prominence, line_metric) for r in recs}


def _line_brem(r, settings, convolve=None):
    """Detected line and brem densities (per eV, before the unit scale) for one
    record, honoring the QE / brem-source flags. ``convolve`` overrides
    settings.convolve_with_det when given (True/False), so a caller can draw the
    response-free source (convolve=False) and detector-convolved (convolve=True) spectra
    side by side."""
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    detector = Detector(response=LegacyEDS(apply_qe=settings.apply_detector_qe, convolve=do_conv))
    line_det = detector.score(r["E_grid"], line_spectrum(r), fwhm_eV=r["fwhm"])
    brem_det = detected_background(r, settings, convolve=do_conv) / r["scale"]
    return line_det, brem_det


# ---- Timepix3 / Eagle XO detector response prep -------------------------------
# Renderer-neutral per-record detector response builders, shared by
# :mod:`pyrite.plots.mpl.detectors` (matplotlib) and
# :mod:`pyrite.plots.altair.detectors` (Altair) -- the same physics/units, only
# the renderer differs.
SI_K_EDGE_EV = 1839.0  # silicon K absorption edge -> the QE notch the lines cross


def _thr_keV():
    return tpx.THRESHOLD_E * tpx.W_EHP_EV / 1e3


def _tpx_detected(r, settings, thickness_um, bias_v, n_mc, seed):
    """Incident and Timepix3-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by tpx.get_response."""
    incident = incident_spectrum(r) * r["scale"]
    detector = Detector(
        response=Timepix3(
            n_mc=n_mc,
            seed=seed,
            thickness_um=thickness_um,
            bias_v=bias_v,
        )
    )
    return incident, detector.score(r["E_grid"], incident_spectrum(r), scale=r["scale"])


def _eag_detected(r, settings, coating="BN", resolve_energy=False):
    """Incident and Eagle-detected (line + brem) [Phs/eV/s/nA] on r['E_grid'];
    the per-grid response is cached by eag.get_response."""
    incident = incident_spectrum(r) * r["scale"]
    detector = Detector(response=EagleXO(coating=coating, resolve_energy=resolve_energy))
    return incident, detector.score(r["E_grid"], incident_spectrum(r), scale=r["scale"])


def _eag_wide_brem(r, coating="BN"):
    """Wide-grid Eagle XO incident and detected brem [Phs/eV/s/nA]."""
    E = np.asarray(r["E_grid_brem"], dtype=float)
    incident = np.asarray(r["brem_wide"], dtype=float) * r["scale"]
    detector = Detector(response=EagleXO(coating=coating))
    detected = detector.score(E, r["brem_wide"], scale=r["scale"])
    # EagleResponse sanitizes non-finite response-matrix inputs. This wide-grid
    # diagnostic historically exposed missing input bins as NaN; retain that
    # presentation contract after routing the actual response through Detector.
    detected = np.where(np.isnan(incident), np.nan, detected)
    return E, incident, detected


def _eag_wide_charge(r, coating="BN", beam_current_na=1.0):
    """Wide-grid Eagle XO charge density [e-/eV/s] from brem."""
    E = np.asarray(r["E_grid_brem"], dtype=float)
    incident = np.nan_to_num(np.asarray(r["brem_wide"], dtype=float) * r["scale"])
    return E, incident * eag.qe(E, coating) * (E / eag.W_EHP_EV) * beam_current_na


# ---- material-comparison drop reasons ------------------------------------------
_COMPARISON_DROP_REASONS = {
    "quality_floor": "no candidate line met the quality floor",
    "beam_energy": "no checkpoint record exists at the selected beam energy",
    "nonfinite_ratio": "local line-to-bremsstrahlung ratio is undefined or non-finite",
}


def _comparison_drop_message(dropped):
    """Format material labels grouped with their exact exclusion reasons."""
    if not dropped:
        return ""
    if not hasattr(dropped, "items"):
        return f"{', '.join(dropped)} -- no candidate line met the gate."
    return "; ".join(
        f"{label} -- {_COMPARISON_DROP_REASONS.get(reason, reason)}"
        for label, reason in dropped.items()
    )


# ---- trajectory/groove geometry helpers ----------------------------------------
# Renderer-neutral case/geometry helpers for the electron-penetration figures,
# shared by :mod:`pyrite.plots.mpl.trajectories`, :mod:`pyrite.plots.altair.trajectories`
# and :mod:`pyrite.plots.plotly.trajectories`.
def _case_of(rec_or_case):
    """Accept either a results record (carries 'case') or a raw case dict."""
    return rec_or_case.get("case", rec_or_case)


def _groove_spec(case):
    """Blazed :class:`~pyrite.montecarlo.groove.GrooveSpec` for a case carrying a
    ``groove_spacing_ang`` knob, or ``None`` when it is absent.

    Mirrors ``montecarlo.runner._transport_case``'s spec construction exactly
    (``blazed_groove_spec(spacing, theta_obs_rad, tilt_polar_rad,
    tilt_azim_rad)``) so the penetration figures transport electrons through the
    SAME relief facets the spectrum runner does -- no new physics, no new
    convention. ``blazed_groove_spec`` validates the restricted geometry
    (theta_obs = 90 deg, tilt_azim = 180 deg, 0 < tilt_polar < 90 deg) and
    raises ``ValueError`` otherwise; callers that build cases via
    ``sweep.build_cases`` never hit that because it rejects the same geometries
    up front."""
    spacing = case.get("groove_spacing_ang")
    if spacing is None:
        return None
    from ..montecarlo import blazed_groove_spec

    return blazed_groove_spec(
        spacing,
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )


def groove_profile_z(x_ang, spec):
    """Sawtooth surface depth [Ang] into the slab at sample-frame lateral position
    ``x_ang`` [Ang].

    Apexes (``z = 0``) sit at ``x = k * spacing``; the valley floor is at the
    groove depth ``spec.depth_ang``. This is the SAME closed-form profile the
    ``_z_surf`` helper in ``tests/montecarlo/test_groove.py`` evaluates (single geometric
    source: :mod:`pyrite.montecarlo.groove`), reused here only to DRAW the
    surface -- it introduces no new physics. numpy ufuncs only, so array input
    works elementwise."""
    return surface_depth_ang(x_ang, spec)


def groove_profile_knots(x_lo_ang, x_hi_ang, spec, *, max_periods=200):
    """Minimal sample-frame sawtooth vertices ``(x_ang, z_ang)`` covering
    ``[x_lo_ang, x_hi_ang]``: two knots per period (apex at ``z = 0``, valley
    floor at ``z = depth``), so a corrugated surface needs only ~2 vertices per
    groove instead of a dense sweep.

    Returns ``None`` when the requested span exceeds ``max_periods`` grooves --
    the caller then falls back to the flat entrance face (drawing thousands of
    teeth is neither legible nor cheap). ``x`` is returned strictly increasing so
    the vertices trace the profile directly as a polyline."""
    lam = spec.spacing_ang
    k0 = int(np.floor(x_lo_ang / lam))
    k1 = int(np.ceil(x_hi_ang / lam))
    if k1 - k0 > max_periods:
        return None
    x_valley = spec.depth_ang * np.tan(spec.tilt_polar_rad)
    xs = np.empty(2 * (k1 - k0 + 1))
    ks = np.arange(k0, k1 + 1)
    xs[0::2] = ks * lam  # apexes (z = 0)
    xs[1::2] = ks * lam + x_valley  # valley floors (z = depth)
    zs = groove_profile_z(xs, spec)
    return xs, zs


def _beam_detector_basis(beam, n_hat):
    """Orthonormal 2D basis of the BEAM-DETECTOR plane: e1 = beam (-> +x, into the
    slab); e2 = the in-plane part of the detector direction (-> +y, "up"). Working
    in this plane (rather than a fixed x-z slice) keeps the beam horizontal AND the
    detector arrow pointing the right way for ANY polar/azimuthal tilt."""
    e1 = np.asarray(beam, float)
    e1 = e1 / np.linalg.norm(e1)
    nh = np.asarray(n_hat, float)
    nh = nh / np.linalg.norm(nh)
    perp = nh - np.dot(nh, e1) * e1
    if np.linalg.norm(perp) < 1e-9:  # detector ~parallel to beam: any in-plane up
        for ref in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            perp = ref - np.dot(ref, e1) * e1
            if np.linalg.norm(perp) > 1e-9:
                break
    return e1, perp / np.linalg.norm(perp)


def _beam_phase_space(case):
    """The case's own beam phase space, for the transport behind a plot.

    A trajectory plot that draws a point source for a beam the run treated as
    having finite emittance, bunch length or energy spread is a picture of a
    different beam than the one that made the spectrum. These keys are absent
    unless a profile deliberately set them, so every case built before the beam
    block existed still transports as the legacy point bunch, bit-for-bit.

    The Gaussian spot (``beam_fwhm_mm``) is deliberately NOT included: it stays
    under the caller's control, because the 2D cross-section wants a point
    source and the 3D view supplies its own display width.
    """
    return dict(
        bunch_length_fs=case.get("bunch_length_fs"),
        long_shape=case.get("long_shape", "gaussian"),
        long_offsets_fs=case.get("long_offsets_fs"),
        longitudinal_distribution=case.get("longitudinal_distribution"),
        transverse_distribution=case.get("transverse_distribution"),
        energy_spread_frac=case.get("energy_spread_frac"),
    )
