"""
results.store
==============

The ``results`` post-processing store: ``{config_name: {E0_keV: record}}``,
each record produced by :func:`store_result` from one finished MC case
(spectrum, brem, detector FWHM, unit scale, and the originating ``case``).
Also holds :class:`Settings`, the analysis/detector/unit knobs threaded
through post-processing and plots, and :func:`detected_background`, the
DETECTED-units bremsstrahlung curve built from those knobs.
"""

from dataclasses import dataclass

import numpy as np

from ..montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    convolve_detector,
    detector_efficiency,
    eds_fwhm_eV,
    load_external_brem,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
)

PER_NA = 6.2415e9  # electrons/s at 1 nA


@dataclass
class Settings:
    """Analysis / detector / unit knobs shared by post-processing and plots."""

    beam_current_na: float = 5.0
    # Legacy EDS/SDD polymer-window QE. The detector is now the Timepix3 quad or
    # the Eagle XO (each carries its OWN QE in its forward model), so this is OFF
    # by default -- the "intrinsic" spectra are then genuinely what leaves the
    # sample, not silently filtered by an unused SDD window. Leave False unless
    # you specifically want the old polymer-window SDD lens.
    apply_detector_qe: bool = False
    convolve_with_det: bool = False  # Gaussian EDS-resolution convolution
    brem_source: str = "mc"  # "mc" | "external" | "none"
    n_electrons: int = 450  # transport electrons for the lines
    n_electrons_brem: int = 100  # transport electrons for the background


def line_fwhm_eV(case: dict, E_pk: float, mosaic_rad: float | None) -> float:
    """Combined EDS, aperture, and optional capped-mosaic line FWHM [eV]."""
    fwhm_sq = (
        eds_fwhm_eV(E_pk) ** 2
        + aperture_fwhm_eV(
            E_pk, beta_from_keV(case["E0_keV"]), case["theta_obs_rad"], case["dtheta_obs_rad"]
        )
        ** 2
    )
    if mosaic_rad:
        psi = mosaic_psi_rad(case, E_pk)
        if psi is not None:
            fwhm_sq += min(mosaic_fwhm_eV(E_pk, psi, mosaic_rad), E_pk) ** 2
    return float(np.sqrt(fwhm_sq))


# ---- results store -----------------------------------------------------------
def store_result(results, case, out):
    """Post-process one finished case into ``results[name][E0]`` (in place)."""
    name, E0 = case["name"], case["E0_keV"]
    E_grid = out["E_grid"]
    E_pk = E_grid[np.argmax(out["spec"])]
    # crystal mosaicity (analytic): add the mosaic broadening in quadrature when the
    # run enabled it (case["mosaic_fwhm_rad"] from build_cases; None/absent -> skip,
    # an exact no-op so old checkpoints and mosaic=False runs are unchanged). The
    # linearization diverges as psi -> 90 deg, so cap the term at E_pk -- beyond
    # FWHM ~ E the line is washed out and the model is meaningless anyway.
    fwhm = line_fwhm_eV(case, E_pk, case.get("mosaic_fwhm_rad"))
    results.setdefault(name, {})[E0] = dict(
        E_grid=E_grid,
        spec=out["spec"],
        brem=out["brem"],
        E_grid_brem=out.get("E_grid_brem"),  # wide coarse grid (full range)
        brem_wide=out.get("brem_wide"),  # bremsstrahlung out to the beam energy
        E_pk=E_pk,
        fwhm=fwhm,
        eta=out["eta"],
        # finite-crystal footprint-hit fraction (NaN on pre-feature checkpoints
        # that never recorded it); surfaced as the "hit_frac" heatmap quantity.
        hit_frac=out.get("hit_frac", float("nan")),
        scale=case["domega_sr"] * PER_NA,  # (per e per sr) -> (per s per nA)
        case=case,
    )


def detected_background(r, settings, convolve=None):
    """Bremsstrahlung background in DETECTED units (Phs/eV/s/nA) on r['E_grid'],
    honoring the brem source + QE flags in ``settings``. ``convolve`` overrides
    settings.convolve_with_det when given (True/False) -- lets a caller draw the
    intrinsic and detector-convolved background side by side."""
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    E = r["E_grid"]
    if settings.brem_source == "none":
        return np.zeros_like(E)
    if settings.brem_source == "external":
        path = r["case"].get("brem_file")
        return load_external_brem(path, E) if path else np.zeros_like(E)
    qe = detector_efficiency(E) if settings.apply_detector_qe else 1.0
    b = r["brem"] * qe
    if do_conv:
        b = convolve_detector(E, b, r["fwhm"])
    return b * r["scale"]


def _detected_background_wide(r, settings, convolve=None):
    """Bremsstrahlung background in detected units on the widest available grid.

    Returns ``(E, brem)`` where ``brem`` is already scaled to Phs/eV/s/nA.
    Slimmed checkpoints without ``E_grid_brem``/``brem_wide`` fall back to the
    line grid used by :func:`detected_background`.
    """
    E_wide = r.get("E_grid_brem")
    brem_wide = r.get("brem_wide")
    if E_wide is None or brem_wide is None:
        E = np.asarray(r["E_grid"], dtype=float)
        return E, detected_background(r, settings, convolve=convolve)

    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    E = np.asarray(E_wide, dtype=float)
    if settings.brem_source == "none":
        return E, np.zeros_like(E)
    if settings.brem_source == "external":
        path = r["case"].get("brem_file")
        b = load_external_brem(path, E) if path else np.zeros_like(E)
    else:
        qe = detector_efficiency(E) if settings.apply_detector_qe else 1.0
        b = np.asarray(brem_wide, dtype=float) * qe
    if do_conv:
        b = convolve_detector(E, b, r["fwhm"])
    return E, b * r["scale"]
