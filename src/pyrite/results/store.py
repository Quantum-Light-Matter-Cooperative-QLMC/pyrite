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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

EmissionMode = Literal["incoherent", "coherent", "both"]

import numpy as np

from ..detectors import Detector, LegacyEDS
from ..montecarlo import (
    aperture_fwhm_eV,
    beta_from_keV,
    eds_fwhm_eV,
    load_external_brem,
    mosaic_fwhm_eV,
    mosaic_psi_rad,
)

PER_NA = 6.2415e9  # electrons/s at 1 nA
DEFAULT_BUNCH_CHARGE_PC = 1.0
DEFAULT_REP_RATE_HZ = 5000.0
DEFAULT_BEAM_CURRENT_NA = DEFAULT_BUNCH_CHARGE_PC * DEFAULT_REP_RATE_HZ / 1000.0


def _pulse_current_na(case: Mapping[str, Any]) -> float:
    charge_pc = float(case.get("bunch_charge_pc", DEFAULT_BUNCH_CHARGE_PC))
    rep_rate_hz = float(case.get("rep_rate_hz", DEFAULT_REP_RATE_HZ))
    return charge_pc * rep_rate_hz / 1000.0


def beam_current_na(record_or_case: Mapping[str, Any] | None, settings=None) -> float:
    """Average source current [nA] for a result record or case.

    New cases carry pulsed-source parameters, for which ``1 pC * 1 kHz =
    1 nA`` and therefore ``I[nA] = Q[pC] * f[kHz]``. Records made before
    pulsed-beam support lack ``source_current_na``; retain their explicit
    Settings value as the compatibility fallback. New records stamp current
    outside ``case`` so default case identity remains legacy-compatible.

    Validation: beam-phase-space-metrics
    """
    if record_or_case is not None:
        if "source_current_na" in record_or_case:
            return float(record_or_case["source_current_na"])
        if "case" in record_or_case:
            case = record_or_case["case"]
            if isinstance(case, Mapping) and ("bunch_charge_pc" in case or "rep_rate_hz" in case):
                return _pulse_current_na(cast(Mapping[str, Any], case))
        else:
            return _pulse_current_na(record_or_case)
    if settings is not None:
        return float(settings.beam_current_na)
    return DEFAULT_BEAM_CURRENT_NA


@dataclass
class Settings:
    """Analysis / detector / unit knobs shared by post-processing and plots.

    ``beam_current_na`` is a compatibility fallback for pre-pulse checkpoints.
    Cases carrying ``bunch_charge_pc`` / ``rep_rate_hz`` derive their current
    from those BeamSpec fields instead.

    Parameters
    ----------
    beam_current_na
        Fallback average current in nA for legacy checkpoints.
    apply_detector_qe
        Apply the historical polymer-window EDS efficiency during analysis.
    convolve_with_det
        Convolve read-time spectra with the stored detector FWHM.
    brem_source
        ``"mc"``, ``"external"``, or ``"none"`` background selection.
    n_electrons, n_electrons_brem
        Default line and continuum macro-electron counts.
    straggling
        Enable stochastic transport energy-loss straggling.
    energy_model, max_dE_frac
        Flight integration rule and optional fractional-loss substep cap.
    emission
        ``"incoherent"``, ``"coherent"``, or ``"both"`` line policy.
    """

    beam_current_na: float = DEFAULT_BEAM_CURRENT_NA
    # Legacy EDS/SDD polymer-window QE. The detector is now the Timepix3 quad or
    # the Eagle XO (each carries its OWN QE in its forward model), so this is OFF
    # by default -- the response-free spectra then represent what leaves the
    # sample, not silently filtered by an unused SDD window. Leave False unless
    # you specifically want the old polymer-window SDD lens.
    apply_detector_qe: bool = False
    convolve_with_det: bool = False  # Gaussian EDS-resolution convolution
    brem_source: str = "mc"  # "mc" | "external" | "none"
    n_electrons: int = 450  # transport electrons for the lines
    n_electrons_brem: int = 100  # transport electrons for the background
    straggling: bool = False
    energy_model: Literal["frozen", "midpoint"] = "frozen"
    max_dE_frac: float = 0.0
    # Emission policy (tri-state). "incoherent" (default) is the incoherent line
    # spectrum, bit-for-bit; "coherent" is the phased segment sum in mc_spectrum;
    # "both" runs one transport and stores both spectra. Run-affecting, so
    # dataset_identity joins it into the hash ONLY when it diverges from
    # "incoherent" (divergence-only rule); an incoherent run's parameter_sha256
    # -- and its checkpoint stem -- stays unchanged.
    emission: EmissionMode = "incoherent"

    def __post_init__(self) -> None:
        if not isinstance(self.straggling, bool):
            raise ValueError("straggling must be a bool")
        if self.energy_model not in {"frozen", "midpoint"}:
            raise ValueError("energy_model must be 'frozen' or 'midpoint'")
        if not np.isfinite(self.max_dE_frac) or self.max_dE_frac < 0.0:
            raise ValueError("max_dE_frac must be finite and non-negative")
        if self.max_dE_frac > 0.0 and self.energy_model != "midpoint":
            raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")

    @property
    def coherent_emission(self) -> bool:
        """Derived: whether the emission policy runs the coherent kernel.

        Kept so existing transport-side readers (``build_cases``, ``blaze``,
        ``prune``, scan progress) stay unchanged while the tri-state
        ``emission`` field is the single source of truth."""
        return self.emission in {"coherent", "both"}


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
        # Stored outside ``case``: current is reporting metadata, while the
        # legacy-shaped case remains stable for checkpoint matching/identity.
        source_current_na=_pulse_current_na(case),
        case=case,
    )
    # Coherent-kernel companion spectrum, present only for an emission
    # "coherent"/"both" transport (runner attaches out["spec_coherent"] from the
    # SAME segments as ``spec``). Conditional so an incoherent record grows no
    # key at all -- every reader downstream (analyze.emission_menu, the altair
    # overlay, run.repair_line_spec) gates on its presence.
    if out.get("spec_coherent") is not None:
        results[name][E0]["spec_coherent"] = out["spec_coherent"]


def detected_background(r, settings, convolve=None):
    """Bremsstrahlung background in DETECTED units (Phs/eV/s/nA) on r['E_grid'],
    honoring the brem source + QE flags in ``settings``. ``convolve`` overrides
    settings.convolve_with_det when given (True/False) -- lets a caller draw the
    response-free source and detector-convolved background side by side."""
    do_conv = getattr(settings, "convolve_with_det", False) if convolve is None else convolve
    E = r["E_grid"]
    if settings.brem_source == "none":
        return np.zeros_like(E)
    if settings.brem_source == "external":
        path = r["case"].get("brem_file")
        return load_external_brem(path, E) if path else np.zeros_like(E)
    detector = Detector(response=LegacyEDS(apply_qe=settings.apply_detector_qe, convolve=do_conv))
    return detector.score(E, r["brem"], fwhm_eV=r["fwhm"], scale=r["scale"])


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
        b = np.asarray(brem_wide, dtype=float)
    detector = Detector(
        response=LegacyEDS(
            apply_qe=settings.apply_detector_qe and settings.brem_source != "external",
            convolve=do_conv,
        )
    )
    return E, detector.score(E, b, fwhm_eV=r["fwhm"], scale=r["scale"])
