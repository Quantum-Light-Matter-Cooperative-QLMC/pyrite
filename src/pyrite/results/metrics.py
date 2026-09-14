"""
results.metrics
================

Per-record scalar metrics for the parametric heatmaps: peak-finding on the
coherent LINE spectrum (:func:`line_index`, :func:`line_quality`) and the
full metrics bundle (:func:`line_metrics`) they feed.
"""

import warnings

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .store import beam_current_na


def _find_peaks_props(spec, rel_prominence):
    """One peak-finding pass shared by line_index/line_quality/line_metrics:
    returns (peaks, props, smax). props carries 'prominences' and 'widths'
    (FWHM in samples). Empty peaks array if the spectrum is flat/zero."""
    spec = np.asarray(spec, dtype=float)
    smax = float(spec.max()) if spec.size else 0.0
    if smax <= 0:
        return np.array([], dtype=int), {}, smax
    peaks, props = find_peaks(spec, prominence=rel_prominence * smax, width=0)
    return peaks, props, smax


def line_index(spec, rel_prominence=0.03, metric="sharpness"):
    """Index of the coherent LINE in a spectrum. Operates on the LINE spectrum
    only (r['spec']); the incoherent brem never enters here.

    ``metric`` chooses which detected peak is "the line":
      "sharpness"  : largest prominence/width -- the narrowest prominent peak, so
                     a sharp PXR line beats a broader coherent feature. Can grab a
                     short, sharp secondary line at transition geometries (default).
      "prominence" : largest prominence -- the DOMINANT line (~ the tallest peak).
                     Smoother heatmaps; only moves where the dominant line truly
                     shifts. Recommended if "sharpness" picks spurious side lines.
      "max"        : the global argmax, no peak finding.
    ``rel_prominence`` is the prominence floor as a fraction of the spectrum's max
    (raise it to ignore more small wiggles). Falls back to argmax if no peak
    clears the floor.

    NOTE: this ALWAYS returns an index, even when the spectrum has no
    well-defined line (a broad ramp, or many comparable peaks) -- it just falls
    back to argmax. Use line_quality() to decide whether that index is
    meaningful before trusting line_eV / fwhm_eV."""
    spec = np.asarray(spec, dtype=float)
    if spec.size == 0:
        return 0
    peaks, props, smax = _find_peaks_props(spec, rel_prominence)
    if smax <= 0 or metric == "max" or peaks.size == 0:
        return int(np.argmax(spec))
    if metric == "prominence":
        score = props["prominences"]
    else:  # "sharpness"
        score = props["prominences"] / np.maximum(props["widths"], 1.0)
    return int(peaks[int(np.argmax(score))])


def line_quality(spec, rel_prominence=0.03, rel_width_max=0.10):
    """How "well-defined" the dominant coherent line is, as a score in [0, 1].
    Designed to flag the geometries where there is NO single meaningful line, so
    the heatmaps can blank them instead of plotting a peak-finder artifact.

    It is the product of three independent factors (all in [0, 1], so every one
    must be good for a high score), each catching a distinct failure mode:

      dominance  = top_prominence / sum(all prominences)
                   -> ~1 for a lone peak, ~1/N for N comparable peaks.
                   Catches the "many small sharp peaks of similar size" case.
      contrast   = top_prominence / max(spec)
                   -> ~1 when the peak rises from baseline, small when it is a
                   wiggle riding on a broad pedestal. Catches the "large slow
                   change with a tiny bump" case.
      narrowness = 1 - (FWHM_samples / size) / rel_width_max, clipped to [0, 1]
                   -> ~1 for a sharp line, 0 once the peak is wider than
                   rel_width_max of the whole grid. Catches the "broad smeared
                   blob" (heavily Doppler-scattered bulk) case.

    Returns 0 when no peak clears the prominence floor at all (flat / zero
    spectrum). rel_width_max is the broadness cutoff as a fraction of the grid
    (0.10 -> a line spanning >10% of the energy window scores 0 on narrowness)."""
    peaks, props, smax = _find_peaks_props(spec, rel_prominence)
    if peaks.size == 0:
        return 0.0
    proms = props["prominences"]
    k = int(np.argmax(proms))
    top = float(proms[k])
    dominance = top / float(proms.sum())
    contrast = top / smax
    relwidth = float(props["widths"][k]) / np.asarray(spec).size
    narrowness = float(np.clip(1.0 - relwidth / rel_width_max, 0.0, 1.0))
    return float(dominance * contrast * narrowness)


def _sample_energy(E: np.ndarray, sample_position: float) -> float:
    """Map scipy's fractional sample coordinate onto a physical energy axis."""
    return float(np.interp(sample_position, np.arange(E.size, dtype=float), E))


def _peak_sample_spacing(E: np.ndarray, idx: int) -> float:
    """Local energy interval represented by a sampled peak node."""
    if E.size < 2:
        return float("nan")
    if idx == 0:
        return float(E[1] - E[0])
    if idx == E.size - 1:
        return float(E[-1] - E[-2])
    return float((E[idx + 1] - E[idx - 1]) / 2.0)


def _integrate_energy_window(
    values: np.ndarray, E: np.ndarray, lower_eV: float, upper_eV: float
) -> float:
    """Trapezoidal integral over exact physical-energy bounds."""
    lower_eV = max(lower_eV, float(E[0]))
    upper_eV = min(upper_eV, float(E[-1]))
    if lower_eV >= upper_eV:
        return 0.0
    interior = (E > lower_eV) & (E < upper_eV)
    energies = np.concatenate(([lower_eV], E[interior], [upper_eV]))
    samples = np.concatenate(
        (
            [np.interp(lower_eV, E, values)],
            values[interior],
            [np.interp(upper_eV, E, values)],
        )
    )
    return float(np.trapezoid(samples, energies))


def line_metrics(r, settings, rel_prominence=0.03, n_fwhm=3.0, metric="sharpness"):
    """Scalar metrics for one record, used by the heatmaps. Two flux quantities
    are deliberately distinct:

    * ``peak_spectral_flux_density`` is the tallest sampled line density in
      photons/eV/s. ``peak_flux`` is its compatibility alias, not an
      integrated flux; ``peak_sample_spacing_eV`` records its local sampling.
    * ``coherent_flux`` integrates every line over the complete line grid.
    * ``line_flux`` integrates only the dominant peak over ``n_fwhm`` widths.
    * ``line_eV`` and ``fwhm_eV`` locate and characterize that dominant peak.
    * ``line_frac`` is the dominant line's share of line-plus-background flux.
    * ``total_flux`` integrates line plus background over the line grid.
    * ``coherent_brem_ratio`` compares all line flux to all local-grid background.
    * ``line_brem_ratio`` compares the dominant peak to its local background.
    * ``line_quality`` is the dominant-line definition score in ``[0, 1]``.

    The coherent-line quantities (peak_flux, coherent_flux, line_eV, fwhm_eV,
    and line_flux) are calculated from the LINE spectrum r['spec']. Brem enters
    total_flux, the line_frac denominator, coherent_brem_ratio (over the full
    line grid), and line_brem_ratio (over the dominant line's local window).
    peak_flux / coherent_flux / total_flux need no peak and stay valid
    everywhere; the line_index-based quantities are unreliable where
    line_quality is low (broad ramps, or many comparable peaks).

    Parameters
    ----------
    r
        Result record containing line/background arrays, grid, scale, and case.
    settings
        Read-time detector and current compatibility controls.
    rel_prominence
        Peak prominence relative to the spectrum maximum.
    n_fwhm
        Width of the dominant-line integration window in FWHM units.
    metric
        Peak-selection metric forwarded to :func:`line_index`.

    Returns
    -------
    dict
        Current-normalized source-yield and absolute flux metrics, line position/width/quality,
        line-to-background ratios, and finite-footprint hit fraction.
    """
    E = np.asarray(r["E_grid"], dtype=float)
    spec = np.asarray(r["spec"], dtype=float)
    brem = np.asarray(r["brem"], dtype=float)
    cur, sc = beam_current_na(r, settings), r["scale"]
    smax = float(spec.max()) if spec.size else 0.0
    idx = line_index(spec, rel_prominence, metric)
    try:
        # a zero-prominence/zero-width peak (argmax fallback, single-sample spike)
        # is expected at ill-defined geometries -- line_quality already gates those,
        # so silence scipy's per-peak PeakPropertyWarning (private class, hence the
        # broad filter on this one call) rather than spamming the scan log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widths = peak_widths(spec, [idx], rel_height=0.5)
            left_eV = _sample_energy(E, float(widths[2][0]))
            right_eV = _sample_energy(E, float(widths[3][0]))
            fwhm_eV = right_eV - left_eV
    except Exception:
        fwhm_eV = 0.0
    half_window_eV = n_fwhm * fwhm_eV / 2.0
    lower_eV = float(E[idx]) - half_window_eV
    upper_eV = float(E[idx]) + half_window_eV
    line_int = _integrate_energy_window(spec, E, lower_eV, upper_eV)
    brem_line_int = _integrate_energy_window(brem, E, lower_eV, upper_eV)
    coh_int = float(np.trapezoid(spec, E)) if spec.size else 0.0
    brem_int = float(np.trapezoid(brem, E))
    total_int = coh_int + brem_int
    return {
        # Per-nA values are normalized MC yields. Existing absolute rates
        # use case-derived pulsed-source current: photons/s at its rep rate.
        "peak_spectral_flux_density_per_na": smax * sc,
        "peak_sample_spacing_eV": _peak_sample_spacing(E, idx),
        # Compatibility aliases. Despite their historical names these are
        # sampled spectral densities, not energy-integrated fluxes.
        "peak_flux_per_na": smax * sc,
        "coherent_flux_per_na": coh_int * sc,
        "line_flux_per_na": line_int * sc,
        "total_flux_per_na": total_int * sc,
        "peak_spectral_flux_density": smax * sc * cur,
        "peak_flux": smax * sc * cur,
        "coherent_flux": coh_int * sc * cur,
        "line_eV": float(E[idx]),
        "fwhm_eV": fwhm_eV,
        "line_flux": line_int * sc * cur,
        "line_frac": (line_int / total_int) if total_int > 0 else float("nan"),
        "total_flux": total_int * sc * cur,
        "coherent_brem_ratio": (coh_int / brem_int) if brem_int > 0 else float("nan"),
        "line_brem_ratio": (line_int / brem_line_int) if brem_line_int > 0 else float("nan"),
        "line_quality": line_quality(spec, rel_prominence),
        # geometry diagnostic (not spectrum-derived): finite-crystal footprint-hit
        # fraction carried straight from the record. NaN on pre-feature checkpoints.
        "hit_frac": float(r.get("hit_frac", float("nan"))),
    }
