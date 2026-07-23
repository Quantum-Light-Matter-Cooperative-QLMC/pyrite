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


def line_metrics(r, settings, rel_prominence=0.03, n_fwhm=3.0, metric="sharpness"):
    """Scalar metrics for one record, used by the heatmaps. Two flux quantities
    are deliberately distinct -- see the note below on which integrates what:

      peak_flux     : max(spec) * scale * current   [Phs/eV/s]. The tallest point
                      of the coherent (line) spectral DENSITY, in absolute
                      detected-rate-per-eV units. No peak finding -- just the max.
      coherent_flux : trapz(spec) over the WHOLE line grid * scale * current
                      [Phs/s]. ALL coherent flux (every line in the window), with
                      no peak finding -- the robust, always-well-defined total.
      line_flux     : trapz(spec) over ONLY the +-n_fwhm/2 window around the
                      single dominant found line * scale * current [Phs/s]. This
                      is one line, not the whole coherent spectrum, so it is only
                      meaningful where line_quality is high.
      line_eV       : energy of that dominant found line [eV] (see line_index).
      fwhm_eV       : spectral FWHM of that line [eV] (peak_widths at half height).
      line_frac     : line_flux / trapz(spec + brem) over the line grid -- the
                      dominant line's share of the total (lines + brem) flux.
      total_flux    : trapz(spec + brem) over the line grid * scale * current
                      [Phs/s]. (brem here is the line-grid brem, not the wide
                      grid -- this is the total IN the line window, not to E0.)
      coherent_brem_ratio : trapz(spec) / trapz(brem) over the line grid -- ALL
                      coherent (CXR) flux relative to the incoherent brem beneath
                      it (a ratio, so scale/current cancel). NaN if there's no brem.
      line_brem_ratio : trapz(spec) / trapz(brem) over ONLY the dominant line's
                      local peak window. Unlike coherent_brem_ratio, this compares
                      one line against its local brem, not all coherent flux over
                      the full line grid. NaN if that window has no brem.
      line_quality  : [0, 1] definition score of the dominant line (line_quality);
                      the heatmaps gate the line-characterization maps on it.

    The coherent-line quantities (peak_flux, coherent_flux, line_eV, fwhm_eV,
    and line_flux) are calculated from the LINE spectrum r['spec']. Brem enters
    total_flux, the line_frac denominator, coherent_brem_ratio (over the full
    line grid), and line_brem_ratio (over the dominant line's local window).
    peak_flux / coherent_flux / total_flux need no peak and stay valid
    everywhere; the line_index-based quantities are unreliable where
    line_quality is low (broad ramps, or many comparable peaks).
    """
    E = np.asarray(r["E_grid"], dtype=float)
    spec = np.asarray(r["spec"], dtype=float)
    brem = np.asarray(r["brem"], dtype=float)
    dE = float(E[1] - E[0])
    cur, sc = settings.beam_current_na, r["scale"]
    smax = float(spec.max()) if spec.size else 0.0
    idx = line_index(spec, rel_prominence, metric)
    try:
        # a zero-prominence/zero-width peak (argmax fallback, single-sample spike)
        # is expected at ill-defined geometries -- line_quality already gates those,
        # so silence scipy's per-peak PeakPropertyWarning (private class, hence the
        # broad filter on this one call) rather than spamming the scan log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_samp = float(peak_widths(spec, [idx], rel_height=0.5)[0][0])
    except Exception:
        w_samp = 0.0
    half = max(round(n_fwhm * w_samp / 2.0), 1)
    lo, hi = max(idx - half, 0), min(idx + half + 1, spec.size)
    line_int = float(np.trapezoid(spec[lo:hi], E[lo:hi])) if hi > lo else 0.0
    brem_line_int = float(np.trapezoid(brem[lo:hi], E[lo:hi])) if hi > lo else 0.0
    coh_int = float(np.trapezoid(spec, E)) if spec.size else 0.0
    brem_int = float(np.trapezoid(brem, E))
    total_int = coh_int + brem_int
    return {
        "peak_flux": smax * sc * cur,
        "coherent_flux": coh_int * sc * cur,
        "line_eV": float(E[idx]),
        "fwhm_eV": w_samp * dE,
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
