"""Fixed-population spectral roulette diagnostics; no production estimator.

Rows are contributions of whole primary histories, already divided by the
ORIGINAL incident count. Columns retain every production spectral bin. All
errors below are conditional on these captured histories and this frozen grid;
they say nothing about an unresolved rare-history population.
"""

import numpy as np


def positive_probabilities(scores, fraction, floor, *, threshold=None):
    """Two strata: highest scores kept surely, every other primary with p_min.

    A supplied threshold must have been frozen before held-out predictions.
    Ties are kept together, so the retained fraction can exceed the request.
    """
    values = np.asarray(scores, dtype=float)
    if not 0 < floor <= 1 or not 0 < fraction <= 1:
        raise ValueError("fraction and probability floor must be in (0, 1]")
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a nonempty finite vector")
    cut = float(np.quantile(values, 1 - fraction)) if threshold is None else float(threshold)
    if not np.isfinite(cut):
        raise ValueError("threshold must be finite")
    return np.where(values >= cut, 1.0, floor), cut


def angle_features(segments, n_hat, n):
    """Cheap, reflection-free angular/path features, grouped by primary ID.

    All tracks of one primary stay together. ``theta*gamma`` is a feature,
    never a hard support cut: #203's dominant primary lies outside 1/gamma.
    """
    ids = np.asarray(segments.get("electron_id", segments.get("elec_id")), dtype=int)
    if np.any((ids < 0) | (ids >= n)):
        raise ValueError("primary IDs outside original incident population")
    direction = np.asarray(segments["v_hat"], dtype=float)
    energy = np.asarray(segments.get("E_repr_keV", segments["E_keV"]), dtype=float)
    length = np.asarray(segments["L_ang"], dtype=float)
    theta = np.arccos(np.clip(direction @ np.asarray(n_hat), -1, 1))
    local = theta * (1 + energy / 510.99895)
    minimum = np.full(n, np.inf)
    np.minimum.at(minimum, ids, local)
    minimum[~np.isfinite(minimum)] = 1e6
    path = np.bincount(ids, weights=length, minlength=n)
    beamed_path = np.bincount(ids, weights=length / (1 + local**2) ** 2, minlength=n)
    rows = np.bincount(ids, minlength=n)
    return np.column_stack((np.log1p(minimum), np.log1p(path), np.log1p(beamed_path))), rows


def fit_predictor(features, pilot_mass):
    """Ridge fit on PILOT data only; return a serializable frozen predictor."""
    x = np.asarray(features, dtype=float)
    mass = np.asarray(pilot_mass, dtype=float)
    if x.shape[0] != mass.size or mass.size < x.shape[1] + 2 or np.any(mass < 0):
        raise ValueError("insufficient or invalid pilot contributions")
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(mass.size), (x - center) / scale))
    offset = max(float(mass.max()) * 1e-12, np.finfo(float).tiny)
    ridge = np.diag([0.0, *([1e-3] * x.shape[1])])
    coef = np.linalg.solve(design.T @ design + ridge, design.T @ np.log(mass + offset))
    return {"center": center.tolist(), "scale": scale.tolist(), "coef": coef.tolist()}


def predict(features, model):
    x = (np.asarray(features) - model["center"]) / model["scale"]
    return np.column_stack((np.ones(x.shape[0]), x)) @ model["coef"]


def tail_statistics(values):
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    squares = float(values @ values)
    return {
        "max_share": float(values.max(initial=0) / total) if total > 0 else None,
        "ess": total**2 / squares if squares > 0 else 0.0,
        "total": total,
    }


def band_weights(edges, bands):
    """Exact overlap widths of declared bands with the frozen bin-mean cells."""
    edges = np.asarray(edges)
    return np.asarray(
        [np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)) for lo, hi in bands]
    )


def spectral_importance(contributions, widths, chunk=2048):
    """Post-hoc full-bin heuristic; not an optimal policy or opportunity bound."""
    f = contributions
    reference = np.asarray(f.sum(axis=0))
    floor = max(float(reference.max()) * 1e-6, np.finfo(float).tiny)
    score = np.zeros(f.shape[0])
    for first in range(0, f.shape[1], chunk):
        stop = first + chunk
        scaled = f[:, first:stop] / np.maximum(reference[first:stop], floor)
        score += (scaled**2) @ widths[first:stop]
    return np.sqrt(score)


def conditional_errors(contributions, probabilities, widths, bands, *, repeats=256, seed=879):
    """Exact per-bin variance plus repeated conditional spectral errors.

    Independent Bernoulli draws occur once per PRIMARY, shared by all bins.
    Full-grid calculations are chunked; no repeats x full-grid array is kept.
    Band errors use exact cell overlaps; bin RSE summaries declare their
    reference support (>= 1e-6 of the peak), and all bins enter L1 errors.
    """
    f = contributions
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (f.shape[0],) or not np.all((p > 0) & (p <= 1)):
        raise ValueError("one strictly positive probability per primary required")
    rng = np.random.default_rng(seed)
    coefficients = (rng.random((repeats, p.size)) < p) / p
    delta = coefficients - 1
    reference = np.asarray(f.sum(axis=0))
    peak = float(reference.max(initial=0))
    total = float(reference @ widths)
    if total <= 0:
        raise ValueError("captured spectrum has zero mass")
    variance = np.zeros(reference.size)
    l1 = np.zeros(repeats)
    worst = np.zeros(repeats)
    for first in range(0, reference.size, 2048):
        stop = first + 2048
        block = f[:, first:stop]
        variance[first:stop] = (1 / p - 1) @ (block**2)
        error = np.abs(delta @ block)
        l1 += error @ widths[first:stop]
        worst = np.maximum(worst, error.max(axis=1))
    mass = np.asarray(f @ widths)
    sampled_mass = coefficients * mass
    sample_total = sampled_mass.sum(axis=1)
    max_share = np.divide(
        sampled_mass.max(axis=1), sample_total, out=np.zeros(repeats), where=sample_total > 0
    )
    squared = (sampled_mass**2).sum(axis=1)
    ess = np.divide(sample_total**2, squared, out=np.zeros(repeats), where=squared > 0)
    support = reference >= peak * 1e-6
    rse = np.sqrt(variance[support]) / reference[support]
    band_values = np.asarray(f @ bands.T)
    band_reference = band_values.sum(axis=0)
    band_sd = np.sqrt((1 / p - 1) @ (band_values**2))
    band_error = delta @ band_values
    summaries = []
    for index, ref in enumerate(band_reference):
        summaries.append(
            {
                "reference_mass": float(ref),
                "conditional_rse": float(band_sd[index] / ref) if ref > 0 else None,
                "abs_relative_error_q95": float(
                    np.quantile(np.abs(band_error[:, index]), 0.95) / ref
                )
                if ref > 0
                else None,
                **tail_statistics(band_values[:, index]),
            }
        )
    summary = {
        "conditional_only": True,
        "repeats": repeats,
        "seed": seed,
        "expected_primaries": float(p.sum()),
        "probability_min": float(p.min()),
        "reference_tail": tail_statistics(mass),
        "bands": summaries,
        "bin_support_peak_fraction": 1e-6,
        "supported_bins": int(support.sum()),
        "conditional_bin_rse_max": float(rse.max(initial=0)),
        "conditional_bin_rse_q95": float(np.quantile(rse, 0.95)),
        "conditional_bin_sd_peak_normalized_max": float(np.sqrt(variance).max() / peak),
        "resampled_l1_relative_q50_q95_max": np.quantile(l1 / total, [0.5, 0.95, 1]).tolist(),
        "resampled_bin_error_peak_normalized_q95": float(np.quantile(worst / peak, 0.95)),
        "resampled_max_share_q50_q95": np.quantile(max_share, [0.5, 0.95]).tolist(),
        "resampled_ess_min_q50": np.quantile(ess, [0, 0.5]).tolist(),
        "zero_retained_signal_replicates": int(np.count_nonzero(sample_total == 0)),
    }
    tail = summary["reference_tail"]
    summary["declared_population_tail_guards"] = {
        "max_share_limit": 0.1,
        "minimum_ess": 20.0,
        "passed": bool(tail["max_share"] <= 0.1 and tail["ess"] >= 20),
        "population_convergence_established": False,
    }
    summary["declared_conditional_spectral_margin"] = {
        "relative_margin": 0.01,
        "passed_l1_and_peak_bin_q95": bool(
            summary["resampled_l1_relative_q50_q95_max"][1] <= 0.01
            and summary["resampled_bin_error_peak_normalized_q95"] <= 0.01
        ),
        "passed_every_nonzero_band_rse": bool(
            all(
                band["conditional_rse"] is None or band["conditional_rse"] <= 0.01
                for band in summaries
            )
        ),
        "deployment_recommendation": False,
    }
    return summary, variance
