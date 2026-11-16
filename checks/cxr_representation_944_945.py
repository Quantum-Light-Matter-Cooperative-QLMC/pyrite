"""Offline line-population and shape-convergence study for #944/#945.

Host FP64 only; no GPU, no transport. Inputs are saved bin-mean flush captures
(``E_r``, ``aw``, ``w`` arrays from ``checks/line_policy_286.py
--compare-reducer-routes``) and, optionally, the production spectra saved with
them. A line has density ``w sinc^2(aw (E - E_r) / pi)``, first-zero half width
``W = pi / aw`` and mass ``w pi / aw``.

``population`` characterizes each distinct line set (#945 Plan 1): exact and
``(E_r, aw)`` duplicates, width and weight distributions, neighbour overlap, and
the line counts a greedy energy/width merge would leave at several tolerances.
A merge count is a compression *potential*, not a cost or accuracy claim.

``shape`` scores uniform bin-mean axes and saved production spectra against a
refined exact bin-mean reference from the same lines (#944 Plans 1-2):
yield, centroid, peak height, dominant and per-peak FWHM, and the 100 eV mass
L1 under the #1143 margins. Intrinsic source spectra only; no detector response.
Grid error is isolated: every arm uses the identical line set, so Monte Carlo
error cancels; reference refinement is reported, not assumed. Measurement only;
no physics claim or independent validation verdict.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from checks.defaults_spectral_gate_642 import MARGINS, band_edges, versus

#: Greedy merge tolerances: (energy, as a fraction of the narrower W; width,
#: as an absolute difference of ln aw).
MERGE_TOLERANCES = ((0.0, 0.0), (0.01, 0.01), (0.05, 0.05), (0.1, 0.1), (0.25, 0.25))
#: Neighbour-overlap thresholds, in first-zero widths W of the narrower line.
OVERLAP_WIDTHS = (1.0, 4.0, 64.0)
#: Uniform bin-mean spacings [eV] for the convergence ladder.
LADDER_EV = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0)
#: Peaks reported per spectrum, by reference height.
N_PEAKS = 8


def load_line_sets(capture_dirs):
    """``{digest: dict(E, a, w, sources, edges)}`` of distinct flush line sets."""
    sets = {}
    for source in map(Path, capture_dirs):
        paths = [source] if source.is_file() else sorted(source.glob("flush-*.npz"))
        for path in paths:
            z = np.load(path)
            E, a, w = (np.asarray(z[k], dtype=np.float64) for k in ("E_r", "aw", "w"))
            digest = hashlib.sha256(
                b"".join(np.asarray(z[k]).tobytes() for k in ("E_r", "aw", "w"))
            ).hexdigest()[:16]
            entry = sets.setdefault(digest, dict(E=E, a=a, w=w, sources=[], axes=set()))
            for key in ("electron", "reflection", "grid"):
                if key in z.files:
                    entry[key] = np.asarray(z[key])
            entry["sources"].append(str(path))
            entry["axes"].add(int(z["edges"].size - 1))
    return sets


def weighted_quantiles(values, weights, qs=(0.01, 0.1, 0.5, 0.9, 0.99)):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return {f"q{q:g}": float(values[order][np.searchsorted(cdf, q)]) for q in qs}


def greedy_merge_count(E, a, mass, e_tol, w_tol):
    """Lines left by a single energy-sorted greedy pass.

    A line joins the open cluster when its resonance lies within ``e_tol`` of
    the cluster's mass centroid, in units of the narrower first-zero width,
    and its ``ln aw`` within ``w_tol`` of the cluster's. ``e_tol = w_tol = 0``
    counts exact ``(E, aw)`` duplicates. Upper bound on achievable count for
    these rules; a heuristic, not an optimal clustering.
    """
    order = np.lexsort((a, E))
    E, a, mass = E[order], a[order], mass[order]
    log_a = np.log(a)
    clusters = 0
    c_e = c_la = c_m = None
    for e, la, m in zip(E, log_a, mass, strict=True):
        if c_m is not None:
            width = np.pi / np.exp(max(la, c_la))
            same = (
                (e == c_e and la == c_la)
                if e_tol == 0
                else (abs(e - c_e) <= e_tol * width and abs(la - c_la) <= w_tol)
            )
            if same:
                total = c_m + m
                c_e = (c_e * c_m + e * m) / total if total > 0 else c_e
                c_la = (c_la * c_m + la * m) / total if total > 0 else c_la
                c_m = total
                continue
        clusters += 1
        c_e, c_la, c_m = e, la, m
    return clusters


def overlap_fractions(E, a, mass):
    """Mass fraction of lines whose nearest neighbour lies within k widths."""
    order = np.argsort(E)
    E, W, mass = E[order], (np.pi / a)[order], mass[order]
    gap = np.full(E.size, np.inf)
    d = np.diff(E)
    narrower = np.minimum(W[1:], W[:-1])
    rel = d / narrower
    gap[1:] = np.minimum(gap[1:], rel)
    gap[:-1] = np.minimum(gap[:-1], rel)
    total = mass.sum()
    return {f"within_{k:g}W": float(mass[gap <= k].sum() / total) for k in OVERLAP_WIDTHS}


def attribution(entry, mass):
    """Per-primary and per-reflection mass shares, and within-primary merges.

    A merge across primaries erases per-primary contributions, so the
    within-primary greedy count is the compression available without losing
    variance/ESS diagnostics.
    """
    if "electron" not in entry:
        return dict(lost_diagnostics=["primary/history index", "reflection row", "material"])
    E, a, el = entry["E"], entry["a"], entry["electron"]
    ids, inverse = np.unique(el, return_inverse=True)
    per = np.bincount(inverse, weights=mass)
    out = dict(
        electrons_with_lines=int(ids.size),
        electron_mass=dict(
            max_share=float(per.max() / per.sum()),
            effective_electrons=float(per.sum() ** 2 / np.sum(per**2)),
            relative_se_among_lined_electrons=float(
                np.sqrt(ids.size / (ids.size - 1)) * np.std(per) / np.mean(per) / np.sqrt(ids.size)
            )
            if ids.size > 1
            else None,
        ),
        lines_per_electron_q=weighted_quantiles(
            np.bincount(inverse).astype(float), np.ones(ids.size)
        ),
    )
    if "reflection" in entry:
        rid, rinv = np.unique(entry["reflection"], return_inverse=True)
        rmass = np.bincount(rinv, weights=mass)
        out["reflections"] = dict(
            count=int(rid.size),
            top_share=float(rmass.max() / rmass.sum()),
            effective=float(rmass.sum() ** 2 / np.sum(rmass**2)),
        )
    within = {}
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(ids.size + 1))
    for e_tol, w_tol in MERGE_TOLERANCES[2:]:
        kept = 0
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            sel = order[lo:hi]
            kept += greedy_merge_count(E[sel], a[sel], mass[sel], e_tol, w_tol)
        within[f"e{e_tol:g}_w{w_tol:g}"] = E.size / kept
    out["merge_ratio_within_electron"] = within
    return out


def fourier_plan(entry, band, spacing):
    """Host replica of the default device Fourier plan on a uniform ``band`` axis.

    Mirrors ``cuda_fourier_bin_mean_reduce``/``cuda_positive_modes`` planning
    (broad split, period, per-line last mode, cutoff rectangles and the
    NUFFT/direct split) without evaluating anything. ``direct_pairs`` and the
    NUFFT source/mode totals are work proxies, not timings.
    """
    from pyrite.montecarlo.spectrum.cxr._bin_broad import broad_width_for_axis
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL, fourier_period
    from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import cutoff_rectangle_array

    _, grid = uniform_axis(band, spacing)
    h, n_bins, c0 = float(grid[1] - grid[0]), int(grid.size), float(grid[0])
    broad_w = broad_width_for_axis(h, (n_bins - 1) * h, FOURIER_RESIDUAL_TOL, 1000.0)
    E, a = entry["E"], entry["a"]
    broad = a <= np.pi / broad_w
    E, a = E[~broad], a[~broad]
    n_fft, P, span = fourier_period(
        np.array([E.min(), E.max()]), np.array([a.min()]), c0, h, n_bins
    )
    counts = np.sort(np.floor(2.0 * a / (2.0 * np.pi / P)).astype(np.int64))
    rects = cutoff_rectangle_array(counts)
    modes, sources = rects[:, 1] - rects[:, 0], rects[:, 3] - rects[:, 2]
    large = (sources >= 64) & (modes >= 32) & (sources * modes >= 262144)
    return dict(
        spacing_eV=h,
        bins=n_bins,
        broad_width_eV=broad_w,
        broad_lines=int(broad.sum()),
        period_eV=P,
        span_eV=span,
        n_fft=int(n_fft),
        max_mode=int(counts[-1]),
        max_mode_over_n_fft=float(counts[-1] / n_fft),
        rectangles=int(rects.shape[0]),
        nufft_rectangles=int(large.sum()),
        nufft_sources=int(sources[large].sum()),
        nufft_modes=int(modes[large].sum()),
        direct_pairs=int((sources * modes)[~large].sum()),
        narrowest_W_eV=float(np.pi / a.max()),
        widest_W_eV=float(np.pi / a.min()),
    )


def characterize(entry):
    E, a, w = entry["E"], entry["a"], entry["w"]
    W = np.pi / a
    mass = w * W
    triples = np.stack([E, a, w], axis=1)
    n_unique_triples = int(np.unique(triples, axis=0).shape[0])
    t0 = time.perf_counter()
    merges = {
        f"e{e:g}_w{lw:g}": greedy_merge_count(E, a, mass, e, lw) for e, lw in MERGE_TOLERANCES
    }
    merge_s = time.perf_counter() - t0
    return dict(
        n_lines=int(E.size),
        sources=len(entry["sources"]),
        axes=sorted(entry["axes"]),
        total_mass=float(mass.sum()),
        nonpositive_weight=int(np.sum(w <= 0)),
        resonance_eV=dict(
            min=float(E.min()), max=float(E.max()), mass_q=weighted_quantiles(E, mass)
        ),
        first_zero_width_eV=dict(
            min=float(W.min()),
            max=float(W.max()),
            mass_q=weighted_quantiles(W, mass),
            count_q=weighted_quantiles(W, np.ones_like(W)),
        ),
        mass_share=dict(
            top1=float(np.sort(mass)[-1] / mass.sum()),
            top1pct_lines=float(np.sort(mass)[-max(1, E.size // 100) :].sum() / mass.sum()),
            effective_lines=float(mass.sum() ** 2 / np.sum(mass**2)),
        ),
        exact_duplicate_ratio=E.size / n_unique_triples,
        overlap=overlap_fractions(E, a, mass),
        merge_counts=merges,
        merge_ratio={k: E.size / v for k, v in merges.items()},
        merge_host_s=merge_s,
        **attribution(entry, mass),
        fourier_plan={f"{h:g}": fourier_plan(entry, _plan_band(entry), h) for h in (0.2, 0.05)},
    )


def _plan_band(entry):
    if "grid" in entry:
        return float(entry["grid"][0]), float(entry["grid"][-1])
    return 50.0, 8700.0


def cumulative_mass(entry, edges, block=1 << 24):
    """Exact line mass below each edge, ``sum_l w_l (pi/a_l) (F(x_l) + 1/2)``.

    Direct host FP64 sum over every line and edge (no envelope, no period), so
    any bin mass is a difference of two values. Cost is lines x edges.
    """
    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import sincsq_tail

    E, a, w = entry["E"], entry["a"], entry["w"]
    scale = w * np.pi / a
    edges = np.asarray(edges, dtype=np.float64)
    out = np.zeros(edges.size)
    step = max(1, block // max(1, E.size))
    for i in range(0, edges.size, step):
        x = a[:, None] * (edges[None, i : i + step] - E[:, None]) / np.pi
        out[i : i + step] = scale @ (sincsq_tail(x) + 0.5 * np.sign(x))
    return out + 0.5 * scale.sum()


def masses_from(edges, cumulative):
    return np.diff(cumulative)


def grid_masses(grid, density, edges):
    """Bin-mean spectrum mass in ``edges`` bins (``bin_axis`` cells, linear split)."""
    from pyrite.energy_grid.bandwidth_check import _bin_masses

    return _bin_masses(grid, density, edges, quadrature="bin-mean")


def band_summary(edges100, m100, edges_c, m_c):
    """Yield, 100 eV masses and centroid from ``edges_c`` midpoint masses."""
    mid = 0.5 * (edges_c[1:] + edges_c[:-1])
    return dict(
        yield_per_primary=float(m100.sum()),
        centroid_eV=float((m_c * mid).sum() / m_c.sum()),
        band_masses_100eV=m100,
    )


def peak_rows(grid, density, windows):
    """Height, position and FWHM of the highest point in each window."""
    rows = []
    for lo, hi in windows:
        sel = np.flatnonzero((grid >= lo) & (grid <= hi))
        if sel.size < 3:
            rows.append(None)
            continue
        g, d = grid[sel], density[sel]
        k = int(np.argmax(d))
        half = 0.5 * d[k]
        left = np.flatnonzero(d[:k] < half)
        right = np.flatnonzero(d[k:] < half)
        fwhm = None
        if left.size and right.size:
            i, j = int(left[-1]), k + int(right[0])
            x_l = g[i] + (half - d[i]) * (g[i + 1] - g[i]) / (d[i + 1] - d[i])
            x_r = g[j - 1] + (half - d[j - 1]) * (g[j] - g[j - 1]) / (d[j] - d[j - 1])
            fwhm = float(x_r - x_l)
        rows.append(dict(position_eV=float(g[k]), height=float(d[k]), fwhm_eV=fwhm))
    return rows


def peak_windows(grid, density, n_peaks, half_width_eV):
    """Windows around the highest separated local maxima of a survey spectrum."""
    interior = np.flatnonzero((density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])) + 1
    interior = interior[np.argsort(density[interior])[::-1]]
    centres = []
    for idx in interior:
        if len(centres) >= n_peaks:
            break
        if all(abs(grid[idx] - c) > 2 * half_width_eV for c in centres):
            centres.append(float(grid[idx]))
    return [(c - half_width_eV, c + half_width_eV) for c in sorted(centres)]


def exact_peaks(entry, windows, spacing):
    rows = []
    for lo, hi in windows:
        edges = np.arange(lo, hi + 0.5 * spacing, spacing)
        cells = masses_from(edges, cumulative_mass(entry, edges)) / spacing
        rows.extend(peak_rows(0.5 * (edges[1:] + edges[:-1]), cells, [(lo, hi)]))
    return rows


def compare(candidate, reference, c_peaks, r_peaks):
    out = versus(
        dict(candidate, peak_density_per_eV=None, dominant_fwhm_eV=None),
        dict(reference, peak_density_per_eV=None, dominant_fwhm_eV=None),
    )
    peaks = []
    for c, r in zip(c_peaks, r_peaks, strict=True):
        if c is None or r is None:
            peaks.append(None)
            continue
        peaks.append(
            dict(
                shift_eV=c["position_eV"] - r["position_eV"],
                height_rel=(c["height"] - r["height"]) / r["height"],
                fwhm_rel=None
                if c["fwhm_eV"] is None or r["fwhm_eV"] is None
                else (c["fwhm_eV"] - r["fwhm_eV"]) / r["fwhm_eV"],
            )
        )

    def worst(key):
        vals = [abs(p[key]) for p in peaks if p is not None and p[key] is not None]
        return max(vals) if vals else None

    out.update(peaks=peaks, worst_height_rel=worst("height_rel"), worst_fwhm_rel=worst("fwhm_rel"))
    return out


def uniform_axis(band, spacing):
    n = int(np.ceil((band[1] - band[0]) / spacing))
    edges = np.linspace(band[0], band[1], n + 1)
    return edges, 0.5 * (edges[1:] + edges[:-1])


def local_shape_reference(entry, windows, spacing):
    """Exact cell masses on fixed local windows; independent of peak crossings."""
    references = []
    for band in windows:
        edges, _ = uniform_axis(band, spacing)
        references.append((edges, masses_from(edges, cumulative_mass(entry, edges))))
    return references


def local_shape_errors(grid, density, references):
    """Window L1/mass and L∞/peak for piecewise-constant bin-mean spectra.

    Both spectra are integrated onto the same reference cells. This tests
    local shape without choosing a half-maximum crossing; its reference-cell
    approximation must be checked by halving the reference spacing.
    """
    rows = []
    for edges, reference in references:
        candidate = grid_masses(grid, density, edges)
        delta = np.abs(candidate - reference)
        mass = float(reference.sum())
        peak = float(np.max(reference / np.diff(edges)))
        rows.append(
            dict(
                l1=float(delta.sum() / mass) if mass > 0.0 else None,
                linf=float(np.max(delta / np.diff(edges)) / peak) if peak > 0.0 else None,
            )
        )
    return dict(
        windows=rows,
        worst_l1=max((r["l1"] for r in rows if r["l1"] is not None), default=None),
        worst_linf=max((r["linf"] for r in rows if r["linf"] is not None), default=None),
    )


def shape_study(entry, band, ladder, args, saved):
    """Exact references, exact uniform ladder and saved production spectra.

    The reference masses are exact; refinement only affects the centroid
    midpoint rule and the peak-window FWHM/height, both reported at two
    spacings. Candidates are scored with the same masses/centroid rule.
    """
    edges100 = band_edges(*band)
    timers = {}
    t0 = time.perf_counter()
    e_c, _ = uniform_axis(band, args.centroid_spacing)
    m_c = masses_from(e_c, cumulative_mass(entry, e_c))
    m100 = grid_masses(0.5 * (e_c[1:] + e_c[:-1]), m_c / np.diff(e_c), edges100)
    m100_exact = masses_from(edges100, cumulative_mass(entry, edges100))
    reference = band_summary(edges100, m100_exact, e_c, m_c)
    e_c2, _ = uniform_axis(band, 2 * args.centroid_spacing)
    coarse_ref = band_summary(
        edges100, m100_exact, e_c2, masses_from(e_c2, cumulative_mass(entry, e_c2))
    )
    timers["band_s"] = time.perf_counter() - t0
    survey = np.load(saved[args.survey]) if args.survey in saved else None
    if survey is None:
        e_s, g_s = uniform_axis(band, 0.2)
        d_s = masses_from(e_s, cumulative_mass(entry, e_s)) / np.diff(e_s)
    else:
        g_s, d_s = np.asarray(survey["grid_eV"], float), np.asarray(survey["density"], float)
    windows = peak_windows(g_s, d_s, args.peaks, args.window)
    t0 = time.perf_counter()
    r_peaks = exact_peaks(entry, windows, args.peak_spacing)
    r_peaks2 = exact_peaks(entry, windows, 2 * args.peak_spacing)
    local_ref = local_shape_reference(entry, windows, args.peak_spacing)
    local_coarse = local_shape_reference(entry, windows, 2 * args.peak_spacing)
    timers["peaks_s"] = time.perf_counter() - t0
    arms = {"reference_coarsened": compare(coarse_ref, reference, r_peaks2, r_peaks)}
    arms["reference_coarsened"]["note"] = "2x centroid and peak spacing; reference refinement"
    arms["reference_coarsened"]["local_shape"] = [
        local_shape_errors(0.5 * (edges[1:] + edges[:-1]), mass / np.diff(edges), [fine])
        for (edges, mass), fine in zip(local_coarse, local_ref, strict=True)
    ]
    arms["reference_coarsened"]["m100_interp_l1"] = float(
        np.abs(m100 / m100.sum() - m100_exact / m100_exact.sum()).sum()
    )
    for spacing in ladder:
        edges, grid = uniform_axis(band, spacing)
        t0 = time.perf_counter()
        density = masses_from(edges, cumulative_mass(entry, edges)) / np.diff(edges)
        cand = band_summary(
            edges100, grid_masses(grid, density, edges100), e_c, grid_masses(grid, density, e_c)
        )
        arms[f"exact_uniform_{spacing:g}"] = compare(
            cand, reference, peak_rows(grid, density, windows), r_peaks
        )
        arms[f"exact_uniform_{spacing:g}"].update(
            nodes=int(grid.size),
            host_s=time.perf_counter() - t0,
            local_shape=local_shape_errors(grid, density, local_ref),
        )
    for name, path in saved.items():
        z = np.load(path)
        grid, density = np.asarray(z["grid_eV"], float), np.asarray(z["density"], float)
        if grid[0] > band[0] + 5 or grid[-1] < band[1] - 5:
            arms[f"saved_{name}"] = dict(
                skipped=f"axis {grid[0]:.2f}-{grid[-1]:.2f} eV does not cover band"
            )
            continue
        cand = band_summary(
            edges100, grid_masses(grid, density, edges100), e_c, grid_masses(grid, density, e_c)
        )
        arms[f"saved_{name}"] = compare(cand, reference, peak_rows(grid, density, windows), r_peaks)
        arms[f"saved_{name}"]["nodes"] = int(grid.size)
        arms[f"saved_{name}"]["local_shape"] = local_shape_errors(grid, density, local_ref)
    return dict(
        band_eV=list(band),
        timers=timers,
        windows_eV=windows,
        reference=dict(reference, band_masses_100eV=None),
        reference_peaks=r_peaks,
        arms=arms,
        margins=MARGINS,
    )


def replay_axes(band, spacings, saved_edges):
    """``{name: (grid, edges)}``: uniform ``c_k +- h/2`` axes and saved flush axes."""
    from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

    axes = {}
    for spacing in spacings:
        edges, grid = uniform_axis(band, spacing)
        axes[f"uniform_{spacing:g}"] = (grid, edges)
    for name, path in saved_edges.items():
        z = np.load(path)
        grid = np.asarray(z["grid_eV" if "grid_eV" in z.files else "grid"], dtype=np.float64)
        axes[name] = (grid, np.asarray(_energy_bin_edges_and_widths(grid)[0], dtype=np.float64))
    return axes


def replay(entry, axes, repeats, out_dir, digest, near_far=True, broad_floors=()):
    """Time production-resolved and forced near/far bin-mean reductions; CUDA/ROCm.

    One flush of every line per call, as in the #286 captures. The resolved
    method is what production picks for that axis (``fourier`` on uniform
    axes when the backend is available, else ``auto`` near/far). Timings are
    synchronized warm medians after one untimed call; spectra are saved as
    ``grid_eV``/``density`` (flush lines, unscaled) for the ``shape`` mode.
    """
    import cupy as cp

    from pyrite.montecarlo.spectrum.cxr._bin_quadrature import (
        resolve_bin_reduction,
        run_bin_mean_reduction_kernel,
    )
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import cuda_fourier_bin_mean_reduce

    e, a, w = (cp.asarray(entry[k], dtype=cp.float32) for k in ("E", "a", "w"))
    rows = {}
    for name, (grid, edges) in axes.items():
        d_edges = cp.asarray(edges)
        inv = cp.asarray(1.0 / np.diff(edges))
        method, axis, _ = resolve_bin_reduction(edges)
        arms = {"resolved": (method, axis)}
        if method == "fourier" and near_far:
            arms["near_far"] = ("auto", None)
        if method == "fourier":
            arms.update({f"broad_{b:g}": ("broad", b) for b in broad_floors})
        row = rows.setdefault(name, dict(nodes=int(grid.size), route=method))
        for arm, (kind, fourier_axis) in arms.items():
            times = []
            for repeat in range(repeats + 1):
                out = cp.zeros(grid.size, dtype=cp.float64)
                cp.cuda.Device().synchronize()
                t0 = time.perf_counter()
                if kind == "broad":
                    out += cuda_fourier_bin_mean_reduce(e, a, w, axis, broad_width_eV=fourier_axis)
                else:
                    run_bin_mean_reduction_kernel(
                        e, a, w, d_edges, inv, out=out, method=kind, fourier_axis=fourier_axis
                    )
                cp.cuda.Device().synchronize()
                if repeat:
                    times.append(time.perf_counter() - t0)
            host = cp.asnumpy(out)
            path = out_dir / f"replay-{digest}-{name}-{arm}.npz"
            np.savez(path, grid_eV=grid, density=host)
            row[arm] = dict(warm_s=times, median_s=float(np.median(times)), spectrum=str(path))
            print(name, arm, kind, f"{np.median(times):.4f}s", flush=True)
    return dict(
        axes=rows,
        n_lines=int(e.size),
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    )


def _layered_case(args):
    """#642/#1173 stack (2000 A MoS2 on 20 um Si, 30 keV) under the adaptive bin-mean policy.

    Reflection indices are per radiator layer; the capture does not record
    which layer emitted a line.
    """
    import dataclasses

    from checks.defaults_spectral_gate_642 import scene_sweep
    from pyrite.campaign.sweep import build_cases
    from pyrite.energy_grid.bandwidth_check import RESONANCE_BANDWIDTH_POLICY

    sweep = dataclasses.replace(
        scene_sweep("mos2-on-si", 30.0, None),
        cxr_grid_policy={
            "bandwidth": RESONANCE_BANDWIDTH_POLICY,
            "resolution": "resonance-local",
            "quadrature": "bin-mean",
        },
    )
    n = int(args.electrons)
    case = dict(build_cases(sweep, n_electrons=n, n_electrons_brem=n)[0])
    case["seed"] = int(args.seed)
    return case


def capture(args):
    """Transport one production case and save every queued incoherent line.

    Hooks the batched route's attribution call site, which sees each block's
    surviving lines with their segment rows, so each line keeps its electron
    (``st.seg_elec_id``) and reflection column (``gm_idx % N_g``). The case
    runs the ``resonance-population``/``resonance-local`` bin-mean policy; its
    line axis is saved as ``grid``. A per-line count of lines handed to the
    reduction batch checks that the capture is complete. CUDA/ROCm only.
    """
    import cupy as cp

    from pyrite.energy_grid.bandwidth_check import build_case
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.cxr import _batched

    config = dict(
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        tilt_azim_deg=args.azimuth,
        n_electrons=args.electrons,
    )
    if args.material == "mos2-on-si":
        built = _layered_case(args)
    else:
        built = build_case(args.material, args.energy, config, seed=args.seed, resolution="local")
    rows, queued = [], [0]
    original_record, original_append = _batched.record_cxr_attribution, _batched._CxrBatch.append

    def record(audit, st, bt, blk, gm_idx, E_r, a_width, weight, factors):
        if _batched._audit_skips_reduction(audit):
            # Population-audit pass: same lines, never reduced.
            return original_record(audit, st, bt, blk, gm_idx, E_r, a_width, weight, factors)
        electron = st.seg_elec_id[blk.sb][gm_idx // bt.N_g]
        rows.append(
            tuple(
                cp.asnumpy(v)
                for v in (
                    E_r.astype(cp.float32),
                    a_width.astype(cp.float32),
                    weight.astype(cp.float32),
                    electron.astype(cp.int32),
                    (gm_idx % bt.N_g).astype(cp.int32),
                )
            )
        )
        return original_record(audit, st, bt, blk, gm_idx, E_r, a_width, weight, factors)

    def append(self, E_r, aw, w):
        queued[0] += int(E_r.size)
        return original_append(self, E_r, aw, w)

    _batched.record_cxr_attribution, _batched._CxrBatch.append = record, append
    t0 = time.perf_counter()
    try:
        result = runner.run_case(built)
    finally:
        _batched.record_cxr_attribution, _batched._CxrBatch.append = (
            original_record,
            original_append,
        )
    wall = time.perf_counter() - t0
    if not rows:
        raise RuntimeError("no batched incoherent lines captured (route not batched?)")
    cols = [np.concatenate([r[k] for r in rows]) for k in range(5)]
    from pyrite._backend import _to_cpu

    grid = np.asarray(_to_cpu(result["E_grid"]), dtype=np.float64)
    spec = result.get("spec")
    density = np.asarray(_to_cpu(spec), dtype=np.float64) if spec is not None else np.zeros(0)
    if density.shape != grid.shape:
        density = np.zeros(0)
    from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

    edges, widths = _energy_bin_edges_and_widths(grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        E_r=cols[0],
        aw=cols[1],
        w=cols[2],
        electron=cols[3],
        reflection=cols[4],
        edges=edges,
        inv_width=1.0 / widths,
        grid=grid,
        density=density,
    )
    meta = dict(
        material=args.material,
        energy_keV=args.energy,
        thickness_ang=args.thickness,
        tilt_deg=args.tilt,
        azimuth_deg=args.azimuth,
        electrons=args.electrons,
        seed=args.seed,
        lines=int(cols[0].size),
        queued_lines=queued[0],
        complete=queued[0] == cols[0].size,
        bins=int(grid.size),
        wall_s=wall,
        result_keys=sorted(result)[:80],
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("lines", "queued_lines", "complete", "bins", "wall_s")}))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("population", "shape", "replay", "capture"))
    parser.add_argument(
        "captures", nargs="*", type=Path, help="directories of flush-*.npz or capture .npz files"
    )
    parser.add_argument("--material", default="hopg")
    parser.add_argument("--energy", type=float, default=300.0, help="beam energy [keV]")
    parser.add_argument("--thickness", type=float, default=1e7, help="[Angstrom]")
    parser.add_argument("--tilt", type=float, default=45.0)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument(
        "--electrons",
        type=int,
        default=1,
        help="capture: primaries; shape: saved spectra are flush lines / electrons",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--band", nargs=2, type=float, default=(50.0, 8700.0))
    parser.add_argument(
        "--auto-band", action="store_true", help="replay: band and axis from the capture grid"
    )
    parser.add_argument("--ladder", nargs="*", type=float, default=LADDER_EV)
    parser.add_argument("--centroid-spacing", type=float, default=0.5)
    parser.add_argument("--peak-spacing", type=float, default=0.01)
    parser.add_argument("--peaks", type=int, default=N_PEAKS)
    parser.add_argument("--window", type=float, default=15.0, help="peak half window [eV]")
    parser.add_argument("--survey", default="full_bin", help="saved arm used to locate peaks")
    parser.add_argument("--saved", nargs="*", default=(), help="name=spectrum.npz for one line set")
    parser.add_argument("--only", help="line-set digest prefix to analyse")
    parser.add_argument("--spacings", nargs="*", type=float, default=(0.2, 0.5, 1.0, 3.0))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--broad-floors",
        nargs="*",
        type=float,
        default=(),
        help="replay: extra Fourier arms with these broad-line floors [eV]",
    )
    parser.add_argument("--skip-near-far", action="store_true", help="replay: Fourier axes only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "capture":
        capture(args)
        return
    sets = load_line_sets(args.captures)
    if args.only:
        sets = {k: v for k, v in sets.items() if k.startswith(args.only)}
    out = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, sets={}
    )
    for digest, entry in sets.items():
        if args.mode == "population":
            out["sets"][digest] = characterize(entry)
        elif args.mode == "replay":
            saved = dict(item.split("=", 1) for item in args.saved)
            band = tuple(args.band)
            if "grid" in entry and args.auto_band:
                band = (float(entry["grid"][0]), float(entry["grid"][-1]))
                saved.setdefault("capture_axis", entry["sources"][0])
            axes = replay_axes(band, args.spacings, saved)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out["sets"][digest] = replay(
                entry,
                axes,
                args.repeats,
                args.output.parent,
                digest,
                not args.skip_near_far,
                args.broad_floors,
            )
        else:
            saved = dict(item.split("=", 1) for item in args.saved)
            entry = dict(entry, w=entry["w"] / args.electrons)
            band = tuple(args.band)
            if "grid" in entry and args.auto_band:
                band = (float(entry["grid"][0]), float(entry["grid"][-1]))
            out["sets"][digest] = shape_study(entry, band, args.ladder, args, saved)
        print(digest, json.dumps(_jsonable(out["sets"][digest]))[:400], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(out), indent=1))


if __name__ == "__main__":
    main()
