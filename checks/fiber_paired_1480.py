"""Paired frozen-transport CXR/BremsLib diagnostic for #1480 (HOPG).

Research only: no production selectors, material data or defaults are changed.
See docs/research/physics/fiber-paired-1480.md for assumptions and commands.
"""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def photon_axis(lo, hi, spacing):
    """Uniform cell centres and edges spanning the requested band exactly."""
    if not (math.isfinite(lo) and math.isfinite(hi) and 10.0 <= lo < hi):
        raise ValueError("require finite band with 10 <= start < stop")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("require 10 <= start < stop and positive finite spacing")
    count = max(2, math.ceil((hi - lo) / spacing))
    edges = np.linspace(lo, hi, count + 1)
    return (edges[:-1] + edges[1:]) / 2, edges


def reflection_domain(crystal, cutoff):
    """Production candidate enumeration with only the fiber restriction lifted.

    Use B=0 for enumeration, so all arms share a thermal-independent set.
    Keep both Friedel mates and every member of each retained family.
    """
    from pyrite.materials.crystal import CRYSTALS, dominant_reflections
    from pyrite.montecarlo.runner.reflection_cutoff import F0_RANGE_INVANG

    if not 0 < cutoff <= F0_RANGE_INVANG:
        raise ValueError(f"cutoff must be in (0, {F0_RANGE_INVANG}] 1/Angstrom")
    info = CRYSTALS[crystal]
    if info.get("texture") != "fiber":
        raise ValueError("expected a fiber-textured catalog crystal")
    with patch.dict(CRYSTALS, {crystal: dict(info, texture=None)}):
        hkls = dominant_reflections(crystal, n_families=10**9, B_ang2=0, g_max_invang=cutoff)
    full = sorted({tuple(int(i) for i in h) for h in hkls})
    basal = [h for h in full if h[:2] == (0, 0)]
    if not basal or len(basal) == len(full):
        raise ValueError("cutoff must contain both basal and non-basal reflections")
    return basal, full


def tensor_groups(crystal, hkls, site, temperature):
    """Group rows with equal projected B; no symmetry/multiplicity reduction."""
    from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector

    lattice = CRYSTALS[crystal]["lattice"]
    axis, norm = reciprocal_g_vector((0, 0, 1), lattice)
    axis = axis / norm
    groups = {}
    for hkl in hkls:
        g, norm = reciprocal_g_vector(hkl, lattice)
        # Quantize only floating-point noise in symmetry-equivalent directions.
        cos2 = round(float((g @ axis / norm) ** 2), 14)
        b = float(site.b_factor_along(cos2, temperature))
        groups.setdefault(b, []).append(hkl)
    return groups


def merge_shell_weights(parts):
    """Sum disjoint reflection groups BEFORE the nonlinear DDCS clamp."""
    shells = {}
    for g, weights in parts:
        for i, magnitude in enumerate(g):
            row = shells.setdefault(float(magnitude), dict.fromkeys(weights, 0.0))
            if row.keys() != weights.keys():
                raise ValueError("inconsistent element allocation across groups")
            for el, w in weights.items():
                row[el] += float(w[i])
    g = np.array(sorted(shells), dtype=float)
    if not g.size:
        raise ValueError("empty shell sum")
    return g, {el: np.array([shells[x][el] for x in g]) for el in shells[g[0]]}


def shell_weights(case, groups):
    from pyrite.montecarlo.spectrum.brem_lattice_share import reflection_shell_weights

    return merge_shell_weights(
        reflection_shell_weights(case["crystal"], hkls, b, case["composition"])
        for b, hkls in groups.items()
    )


def corrected_tables(case, tables, kernel, shell):
    """Use #1250's existing isotropic-orientation shell kernel and one clamp."""
    from pyrite.montecarlo.spectrum.brem_lattice_share import _lattice_share_tables

    g, weights = shell
    if g[0] < kernel.g_invang[0] or g[-1] > kernel.g_invang[-1]:
        raise ValueError("reflection set extends beyond released lattice-share kernel")
    composition = [(str(el), float(n)) for el, n in case["composition"]]
    entries = [(tables[el], el, n, weights[el]) for el, n in composition]
    results = _lattice_share_tables(
        entries, kernel, g, T_floor_keV=float(case.get("E_cut_lines_keV", 5.0))
    )
    out, record = dict(tables), []
    for (_, el, _, _), (table, info) in zip(entries, results, strict=True):
        out[el] = table
        record.append(
            dict(
                element=el, removed_fraction=info.removed_fraction, clamped_nodes=info.clamped_nodes
            )
        )
    return out, dict(
        shells=int(g.size),
        sum_U2_eV2=float(sum(w.sum() for w in weights.values())),
        elements=record,
    )


def host_density(value, grid):
    from pyrite._backend import _to_cpu

    result = np.asarray(_to_cpu(value), dtype=float)
    if result.shape != grid.shape or not np.isfinite(result).all():
        raise RuntimeError("invalid spectrum returned by production scorer")
    return result


def fwhm(grid, y):
    """Width of the contiguous half-height peak, or None if band-truncated."""
    peak = int(np.argmax(y))
    if y[peak] <= 0:
        return None
    half = y[peak] / 2
    left = right = peak
    while left > 0 and y[left - 1] >= half:
        left -= 1
    while right + 1 < y.size and y[right + 1] >= half:
        right += 1
    if left == 0 or right == y.size - 1:
        return None
    low = np.interp(half, y[left - 1 : left + 1], grid[left - 1 : left + 1])
    high = np.interp(half, y[right : right + 2][::-1], grid[right : right + 2][::-1])
    return float(high - low)


def observables(grid, widths, cxr, brem):
    line = float(cxr @ widths)
    background = float(brem @ widths)
    peak = int(np.argmax(cxr))
    return dict(
        cxr_yield=line,
        brem_yield=background,
        total_yield=line + background,
        cxr_to_brem=line / background if background > 0 else None,
        cxr_centroid_eV=float((cxr * grid) @ widths / line) if line > 0 else None,
        cxr_peak_eV=float(grid[peak]) if line > 0 else None,
        cxr_peak_density=float(cxr[peak]),
        cxr_fwhm_eV=fwhm(grid, cxr),
        peak_cxr_to_brem=float(cxr[peak] / brem[peak]) if line > 0 and brem[peak] > 0 else None,
    )


def paired_differences(metrics):
    out = {}
    for a, b in (("B", "A"), ("C", "B"), ("C", "A"), ("A", "D"), ("B", "D"), ("C", "D")):
        row = {}
        for key in (
            "cxr_yield",
            "brem_yield",
            "total_yield",
            "cxr_peak_density",
            "cxr_centroid_eV",
            "cxr_peak_eV",
            "cxr_fwhm_eV",
        ):
            x, y = metrics[a][key], metrics[b][key]
            delta = None if x is None or y is None else x - y
            row[key] = dict(absolute=delta, relative=delta / y if delta is not None and y else None)
        out[f"{a}-{b}"] = row
    return out


def azimuth_average(score, groups, angles, grid):
    """Average INTENSITIES, never amplitudes, with identical angular nodes."""
    total = np.zeros_like(grid, dtype=float)
    samples = []
    for phi in angles:
        spectrum = np.zeros_like(grid, dtype=float)
        for b, hkls in groups.items():
            spectrum += score(b, hkls, float(phi))
        total += spectrum / len(angles)
        samples.append(spectrum)
    return total, np.stack(samples)


def validate_case(case, tp):
    if case.get("crystal") != "hopg":
        raise ValueError("this slice supports HOPG only: no verified hBN tensor in #535")
    if case.get("channeling") not in (None, "off") or case.get("transport_mode") == "mixed":
        raise ValueError("only ordinary Born-line transport is supported")
    if case.get("coherent_emission") or case.get("cxr_source") == "none":
        raise ValueError("expected incoherent CXR line scoring")
    if (case.get("transition_radiation") or {}).get("model") == "two-beam":
        raise ValueError("two-beam line scoring is not supported")
    if int(tp["segs"].get("n_layers", 1)) != 1 or tp["segs"].get("cone_split") is not None:
        raise ValueError("only single-slab unsplit trajectories are supported")
    if int(tp["Ne_lines"]) != int(tp["Ne_brem"]) or int(tp["Ne_lines"]) != int(tp["segs"]["Ne"]):
        raise ValueError("CXR and continuum must use the same complete electron population")
    if int(tp["Ne_lines"]) <= 0:
        raise ValueError("empty electron population")
    if case.get("radiative_model") is not None and case.get("bremsstrahlung_model") != "bremslib":
        raise ValueError("coupled transport must already use BremsLib")
    radiator = case
    if case.get("layer_radiators") is not None:
        radiators = case["layer_radiators"]
        if len(radiators) != 1 or radiators[0] is None:
            raise ValueError("expected one crystalline radiator")
        radiator = radiators[0]
    for key in ("beam_uvw", "surface_hkl"):
        axis = radiator.get(key)
        if axis is not None and (axis[0] != 0 or axis[1] != 0 or axis[2] <= 0):
            raise ValueError("fiber roll requires the positive c axis along the sample normal")
    if {el for el, _ in case["composition"]} != {"C"}:
        raise ValueError("the tensor projection assumes one common carbon displacement tensor")
    return {
        **case,
        **radiator,
        "layer_radiators": None,
        "reflection_selection": None,
        "bremsstrahlung_model": "bremslib",
        "line_quadrature": "bin-mean",
        "coherent_emission": False,
        "allow_physics_fallback": False,
    }


def load_source(args):
    from pyrite.montecarlo import runner

    if args.artifact is not None:
        from pyrite.montecarlo.runner.artifacts import _checked_case_and_inputs
        from pyrite.montecarlo.trajectories import read_trajectory_artifact

        artifact = read_trajectory_artifact(args.artifact, load_transport=True)
        case, inputs = _checked_case_and_inputs(artifact, None)
        if int(artifact.transport["L_ang"].size) != int(artifact.attrs["segment_count"]):
            raise ValueError("trajectory artifact is truncated")
        return (
            dict(case),
            dict(inputs, segs=artifact.transport),
            dict(artifact=str(args.artifact), sha256=file_digest(args.artifact), transported=False),
        )
    from pyrite.energy_grid.convergence_case import build_ladder_case

    case = build_ladder_case(
        "hopg",
        args.energy,
        args.tilt,
        args.tilt_azimuth,
        thickness_ang=args.thickness,
        n_electrons=args.electrons,
        seed=args.seed,
    )
    case.update(
        bremsstrahlung_model="bremslib",
        reflection_selection=None,
        line_grid_policy=None,
        allow_physics_fallback=False,
    )
    if args.detector is not None:
        case["theta_obs_rad"] = np.deg2rad(args.detector)
    tp = runner._transport_case(case)
    return case, tp, dict(transported=True, seed=args.seed)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    from pyrite.energy_grid.convergence import require_identical_segments, segment_fingerprint
    from pyrite.materials.channeling.thermal import UniaxialThermalSite, thermal_site
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.case_tables import _case_bremslib_tables
    from pyrite.montecarlo.runner.emission import _layer_brem_from_segments
    from pyrite.montecarlo.runner.lattice_share import _kernel
    from pyrite.montecarlo.runner.oom import _ensure_pool_limit

    grid, edges = photon_axis(*args.band, args.spacing)
    widths = np.diff(edges)
    if not math.isfinite(args.azimuth_offset):
        raise ValueError("azimuth offset must be finite")
    if args.azimuths < 1:
        raise ValueError("azimuth count must be positive")
    if args.artifact is not None and args.detector is not None:
        raise ValueError("artifact detector geometry is fixed; omit --detector")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("spectra.npz", "summary.json", "SHA256SUMS")):
        raise FileExistsError("use a fresh output directory for each paired run")
    _ensure_pool_limit()
    started = time.perf_counter()
    source_case, tp, source = load_source(args)
    case = validate_case(source_case, tp)
    source["load_or_transport_s"] = time.perf_counter() - started
    source["diagnostic_sha256"] = file_digest(Path(__file__))
    fingerprint = segment_fingerprint(tp["segs"])
    segments = runner._segments_on_device(tp["segs"])
    staged_fingerprint = segment_fingerprint(segments)
    n_hat, groove, count = tp["n_hat"], tp.get("groove"), int(tp["Ne_lines"])
    basal, full = reflection_domain(case["crystal"], args.cutoff)
    nonbasal = [h for h in full if h[:2] != (0, 0)]
    site = thermal_site(case["crystal"], "C")
    if not isinstance(site, UniaxialThermalSite):
        raise ValueError("verified uniaxial carbon tensor required")
    b33 = float(site.b_factor_along(1.0, args.temperature))
    groups = tensor_groups(case["crystal"], nonbasal, site, args.temperature)
    tensor = {b: list(hkls) for b, hkls in groups.items()}
    tensor.setdefault(b33, []).extend(basal)
    angles = (
        float(case.get("azimuth_rad", 0.0))
        + 2 * np.pi * (np.arange(args.azimuths) + args.azimuth_offset) / args.azimuths
    )
    tables = _case_bremslib_tables(case)
    if tables is None or any(el not in tables for el, _ in case["composition"]):
        raise ValueError("all BremsLib tables must be present; no EEDL fallback allowed")
    kernel = _kernel()
    floor = float(case.get("E_cut_lines_keV", 5.0))
    from pyrite._backend import _to_cpu

    energies = [
        np.asarray(_to_cpu(tp["segs"][key]), dtype=float)
        for key in ("E_keV", "E_repr_keV")
        if tp["segs"].get(key) is not None
    ]
    max_energy = max([float(case["E0_keV"])] + [float(e.max()) for e in energies if e.size])
    if floor < kernel.T_keV[0] or max_energy > kernel.T_keV[-1]:
        raise ValueError("line-active electron energies may fall outside the subtraction kernel")
    timings, corrections = {}, {}
    arrays = dict(
        grid_eV=grid,
        edges_eV=edges,
        azimuth_rad=angles,
        basal_hkls=np.asarray(basal),
        full_hkls=np.asarray(full),
    )

    def timed(label, fn):
        t0 = time.perf_counter()
        value = fn()
        timings[label] = time.perf_counter() - t0
        print(f"{label}: {timings[label]:.2f}s", flush=True)
        return value

    def score_line(b, hkls, phi):
        pilot = dict(case, hkl_list=hkls, B_ang2=b, azimuth_rad=phi)
        return host_density(
            runner._cxr_for_segments(
                segments,
                grid,
                pilot,
                n_hat,
                case.get("abs_layers"),
                groove,
                coherent=False,
                Ne=count,
                table_cache={},
            ),
            grid,
        )

    def score_brem(tables_):
        return host_density(
            _layer_brem_from_segments(
                segments,
                grid,
                case,
                n_hat,
                case.get("abs_layers"),
                layer=None,
                tables=tables_,
                groove=groove,
                Ne=count,
            ),
            grid,
        )

    def corrected(arm, groups_):
        table, record = corrected_tables(case, tables, kernel, shell_weights(case, groups_))
        corrections[arm] = record
        return score_brem(table)

    # Basal is unchanged by a fiber-axis roll; exact B33 shared across A/B/C.
    arrays["D_cxr"] = np.zeros_like(grid)
    arrays["D_brem"] = timed("D_brem", lambda: score_brem(tables))
    arrays["A_cxr"] = timed("A_cxr", lambda: score_line(b33, basal, angles[0]))
    arrays["A_brem"] = timed("A_brem", lambda: corrected("A", {b33: basal}))
    for arm, line_groups, all_groups in (
        ("B", {b33: nonbasal}, {b33: full}),
        ("C", groups, tensor),
    ):
        extra, samples = timed(
            f"{arm}_cxr", lambda g=line_groups: azimuth_average(score_line, g, angles, grid)
        )
        arrays[f"{arm}_cxr"] = arrays["A_cxr"] + extra
        arrays[f"{arm}_nonbasal_by_azimuth"] = samples
        arrays[f"{arm}_brem"] = timed(f"{arm}_brem", lambda a=arm, g=all_groups: corrected(a, g))
    require_identical_segments(fingerprint, tp["segs"])
    require_identical_segments(staged_fingerprint, segments)
    metrics = {}
    for arm in "ABCD":
        cxr, brem = arrays[f"{arm}_cxr"], arrays[f"{arm}_brem"]
        arrays[f"{arm}_total"] = cxr + brem
        arrays[f"{arm}_removed_brem"] = arrays["D_brem"] - brem
        metrics[arm] = observables(grid, widths, cxr, brem)
        metrics[arm]["removed_brem_yield"] = float(arrays[f"{arm}_removed_brem"] @ widths)
        metrics[arm]["nonbasal_cxr_yield"] = (
            float((cxr - arrays["A_cxr"]) @ widths) if arm in "BC" else 0.0
        )
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_sha = None
    summary = dict(
        issue=1480,
        complete=True,
        code_sha=code_sha,
        source=source,
        fingerprint=fingerprint,
        staged_fingerprint=staged_fingerprint,
        electrons=count,
        E0_keV=float(case["E0_keV"]),
        tilt_deg=case.get("tilt_deg"),
        tilt_azim_deg=case.get("tilt_azim_deg"),
        thickness_ang=case.get("thickness_ang"),
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
        mosaic_sampling=case.get("mosaic_mc_sampling"),
        detector_n_hat=np.asarray(n_hat).tolist(),
        band_eV=[float(edges[0]), float(edges[-1])],
        bins=int(grid.size),
        spacing_eV=float(widths[0]),
        cutoff_invang=args.cutoff,
        azimuth_count=args.azimuths,
        azimuth_offset=args.azimuth_offset,
        basal_reflections=len(basal),
        full_reflections=len(full),
        tensor_groups=len(tensor),
        temperature_K=args.temperature,
        B33_ang2=b33,
        catalog_B_ang2=float(case["B_ang2"]),
        B11_ang2=float(site.b_factor_along(0.0, args.temperature)),
        thermal_source=site.source,
        thermal_doi=site.doi,
        continuum_kernel="production isotropic orientation average",
        line_quadrature="bin-mean",
        continuum_quadrature="cell midpoint",
        line_cutoff_keV=floor,
        brem_cutoff_keV=case.get("E_cut_brem_keV", 1.0),
        timings_s=timings,
        corrections=corrections,
        arms=metrics,
        differences=paired_differences(metrics),
    )
    np.savez_compressed(output / "spectra.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{file_digest(output / name)}  {name}\n" for name in ("spectra.npz", "summary.json")
        )
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


def ladder_bands(domains):
    """Split nested fiber domains into disjoint |g| bands (each roll/Friedel closed).

    ``domains`` are prepared ``FiberReflectionSet`` objects at increasing
    cutoffs. Band ``i`` holds the rows of domain ``i`` absent from ``i - 1``;
    each band keeps the projected B of its own domain. Nesting is asserted so
    that cumulative band sums reproduce every prepared domain exactly.
    """
    bands, previous = [], set()
    for domain in domains:
        rows = set(domain.hkls)
        if not previous <= rows:
            raise ValueError("fiber domains are not nested in cutoff")
        b = dict(zip(domain.hkls, domain.b_ang2, strict=True))
        new = sorted(rows - previous)
        bands.append(
            {
                part: [(h, b[h]) for h in new if (h[:2] == (0, 0)) == (part == "basal")]
                for part in ("basal", "nonbasal")
            }
        )
        previous = rows
    return bands


def component_score(segments, grid, case, hkls, lookup, phi, n_hat, count, groove):
    """Single-radiator incoherent CXR with its PXR and CBS parts, shape (3, n).

    Mirrors the single-slab branch of the runner's ``_cxr_for_segments_once``
    (same case keys, mosaic and line options) with ``components=True``. Row
    0 is the full row intensity including PXR-CBS interference, so it must
    equal the runner route; the ladder integrity check enforces that on the
    smallest cutoff.
    """
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo.runner.chunking import _RESOURCE_POLICY, _adaptive_chunk, _admit_chunk
    from pyrite.montecarlo.spectrum import mc_cxr_spectrum

    mosaic = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"), mosaic_nodes=case.get("mosaic_mc_nodes", 1)
    )
    if case.get("mosaic_mc_sampling") == "history":
        mosaic.update(mosaic_sampling="history", mosaic_seed=int(case["seed"]))
    chunk = _admit_chunk(
        case.get("spec_chunk") or _RESOURCE_POLICY.spec_chunk or _adaptive_chunk(grid.size),
        grid.size,
    )
    parts = mc_cxr_spectrum(
        segments,
        grid,
        crystal=case["crystal"],
        hkl_list=hkls,
        n_hat=n_hat,
        B_ang2=lookup.__getitem__,
        composition=case["composition"],
        beam_uvw=case.get("beam_uvw"),
        surface_hkl=case.get("surface_hkl"),
        azimuth_rad=phi,
        recip_miscut_rad=case.get("recip_miscut_rad"),
        sinc_cutoff=case.get("sinc_cutoff"),
        node_support_cutoff=case.get("node_support_cutoff", "auto"),
        chunk=chunk,
        layers=case.get("abs_layers"),
        groove=groove,
        coherent=False,
        electron_limit=count,
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        line_quadrature=case.get("line_quadrature", "bin-mean"),
        components=True,
        **mosaic,
    )
    return np.stack([np.asarray(_to_cpu(p), dtype=float) for p in parts])


def run_ladder(args):
    """Reciprocal-cutoff ladder of the opt-in model with its matched continuum.

    One frozen transport; each |g| band is scored once (basal and non-basal
    separately) through the same runner route as ``score_fiber_case``, and
    each cutoff's continuum is corrected with its own full prefix domain.
    The smallest cutoff is cross-checked against ``score_fiber_case``.
    """
    import resource

    from pyrite._backend import _to_cpu
    from pyrite.config import FiberAverage
    from pyrite.energy_grid.convergence import require_identical_segments, segment_fingerprint
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.case_tables import _case_bremslib_tables
    from pyrite.montecarlo.runner.emission import _layer_brem_from_segments
    from pyrite.montecarlo.runner.oom import _ensure_pool_limit
    from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel, FiberReflectionSet
    from pyrite.montecarlo.spectrum.cxr.fiber_case import _max_energy, score_fiber_case

    cuts = sorted(set(args.ladder))
    grid, edges = photon_axis(*args.band, args.spacing)
    widths = np.diff(edges)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("spectra.npz", "summary.json", "SHA256SUMS")):
        raise FileExistsError("use a fresh output directory for each ladder run")
    _ensure_pool_limit()
    started = time.perf_counter()
    source_case, tp, source = load_source(args)
    case = validate_case(source_case, tp)
    source["load_or_transport_s"] = time.perf_counter() - started
    source["diagnostic_sha256"] = file_digest(Path(__file__))
    fingerprint = segment_fingerprint(tp["segs"])
    models = [
        FiberCXRModel(
            FiberAverage(
                cutoff_invang=cut, azimuth_count=args.azimuths, reflection_batch_size=args.batch
            ),
            args.temperature,
        )
        for cut in cuts
    ]
    domains = [m.prepare() for m in models]
    bands = ladder_bands(domains)
    pilot = dict(case, reflection_selection=None, line_quadrature="bin-mean")
    if pilot.get("beam_uvw") is None and pilot.get("surface_hkl") is None:
        pilot["surface_hkl"] = (0, 0, 1)
    floor = float(pilot.get("E_cut_lines_keV", 5.0))
    max_energy = _max_energy(pilot, tp["segs"])
    tables0 = _case_bremslib_tables(pilot)
    segments = runner._segments_on_device(tp["segs"])
    n_hat, groove, count = tp["n_hat"], tp.get("groove"), int(tp["Ne_lines"])
    phi0 = float(pilot.get("azimuth_rad", 0.0))

    parts = ("cxr", "pxr", "cbs") if args.components else ("cxr",)
    zero = np.zeros((len(parts), grid.size))

    def score_rows(model, rows):
        if not rows:
            return zero.copy()
        band = FiberReflectionSet(model, tuple(h for h, _ in rows), tuple(b for _, b in rows))
        lookup = band._thermal_lookup()

        def score(hkls, phi):
            if args.components:
                return component_score(
                    segments, grid, pilot, hkls, lookup, phi, n_hat, count, groove
                )
            return _to_cpu(
                runner._cxr_for_segments(
                    segments,
                    grid,
                    dict(pilot, hkl_list=hkls, B_ang2=lookup.__getitem__, azimuth_rad=phi),
                    n_hat,
                    pilot.get("abs_layers"),
                    groove,
                    coherent=False,
                    Ne=count,
                    table_cache=None,
                )
            )

        value = band._average(score, zero.shape if args.components else grid.shape, phi0)
        return np.stack([host_density(v, grid) for v in np.reshape(value, zero.shape)])

    def score_brem(tables):
        return host_density(
            _layer_brem_from_segments(
                segments,
                grid,
                pilot,
                n_hat,
                pilot.get("abs_layers"),
                layer=None,
                tables=tables,
                groove=groove,
                Ne=count,
            ),
            grid,
        )

    arrays = dict(grid_eV=grid, edges_eV=edges, cutoffs_invang=np.asarray(cuts))
    arrays["D_brem"] = score_brem(tables0)
    rows, basal_parts, nonbasal_parts = [], zero.copy(), zero.copy()
    for cut, model, domain, band in zip(cuts, models, domains, bands, strict=True):
        t0 = time.perf_counter()
        basal_parts = basal_parts + score_rows(model, band["basal"])
        nonbasal_parts = nonbasal_parts + score_rows(model, band["nonbasal"])
        basal_sum, nonbasal_sum = basal_parts[0], nonbasal_parts[0]
        t_lines = time.perf_counter() - t0
        tables, correction = domain.corrected_bremslib_tables(
            tables0, pilot["composition"], E_cut_keV=floor, max_energy_keV=max_energy
        )
        brem = score_brem(tables)
        cxr = basal_sum + nonbasal_sum
        tag = f"g{cut:g}"
        arrays[f"{tag}_basal_cxr"], arrays[f"{tag}_nonbasal_cxr"] = basal_sum, nonbasal_sum
        arrays[f"{tag}_brem"] = brem
        split = {}
        for i, part in enumerate(parts[1:], start=1):
            arrays[f"{tag}_basal_{part}"] = basal_parts[i]
            arrays[f"{tag}_nonbasal_{part}"] = nonbasal_parts[i]
            split[f"basal_{part}_yield"] = float(basal_parts[i] @ widths)
            split[f"nonbasal_{part}_yield"] = float(nonbasal_parts[i] @ widths)
        row = dict(
            cutoff_invang=cut,
            reflections=len(domain.hkls),
            band_reflections=len(band["basal"]) + len(band["nonbasal"]),
            band_lines_s=t_lines,
            continuum_s=time.perf_counter() - t0 - t_lines,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            basal_cxr_yield=float(basal_sum @ widths),
            nonbasal_cxr_yield=float(nonbasal_sum @ widths),
            continuum_correction=correction,
            **split,
            **observables(grid, widths, cxr, brem),
        )
        if rows:
            prev = rows[-1]
            for key in ("cxr_yield", "total_yield", "brem_yield"):
                row[f"{key}_rel_step"] = (row[key] - prev[key]) / prev[key] if prev[key] else None
            prev_cxr = (
                arrays[f"g{prev['cutoff_invang']:g}_basal_cxr"]
                + arrays[f"g{prev['cutoff_invang']:g}_nonbasal_cxr"]
            )
            row["cxr_l1_step"] = float(np.abs(cxr - prev_cxr) @ widths / (cxr @ widths))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "continuum_correction"}), flush=True)
    # Integrity: the smallest rung must equal the public paired entry point.
    t0 = time.perf_counter()
    check = score_fiber_case(models[0], source_case, tp, grid)
    first = f"g{cuts[0]:g}"
    ref_cxr = arrays[f"{first}_basal_cxr"] + arrays[f"{first}_nonbasal_cxr"]
    scale = max(float(np.max(np.abs(check["cxr"]))), 1e-300)
    integrity = dict(
        cxr_max_rel=float(np.max(np.abs(check["cxr"] - ref_cxr)) / scale),
        brem_max_rel=float(
            np.max(np.abs(check["brem"] - arrays[f"{first}_brem"]))
            / max(float(np.max(np.abs(check["brem"]))), 1e-300)
        ),
        seconds=time.perf_counter() - t0,
    )
    print("integrity", json.dumps(integrity), flush=True)
    require_identical_segments(fingerprint, tp["segs"])
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_sha = None
    summary = dict(
        issue=1480,
        mode="cutoff-ladder",
        complete=True,
        code_sha=code_sha,
        source=source,
        fingerprint=fingerprint,
        electrons=count,
        E0_keV=float(case["E0_keV"]),
        tilt_deg=case.get("tilt_deg"),
        thickness_ang=case.get("thickness_ang"),
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
        detector_n_hat=np.asarray(_to_cpu(n_hat)).tolist(),
        band_eV=[float(edges[0]), float(edges[-1])],
        spacing_eV=float(widths[0]),
        azimuth_count=args.azimuths,
        reflection_batch_size=args.batch,
        components=bool(args.components),
        temperature_K=args.temperature,
        model=domains[-1].record()["model"],
        D=observables(grid, widths, np.zeros_like(grid), arrays["D_brem"]),
        ladder=rows,
        integrity=integrity,
        wall_s=time.perf_counter() - started,
    )
    np.savez_compressed(output / "spectra.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{file_digest(output / name)}  {name}\n" for name in ("spectra.npz", "summary.json")
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, help="existing schema-2 trajectory artifact")
    parser.add_argument("--energy", type=float, default=300.0, help="fresh transport energy [keV]")
    parser.add_argument(
        "--thickness", type=float, default=5000.0, help="fresh thickness [Angstrom]"
    )
    parser.add_argument("--tilt", type=float, default=45.0)
    parser.add_argument("--tilt-azimuth", type=float, default=180.0)
    parser.add_argument("--detector", type=float, help="fresh detector polar angle [degrees]")
    parser.add_argument("--electrons", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1480)
    parser.add_argument("--band", type=float, nargs=2, default=(50.0, 3000.0))
    parser.add_argument("--spacing", type=float, default=1.0, help="maximum bin width [eV]")
    parser.add_argument("--cutoff", type=float, default=8.0, help="common |g| cap [1/Angstrom]")
    parser.add_argument("--azimuths", type=int, default=8)
    parser.add_argument("--azimuth-offset", type=float, default=0.0, help="fractional angular step")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ladder",
        type=float,
        nargs="+",
        help="instead of arms A-D: opt-in model cutoff ladder [1/Angstrom], sector rolls",
    )
    parser.add_argument("--batch", type=int, default=256, help="ladder reflection batch size")
    parser.add_argument(
        "--components", action="store_true", help="ladder: also record PXR and CBS parts"
    )
    args = parser.parse_args(argv)
    if args.ladder:
        run_ladder(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
