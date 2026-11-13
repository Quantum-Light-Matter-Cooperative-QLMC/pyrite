"""Remote-only bounded transport/spectrum sensitivity pilot (#1142, #1143).

No default or physics verdict. Fixed 0.5--10 keV reporting band; reference
refinement and changed-transport statistical qualification remain separate.
Our own pickle is retained for trusted, same-source replay only.
"""

import argparse
import hashlib
import json
import os
import pickle
import resource
import runpy
import time
from pathlib import Path

import numpy as np


def axis(start, stop, spacing, quadrature):
    """Node support or midpoint-bin support exactly spans the same interval."""
    n = int(np.ceil((stop - start) / spacing))
    if quadrature == "node":
        return np.linspace(start, stop, n + 1)
    return start + (np.arange(n) + 0.5) * ((stop - start) / n)


def metrics(grid, density, quadrature, edges=None):
    from pyrite.energy_grid.bandwidth_check import _bin_masses, _dominant_fwhm_eV
    from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

    if quadrature == "node":
        total = float(np.trapezoid(density, grid))
        first = float(np.trapezoid(density * grid, grid))
    else:
        cells, widths = _energy_bin_edges_and_widths(grid)
        mass = density * widths
        total = float(mass.sum())
        first = float(np.sum(mass * (cells[:-1] + cells[1:]) / 2))
    if total <= 0 or not np.isfinite(density).all():
        raise ValueError("nonpositive band yield or nonfinite spectrum")
    fwhm = float(_dominant_fwhm_eV(grid, density))
    return dict(
        yield_per_primary=total,
        centroid_eV=first / total,
        peak_density_per_eV=float(density.max()),
        dominant_fwhm_eV=fwhm if np.isfinite(fwhm) else None,
        band_masses_100eV=_bin_masses(
            grid,
            density,
            np.linspace(500, 10000, 96) if edges is None else edges,
            quadrature=quadrature,
        ).tolist(),
    )


CRYSTAL = {"mos2-on-si": "mos2"}


def scene_sweep(material, energy_keV, thickness_ang):
    """The frozen scene's sweep: single slab at 45/180 deg, or the layered stack."""
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.geometry import Stack

    if material == "mos2-on-si":
        target = Stack.on_substrate(
            "mos2", 2000.0, "silicon", 2e5, tilt_deg=45, tilt_azim_deg=180, footprint=None
        )
        return material_sweep("mos2", target=target, energy_keV=30.0)
    return material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=45,
        tilt_azim_deg=180,
    )


def layered_case(histories, seed):
    """Supported two-crystal stack: 2000 A MoS2 on a 20 um Si substrate at 30 keV.

    Both layers have shell records, so shell soft/hard transport admits it; the
    substrate stops the beam (near-range/backscatter scene) and also radiates.
    """
    from pyrite.campaign.sweep import build_cases

    sweep = scene_sweep("mos2-on-si", 30.0, None)
    case = dict(build_cases(sweep, n_electrons=histories, n_electrons_brem=histories)[0])
    case["seed"] = int(seed)
    return case


def mosaic_mc_case(case, nodes):
    """Exact MC mosaic average at the crystal's catalog FWHM; analytic term off.

    The analytic route only broadens at detector convolution, so it leaves the
    intrinsic source spectrum unchanged; only this route is a source-level ladder.
    """
    from pyrite.materials import CATALOG

    fwhm_deg = CATALOG.crystal(case["crystal"]).mosaic_fwhm_deg
    if not fwhm_deg:
        raise ValueError(f"{case['crystal']!r} has no catalog mosaic FWHM")
    return dict(
        case,
        mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=float(np.deg2rad(fwhm_deg)),
        mosaic_mc_nodes=int(nodes),
    )


#: Production automatic line-grid policies: arm name -> profile selector table.
#: ``sinc-*`` keep the kinematic-ceiling bandwidth; ``measured*`` use the
#: ``resonance-population`` edge; ``measured-local`` adds ``resonance-local``.
POLICY_ARMS = {
    "sinc": {"resolution": "sinc-nyquist"},
    "sinc-bin": {"resolution": "sinc-nyquist", "quadrature": "bin-mean"},
    "measured": {"bandwidth": "resonance-population"},
    "measured-bin": {"bandwidth": "resonance-population", "quadrature": "bin-mean"},
    "measured-local": {
        "bandwidth": "resonance-population",
        "resolution": "resonance-local",
        "quadrature": "bin-mean",
    },
}
#: Predeclared same-track numerical margins (#1143); ``mass_l1`` is the L1
#: distance of the two normalized 100 eV source-band mass vectors.
MARGINS = dict(
    yield_per_primary=1e-3,
    centroid_eV=1e-3,
    peak_density_per_eV=1e-2,
    dominant_fwhm_eV=1e-2,
    mass_l1=1e-2,
)
BAND = (500.0, 10000.0)


def band_edges(start, stop):
    """100 eV source-band edges inside ``[start, stop]``, aligned to 100 eV."""
    return np.arange(np.ceil(start / 100) * 100, np.floor(stop / 100) * 100 + 1, 100.0)


def versus(candidate, reference):
    """Separate signed relative changes against a same-track reference.

    Each observable is judged on its own margin, so conserved mass cannot hide
    a shape failure. An undefined FWHM on either side stays ``None``.
    """
    out = {}
    for key in ("yield_per_primary", "centroid_eV", "peak_density_per_eV", "dominant_fwhm_eV"):
        c, r = candidate[key], reference[key]
        out[key] = None if c is None or r is None else (c - r) / r
    cm = np.asarray(candidate["band_masses_100eV"], dtype=float)
    rm = np.asarray(reference["band_masses_100eV"], dtype=float)
    out["mass_l1"] = float(np.abs(cm / cm.sum() - rm / rm.sum()).sum())
    out["within_margin"] = {
        key: None if out[key] is None else bool(abs(out[key]) <= margin)
        for key, margin in MARGINS.items()
    }
    return out


def _jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)


def policy_case(case, sweep, selectors):
    """``case`` carrying the production policy payload ``build_cases`` attaches.

    Only the policy and its quadrature come from the rebuilt case; transport
    inputs stay those of the (possibly replayed) frozen case.
    """
    from dataclasses import replace

    from pyrite.campaign.sweep import build_cases

    built = build_cases(
        replace(sweep, cxr_grid_policy=dict(selectors)),
        n_electrons=int(case["Ne"]),
        n_electrons_brem=int(case["Ne"]),
    )[0]
    return dict(
        case,
        line_grid_policy=built["line_grid_policy"],
        line_quadrature=built.get("line_quadrature", "node"),
        _profile_line_grid_stages=True,
    )


def matched_grid(grid, record, quadrature, backbone_eV):
    """The policy's spacing on the fixed band.

    Uniform policies: their resolved spacing on the band. ``resonance-local``:
    its own nodes inside the band, continued past a measured edge below the
    band top at the policy's backbone spacing (what it would place there).
    """
    lo, hi = BAND
    if record.get("resolution_policy") == "resonance-local" or "actual_spacing_eV" not in record:
        inner = grid[(grid > lo) & (grid < hi)]
        if grid[-1] < hi:
            tail = axis(float(grid[-1]), hi, backbone_eV, "node")[1:-1]
            inner = np.r_[inner, tail[tail > inner[-1]]]
        return np.r_[lo, inner, hi]
    return axis(lo, hi, float(record["actual_spacing_eV"]), quadrature)


def score_policies(case, sweep, ladder, names, reference_spacing, sync):
    """Resolve each policy from the saved trajectories and score it twice.

    ``native`` is the production axis (own support) against a node reference
    at ``reference_spacing`` on that same support. ``matched`` puts the
    policy's spacing on the fixed band, against the band reference, isolating
    support from spacing/quadrature.
    """
    from pyrite.energy_grid.bandwidth_check import _piecewise
    from pyrite.energy_grid.convergence_case import CaseLadder
    from pyrite.montecarlo.runner.cxr_grid import resolve_cxr_grid

    tp = ladder.transport
    node = CaseLadder(dict(case, line_quadrature="node"), transport=tp)
    references = {}

    def reference(start, stop):
        if (start, stop) not in references:
            grid = axis(start, stop, reference_spacing, "node")
            density = _piecewise(node.lines, grid)
            edges = band_edges(start, stop)
            references[start, stop] = metrics(grid, density, "node", edges=edges)
        return references[start, stop]

    rows = []
    for name in names:
        pcase = policy_case(case, sweep, POLICY_ARMS[name])
        quadrature = pcase["line_quadrature"]
        row = dict(policy=name, selectors=POLICY_ARMS[name], quadrature=quadrature)
        try:
            sync()
            begin = time.perf_counter()
            grid, record = resolve_cxr_grid(
                pcase,
                tp["segs"],
                tp["n_hat"],
                int(tp["Ne_lines"]),
                np.asarray(tp["E_grid"], dtype=float),
                case.get("abs_layers"),
                tp.get("groove"),
            )
            row["resolution_s"] = time.perf_counter() - begin
        except ValueError as exc:  # CxrGridToleranceError and admission refusals
            rows.append(dict(row, refused=f"{type(exc).__name__}: {exc}"))
            continue
        grid = np.asarray(grid, dtype=float)
        row["record"] = json.loads(json.dumps(record, default=_jsonable))
        scorer = CaseLadder(pcase, transport=tp)
        backbone = float(pcase["line_grid_policy"]["resolution"]["max_spacing_eV"])
        start, stop = float(grid[0]), float(grid[-1])
        for label, g, ref, edges in (
            ("native", grid, reference(start, stop), band_edges(start, stop)),
            (
                "matched",
                matched_grid(grid, record, quadrature, backbone),
                reference(*BAND),
                None,
            ),
        ):
            sync()
            begin = time.perf_counter()
            density = scorer.lines(g)
            sync()
            m = metrics(g, density, quadrature, edges=edges)
            row[label] = dict(
                nodes=int(g.size),
                support_eV=[float(g[0]), float(g[-1])],
                min_spacing_eV=float(np.diff(g).min()),
                max_spacing_eV=float(np.diff(g).max()),
                first_s=time.perf_counter() - begin,
                metrics=m,
                versus_reference=versus(m, ref),
            )
        rows.append(row)
    return dict(
        reference_spacing_eV=reference_spacing,
        references={f"{a:g}-{b:g}": m for (a, b), m in references.items()},
        margins=MARGINS,
        policies=rows,
    )


def runtime():
    """Host replay must not initialize a GPU or change saved transport."""
    from pyrite._backend import BACKEND

    if BACKEND.name == "cpu":
        return lambda: None, "cpu", lambda: 0, None
    import cupy

    return (
        cupy.cuda.Device().synchronize,
        str(cupy.cuda.runtime.getDeviceProperties(0)["name"]),
        lambda: cupy.get_default_memory_pool().total_bytes(),
        device_peak_probe(cupy),
    )


def device_peak_probe(cupy):
    """One untimed call: peak CuPy-pool bytes in use, and pool reservation.

    The pool is emptied first, so its reservation after the call is the
    call's reservation high-water. A hook tracks rounded in-use bytes; it
    slows malloc, so it never wraps a timed call.
    """
    pool = cupy.get_default_memory_pool()

    class Peak(cupy.cuda.MemoryHook):
        name = "Peak"

        def __init__(self):
            self.live = self.peak = 0

        def malloc_postprocess(self, **kw):
            if kw["mem_ptr"]:
                self.live += kw["mem_size"]
                self.peak = max(self.peak, self.live)

        def free_postprocess(self, **kw):
            self.live -= kw["mem_size"]

    def probe(call, *args):
        cupy.cuda.Device().synchronize()
        pool.free_all_blocks()
        base = pool.used_bytes()
        hook = Peak()
        with hook:
            call(*args)
            cupy.cuda.Device().synchronize()
        return dict(
            device_peak_in_use_bytes=int(base + hook.peak),
            device_pool_reserved_after_emptied_call_bytes=int(pool.total_bytes()),
            device_in_use_before_call_bytes=int(base),
        )

    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", choices=["hopg", "silicon", "mos2-on-si"])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--arm",
        choices=[
            "baseline",
            "half-cap",
            "quarter-cap",
            "cut25",
            "cut35",
            "cut100",
            "mixed-precision",
            "mixed-elastic",
        ],
        default="baseline",
    )
    parser.add_argument("--seed", type=int, default=642501)
    parser.add_argument("--histories", type=int, default=64)
    parser.add_argument("--replay", type=Path, help="trusted same-source transport pickle")
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument(
        "--node-spacings",
        type=lambda text: [float(x) for x in text.split(",") if x],
        default=[0.75, 0.375, 0.1875],
        help="comma-separated node spacings in eV (reference-refinement ladder); empty for none",
    )
    parser.add_argument(
        "--bin-mean-spacings",
        type=lambda text: [float(x) for x in text.split(",") if x],
        default=None,
        help="comma-separated bin-mean spacings in eV; baseline default 0.375,3,12; empty for none",
    )
    parser.add_argument("--energy-keV", type=float, default=None, help="override beam energy")
    parser.add_argument(
        "--thickness-ang", type=float, default=None, help="override single-slab thickness"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="scoring calls per grid (first + warm); 1 for slow host-grouped replays",
    )
    parser.add_argument(
        "--mosaic-mc-nodes",
        type=int,
        default=None,
        help="score with the exact MC mosaic route at the catalog FWHM and N nodes per axis",
    )
    parser.add_argument(
        "--policy-arms",
        type=lambda text: [x for x in text.split(",") if x],
        default=[],
        help=f"comma-separated automatic grid policies: {','.join(POLICY_ARMS)}",
    )
    parser.add_argument(
        "--policy-reference-spacing",
        type=float,
        default=0.09375,
        help="node reference spacing (eV) for policy arms; 0.09375 passed refinement",
    )
    args = parser.parse_args()
    unknown = sorted(set(args.policy_arms) - set(POLICY_ARMS))
    if unknown:
        parser.error(f"unknown policy arms {unknown}")
    os.environ["PYRITE_MC_TRANSPORT_PRECISION"] = (
        "mixed" if args.arm == "mixed-precision" else "fp64"
    )
    from pyrite._backend import BACKEND
    from pyrite._energy_grid_encoding import encode_energy_grid
    from pyrite.energy_grid.convergence import segment_fingerprint
    from pyrite.energy_grid.convergence_case import CaseLadder, build_ladder_case

    if args.material == "mos2-on-si":
        if args.energy_keV is not None or args.thickness_ang is not None:
            parser.error("the layered scene is frozen; no energy/thickness overrides")
        case = layered_case(args.histories, args.seed)
    else:
        energy, thickness = (30.0, 20000.0) if args.material == "hopg" else (300.0, 100000.0)
        energy = energy if args.energy_keV is None else args.energy_keV
        thickness = thickness if args.thickness_ang is None else args.thickness_ang
        case = build_ladder_case(
            args.material,
            energy,
            45,
            180,
            thickness_ang=thickness,
            n_electrons=args.histories,
            seed=args.seed,
        )
    case.pop("line_grid_policy", None)
    carrier = encode_energy_grid(axis(500, 10000, 3, "node"))
    case.update(
        E_grid=carrier,
        E_grid_line=carrier,
        line_quadrature="node",
        emission="incoherent",
        transport_mode="single-scattering",
        transport_precision="fp64",
        max_dE_frac=0.0,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        straggling=False,
        elastic_model="elsepa",
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        atomic_electron_deflection="kawrakow",
    )
    if args.arm in ("half-cap", "quarter-cap"):
        case["max_dE_frac"] = 0.00025 if args.arm == "half-cap" else 0.000125
    elif args.arm in ("cut25", "cut35", "cut100"):
        case["inelastic_cutoff_eV"] = {"cut25": 25.0, "cut35": 35.0, "cut100": 100.0}[args.arm]
    elif args.arm == "mixed-precision":
        case["transport_precision"] = "mixed"
    elif args.arm == "mixed-elastic":
        # Use the model's own validated tuning defaults; admission may refuse.
        case.update(transport_mode="mixed", mixed_c1=0.05, mixed_c2=0.05)
    sync, device, pool_bytes, peak_probe = runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    sync()
    start = time.perf_counter()
    if args.replay:
        saved = pickle.loads(args.replay.read_bytes())  # noqa: S301 - explicit trusted study artifact
        case = saved["case"]
        if (
            case["crystal"] != CRYSTAL.get(args.material, args.material)
            or case["seed"] != args.seed
            or case["Ne"] != args.histories
        ):
            raise ValueError("replay material/seed/population mismatch")
        ladder = CaseLadder(case, transport=saved["transport"])
    else:
        ladder = CaseLadder(case, transport_core="cuda")
    sync()
    transport_s = time.perf_counter() - start
    payload = args.output / "transport.pkl"
    payload.write_bytes(pickle.dumps(dict(case=case, transport=ladder.transport), protocol=5))
    score = runpy.run_path(str(Path(__file__).with_name("straggling_defaults") / "study.py"))[
        "primary_scores"
    ]
    primary, residual = score(ladder.segments, args.histories, float(case["E0_keV"]))
    np.savez_compressed(args.output / "primary.npz", **primary)
    records = []
    if args.mosaic_mc_nodes is not None:
        case = mosaic_mc_case(case, args.mosaic_mc_nodes)
    arms = [("node", h) for h in args.node_spacings]
    bin_mean = args.bin_mean_spacings
    if bin_mean is None:
        bin_mean = [0.375, 3.0, 12.0] if args.arm == "baseline" else []
    arms += [("bin-mean", h) for h in bin_mean]
    report = dict(
        schema="pyrite.defaults-spectral-pilot.v1",
        material=args.material,
        E0_keV=float(case["E0_keV"]),
        thickness_ang=float(case["thickness_ang"]),
        mosaic_mc_nodes=args.mosaic_mc_nodes,
        mosaic_mc_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        arm=args.arm,
        seed=args.seed,
        histories=args.histories,
        backend=BACKEND.name,
        device=device,
        replay_transport_sha256=None
        if args.replay is None
        else hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        band_eV=[500, 10000],
        source_revision=Path("SOURCE_SHA").read_text().strip(),
        transport_s_including_first_setup=None if args.replay else transport_s,
        replay_adoption_s=transport_s if args.replay else None,
        fingerprint=ladder.fingerprint,
        transport_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        qualification="sensitivity only; no accuracy or equal-precision verdict",
        diagnostics=dict(
            rows=len(ladder.segments["L_ang"]),
            energy_residual_keV=residual,
            primary_means={k: float(v.mean()) for k, v in primary.items()},
        ),
        spectra=records,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.transport_only:
        return
    for quadrature, spacing in arms:
        scoring_case = dict(case, line_quadrature=quadrature)
        scorer = CaseLadder(scoring_case, transport=ladder.transport)
        grid = axis(500, 10000, spacing, quadrature)
        times = []
        for _ in range(max(1, args.repeats)):
            sync()
            begin = time.perf_counter()
            density = scorer.lines(grid)
            sync()
            times.append(time.perf_counter() - begin)
        label = f"{quadrature}-{spacing}"
        memory = None if peak_probe is None else peak_probe(scorer.lines, grid)
        np.savez_compressed(args.output / f"{label}.npz", grid_eV=grid, density=density)
        record = dict(
            quadrature=quadrature,
            spacing_eV=spacing,
            nodes=len(grid),
            first_s=times[0],
            warm_s=times[1:],
            metrics=metrics(grid, density, quadrature),
            device_memory=memory,
        )
        records.append(record)
        assert segment_fingerprint(ladder.segments) == ladder.fingerprint
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["device_pool_reserved_bytes_not_peak"] = pool_bytes()
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(dict(arm=args.arm, spectrum=label, metrics=record["metrics"])), flush=True)
    if args.policy_arms:
        sweep = scene_sweep(args.material, float(case["E0_keV"]), float(case["thickness_ang"]))
        report["policy_arms"] = score_policies(
            case, sweep, ladder, args.policy_arms, args.policy_reference_spacing, sync
        )
        assert segment_fingerprint(ladder.segments) == ladder.fingerprint
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
