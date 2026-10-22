"""Bounded primary-only PyRITE adapter for #510; no crystal radiation.

Arms: ``source`` (default, the retained pilot record: coupled transport with
creation-level hard-photon counts), ``transport`` (terminal fractions and
transmitted energy only, no radiation scoring) and ``detector`` (coupled
transport plus the production weighted per-segment estimator integrated over a
matrix detector cone). Scenes are pilot ``manifest.json`` ids or frozen
``matrix.json`` case ids.
"""

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

START = time.perf_counter()
HERE = Path(__file__).parent
ARMS = ("source", "transport", "detector")
# What stays in transport for the transport-only arm; recorded verbatim so the
# Geant4/PenRed arm can configure the same treatment.
RADIATIVE_LOSS = {
    "none": {
        "radiative_model": "uncoupled",
        "transport": "SBETHE collision stopping only; no bremsstrahlung energy loss and no "
        "hard-photon sampling (radiative_model='uncoupled')",
        "comparator_match": "remove electron bremsstrahlung (e.g. Geant4 eBrem) from transport",
    },
    "bremslib-soft-hard": {
        "radiative_model": "bremslib-soft-hard",
        "transport": "SBETHE collision stopping plus BremsLib soft radiative loss below the hard "
        "cutoff and sampled hard-photon energy debits above it; photons are not scored",
        "comparator_match": "keep electron bremsstrahlung with the same photon production "
        "threshold and kill photons at creation without scoring",
    },
}
UNSUPPORTED_BEAM = ("spot_rms_mm", "divergence_rms_mrad", "energy_spread_rms_frac")
UNSUPPORTED_PHYSICS = ("secondary_transport", "photon_transport", "characteristic", "pxr", "cbs")


def transport_scores(r, n_primaries):
    """Terminal indicators and transmitted energy per incident primary."""
    import numpy as np

    from pyrite.perf.precision import grouped_scores

    ids = np.asarray(r["electron_id"], dtype=int)
    ev = np.asarray(r["event_kind"])
    terminal = np.isin(ev, [4, 5, 6, 7])
    if not np.array_equal(np.sort(ids[terminal]), np.arange(n_primaries)):
        raise RuntimeError("incomplete or duplicate primary terminal records")
    scores = {}
    for name, code in [
        ("transmitted", 5),
        ("backscattered", 4),
        ("side_exited", 6),
        ("stopped", 7),
    ]:
        scores[name] = grouped_scores(
            ids[ev == code], np.ones(np.count_nonzero(ev == code)), n_primaries
        )
    transmitted = ev == 5
    scores["transmitted_energy_keV_per_primary"] = grouped_scores(
        ids[transmitted], np.asarray(r["E_end_keV"])[transmitted], n_primaries
    )
    return scores


def score_segments(r, n_primaries):
    """Reduce one independent transport block to incident-primary scores."""
    import numpy as np

    from pyrite.perf.precision import grouped_scores

    scores = transport_scores(r, n_primaries)
    ids = np.asarray(r["electron_id"], dtype=int)
    k = np.asarray(r["hard_radiative_k_eV"])
    for name, low, high in [
        ("photons_ge10kev", 10000, float("inf")),
        ("photons_10_50kev", 10000, 50000),
        ("photons_50_100kev", 50000, 100000),
        ("photons_ge100kev", 100000, float("inf")),
    ]:
        mask = (k >= low) & (k < high)
        scores[name] = grouped_scores(ids[mask], np.ones(np.count_nonzero(mask)), n_primaries)
    return scores


def catalog_composition(key):
    """Exact number densities of a catalog crystal or medium [atoms/Å³]."""
    from pyrite.xsgen.sbethe.catalog import CATALOG

    entry = CATALOG.media[key] if key in CATALOG.media else CATALOG.crystal(key)
    return [(str(element), float(n)) for element, n in entry.composition]


def case_unsupported(case):
    """Matrix-case features this adapter cannot express; empty when runnable."""
    physics = case["physics"]
    out = [f"beam {k}={case['beam'][k]}" for k in UNSUPPORTED_BEAM if case["beam"][k]]
    out += [k for k in UNSUPPORTED_PHYSICS if physics[k]]
    if physics["inelastic_model"] != "continuous":
        out.append(f"inelastic_model={physics['inelastic_model']}")
    return out


def case_arms(case):
    """Adapter arms that can run a matrix case."""
    if case_unsupported(case):
        return []
    return ["source", "transport"] + (["detector"] if case.get("detector") else [])


def load_scene(scene_id):
    """Return ``(raw, spec, path)`` for a pilot manifest id or a matrix case id.

    ``spec`` is the engine-neutral scene the adapter runs; ``raw`` is the
    unchanged source entry and ``path`` its file.
    """
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for scene in manifest["scenes"]:
        if scene["id"] == scene_id:
            return (
                scene,
                {
                    "id": scene_id,
                    "source": "manifest",
                    "catalog_key": scene["catalog_key"],
                    # Frozen elemental number densities shared with the explicit Geant4 materials.
                    "composition": [(scene["element"], scene["number_density_ang3"])],
                    "energy_keV": scene["energy_keV"],
                    "thickness_um": scene["thickness_um"],
                    "angle_deg": scene["angle_deg"],
                    "width_mm": manifest["width_mm"],
                    "electron_cutoff_keV": manifest["electron_cutoff_keV"],
                    "hard_photon_cutoff_eV": manifest["hard_photon_cutoff_eV"],
                    "straggling": manifest["straggling"],
                    "detector": None,
                    "unsupported": [],
                },
                manifest_path,
            )
    matrix_path = HERE / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    case = next((c for c in matrix["cases"] if c["id"] == scene_id), None)
    if case is None:
        raise SystemExit(f"unknown scene {scene_id!r}: not in manifest.json or matrix.json")
    physics = case["physics"]
    spec = {
        "id": scene_id,
        "source": "matrix",
        "catalog_key": case["material"],
        "composition": catalog_composition(case["material"]),
        "energy_keV": case["energy_keV"],
        "thickness_um": case["thickness_um"],
        "angle_deg": case["angle_deg"],
        "width_mm": case["width_mm"],
        "electron_cutoff_keV": physics["electron_cutoff_keV"],
        "hard_photon_cutoff_eV": physics["hard_photon_cutoff_eV"],
        "straggling": physics["straggling"],
        "detector": case["detector"],
        "unsupported": case_unsupported(case),
    }
    return case, spec, matrix_path


def table_provenance(entries, stopping, brem, elements):
    """Exact identities of every table the transport/scoring consumes."""
    bremslib = [
        {"key": brem[el].key, "manifest_sha256": brem[el].digest} for el in elements if el in brem
    ]
    return {
        "elsepa": [
            [
                {
                    "key": t.key,
                    "manifest_sha256": t.digest,
                    "arrays_sha256": t.manifest["arrays_sha256"],
                }
                for t in e.tables
            ]
            for e in entries
        ],
        "sbethe": {
            "key": stopping.key,
            "manifest_sha256": stopping.digest,
            "arrays_sha256": stopping.manifest["arrays_sha256"],
        },
        "bremslib": bremslib[0] if len(bremslib) == 1 else bremslib,
    }


def radiative_arguments(arm, radiative_loss, cutoff_eV, brem):
    """``simulate_trajectories`` radiative keywords for one arm."""
    if arm == "transport" and RADIATIVE_LOSS[radiative_loss]["radiative_model"] == "uncoupled":
        return dict(radiative_model="uncoupled", radiative_cutoff_eV=None, bremslib_tables=None)
    return dict(
        radiative_model="bremslib-soft-hard", radiative_cutoff_eV=cutoff_eV, bremslib_tables=brem
    )


# Geometry helpers the bremsstrahlung estimator imports from the CXR package;
# they clip/orient segments and compute no PXR/CBS amplitude or line.
SHARED_GEOMETRY = {
    "pyrite.montecarlo.spectrum.cxr._kernels." + name
    for name in (
        "_clip_segments_to_cutoff",
        "_observation_direction",
        "_validate_groove_escape_direction",
    )
}


def forbidden_module(module, arm, radiative_loss="none", name=""):
    """True for a module (function ``name``) whose computation an arm must never run.

    Every arm excludes PXR/CBS, apart from the shared geometry helpers the
    bremsstrahlung estimator calls. The transport arm also excludes every
    emission scorer; with retained BremsLib loss, transport itself samples
    hard-photon energies/angles from the staged BremsLib tables, so only that
    table module is allowed.
    """
    crystal = module.startswith("pyrite.montecarlo.spectrum.cxr") or module.startswith(
        "pyrite.montecarlo.spectrum.lines"
    )
    if arm == "detector" and f"{module}.{name}" in SHARED_GEOMETRY:
        return False
    if arm != "transport":
        return crystal
    sampling = radiative_loss != "none" and module == "pyrite.montecarlo.spectrum.brem_bremslib"
    scorer = module.startswith(("pyrite.montecarlo.spectrum", "pyrite.perf.detector"))
    return crystal or (scorer and not sampling)


def check_transport_only(result, radiative_loss):
    """Refuse a transport-only block that carries photon or emission work."""
    import numpy as np

    k = result.get("hard_radiative_k_eV")
    if radiative_loss == "none" and (
        result.get("radiative") or (k is not None and np.any(np.asarray(k) != 0.0))
    ):
        raise RuntimeError("transport-only arm produced radiative rows")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--core", choices=["per-electron", "cuda"], required=True)
    parser.add_argument("--arm", choices=ARMS, default="source")
    parser.add_argument(
        "--radiative-loss", choices=sorted(RADIATIVE_LOSS), default="bremslib-soft-hard"
    )
    parser.add_argument("--max-de-frac", type=float, default=0.05)
    parser.add_argument("--cutoff-ev", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-cxr", action="store_true")
    parser.add_argument("--batch-histories", type=int, default=0)
    parser.add_argument("--scratch-mib", type=int, default=512)
    parser.add_argument("--probe-electrons", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--cone-rtol", type=float, default=1e-3)
    parser.add_argument("--cone-max-order", type=int, default=32)
    parser.add_argument(
        "--straggling",
        choices=["scene", "on", "off"],
        default="scene",
        help="override the scene's straggling switch (recorded); e.g. to pair with PenRed",
    )
    args = parser.parse_args()
    if args.histories < 2 or args.batch_histories < 0 or args.scratch_mib < 1 or args.max_steps < 1:
        parser.error("invalid history count or scratch budget")
    if not args.cone_rtol > 0 or args.cone_max_order < 4:
        parser.error("invalid cone quadrature controls")
    return args


def write_unsupported(args, spec, raw_path, reason):
    """Retain an explicit outcome for a scene the adapter cannot express."""
    from pyrite.perf.comparison import comparison_record

    record = comparison_record(
        case_id=spec["id"],
        engine="pyrite",
        arm=args.core,
        kind=args.arm,
        status="unsupported-physics",
        n_primaries=args.histories,
        energy_keV=spec["energy_keV"],
        seed=args.seed,
        reason=reason,
        provenance={"scene_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": record["status"]}))


def main():
    args = parse_args()
    raw, spec, raw_path = load_scene(args.scene)
    if args.straggling != "scene":
        spec["straggling_scene"] = spec["straggling"]
        spec["straggling"] = args.straggling == "on"
    if args.arm == "detector" and not spec["detector"]:
        raise SystemExit(f"{args.scene} has no detector cone; the detector arm needs a matrix case")
    if spec["unsupported"]:
        write_unsupported(
            args, spec, raw_path, "adapter cannot express: " + ", ".join(spec["unsupported"])
        )
        return
    # An override never writes the unchanged pilot record, whose scene omits it.
    legacy = spec["source"] == "manifest" and args.arm == "source" and args.straggling == "scene"
    cutoff_eV = spec["hard_photon_cutoff_eV"] if args.cutoff_ev is None else args.cutoff_ev
    setup = time.perf_counter()
    import numpy as np

    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
    from pyrite.perf.precision import primary_moments
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    comp = spec["composition"]
    elements = [element for element, _ in comp]
    entries = resolve_layer_tables(comp)
    stopping = resolve_composition_table(spec["catalog_key"], comp)
    uncoupled = args.arm == "transport" and args.radiative_loss == "none"
    brem = {} if uncoupled else load_bremsstrahlung_tables(elements)
    tables = table_provenance(entries, stopping, brem, elements)
    angle = math.radians(spec["angle_deg"])
    if args.core == "cuda":
        import cupy as cp

        cp.cuda.Device().synchronize()
    cxr_calls = []

    def audit(frame, event, arg):
        module = frame.f_globals.get("__name__", "") or ""
        name = frame.f_code.co_name
        import_scaffolding = (
            name.startswith("<") or name[:1].isupper() or name in {"__annotate__", "__create_fn__"}
        )
        if (
            event == "call"
            and not import_scaffolding
            and forbidden_module(module, args.arm, args.radiative_loss, name)
        ):
            cxr_calls.append(module + "." + frame.f_code.co_name)

    if args.audit_cxr:
        sys.setprofile(audit)
    transport_args = dict(
        E0_keV=spec["energy_keV"],
        Ne=args.histories,
        composition=comp,
        thickness_ang=spec["thickness_um"] * 1e4,
        crystal_width_mm=spec["width_mm"],
        crystal_height_mm=spec["width_mm"],
        beam_dir=np.array([math.sin(angle), 0, math.cos(angle)]),
        E_cut_keV=spec["electron_cutoff_keV"],
        seed=args.seed,
        energy_model="midpoint",
        max_dE_frac=args.max_de_frac,
        max_steps=args.max_steps,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in entries]],
        stopping_tables=[stopping.arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=spec["straggling"],
        **radiative_arguments(args.arm, args.radiative_loss, cutoff_eV, brem),
        transport_core=args.core,
        transport_lut_config=TransportLUTConfig(enabled=False),
        per_electron_config=PerElectronTransportConfig(
            scratch_budget_bytes=args.scratch_mib * 1024**2,
            probe_electrons=args.probe_electrons,
        ),
    )
    batch_histories = args.batch_histories or args.histories
    acceptance = []

    def score_block(result, count):
        if args.arm == "source":
            return score_segments(result, count)
        if args.arm == "transport":
            check_transport_only(result, args.radiative_loss)
            return transport_scores(result, count)
        from pyrite.perf.detector_scoring import cone_detector_scores

        cone = cone_detector_scores(
            result,
            composition=comp,
            energy_keV=spec["energy_keV"],
            center_deg=spec["detector"]["center_deg"],
            half_angle_deg=spec["detector"]["half_angle_deg"],
            bremslib_tables=brem,
            rtol=args.cone_rtol,
            max_order=args.cone_max_order,
        )
        acceptance.append(cone["acceptance"])
        return transport_scores(result, count) | cone["scores"]

    def run_request(run_seed):
        combined = {}
        compute_s = reduction_s = 0.0
        dtypes = None
        acceptance.clear()
        request_start = time.perf_counter()
        for offset in range(0, args.histories, batch_histories):
            count = min(batch_histories, args.histories - offset)
            # Counter-addressed global primary indices preserve streams across blocks.
            kwargs = dict(transport_args, Ne=count, seed=run_seed, _electron_start=offset)
            if args.core == "cuda":
                cp.cuda.Device().synchronize()
            compute_start = time.perf_counter()
            result = simulate_trajectories(**kwargs)
            if args.core == "cuda":
                cp.cuda.Device().synchronize()
            compute_end = time.perf_counter()
            compute_s += compute_end - compute_start
            block_scores = score_block(result, count)
            dtypes = {
                name: str(np.asarray(result[name]).dtype)
                for name in ["E_end_keV", "v_hat", "L_ang"]
            }
            del result
            for key, values in block_scores.items():
                if key not in combined:
                    combined[key] = np.empty(args.histories, dtype=np.float64)
                combined[key][offset : offset + count] = values
            reduction_s += time.perf_counter() - compute_end
        return combined, compute_s, reduction_s, time.perf_counter() - request_start, dtypes

    # Warm the complete request with identical batching and numerical controls.
    warmup_start = time.perf_counter()
    try:
        warm, _, _, warmup_s, _ = run_request(args.seed + 1_000_000)
        del warm
        scores, compute_s, reduction_s, request_s, dtypes = run_request(args.seed)
    finally:
        if args.audit_cxr:
            sys.setprofile(None)
    if cxr_calls:
        raise RuntimeError(f"unexpected PXR/CBS or emission computation calls: {cxr_calls}")
    end = time.perf_counter()
    common = dict(
        seed=args.seed,
        histories=args.histories,
        core=args.core,
        cache_state="warmed-in-process",
        warmup={
            "repeats": 1,
            "histories": args.histories,
            "seed": args.seed + 1_000_000,
            "wall_s": warmup_s,
            "same_scene_tables_numerics": True,
            "device_synchronized": args.core == "cuda",
        },
        batch_histories=batch_histories,
        scratch_mib=args.scratch_mib,
        probe_electrons=args.probe_electrons,
        segment_dtypes=dtypes,
        tables=tables,
        max_de_frac=args.max_de_frac,
        max_steps=args.max_steps,
        cutoff_eV=cutoff_eV,
        timings_s={
            "imports_and_tables": warmup_start - setup,
            "warmup": warmup_s,
            "transport_and_emission": compute_s,
            "request_with_reductions": request_s,
            "reductions": reduction_s,
            "adapter_total": end - START,
        },
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        environment={
            k: os.environ.get(k)
            for k in [
                "PYRITE_MC_BACKEND",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            ]
        },
        cxr_execution_audit={"enabled": args.audit_cxr, "calls": cxr_calls},
        numerical_gate="unmeasured",
        equivalence_gate="unmeasured",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_scores = args.output.with_suffix(".npz")
    np.savez_compressed(raw_scores, **scores)
    raw_sha256 = hashlib.sha256(raw_scores.read_bytes()).hexdigest()
    if legacy:
        record = dict(
            schema="pyrite.equal-precision.sample.v1",
            engine="pyrite",
            scene=raw,
            manifest_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            **common,
            observables={k: primary_moments(v) for k, v in scores.items()},
        )
        record["raw_scores_sha256"] = raw_sha256
    else:
        from pyrite.perf.comparison import comparison_record

        common["timings_s"]["scoring"] = common["timings_s"].pop("reductions")
        common["timings_s"]["transport"] = common["timings_s"].pop("transport_and_emission")
        provenance = dict(
            scene=spec,
            scene_file=raw_path.name,
            scene_file_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            **common,
            raw_scores_sha256=raw_sha256,
        )
        if args.arm == "transport":
            provenance["radiative_energy_loss"] = {"mode": args.radiative_loss} | RADIATIVE_LOSS[
                args.radiative_loss
            ]
            provenance["emission_scoring"] = "none: no photon, cone or PXR/CBS scorer is called"
        else:
            provenance["radiative_energy_loss"] = {"mode": "bremslib-soft-hard"} | RADIATIVE_LOSS[
                "bremslib-soft-hard"
            ]
        if args.arm == "detector":
            provenance["acceptance"] = acceptance
            provenance["acceptance_gate"] = (
                "converged" if all(a["converged"] for a in acceptance) else "unconverged"
            )
        record = comparison_record(
            case_id=spec["id"],
            engine="pyrite",
            arm=args.core,
            kind=args.arm,
            status="ok",
            n_primaries=args.histories,
            energy_keV=spec["energy_keV"],
            seed=args.seed,
            scores=scores,
            provenance=provenance,
        )
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "timings_s": common["timings_s"]}))


if __name__ == "__main__":
    main()
