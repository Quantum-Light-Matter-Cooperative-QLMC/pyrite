"""Bounded primary-only coupled-transport/source pilot; no crystal radiation."""

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


def score_segments(r, n_primaries):
    """Reduce one independent transport block to incident-primary scores."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--core", choices=["per-electron", "cuda"], required=True)
    parser.add_argument("--max-de-frac", type=float, default=0.05)
    parser.add_argument("--cutoff-ev", type=float, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-cxr", action="store_true")
    parser.add_argument("--batch-histories", type=int, default=0)
    parser.add_argument("--scratch-mib", type=int, default=512)
    parser.add_argument("--probe-electrons", type=int, default=1024)
    args = parser.parse_args()
    if args.histories < 2 or args.batch_histories < 0 or args.scratch_mib < 1:
        parser.error("invalid history count or scratch budget")
    manifest_path = Path(__file__).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    scene = next(s for s in manifest["scenes"] if s["id"] == args.scene)
    setup = time.perf_counter()
    import numpy as np

    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
    from pyrite.perf.precision import primary_moments
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    # Frozen elemental number densities shared with the explicit Geant4 materials.
    comp = [(scene["element"], scene["number_density_ang3"])]
    entries = resolve_layer_tables(comp)
    stopping = resolve_composition_table(scene["catalog_key"], comp)
    brem = load_bremsstrahlung_tables([scene["element"]])
    tables = {
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
        "bremslib": {
            "key": brem[scene["element"]].key,
            "manifest_sha256": brem[scene["element"]].digest,
        },
    }
    angle = math.radians(scene["angle_deg"])
    if args.core == "cuda":
        import cupy as cp

        cp.cuda.Device().synchronize()
    cxr_calls = []

    def audit(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        import_scaffolding = (
            name.startswith("<") or name[:1].isupper() or name in {"__annotate__", "__create_fn__"}
        )
        if (
            event == "call"
            and not import_scaffolding
            and (
                module.startswith("pyrite.montecarlo.spectrum.cxr")
                or module.startswith("pyrite.montecarlo.spectrum.lines")
            )
        ):
            cxr_calls.append(module + "." + frame.f_code.co_name)

    if args.audit_cxr:
        sys.setprofile(audit)
    transport_args = dict(
        E0_keV=scene["energy_keV"],
        Ne=args.histories,
        composition=comp,
        thickness_ang=scene["thickness_um"] * 1e4,
        crystal_width_mm=manifest["width_mm"],
        crystal_height_mm=manifest["width_mm"],
        beam_dir=np.array([math.sin(angle), 0, math.cos(angle)]),
        E_cut_keV=manifest["electron_cutoff_keV"],
        seed=args.seed,
        energy_model="midpoint",
        max_dE_frac=args.max_de_frac,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in entries]],
        stopping_tables=[stopping.arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=args.cutoff_ev,
        bremslib_tables=brem,
        transport_core=args.core,
        transport_lut_config=TransportLUTConfig(enabled=False),
        per_electron_config=PerElectronTransportConfig(
            scratch_budget_bytes=args.scratch_mib * 1024**2,
            probe_electrons=args.probe_electrons,
        ),
    )
    batch_histories = args.batch_histories or args.histories

    def run_request(run_seed):
        combined = {}
        compute_s = reduction_s = 0.0
        dtypes = None
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
            block_scores = score_segments(result, count)
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
    warm, _, _, warmup_s, _ = run_request(args.seed + 1_000_000)
    del warm
    scores, compute_s, reduction_s, request_s, dtypes = run_request(args.seed)
    if args.audit_cxr:
        sys.setprofile(None)
    if cxr_calls:
        raise RuntimeError(f"unexpected PXR/CBS computation calls: {cxr_calls}")
    end = time.perf_counter()
    record = dict(
        schema="pyrite.equal-precision.sample.v1",
        engine="pyrite",
        scene=scene,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
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
        cutoff_eV=args.cutoff_ev,
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
        observables={k: primary_moments(v) for k, v in scores.items()},
        numerical_gate="unmeasured",
        equivalence_gate="unmeasured",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_scores = args.output.with_suffix(".npz")
    np.savez_compressed(raw_scores, **scores)
    record["raw_scores_sha256"] = hashlib.sha256(raw_scores.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "timings_s": record["timings_s"]}))


if __name__ == "__main__":
    main()
