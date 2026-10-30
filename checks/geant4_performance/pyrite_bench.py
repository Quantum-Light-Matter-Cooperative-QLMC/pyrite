"""PyRITE CUDA arm of the matched-bounds Geant4 benchmark (#821).

Physics matches ``scenes.json`` bounds and the Geant4 app's electron-only
constructor: ELSEPA single scattering, Kawrakow atomic-electron deflection
(Geant4: DPWA scattering-power correction), SBETHE continuous collision
stopping without straggling or delta rays, coupled BremsLib soft/hard
radiation at 1 keV, 10 keV electron cutoff, first-exit termination. No
PXR/CBS, characteristic or secondary transport is requested.

The transport schedule is the production default (``auto``: bounded
continuation, which never replays a history). The full-history fallback is
also made replay-free: initial segment capacity exceeds the longest observed
track and the probe batch is disabled, so no capacity overflow recomputes a
history. Launched versus retained rows are recorded to prove it.

One process measures one (scene, seed): imports and tables, a complete
same-workload warmup at ``seed + 1000000`` (the first, cold request) and the
measured request. Each request runs fixed history blocks and reduces each block
to per-primary scores inside the timer, as the Geant4 app scores in its event
loop. CUDA is synchronized at both ends of every timed request.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import resource
import time
from pathlib import Path

START = time.perf_counter()
HERE = Path(__file__).parent
BANDS = (
    ("photons_1_10keV", 1e3, 1e4),
    ("photons_10_50keV", 1e4, 5e4),
    ("photons_50_100keV", 5e4, 1e5),
    ("photons_ge100keV", 1e5, math.inf),
)


def block_scores(r, n):
    """Per-primary observables and work counts for one block."""
    import numpy as np

    from pyrite.montecarlo.transport.events import SegmentEvent as EventKind
    from pyrite.perf.precision import grouped_scores

    ids = np.asarray(r["electron_id"], dtype=np.int64)
    ev = np.asarray(r["event_kind"])
    ones = np.ones(ids.size)
    out = {}
    for name, code in [
        ("backscattered", EventKind.EXIT_TOP),
        ("transmitted", EventKind.EXIT_BOTTOM),
        ("side_exited", EventKind.EXIT_SIDE),
        ("stopped", EventKind.CUTOFF),
    ]:
        m = ev == code
        out[name] = grouped_scores(ids[m], ones[m], n)
    terminal = out["backscattered"] + out["transmitted"] + out["side_exited"] + out["stopped"]
    if not np.all(terminal == 1.0):
        raise RuntimeError("every primary needs exactly one terminal row")
    m = ev == EventKind.EXIT_BOTTOM
    out["transmitted_energy_keV"] = grouped_scores(ids[m], np.asarray(r["E_end_keV"])[m], n)
    k = np.asarray(r["hard_radiative_k_eV"], dtype=np.float64)
    for name, lo, hi in BANDS:
        m = (k >= lo) & (k < hi)
        out[name] = grouped_scores(ids[m], ones[m], n)
    m = k >= 1e4
    out["photon_keV_ge10keV"] = grouped_scores(ids[m], k[m] / 1e3, n)
    out["path_um"] = grouped_scores(ids, np.asarray(r["L_ang"], dtype=np.float64) / 1e4, n)
    out["rows"] = grouped_scores(ids, ones, n)
    for name, code in [
        ("rows_elastic", EventKind.ELASTIC),
        ("rows_substep", EventKind.SUBSTEP),
        ("rows_hard_radiative", EventKind.HARD_RADIATIVE),
    ]:
        m = ev == code
        out[name] = grouped_scores(ids[m], ones[m], n)
    return out


class Messages(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--block", type=int, default=25000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-de-frac", type=float, default=0.05)
    parser.add_argument("--scratch-mib", type=int, default=4096)
    parser.add_argument("--seg-capacity", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((HERE / "scenes.json").read_text())
    scene = next(s for s in spec["scenes"] if s["id"] == args.scene)
    mat = spec["materials"][scene["material"]]
    bounds = spec["bounds"]

    import cupy as cp
    import numpy as np

    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.batching import (
        DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
        PerElectronTransportConfig,
    )
    from pyrite.perf.precision import primary_moments
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    comp = [(scene["material"], mat["number_density_ang3"])]
    entries = resolve_layer_tables(comp)
    stopping = resolve_composition_table(mat["catalog_key"], comp)
    brem = load_bremsstrahlung_tables([scene["material"]])
    rounds = {"launched_rows": 0, "launches": 0}

    def on_round(info):
        if info.get("phase") == "round":
            rounds["launches"] += 1
            rounds["launched_rows"] += int(info.get("rows", 0))

    default = DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
    from pyrite.montecarlo.transport.continuation import TransportContinuationConfig

    config = PerElectronTransportConfig(
        seg_capacity=args.seg_capacity,
        scratch_budget_bytes=args.scratch_mib * 1024**2,
        probe_electrons=0,
        max_batch=default.max_batch,
        schedule=default.schedule or "auto",
        continuation=TransportContinuationConfig(on_round=on_round),
    )
    log = Messages()
    for name in ("pyrite.montecarlo.transport.continuation",):
        logging.getLogger(name).addHandler(log)
        logging.getLogger(name).setLevel(logging.INFO)
    kwargs = dict(
        E0_keV=float(scene["energy_keV"]),
        composition=comp,
        thickness_ang=scene["thickness_um"] * 1e4,
        crystal_width_mm=bounds["width_mm"],
        crystal_height_mm=bounds["width_mm"],
        beam_dir=np.array([0.0, 0.0, 1.0]),
        E_cut_keV=float(bounds["electron_cutoff_keV"]),
        energy_model="midpoint",
        max_dE_frac=args.max_de_frac,
        max_steps=args.max_steps,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in entries]],
        stopping_tables=[stopping.arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=float(bounds["hard_photon_cutoff_eV"]),
        bremslib_tables=brem,
        radiative_moments="table",
        transport_core="cuda",
        per_electron_config=config,
        # ELSEPA bypasses the LUT anyway; explicit so `auto` cannot fall back on it.
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    tables_s = time.perf_counter() - START

    def request(seed):
        scores = {}
        retained = 0
        rounds.update(launched_rows=0, launches=0)
        log.messages.clear()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        transport_s = 0.0
        for offset in range(0, args.histories, args.block):
            n = min(args.block, args.histories - offset)
            t1 = time.perf_counter()
            r = simulate_trajectories(**kwargs, Ne=n, seed=seed, _electron_start=offset)
            cp.cuda.Device().synchronize()
            transport_s += time.perf_counter() - t1
            retained += int(np.asarray(r["event_kind"]).size)
            for key, v in block_scores(r, n).items():
                scores.setdefault(key, np.empty(args.histories))[offset : offset + n] = v
            del r
        cp.cuda.Device().synchronize()
        wall = time.perf_counter() - t0
        work = dict(
            rounds,
            retained_rows=retained,
            schedule=sorted({m for m in log.messages if "schedule" in m}),
        )
        return scores, wall, transport_s, work

    _, cold_s, cold_transport_s, cold_work = request(args.seed + 1_000_000)
    scores, warm_s, warm_transport_s, work = request(args.seed)
    if work["launches"] and work["launched_rows"] != work["retained_rows"]:
        raise RuntimeError(f"replayed rows detected: {work}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz = args.output.with_suffix(".npz")
    np.savez_compressed(npz, **scores)
    import cupy

    dev = cupy.cuda.runtime.getDeviceProperties(0)
    record = dict(
        schema="pyrite.geant4-performance.sample.v1",
        engine="pyrite-cuda",
        scene=scene,
        histories=args.histories,
        block=args.block,
        seed=args.seed,
        controls=dict(
            max_dE_frac=args.max_de_frac,
            scratch_mib=args.scratch_mib,
            seg_capacity=args.seg_capacity,
            probe_electrons=0,
            max_steps=args.max_steps,
            schedule=config.schedule,
            radiative_moments="table",
        ),
        timings_s=dict(
            imports_and_tables=tables_s,
            cold_request=cold_s,
            cold_transport=cold_transport_s,
            warm_request=warm_s,
            warm_transport=warm_transport_s,
            process_total=time.perf_counter() - START,
        ),
        work=dict(cold=cold_work, warm=work),
        tables=dict(
            sbethe=stopping.key,
            sbethe_digest=stopping.digest,
            bremslib=brem[scene["material"]].key,
            elsepa=[t.key for e in entries for t in e.tables],
        ),
        device=dev["name"].decode(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        environment={
            k: os.environ.get(k) for k in ("PYRITE_MC_BACKEND", "PYRITE_MC_TRANSPORT_SCHEDULE")
        },
        observables={k: primary_moments(v) for k, v in scores.items()},
        raw_scores_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "timings_s": record["timings_s"],
                "work": record["work"]["warm"],
            }
        )
    )


if __name__ == "__main__":
    main()
