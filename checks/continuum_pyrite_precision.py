"""Matched all-direction discrete source spectra with per-primary bin RSE.

This arm counts coupled hard-bremsstrahlung photons at creation. It is a
statistical PyRITE/Geant4 comparison, separate from the weighted-grid cost probe.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=3.5)
    args = ap.parse_args()
    import cupy as cp

    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
    from pyrite.montecarlo.transport.continuation import TransportContinuationConfig
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    cp.get_default_memory_pool().set_limit(fraction=0.7)
    comp = [("C", 0.11363677327755675)]
    elastic = resolve_layer_tables(comp)
    stopping = resolve_composition_table("hopg", comp)
    brem = load_bremsstrahlung_tables(["C"])
    args.output.mkdir(exist_ok=True, parents=True)
    started = time.monotonic()
    for case_index, (energy, thickness) in enumerate(
        (e, t) for e in (300, 3000, 30000, 100000) for t in (100, 1000)
    ):
        path = args.output / f"{energy}-{thickness}.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else dict(histories=0, total=[0.0] * 6, squares=[0.0] * 6, seconds=0.0, converged=False)
        )
        edges = np.geomspace(1e4, energy * 1000.0, 7)
        while not state["converged"]:
            if time.monotonic() - started > args.max_hours * 3600:
                return
            n = 256
            t0 = time.perf_counter()
            r = simulate_trajectories(
                E0_keV=energy,
                composition=comp,
                thickness_ang=thickness * 1e4,
                crystal_width_mm=10.0,
                crystal_height_mm=10.0,
                beam_dir=np.array([0.0, 0.0, 1.0]),
                E_cut_keV=10.0,
                energy_model="midpoint",
                max_dE_frac=0.05,
                max_steps=2000000,
                elastic_model="elsepa",
                elastic_tables=[[dict(e.arrays) for e in elastic]],
                stopping_tables=[stopping.arrays()],
                atomic_electron_deflection="kawrakow",
                straggling=False,
                radiative_model="bremslib-soft-hard",
                radiative_cutoff_eV=1000.0,
                bremslib_tables=brem,
                radiative_moments="table",
                transport_core="cuda",
                transport_lut_config=TransportLUTConfig(enabled=False),
                per_electron_config=PerElectronTransportConfig(
                    continuation=TransportContinuationConfig(
                        rows_per_round=1024, device_staging=True
                    )
                ),
                Ne=n,
                seed=20000 + case_index,
                _electron_start=state["histories"],
            )
            ids = np.asarray(r["electron_id"], dtype=int)
            # Continuation returns request-local primary IDs, as block_scores uses.
            if ids.size and (ids.min() < 0 or ids.max() >= n):
                raise ValueError("unexpected primary ID scope")
            k = np.asarray(r["hard_radiative_k_eV"])
            bins = np.searchsorted(edges, k, side="right") - 1
            bins[k == edges[-1]] = 5
            keep = (bins >= 0) & (bins < 6)
            x = np.bincount(ids[keep] * 6 + bins[keep], minlength=n * 6).reshape(n, 6).astype(float)
            total = np.asarray(state["total"]) + x.sum(0)
            squares = np.asarray(state["squares"]) + (x * x).sum(0)
            count = state["histories"] + n
            mean = total / count
            var = np.maximum(0, (squares - total * total / count) / (count - 1))
            rse = np.divide(np.sqrt(var / count), mean, out=np.full(6, np.inf), where=mean > 0)
            state.update(
                histories=count,
                total=total.tolist(),
                squares=squares.tolist(),
                seconds=state["seconds"] + time.perf_counter() - t0,
                edges_keV=(edges / 1000).tolist(),
                mean_per_primary=mean.tolist(),
                rse=[float(v) if np.isfinite(v) else None for v in rse],
                converged=count >= 10000 and bool(np.all(rse <= 0.02)),
            )
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(path)
            if count % 25600 == 0 or state["converged"]:
                print(path.name, count, state["rse"], state["converged"], flush=True)


if __name__ == "__main__":
    main()
