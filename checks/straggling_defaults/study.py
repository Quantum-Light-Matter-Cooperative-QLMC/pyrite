"""Bounded #642 sensitivity study; production model selection is explicitly pinned.

Run only through the remote batch script. Independent on/off seeds do not pair
histories. Source hard-photon counts are diagnostics, not detector/line yields.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_scores(result, n, energy_keV):
    """Keep hard debits on terminal rows and refuse incomplete populations."""
    from pyrite.montecarlo.transport.events import check_segment_event_contract
    from pyrite.perf.precision import grouped_scores

    check_segment_event_contract(result)
    ids = np.asarray(result["electron_id"], dtype=int)
    kind = np.asarray(result["event_kind"])
    terminal = np.isin(kind, [4, 5, 6, 7])
    if not np.array_equal(np.sort(ids[terminal]), np.arange(n)):
        raise ValueError("incomplete or duplicate primary terminal records")
    start = np.asarray(result["E_start_keV"])
    end = np.asarray(result["E_end_keV"])
    hard = np.asarray(result.get("hard_W_keV", np.zeros(ids.size)))
    photons = np.asarray(result["hard_radiative_k_eV"]) * 1e-3
    remaining = end[terminal] - hard[terminal] - photons[terminal]
    loss = grouped_scores(ids, start - end + hard + photons, n)
    final = grouped_scores(ids[terminal], remaining, n)
    residual = float(np.max(np.abs(loss + final - energy_keV)))
    if residual > 1e-7 or np.any(remaining < -1e-9):
        raise ValueError(f"primary energy ledger failed: {residual} keV")
    scores = {"final_keV": final}
    for name, code in [
        ("backscattered", 4),
        ("transmitted", 5),
        ("side_exited", 6),
        ("stopped", 7),
    ]:
        scores[name] = grouped_scores(ids[kind == code], np.ones(np.count_nonzero(kind == code)), n)
    scores["path_ang"] = grouped_scores(ids, np.asarray(result["L_ang"]), n)
    scores["rows"] = grouped_scores(ids, np.ones(ids.size), n)
    scores["energy_substeps"] = grouped_scores(
        ids[kind == 1], np.ones(np.count_nonzero(kind == 1)), n
    )
    selected = photons >= 10.0
    scores["photons_ge10keV"] = grouped_scores(ids[selected], np.ones(selected.sum()), n)
    scores["hard_photon_energy_keV"] = grouped_scores(ids, photons, n)
    for values in scores.values():
        if not np.isfinite(values).all():
            raise ValueError("nonfinite primary score")
    return scores, residual


def range_ang(table, energy, cutoff):
    """SBETHE CSDA sizing only; freeze the same slab in every comparison arm."""
    grid = np.geomspace(cutoff, energy, 2000)
    nodes = np.asarray(table["stopping_energy_eV"]) * 1e-3
    stopping = np.asarray(table["stopping_eV_per_angstrom"]) * 1e-3
    rates = np.exp(np.interp(np.log(grid), np.log(nodes), np.log(stopping)))
    return float(np.trapezoid(1 / rates, grid))


def transport_inputs(protocol, material, fraction, core, cap, cutoff):
    from pyrite.montecarlo.transport import TransportLUTConfig
    from pyrite.montecarlo.transport.batching import PerElectronTransportConfig
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import CATALOG, resolve_catalog_table

    entry = CATALOG.crystal(material) if material in CATALOG.crystals else CATALOG.media[material]
    comp = entry.composition
    stopping = resolve_catalog_table(material)
    elastic = resolve_layer_tables(comp)
    brem = load_bremsstrahlung_tables([el for el, _ in comp])
    table = stopping.arrays()
    distance = range_ang(table, protocol["energy_keV"], protocol["electron_cutoff_keV"])
    kwargs = dict(
        E0_keV=protocol["energy_keV"],
        Ne=protocol["histories"],
        thickness_ang=fraction * distance,
        composition=comp,
        crystal_width_mm=10.0,
        crystal_height_mm=10.0,
        E_cut_keV=protocol["electron_cutoff_keV"],
        energy_model="midpoint",
        max_dE_frac=cap,
        max_steps=protocol["max_steps"],
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in elastic]],
        stopping_tables=[table],
        atomic_electron_deflection="kawrakow",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=protocol["radiative_cutoff_eV"],
        bremslib_tables=brem,
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=False),
        per_electron_config=PerElectronTransportConfig(
            scratch_budget_bytes=protocol["scratch_mib"] * 1024**2
        ),
    )
    if protocol["materials"][material] == "shell-soft-hard":
        kwargs.update(
            inelastic_model="shell-soft-hard",
            inelastic_materials=[material],
            inelastic_cutoff_eV=cutoff,
        )
    tables = {
        "sbethe": {"key": stopping.key, "digest": stopping.digest},
        "elsepa": [[{"key": t.key, "digest": t.digest} for t in e.tables] for e in elastic],
        "bremslib": {el: {"key": t.key, "digest": t.digest} for el, t in brem.items()},
    }
    return kwargs, tables, distance


def run(args, protocol):
    from pyrite.montecarlo.transport import simulate_trajectories
    from pyrite.perf.precision import primary_moments

    cap = protocol["max_de_frac"] if args.cap is None else args.cap
    cutoff = protocol["inelastic_cutoff_eV"] if args.cutoff is None else args.cutoff
    kwargs, tables, distance = transport_inputs(
        protocol, args.material, args.fraction, args.core, cap, cutoff
    )
    enabled = args.straggling == "on"
    seeds = protocol["seeds"][args.straggling]
    args.output.mkdir(parents=True, exist_ok=False)

    def sync():
        if args.core == "cuda":
            import cupy

            cupy.cuda.Device().synchronize()

    def sample(seed):
        sync()
        begin = time.perf_counter()
        result = simulate_trajectories(**kwargs, seed=seed, straggling=enabled)
        sync()
        elapsed = time.perf_counter() - begin
        scores, residual = primary_scores(result, protocol["histories"], protocol["energy_keV"])
        return scores, elapsed, residual

    _, warmup_s, _ = sample(protocol["warmup_seed"])
    common = dict(
        schema="pyrite.straggling-defaults.sample.v1",
        protocol_sha256=digest(args.protocol),
        material=args.material,
        range_fraction=args.fraction,
        csda_range_ang=distance,
        thickness_ang=kwargs["thickness_ang"],
        inelastic_model=protocol["materials"][args.material],
        straggling=enabled,
        core=args.core,
        max_de_frac=cap,
        inelastic_cutoff_eV=cutoff,
        histories=protocol["histories"],
        tables=tables,
        warmup_s=warmup_s,
        warmup_seed=protocol["warmup_seed"],
        warmup_histories=protocol["histories"],
        peak_rss_kib=0,
        numerical_gate="unqualified",
        accuracy_gate="unqualified",
    )
    for seed in seeds:
        scores, elapsed, residual = sample(seed)
        raw = args.output / f"{seed}.npz"
        np.savez_compressed(raw, **scores)
        record = dict(
            common,
            seed=seed,
            transport_s=elapsed,
            energy_residual_keV=residual,
            raw_sha256=digest(raw),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            observables={k: primary_moments(v) for k, v in scores.items()},
        )
        raw.with_suffix(".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(raw), "transport_s": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material")
    parser.add_argument("fraction", type=float)
    parser.add_argument("straggling", choices=["on", "off"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--core", choices=["cuda", "per-electron"], default="cuda")
    parser.add_argument("--cap", type=float)
    parser.add_argument("--cutoff", type=float)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    if args.material not in protocol["materials"] or args.fraction <= 0:
        parser.error("unknown material or nonpositive thickness")
    run(args, protocol)


if __name__ == "__main__":
    main()
