"""Verify #642 raw evidence, pool independent primaries, report sensitivity only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_groups(root, protocol_path):
    protocol = json.loads(protocol_path.read_bytes())
    expected_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    groups = {}
    for path in sorted(root.rglob("*.json")):
        # Study samples are named by integer seed. Job provenance and source
        # snapshots may include unrelated JSON (or the text sync stamp).
        if not path.stem.isdecimal():
            continue
        record = json.loads(path.read_bytes())
        if (
            not isinstance(record, dict)
            or record.get("schema") != "pyrite.straggling-defaults.sample.v1"
        ):
            continue
        if record["protocol_sha256"] != expected_hash:
            raise ValueError(f"protocol mismatch: {path}")
        raw = path.with_suffix(".npz")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record["raw_sha256"]:
            raise ValueError(f"raw checksum mismatch: {path}")
        switch = "on" if record["straggling"] else "off"
        if (
            record["seed"] not in protocol["seeds"][switch]
            or record["histories"] != protocol["histories"]
        ):
            raise ValueError(f"population/seed mismatch: {path}")
        key = (
            record["material"],
            record["range_fraction"],
            record["core"],
            record["max_de_frac"],
            record["inelastic_cutoff_eV"],
            switch,
        )
        group = groups.setdefault(key, [])
        if any(r["seed"] == record["seed"] for r, _ in group):
            raise ValueError(f"duplicate seed: {path}")
        with np.load(raw) as data:
            scores = {k: data[k].copy() for k in data.files}
        if not scores or any(
            v.shape != (record["histories"],) or not np.isfinite(v).all() for v in scores.values()
        ):
            raise ValueError(f"invalid raw scores: {path}")
        for previous, values in group:
            if previous["tables"] != record["tables"] or values.keys() != scores.keys():
                raise ValueError(f"inconsistent replica tables/observables: {path}")
        group.append((record, scores))
    return protocol, groups


def summarize_group(key, replicas, protocol):
    from pyrite.perf.precision import primary_moments

    material, fraction, core, cap, cutoff, switch = key
    wanted = set(protocol["seeds"][switch])
    complete = {r["seed"] for r, _ in replicas} == wanted
    scores = {k: np.concatenate([s[k] for _, s in replicas]) for k in replicas[0][1]}
    transmitted = scores["transmitted"].astype(bool)
    energies = scores["final_keV"][transmitted]
    stopped_path = scores["path_ang"][scores["stopped"].astype(bool)]
    times = [r["transport_s"] for r, _ in replicas]
    return dict(
        material=material,
        fraction=fraction,
        core=core,
        cap=cap,
        cutoff_eV=cutoff,
        straggling=switch,
        replicas=len(replicas),
        complete=complete,
        tables=replicas[0][0]["tables"],
        observables={k: primary_moments(v) for k, v in scores.items() if k != "final_keV"},
        transmitted_energy=None
        if energies.size < 2
        else dict(
            primary_moments(energies),
            variance_keV2=float(energies.var(ddof=1)),
            quantiles_keV=np.quantile(energies, [0.01, 0.5, 0.99]).tolist(),
            tail_qualification="unqualified",
        ),
        stopped_path_sd_ang=None if stopped_path.size < 2 else float(stopped_path.std(ddof=1)),
        warmed_transport_s=dict(mean=float(np.mean(times)), min=min(times), max=max(times)),
        energy_residual_max_keV=max(r["energy_residual_keV"] for r, _ in replicas),
        peak_rss_kib=max(r["peak_rss_kib"] for r, _ in replicas),
    )


def comparisons(groups, protocol):
    results = []
    for off in groups:
        if off["straggling"] != "off":
            continue
        on = next(
            (
                g
                for g in groups
                if all(g[k] == off[k] for k in ["material", "fraction", "core", "cap", "cutoff_eV"])
                and g["straggling"] == "on"
            ),
            None,
        )
        if on is None:
            continue
        if on["tables"] != off["tables"]:
            raise ValueError("on/off table identities differ")
        differences = {}
        for observable in ["transmitted", "backscattered", "stopped", "transmitted_energy"]:
            a = (
                off.get(observable)
                if observable == "transmitted_energy"
                else off["observables"][observable]
            )
            b = (
                on.get(observable)
                if observable == "transmitted_energy"
                else on["observables"][observable]
            )
            if a is None or b is None:
                differences[observable] = {"outcome": "unresolved"}
                continue
            if observable == "transmitted_energy" and min(a["n"], b["n"]) < 30:
                differences[observable] = {
                    "outcome": "unresolved",
                    "reason": "fewer than 30 transmitted primaries in an arm",
                }
                continue
            delta = b["mean"] - a["mean"]
            half = 1.96 * np.hypot(a["se"], b["se"])
            margin = protocol["diagnostic_margins"]["terminal_fraction_absolute"]
            if observable == "transmitted_energy":
                margin = (
                    abs(a["mean"])
                    * protocol["diagnostic_margins"]["transmitted_energy_mean_relative"]
                )
            inside = abs(delta) + half <= margin
            # Zero observed variance never establishes equivalence.
            usable = off["complete"] and on["complete"] and a["se"] > 0 and b["se"] > 0
            outcome = "unresolved"
            if usable and inside:
                outcome = "inside-diagnostic-margin"
            elif usable and abs(delta) - half > margin:
                outcome = "outside-diagnostic-margin"
            differences[observable] = dict(
                delta=delta,
                approximate_95ci=[float(delta - half), float(delta + half)],
                margin=margin,
                outcome=outcome,
            )
        results.append(
            dict(
                material=off["material"],
                fraction=off["fraction"],
                core=off["core"],
                cap=off["cap"],
                cutoff_eV=off["cutoff_eV"],
                differences=differences,
                warmed_fixed_population_time_ratio=on["warmed_transport_s"]["mean"]
                / off["warmed_transport_s"]["mean"],
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    protocol, raw = read_groups(args.root, args.protocol)
    groups = [summarize_group(k, v, protocol) for k, v in sorted(raw.items())]
    missing = [
        (m, f, s)
        for m in protocol["materials"]
        for f in protocol["range_fractions"]
        for s in ["off", "on"]
        if not any(
            g["material"] == m
            and g["fraction"] == f
            and g["straggling"] == s
            and g["core"] == "cuda"
            and g["cap"] == protocol["max_de_frac"]
            and g["cutoff_eV"] == protocol["inelastic_cutoff_eV"]
            and g["complete"]
            for g in groups
        )
    ]
    report = dict(
        schema="pyrite.straggling-defaults.summary.v1",
        groups=groups,
        comparisons=comparisons(groups, protocol),
        missing_baseline_groups=missing,
        qualification="sensitivity only; no default change or accuracy/equal-precision claim",
        interval_policy="pointwise normal approximation; no multiple-comparison or tail qualification",
        deferred=protocol["deferred"],
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
