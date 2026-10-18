"""Analyze the complete cheap pilot; keep diagnostics separate from qualification."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

OBSERVABLES = [
    "transmitted",
    "backscattered",
    "side_exited",
    "stopped",
    "transmitted_energy_keV_per_primary",
    "photons_ge10kev",
    "photons_10_50kev",
    "photons_50_100kev",
    "photons_ge100kev",
    "photon_energy_keV_per_primary",
    "continuous_loss_keV_per_primary",
    "terminal_energy_keV_per_primary",
    *[f"transmitted_mu_bin_{i}" for i in range(4)],
]


def parse_cut_couples(text):
    """Read actual material-dependent production cuts, including duplicate MT dumps."""
    text = re.sub(r"(?m)^G4WT\d+ >\s*", "", text)
    pattern = (
        r"Material\s*:\s*(\w+)\s*\n\s*Range cuts[^\n]*\n"
        r"\s*Energy thresholds\s*:\s*gamma\s+([\d.eE+-]+)\s+(eV|keV|MeV|GeV)"
        r"\s+e-\s+([\d.eE+-]+)\s+(eV|keV|MeV|GeV)"
    )
    units = {"eV": 0.001, "keV": 1, "MeV": 1000, "GeV": 1000000}
    result = {}
    for material, gamma, gu, electron, eu in re.findall(pattern, text):
        row = dict(
            gamma_production_threshold_keV=float(gamma) * units[gu],
            electron_production_threshold_keV=float(electron) * units[eu],
        )
        if material in result and result[material] != row:
            raise ValueError("inconsistent repeated production-cut dump")
        result[material] = row
    if not result:
        raise ValueError("missing material production-cut dump")
    return result


def check_record(record, scores, protocol, digest, scene):
    if (
        record["validation_protocol_sha256"] != digest
        or record["validation_sources"] != protocol["validation_sources"]
    ):
        raise ValueError("validation protocol/source mismatch")
    if record["arm"] not in protocol["arms"] or record["scene"] != scene:
        raise ValueError("scene/arm mismatch")
    core, n = record["core"], record["histories"]
    control = protocol["arms"][record["arm"]]
    expected = dict(
        E_cut_keV=10,
        energy_model="midpoint",
        max_dE_frac=control["max_de_frac"],
        elastic_model="elsepa",
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=control["cutoff_eV"],
        transport_core=core,
        crystal_width_mm=10,
        crystal_height_mm=10,
    )
    if (
        record["resolved_controls"] != expected
        or record["max_de_frac"] != control["max_de_frac"]
        or record["cutoff_eV"] != control["cutoff_eV"]
    ):
        raise ValueError("resolved control mismatch")
    warm = record["warmup"]
    if (
        record["cache_state"] != "warmed-in-process"
        or warm["repeats"] != 1
        or warm["histories"] != n
        or warm["seed"] != record["seed"] + 1000000
        or warm["same_scene_tables_numerics"] is not True
        or warm["device_synchronized"] != (core == "cuda")
    ):
        raise ValueError("incomplete same-workload warmup")
    audit = record["cxr_execution_audit"]
    if audit["enabled"] is not True or audit["calls"]:
        raise ValueError("missing/failed PXR/CBS exclusion")
    blocks = record["trajectory_blocks"]
    limits = {
        "energy_ledger_max_error_keV": 1e-7,
        "photon_debit_max_error_keV": 1e-9,
        "direction_norm_max_error": 1e-10,
    }
    if [b["primaries"] for b in blocks] != [n, n]:
        raise ValueError("incomplete warmup/measurement trajectory audit")
    if any(
        not math.isfinite(b[k]) or not 0 <= b[k] <= limit
        for b in blocks
        for k, limit in limits.items()
    ):
        raise ValueError("trajectory invariant failed")
    coverage = record["table_coverage"]
    if len(coverage) < 3 or not all(
        c["covered"]
        and c["requested_keV"] == [10, scene["energy_keV"]]
        and c["min_keV"] <= 10 <= scene["energy_keV"] <= c["max_keV"]
        for c in coverage.values()
    ):
        raise ValueError("table energy domain not covered")
    if set(scores) != set(OBSERVABLES) or any(
        v.shape != (n,) or not np.isfinite(v).all() or np.any(v < 0) for v in scores.values()
    ):
        raise ValueError("invalid incident-primary score population")
    terminals = ["transmitted", "backscattered", "side_exited", "stopped"]
    bins = [f"transmitted_mu_bin_{i}" for i in range(4)]
    if any(np.any((scores[k] != 0) & (scores[k] != 1)) for k in terminals + bins):
        raise ValueError("nonbinary terminal/angular score")
    if not np.all(sum(scores[k] for k in terminals) == 1) or not np.array_equal(
        sum(scores[k] for k in bins), scores["transmitted"]
    ):
        raise ValueError("terminal/angular partition mismatch")
    ledger = sum(
        scores[k]
        for k in [
            "continuous_loss_keV_per_primary",
            "photon_energy_keV_per_primary",
            "terminal_energy_keV_per_primary",
        ]
    )
    if np.max(np.abs(ledger - scene["energy_keV"])) > 1e-7:
        raise ValueError("raw primary score energy ledger mismatch")


def main():
    from paired_equivalence import paired_interval, validate_coupling
    from run_cheap_validation import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = json.loads((args.inputs[0] / "source/manifest.json").read_bytes())
    if (
        hashlib.sha256((args.inputs[0] / "source/manifest.json").read_text().encode()).hexdigest()
        != protocol["manifest_text_sha256"]
    ):
        raise ValueError("frozen scene manifest mismatch")
    scenes = {s["id"]: s for s in manifest["scenes"]}

    def key(r):
        scene = r["scene"] if isinstance(r["scene"], str) else r["scene"]["id"]
        return scene, r["histories"], r["seed"], r["core"], r.get("arm", "baseline")

    expected = {key(r) for stage in ["cpu", "gpu"] for r in requests(protocol, stage)}
    samples, failures, comparator = {}, [], []
    for root in args.inputs:
        if (root / "source/cheap-validation.json").read_bytes() != args.protocol.read_bytes():
            raise ValueError("captured protocol mismatch")
        for path in sorted(root.glob("*.json")):
            if path.name.endswith((".failed.json", ".deferred.json")):
                failures.append(dict(path=path.name, **json.loads(path.read_bytes())))
                continue
            record = json.loads(path.read_bytes())
            schema = record.get("schema")
            if schema not in [
                "pyrite.equal-precision.cheap-validation-sample.v1",
                "pyrite.equal-precision.sample.v1",
            ]:
                continue
            if path.with_suffix(".failed.json").exists():
                continue
            if (
                hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest()
                != record["raw_scores_sha256"]
            ):
                raise ValueError("raw score hash mismatch")
            with np.load(path.with_suffix(".npz")) as raw:
                scores = {k: raw[k].copy() for k in raw.files}
            if record["engine"] == "geant4":
                record["core"] = "geant4"
                if (
                    record["executable_sha256"] != protocol["geant4"]["executable_sha256"]
                    or record["physics"] != protocol["geant4"]["physics"]
                ):
                    raise ValueError("Geant4 executable/physics mismatch")
                log = path.with_suffix("") / "raw.log"
                if hashlib.sha256(log.read_bytes()).hexdigest() != record["raw_log_sha256"]:
                    raise ValueError("Geant4 raw log hash mismatch")
                if any(
                    v.shape != (record["histories"],) or not np.isfinite(v).all()
                    for v in scores.values()
                ):
                    raise ValueError("invalid comparator primary grouping")
                text = log.read_text()
                comparator.append(
                    dict(
                        scene=record["scene"]["id"],
                        histories=record["histories"],
                        executable_sha256=record["executable_sha256"],
                        physics=record["physics"],
                        production_cuts=parse_cut_couples(text),
                        log_relative_path=str(log.relative_to(root)),
                        inference="configuration only; no coupled/cross-engine accuracy claim",
                    )
                )
            else:
                check_record(record, scores, protocol, digest, scenes[record["scene"]["id"]])
            k = key(record)
            if k not in expected or k in samples:
                raise ValueError("unexpected/duplicate fixed request")
            samples[k] = (record, scores)
    missing = sorted(expected - set(samples))
    result = dict(
        schema="pyrite.equal-precision.cheap-validation-summary.v1",
        protocol_sha256=digest,
        complete_fixed_design=not failures and not missing,
        samples=len(samples),
        failures=failures,
        missing=missing,
        comparisons=[],
        comparator_configuration=comparator,
        qualified_T_epsilon=False,
        qualified_numerical_convergence=False,
    )
    if not missing and not failures:
        m = protocol["comparisons"]["family_size"]
        if m != len(scenes) * len(OBSERVABLES) * 3:
            raise ValueError("frozen inference family mismatch")
        for scene_id in protocol["scenes"]:
            scene = scenes[scene_id]
            for test in ["half-step", "half-hard-cutoff", "cpu-gpu"]:
                backend = test == "cpu-gpu"
                design = protocol["backend"] if backend else protocol["convergence"]
                pairs = []
                for seed in design["seeds"]:
                    a = samples[
                        (
                            scene_id,
                            design["histories"],
                            seed,
                            "per-electron" if backend else "cuda",
                            "baseline",
                        )
                    ]
                    b = samples[
                        (
                            scene_id,
                            design["histories"],
                            seed,
                            "cuda",
                            "baseline" if backend else test,
                        )
                    ]
                    validate_coupling(a[0], b[0])
                    if (
                        a[0]["tables"] != b[0]["tables"]
                        or a[0]["table_coverage"] != b[0]["table_coverage"]
                    ):
                        raise ValueError("paired table mismatch")
                    pairs.append((a, b))
                for name in OBSERVABLES:
                    a, b = [np.concatenate([p[i][1][name] for p in pairs]) for i in [0, 1]]
                    if name.startswith("photons_"):
                        lower = (
                            100
                            if name == "photons_ge100kev"
                            else 50
                            if name == "photons_50_100kev"
                            else 10
                        )
                        bound = math.floor((scene["energy_keV"] + 1e-6) / lower)
                        margin = protocol["comparisons"]["photon_absolute_margins"][name]
                        if np.any(a != np.floor(a)) or np.any(b != np.floor(b)):
                            raise ValueError("noninteger source counts")
                    elif "energy" in name or "loss" in name:
                        bound = scene["energy_keV"] + 1e-6
                        margin = (
                            scene["energy_keV"]
                            * protocol["comparisons"]["energy_margin_fraction_of_beam"]
                        )
                    else:
                        bound = 1
                        margin = protocol["comparisons"]["fraction_absolute_margin"]
                    row = paired_interval(a, b, bound=bound, margin=margin, comparisons=m)
                    row.update(
                        scene=scene_id,
                        test=test,
                        observable=name,
                        max_observed_primary_difference=float(np.max(np.abs(b - a))),
                    )
                    result["comparisons"].append(row)
        result["statuses"] = dict(Counter(r["status"] for r in result["comparisons"]))
        result["table_coverage"] = {
            scene: samples[(scene, 64, protocol["backend"]["seeds"][0], "cuda", "baseline")][0][
                "table_coverage"
            ]
            for scene in protocol["scenes"]
        }
        result["max_trajectory_errors"] = {
            k: max(
                b[k]
                for r, _ in samples.values()
                if r["engine"] == "pyrite"
                for b in r["trajectory_blocks"]
            )
            for k in [
                "energy_ledger_max_error_keV",
                "photon_debit_max_error_keV",
                "direction_norm_max_error",
            ]
        }
    result["interpretation"] = (
        "Fixed small sensitivity/backend spot-checks only. Two numerical levels, no asymptotic convergence, no CPU/GPU distributional identity, no cross-engine ranking. Audit overhead included in diagnostic times."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ["complete_fixed_design", "samples", "statuses", "max_trajectory_errors"]
            }
        )
    )


if __name__ == "__main__":
    main()
