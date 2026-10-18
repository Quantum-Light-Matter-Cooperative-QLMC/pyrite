"""Check fixed-count, paired-primary numerical equivalence; never pool pilot data."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from paired_equivalence import paired_interval, validate_coupling


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    design = protocol["equivalence"]
    reference, candidate = design["reference"], design["candidate"]
    scenes = list(design["observables"])
    seeds = protocol["heldout"]["seeds"]
    sizes = dict(
        zip(
            protocol["pilot"]["scenes"],
            [protocol["heldout"]["histories"]["si"], protocol["heldout"]["histories"]["w"]],
            strict=True,
        )
    )
    m = sum(len(v) for v in design["observables"].values())
    if m != design["comparisons"] or len(set(seeds)) != len(seeds):
        raise ValueError("inconsistent frozen inference family/seeds")
    probes = []
    for variant in [reference, candidate]:
        probe = read(args.preflight / f"probe-{variant}.json")
        if {r["element"] for r in probe["reports"]} != set(design["elements"]):
            raise ValueError("component preflight material coverage mismatch")
        if probe["source"]["variant_sha256"] != protocol["source_sha256"][variant] or not all(
            r["screening_passed"] and r["finite"] and r["nonnegative"] and r["photon_support"]
            for r in probe["reports"]
        ):
            raise ValueError("full component preflight failed or source mismatch")
        probes.append(probe)
    samples = {}
    failures = []
    for root in args.inputs:
        gate = read(root / "preflight-gate.json")
        if not {reference, candidate} <= set(gate["eligible"]):
            raise ValueError("held-out stage missing required preflight gate")
        for path in sorted(root.glob("*.json")):
            if path.name.endswith((".failed.json", ".deferred.json")):
                failures.append(dict(path=str(path), **read(path)))
                continue
            record = read(path)
            if record.get("schema") != "pyrite.equal-precision.precision-sample.v1":
                continue
            if path.with_suffix(".failed.json").exists():
                continue
            if (
                record.get("study_phase") != "heldout"
                or record["precision_protocol_sha256"] != digest
            ):
                raise ValueError(
                    "pilot/preflight population or protocol mismatch in held-out inputs"
                )
            scene, variant, seed = (
                record["scene"]["id"],
                record["precision_variant"],
                record["seed"],
            )
            if scene not in scenes or variant not in [reference, candidate] or seed not in seeds:
                raise ValueError("record outside frozen scene/arm/seed design")
            key = (scene, variant, seed)
            if key in samples:
                raise ValueError("duplicate held-out replica")
            if (
                record["histories"] != sizes[scene]
                or record["precision_source"]["variant_sha256"]
                != protocol["source_sha256"][variant]
            ):
                raise ValueError("held-out population or source mismatch")
            if (
                record["precision_source"]["reference_sha256"]
                != protocol["source_sha256"][reference]
            ):
                raise ValueError("original FP64 reference source mismatch")
            warm = record["warmup"]
            if (
                warm["histories"] != record["histories"]
                or not warm["same_scene_tables_numerics"]
                or not warm["device_synchronized"]
            ):
                raise ValueError("invalid complete-workload warmup")
            cxr = record["cxr_execution_audit"]
            if cxr["enabled"] or cxr["calls"]:
                raise ValueError("PXR/CBS profiling/work in held-out study")
            raw = path.with_suffix(".npz")
            if hashlib.sha256(raw.read_bytes()).hexdigest() != record["raw_scores_sha256"]:
                raise ValueError("primary score digest mismatch")
            audit = read(path.with_suffix(".audit.json"))
            if sum(b["primaries"] for b in audit["blocks"]) != 2 * record["histories"]:
                raise ValueError("incomplete warmup/measured trajectory audits")
            for block in audit["blocks"]:
                for field, limit in [
                    ("energy_ledger_max_error_keV", "max_energy_ledger_error_keV"),
                    ("photon_debit_max_error_keV", "max_photon_debit_error_keV"),
                    ("direction_norm_max_error", "max_direction_norm_error"),
                ]:
                    if not 0 <= block[field] <= protocol["invariants"][limit]:
                        raise ValueError("trajectory invariant failed")
            with np.load(raw) as arrays:
                values = {name: arrays[name] for name in arrays.files}
            expected = design["observables"][scene]
            if values.keys() != expected.keys():
                raise ValueError("observable schema mismatch")
            for name, array in values.items():
                spec = expected[name]
                if (
                    array.shape != (record["histories"],)
                    or not np.isfinite(array).all()
                    or np.any(array < 0)
                    or np.any(array > spec["bound"])
                ):
                    raise ValueError("known physical score support/population violated")
                if spec["kind"] in ["indicator", "count"] and not np.array_equal(
                    array, np.floor(array)
                ):
                    raise ValueError("non-integer primary event count")
            samples[key] = (record, values)
    missing = [
        dict(scene=s, variant=v, seed=k)
        for s in scenes
        for v in [reference, candidate]
        for k in seeds
        if (s, v, k) not in samples
    ]
    complete = not missing and not failures
    controls = [
        "scene",
        "manifest_sha256",
        "tables",
        "max_de_frac",
        "cutoff_eV",
        "batch_histories",
        "scratch_mib",
        "probe_electrons",
        "core",
        "precision_protocol_sha256",
    ]
    frozen_controls = {}
    for scene in scenes:
        pre = protocol["preflight"]
        pin = read(args.preflight / f"{scene}-{pre['histories']}-{pre['seed']}-{reference}.json")
        if (
            pin.get("study_phase") != "preflight"
            or pin.get("precision_protocol_sha256") != digest
            or not all(k in pin for k in controls)
        ):
            raise ValueError("missing frozen resolved preflight controls")
        frozen_controls[scene] = {k: pin[k] for k in controls}
    for (scene, _, _), (record, _) in samples.items():
        if any(record.get(k) != value for k, value in frozen_controls[scene].items()):
            raise ValueError("held-out controls differ from frozen preflight")
    comparisons = []
    for scene in scenes:
        pairs = []
        for seed in seeds:
            if (scene, reference, seed) not in samples or (scene, candidate, seed) not in samples:
                continue
            first, second = samples[(scene, reference, seed)], samples[(scene, candidate, seed)]
            if any(first[0].get(k) != second[0].get(k) for k in controls):
                raise ValueError("cross-arm input/control mismatch")
            validate_coupling(first[0], second[0])
            pairs.append((first[1], second[1]))
        for name, spec in design["observables"][scene].items():
            if complete:
                a = np.concatenate([p[0][name] for p in pairs])
                b = np.concatenate([p[1][name] for p in pairs])
                result = paired_interval(
                    a,
                    b,
                    bound=spec["bound"],
                    margin=spec["margin"],
                    comparisons=m,
                    alpha=design["alpha"],
                )
            else:
                result = dict(
                    status="unresolved",
                    reason="incomplete fixed-design data; no selected-subset confidence claim",
                )
            comparisons.append(dict(scene=scene, observable=name, **result))
    statuses = dict(Counter(c["status"] for c in comparisons))
    output = dict(
        schema="pyrite.equal-precision.paired-equivalence.v1",
        complete_fixed_design=complete,
        qualified_observable_equivalence=complete and statuses.get("passed", 0) == m,
        qualified_T_epsilon=False,
        protocol_sha256=digest,
        source_sha256=protocol["source_sha256"],
        reference=reference,
        candidate=candidate,
        expected_replicas_per_scene=len(seeds),
        histories_per_replica=sizes,
        comparisons=comparisons,
        statuses=statuses,
        failures=failures,
        missing=missing,
        component_preflight=probes,
        interpretation="Fixed-count finite-sample numerical mean-equivalence on frozen observables/scenes only, under independent pseudorandom primary-pair assumptions. Not physical sign-off, distributional identity, cross-engine validation or a time-to-precision ranking.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                complete=complete,
                qualified_observable_equivalence=output["qualified_observable_equivalence"],
                statuses=statuses,
            )
        )
    )


if __name__ == "__main__":
    main()
