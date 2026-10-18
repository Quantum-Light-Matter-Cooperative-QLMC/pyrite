"""Analyze mixed-precision diagnostics without establishing production equivalence."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from pyrite.perf.precision import primary_moments


def compare_scores(reference, candidate, *, margin, comparisons, min_nonzero=0):
    """Approximate simultaneous difference interval; unknown seed covariance bounded."""
    if not math.isfinite(margin) or margin <= 0 or comparisons < 1:
        raise ValueError("positive finite margin and comparison count required")
    first, second = primary_moments(reference), primary_moments(candidate)
    # The same seed does not ensure paired trajectories across precision. The
    # sum of SEs bounds the difference SE for any cross-arm covariance.
    se = first["se"] + second["se"]
    difference = second["mean"] - first["mean"]
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons))
    interval = [difference - z * se, difference + z * se]
    estimable = (
        first["se"] > 0
        and second["se"] > 0
        and min(first["nonzero"], second["nonzero"]) >= min_nonzero
    )
    status = "unresolved"
    if estimable:
        if interval[0] >= -margin and interval[1] <= margin:
            status = "passed"
        elif interval[0] > margin or interval[1] < -margin:
            status = "failed"
    return dict(
        status=status,
        margin=margin,
        reference_mean=first["mean"],
        candidate_mean=second["mean"],
        difference=difference,
        difference_ci=interval,
        difference_se_bound=se,
        bonferroni_comparisons=comparisons,
        approximation="Normal interval; unresolved zero variance/rare counts. Screening only.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    protocol = json.loads(
        (args.protocol or Path(__file__).with_name("precision-experiment.json")).read_text(
            encoding="utf-8"
        )
    )
    groups = defaultdict(list)
    failures = []
    probes = []
    for root in args.inputs:
        for path in sorted(root.rglob("*.json")):
            if path.name.endswith((".failed.json", ".deferred.json")):
                failures.append({"path": str(path), **json.loads(path.read_text(encoding="utf-8"))})
                continue
            if path.with_suffix(".failed.json").exists():
                continue
            sample = json.loads(path.read_text(encoding="utf-8"))
            if sample.get("schema") != "pyrite.equal-precision.precision-sample.v1":
                if path.name.startswith("probe-") and "reports" in sample:
                    probes.append({"path": str(path), **sample})
                continue
            if (
                args.protocol
                and sample["precision_protocol_sha256"]
                != hashlib.sha256(args.protocol.read_bytes()).hexdigest()
            ):
                raise ValueError("precision protocol hash mismatch")
            raw = path.with_suffix(".npz")
            if hashlib.sha256(raw.read_bytes()).hexdigest() != sample["raw_scores_sha256"]:
                raise ValueError(f"corrupt primary scores: {path}")
            if sample.get("cxr_execution_audit", {}).get("calls"):
                raise ValueError("PXR/CBS execution in precision diagnostic")
            warmup = sample.get("warmup", {})
            if (
                warmup.get("histories") != sample["histories"]
                or not warmup.get("same_scene_tables_numerics")
                or not warmup.get("device_synchronized")
            ):
                raise ValueError("missing same-workload synchronized warmup")
            groups[
                (sample["scene"]["id"], sample["histories"], sample["precision_variant"])
            ].append((path, sample))
    arrays = {}
    results = []
    for key, records in sorted(groups.items()):
        seeds = [s["seed"] for _, s in records]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate replica seed: {key}")
        source_pins = {json.dumps(s["precision_source"], sort_keys=True) for _, s in records}
        controls = (
            "max_de_frac",
            "cutoff_eV",
            "batch_histories",
            "scratch_mib",
            "probe_electrons",
            "precision_protocol_sha256",
        )
        control_pins = {tuple(s.get(k) for k in controls) for _, s in records}
        if len(source_pins) != 1:
            raise ValueError(f"replica source mismatch: {key}")
        if len(control_pins) != 1:
            raise ValueError(f"replica control mismatch: {key}")
        values = defaultdict(list)
        fields = None
        for path, sample in records:
            with np.load(path.with_suffix(".npz")) as scores:
                if fields is not None and set(scores.files) != fields:
                    raise ValueError("replica observable fields differ")
                fields = set(scores.files)
                for name in scores.files:
                    if scores[name].shape != (sample["histories"],):
                        raise ValueError("score population mismatch")
                    values[name].append(scores[name])
        arrays[key] = {name: np.concatenate(v) for name, v in values.items()}
        times = [s["timings_s"]["transport_and_emission"] for _, s in records]
        results.append(
            dict(
                scene=key[0],
                histories_per_replica=key[1],
                variant=key[2],
                replicas=len(records),
                seeds=seeds,
                warmed_transport_median_s=float(np.median(times)),
                warmed_transport_min_s=min(times),
                warmed_transport_max_s=max(times),
                observables={name: primary_moments(v) for name, v in arrays[key].items()},
            )
        )
    pending = []
    for key in sorted(groups):
        scene, n, variant = key
        baseline = (scene, n, "fp64")
        if variant == "fp64" or baseline not in groups:
            continue
        if set(arrays[key]) != set(arrays[baseline]):
            raise ValueError("cross-precision observable fields differ")
        reference_samples = groups[baseline]
        candidate_samples = groups[key]
        manifests = {s["manifest_sha256"] for _, s in reference_samples + candidate_samples}
        sources = {
            s["precision_source"]["reference_sha256"]
            for _, s in reference_samples + candidate_samples
        }
        control_pins = {
            tuple(s.get(k) for k in controls) for _, s in reference_samples + candidate_samples
        }
        if len(manifests) != 1 or len(sources) != 1 or len(control_pins) != 1:
            raise ValueError("cross-precision input/source mismatch")
        energy = reference_samples[0][1]["scene"]["energy_keV"]
        for name, reference in arrays[baseline].items():
            if name == "photons_ge100kev" and energy <= 100:
                continue
            if "keV" in name:
                margin = protocol["ensemble_screen"]["energy_margin_fraction_of_beam"] * energy
            elif name.startswith("photons_"):
                margin = protocol["ensemble_screen"]["yield_relative_margin"] * float(
                    np.mean(reference)
                )
            else:
                margin = protocol["ensemble_screen"]["fraction_absolute_margin"]
            pending.append((key, name, reference, arrays[key][name], margin))
    comparisons = []
    for key, name, reference, candidate, margin in pending:
        comparison = (
            compare_scores(
                reference,
                candidate,
                margin=margin,
                comparisons=max(1, len(pending)),
                min_nonzero=20 if name.startswith("photons_") else 0,
            )
            if margin > 0
            else {
                "status": "unresolved",
                "reason": "zero reference yield; no frozen absolute floor",
            }
        )
        comparisons.append(
            dict(
                scene=key[0],
                histories_per_replica=key[1],
                variant=key[2],
                observable=name,
                **comparison,
            )
        )
    timing = []
    for key, records in sorted(groups.items()):
        if key[2] == "fp64" or key[1] != protocol["pilot"]["histories"]:
            continue
        baseline = (key[0], key[1], "fp64")
        if baseline not in groups:
            continue
        first = {s["seed"]: s["timings_s"]["transport_and_emission"] for _, s in groups[baseline]}
        second = {s["seed"]: s["timings_s"]["transport_and_emission"] for _, s in records}
        seeds = sorted(first.keys() & second.keys())
        a, b = np.array([first[s] for s in seeds]), np.array([second[s] for s in seeds])
        indices = np.random.default_rng(58000).integers(0, len(seeds), size=(10000, len(seeds)))
        ratios = np.median(a[indices], axis=1) / np.median(b[indices], axis=1)
        timing.append(
            dict(
                scene=key[0],
                variant=key[2],
                paired_timing_replicas=len(seeds),
                complete_five_repeat_pilot=len(seeds) == 5,
                reference_median_s=float(np.median(a)),
                candidate_median_s=float(np.median(b)),
                median_ratio=float(np.median(a) / np.median(b)),
                exploratory_ratio_bootstrap95=list(map(float, np.quantile(ratios, [0.025, 0.975]))),
                interpretation="Paired seed timing bootstrap, diagnostic only. Accuracy/equivalence gates remain separate.",
            )
        )
    output = dict(
        schema="pyrite.equal-precision.precision-summary.v1",
        results=results,
        component_probes=probes,
        comparisons=comparisons,
        timing=timing,
        failures=failures,
        qualified_T_epsilon=False,
        interpretation="Precision experiment only. Production defaults and physics unchanged; normal intervals/rare-bin guards are screening diagnostics, not independent physics validation.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                groups=len(results),
                samples=sum(r["replicas"] for r in results),
                failures=len(failures),
                comparison_statuses={
                    status: sum(c["status"] == status for c in comparisons)
                    for status in ["passed", "failed", "unresolved"]
                },
            )
        )
    )


if __name__ == "__main__":
    main()
