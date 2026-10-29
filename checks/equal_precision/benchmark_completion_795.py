"""Measure issue #795 completion vs an unmodified, pinned scalar module.

Run one adapter process per (variant, scene, population, seed), with the
adapter's complete same-workload warmup. Only the measured request contributes
to the summary. The scalar baseline is loaded from an *unmodified* source file
through importlib, without rewriting or checking out the working tree.

The companion .completion.npz retains only positive-photon rows/vectors for
bitwise candidate-vs-baseline comparison; the adapter's .npz holds per-primary
source scores. Neither is reused during a timed request.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import runpy
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, adapter_args):
    from pyrite._backend import BACKEND
    from pyrite.montecarlo.transport import hard_radiative

    original_completion = hard_radiative.complete_hard_radiative_events
    target = original_completion
    source = Path(hard_radiative.__file__)
    if args.variant == "baseline":
        if not args.baseline_source:
            raise SystemExit("baseline requires --baseline-source (unmodified source)")
        source = args.baseline_source.resolve()
        package = importlib.import_module("pyrite.montecarlo.transport")
        name = package.__name__ + "._completion_795_baseline"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        target = module.complete_hard_radiative_events
    if "--output" not in adapter_args:
        raise SystemExit("pass --output PATH to the underlying pyrite_adapter.py")
    output = Path(adapter_args[adapter_args.index("--output") + 1])
    if output.exists():
        raise SystemExit(f"will not overwrite existing measurement: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.with_suffix(".completion.json")
    data_path = output.with_suffix(".completion.npz")
    if audit_path.exists() or data_path.exists():
        raise SystemExit("will not overwrite existing completion audit")
    captures, arrays = [], {}

    def timed_completion(segments, tables, seed):
        index = len(captures)
        rows = np.flatnonzero(np.asarray(segments["hard_radiative_k_eV"]) > 0.0)
        elements, counts = np.unique(
            np.asarray(segments["hard_radiative_Z"])[rows], return_counts=True
        )
        BACKEND.synchronize()
        start = time.perf_counter()
        target(segments, tables, seed)
        BACKEND.synchronize()
        elapsed = time.perf_counter() - start
        captures.append(
            dict(
                index=index,
                seconds=elapsed,
                photons=int(rows.size),
                rows=int(len(segments["hard_radiative_k_eV"])),
                photons_by_z={str(int(z)): int(n) for z, n in zip(elements, counts, strict=True)},
                allocator=BACKEND.allocator_stats(),
            )
        )
        arrays[f"rows_{index}"] = rows
        arrays[f"direction_{index}"] = np.asarray(segments["hard_radiative_direction"])[rows].copy()
        arrays[f"momentum_{index}"] = np.asarray(segments["hard_radiative_target_momentum_eV_c"])[
            rows
        ].copy()

    hard_radiative.complete_hard_radiative_events = timed_completion
    try:
        sys.argv = [sys.argv[0], *adapter_args]
        runpy.run_path(str(HERE / "pyrite_adapter.py"))["main"]()
    finally:
        hard_radiative.complete_hard_radiative_events = original_completion
    adapter = json.loads(output.read_text())
    # Each run_request invokes completion once per block, and pyrite_adapter
    # executes one complete warmup then the measured request.
    histories = int(adapter["histories"])
    block_size = int(adapter["batch_histories"])
    nblocks = (histories + block_size - 1) // block_size
    if len(captures) != 2 * nblocks:
        raise RuntimeError(f"expected {2 * nblocks} completions, got {len(captures)}")
    measured = captures[nblocks:]
    metadata = {
        "schema": "pyrite.issue-795.completion-benchmark.v1",
        "variant": args.variant,
        "scene": adapter["scene"]["id"] if "scene" in adapter else adapter.get("case_id"),
        "histories": histories,
        "seed": adapter["seed"],
        "core": adapter["core"] if "core" in adapter else adapter.get("arm"),
        "backend": BACKEND.name,
        "pyrite_source_sha256": digest_file(hard_radiative.__file__),
        "completion_source_sha256": digest_file(source),
        "adapter_source_sha256": digest_file(HERE / "pyrite_adapter.py"),
        "warmup": captures[:nblocks],
        "measured": measured,
        "measurement_completion_seconds": sum(s["seconds"] for s in measured),
        "measurement_photons": sum(s["photons"] for s in measured),
        "measurement_total_seconds": adapter["timings_s"].get(
            "request_with_reductions", adapter["timings_s"].get("transport", 0.0)
        ),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "allocator": BACKEND.allocator_stats(),
        "environment": {
            key: os.getenv(key)
            for key in ("PYRITE_MC_BACKEND", "PYRITE_FP64", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")
        },
    }
    np.savez_compressed(data_path, **arrays)
    audit_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "completion_seconds": metadata["measurement_completion_seconds"],
                "photons": metadata["measurement_photons"],
            }
        )
    )


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    left = json.loads(reference.with_suffix(".completion.json").read_text())
    right = json.loads(candidate.with_suffix(".completion.json").read_text())
    fields = ("scene", "histories", "seed", "core", "backend", "measurement_photons")
    for key in fields:
        if left[key] != right[key]:
            raise RuntimeError(f"comparison differs in {key}: {left[key]!r} vs {right[key]!r}")
    failures = []
    with (
        np.load(reference.with_suffix(".completion.npz")) as a,
        np.load(candidate.with_suffix(".completion.npz")) as b,
    ):
        if set(a.files) != set(b.files):
            failures.append("different completed block/field keys")
        for key in sorted(set(a.files) & set(b.files)):
            if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
                failures.append(f"{key}: dtype/shape changed")
            elif a[key].tobytes() != b[key].tobytes():
                differences = np.abs(a[key] - b[key])
                count = np.count_nonzero(a[key] != b[key])
                failures.append(
                    f"{key}: {count} different values; max abs={float(np.max(differences)):.17g}"
                )
    # Also check the unchanged measured per-primary scores: a regression in
    # history, transport energy or RNG cannot hide in equivalent vectors.
    with np.load(reference.with_suffix(".npz")) as a, np.load(candidate.with_suffix(".npz")) as b:
        if set(a.files) != set(b.files):
            failures.append("different adapter observable keys")
        for key in sorted(set(a.files) & set(b.files)):
            if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
                failures.append(f"adapter score {key}: dtype/shape changed")
            elif a[key].tobytes() != b[key].tobytes():
                failures.append(f"adapter score {key}: differs")
    result = {
        "baseline": str(reference),
        "candidate": str(candidate),
        "bitwise_equal": not failures,
        "differences": failures,
        "completion_speedup": (
            left["measurement_completion_seconds"] / right["measurement_completion_seconds"]
        ),
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("bitwise equivalence failed; investigate before accepting #795")


def summarize(directory):
    data = defaultdict(
        lambda: {
            "completion_s": [],
            "request_s": [],
            "photons": [],
            "rss_kib": [],
            "peak_pool_mib": [],
        }
    )
    for path in sorted(directory.glob("*.completion.json")):
        record = json.loads(path.read_text())
        key = (record["scene"], record["histories"], record["variant"])
        group = data[key]
        group["completion_s"].append(record["measurement_completion_seconds"])
        group["request_s"].append(record["measurement_total_seconds"])
        group["photons"].append(record["measurement_photons"])
        group["rss_kib"].append(record["peak_rss_kib"])
        group["peak_pool_mib"].append(record["allocator"].get("peak_mib"))
    rows = []
    for (scene, n, variant), group in sorted(data.items()):
        elapsed = group["completion_s"]
        photons = group["photons"]
        rows.append(
            {
                "scene": scene,
                "histories": n,
                "variant": variant,
                "repeats": len(elapsed),
                "median_completion_s": statistics.median(elapsed),
                "median_request_s": statistics.median(group["request_s"]),
                "median_photons": statistics.median(photons),
                "median_us_per_photon": statistics.median(
                    [1e6 * t / p for t, p in zip(elapsed, photons, strict=True) if p]
                )
                if any(photons)
                else None,
                "max_peak_rss_mib": max(group["rss_kib"]) / 1024,
                "max_peak_pool_mib": max(
                    (x for x in group["peak_pool_mib"] if x is not None), default=None
                ),
            }
        )
    indexed = {(row["scene"], row["histories"], row["variant"]): row for row in rows}
    for row in rows:
        if row["variant"] != "candidate":
            continue
        baseline = indexed.get((row["scene"], row["histories"], "baseline"))
        if baseline is not None:
            row["completion_speedup_vs_baseline"] = (
                baseline["median_completion_s"] / row["median_completion_s"]
            )
            row["request_speedup_vs_baseline"] = (
                baseline["median_request_s"] / row["median_request_s"]
            )
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("run")
    measure.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    measure.add_argument("--baseline-source", type=Path)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("directory", type=Path)
    args, rest = parser.parse_known_args()
    if args.command == "run":
        run(args, rest)
    elif rest:
        parser.error("unexpected arguments: " + " ".join(rest))
    elif args.command == "compare":
        compare(args.reference, args.candidate)
    else:
        summarize(args.directory)


if __name__ == "__main__":
    main()
