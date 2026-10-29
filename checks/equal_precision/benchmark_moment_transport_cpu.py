"""Isolated #788 CPU per-electron transport pilot; production stays exact.

The diagnostic rewrites only three frequent CPU total-moment call sites; the
fourth (row-end hard event) stays exact. The source and call inventory are
hashed and checked at runtime. Never import this shim from production code.
Run exact/table/refined in separate processes, with the same scene and seed.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def install_cpu_core(audit):
    """Compile a private, audited core with the event sampler left untouched."""
    package = importlib.import_module("pyrite.montecarlo.transport")
    from pyrite.montecarlo.transport import host_pass

    owner = Path(package.__file__).with_name("cores.py")
    original = owner.read_text()
    name = "radiative_layer_moments_scalar("
    if original.count(name) != 4:
        raise RuntimeError("CPU radiative moment call-site inventory drifted")
    helper_name = package.__name__ + "._moment_transport_cpu_device"
    helper_path = Path(__file__).with_name("moment_transport_cpu_device.py")
    helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper
    helper_spec.loader.exec_module(helper)
    source = original.replace(name, "lookup_layer_moment_cpu(", 3)
    source += "\nfrom ._moment_transport_cpu_device import lookup_layer_moment_cpu\n"
    # Stable content-addressed path is essential for Numba's disk cache;
    # an audit-specific filename would force a full recompilation per seed.
    candidate_path = (
        Path.home() / ".cache" / "pyrite" / "moment-cpu-788" / f"cores-{_sha(source.encode())}.py"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        if candidate_path.read_text() != source:
            raise RuntimeError("diagnostic CPU kernel cache path contains different source")
    else:
        candidate_path.write_text(source)
    module_name = package.__name__ + "._moment_transport_cpu_core"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = candidate
    spec.loader.exec_module(candidate)
    select_exact = host_pass.exact_ungrooved_core

    def select_core(*, per_electron, inelastic, radiative):
        if per_electron and radiative:
            return candidate.exact_ungrooved_core(
                per_electron=True, inelastic=inelastic, radiative=True
            )
        return select_exact(per_electron=per_electron, inelastic=inelastic, radiative=radiative)

    host_pass.exact_ungrooved_core = select_core
    return {
        "reference_sha256": _sha(original.encode()),
        "candidate_sha256": _sha(source.encode()),
        "helper_sha256": _sha(helper_path.read_bytes()),
        "frequent_calls_replaced": 3,
        "exact_event_total_calls_retained": 1,
    }


def install_tables(rtol, audit):
    """Prepare once per physical table identity, retaining the exact row-end plasma."""
    import numpy as np

    from pyrite.montecarlo.transport import hard_radiative
    from pyrite.montecarlo.transport.radiative_moments import (
        MomentPolicy,
        prepare_radiative_moments,
    )

    original = hard_radiative.radiative_core_args
    original_result = hard_radiative.add_radiative_result_fields
    cache = {}

    def arguments(compositions, tables, cutoff, electron_cutoffs, energies, *tail):
        packed = original(compositions, tables, cutoff, electron_cutoffs, energies, *tail)
        lo = float(np.min(electron_cutoffs)) * 1e3
        hi = float(np.max(energies)) * 1e3
        key = (
            tuple(tuple(c) for c in compositions),
            cutoff,
            lo,
            hi,
            tuple((k, id(t), t.digest) for k, t in sorted(tables.items())),
        )
        if key not in cache:
            start = time.perf_counter()
            prepared = prepare_radiative_moments(
                compositions, tables, lo, hi, cutoff, policy=MomentPolicy(rtol=rtol)
            )
            build_s = time.perf_counter() - start
            counts = np.asarray([g.size for g in prepared.energy_eV], dtype=np.int64)
            starts = np.r_[0, np.cumsum(counts[:-1])]
            # Prefix is the *physical* plasma energy, not an approximate input.
            host = np.ascontiguousarray(
                np.r_[
                    packed[-1],
                    starts,
                    counts,
                    *prepared.energy_eV,
                    *prepared.log_energy,
                    np.concatenate(prepared.moments).ravel(),
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(host)):
                raise ValueError("nonfinite CPU radiative moment packing")
            audit.append(
                {
                    "digest": prepared.digest,
                    "identity": json.loads(prepared.identity_json),
                    "held_out_max_error_fraction": list(prepared.max_error_fraction),
                    "build_s": build_s,
                    "upload_s": 0.0,
                    "host_bytes": host.nbytes,
                    "layers": len(compositions),
                }
            )
            cache[key] = host
        return (*packed[:-1], cache[key])

    def result_fields(result, photons, z, tables, cutoff, seed, plasma=None):
        # The final event-row sampler must receive exactly the physical plasma.
        n_layers = audit[-1]["layers"]
        original_result(result, photons, z, tables, cutoff, seed, plasma[:n_layers])
        result["radiative"]["diagnostic_moment_table"] = audit[-1]["digest"]

    hard_radiative.radiative_core_args = arguments
    hard_radiative.add_radiative_result_fields = result_fields


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["exact", "table", "refined"])
    parser.add_argument("audit", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--core", choices=["per-electron"], required=True)
    parser.add_argument("--arm", choices=["source"], default="source")
    parser.add_argument("--cutoff-ev", type=float)
    args, adapter_args = parser.parse_known_args()
    adapter_args.extend(["--core", args.core, "--arm", args.arm])
    if args.cutoff_ev is not None:
        adapter_args.extend(["--cutoff-ev", str(args.cutoff_ev)])
    return args, adapter_args


def main():
    args, adapter_args = parse_args()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    source = {} if args.variant == "exact" else install_cpu_core(args.audit)
    package = importlib.import_module("pyrite.montecarlo.transport")
    transport_dir = Path(package.__file__).parent
    source.setdefault("reference_sha256", _sha((transport_dir / "cores.py").read_bytes()))
    source["wrapper_sha256"] = _sha(Path(__file__).read_bytes())
    source["protocol_sha256"] = _sha(
        Path(__file__).with_name("moment-policy-788.json").read_bytes()
    )
    source["exact_sampler_sha256"] = _sha((transport_dir / "_jit_radiative.py").read_bytes())
    preparations = []
    if args.variant != "exact":
        install_tables(2e-4 if args.variant == "table" else 1e-4, preparations)

    import numpy as np
    from paired_equivalence import coupling_record
    from precision_checks import trajectory_checks

    from pyrite.perf.precision import grouped_scores

    namespace = runpy.run_path(str(Path(__file__).with_name("pyrite_adapter.py")))
    _, scene, _ = namespace["load_scene"](adapter_args[0])
    cutoff = args.cutoff_ev if args.cutoff_ev is not None else scene["hard_photon_cutoff_eV"]
    original_score = namespace["score_segments"]
    blocks = []

    def score(result, count):
        extra, checks = trajectory_checks(result, count, scene["energy_keV"], cutoff)
        ids = np.asarray(result["electron_id"])
        photons = np.asarray(result["hard_radiative_k_eV"])
        extra["hard_count"] = grouped_scores(ids, (photons > 0).astype(float), count)
        extra["path_length"] = grouped_scores(ids, np.asarray(result["L_ang"]), count)
        digest = hashlib.sha256()
        for key, array in sorted(result.items()):
            if isinstance(array, np.ndarray):
                digest.update(key.encode())
                digest.update(array.tobytes())
        blocks.append({"primaries": count, "arrays_sha256": digest.hexdigest(), "checks": checks})
        return original_score(result, count) | extra

    namespace["main"].__globals__["score_segments"] = score
    sys.argv = [sys.argv[0], *adapter_args]
    namespace["main"]()
    sample_path = Path(adapter_args[adapter_args.index("--output") + 1])
    sample = json.loads(sample_path.read_text())
    replay = None
    if args.replay:
        prior = list(blocks)
        alternate = list(adapter_args)
        alternate[alternate.index("--output") + 1] = str(sample_path.with_suffix(".replay.json"))
        sys.argv = [sys.argv[0], *alternate]
        namespace["main"]()
        # Each complete adapter invocation runs one warmup and one measurement.
        first = [b["arrays_sha256"] for b in prior[len(prior) // 2 :]]
        second_blocks = blocks[len(prior) :]
        second = [b["arrays_sha256"] for b in second_blocks[len(second_blocks) // 2 :]]
        if first != second or not first:
            raise ValueError("CPU within-policy same-seed replay changed trajectory arrays")
        replay = {"passed": True, "blocks": len(first), "hashes": first}
    metadata = sample.get("provenance", {}) | sample
    sample.update(
        schema="radiative-moment-transport-diagnostic-v1",
        moment_variant=args.variant,
        moment_source=source,
        preparations=preparations,
        same_seed_replay=replay,
        primary_coupling=coupling_record(metadata["seed"], metadata["histories"]),
    )
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")
    args.audit.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "core": "per-electron",
                "source": source,
                "preparations": preparations,
                "blocks": blocks,
                "same_seed_replay": replay,
                "scope": "CPU-only diagnostic; exact event-row total, conditional sampler, photon debit and RNG unchanged",
                "limitations": "cold JIT and host allocation measured only by adapter; no profiler register data on CPU",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
