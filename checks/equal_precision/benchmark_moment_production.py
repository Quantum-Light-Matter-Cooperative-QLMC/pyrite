"""#788 production-path moment qualification and timing on CPU or CUDA.

Unlike the retired pre-#805 diagnostic shims, nothing here rewrites a kernel:
``exact`` and ``table`` select the shipped ``radiative_moments`` policies, so
every variant runs the same committed source. ``refined`` is the shipped
table path with the protocol's tighter ``rtol=1e-4`` preparation, the only
patched symbol. Run each variant in its own process with matched scene/seed.
Output keeps the diagnostic record schema read by the existing analyzers.
"""

import argparse
import hashlib
import importlib
import json
import runpy
import sys
import time
from pathlib import Path

HARNESS = "radiative-moment-production-v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def install_preparation(variant, audit):
    """Record each real preparation; ``refined`` tightens only the tolerance."""
    from pyrite.montecarlo.transport import radiative_moment_transport as transport
    from pyrite.montecarlo.transport.radiative_moments import MomentPolicy

    original = transport.prepare_radiative_moments
    policy = MomentPolicy(rtol=1e-4) if variant == "refined" else MomentPolicy()

    def prepare(*args, **kwargs):
        start = time.perf_counter()
        prepared = original(*args, policy=policy, **kwargs)
        audit.append(
            {
                "digest": prepared.digest,
                "identity": json.loads(prepared.identity_json),
                "held_out_max_error_fraction": list(prepared.max_error_fraction),
                "build_s": time.perf_counter() - start,
                "layers": len(prepared.energy_eV),
                "nodes": [int(g.size) for g in prepared.energy_eV],
            }
        )
        return prepared

    transport.prepare_radiative_moments = prepare


def install_cuda_timing(launches, kernels):
    """Synchronized wall time and launch work per CUDA core call.

    Records the ``run`` tuple's electron count and per-electron row capacity,
    so launched electrons beyond the primaries expose capacity replays, and
    collects every specialized kernel variant the shipped launch selects.
    """
    import cupy as cp

    from pyrite.montecarlo.transport import _jit_launch

    factory = _jit_launch.make_cuda_transport_core
    variant = _jit_launch.transport_kernel_variant

    def recorded_variant(switches, *v_args, **v_kwargs):
        kernel = variant(switches, *v_args, **v_kwargs)
        kernels.setdefault(repr(tuple(switches)) + repr(v_args), kernel)
        return kernel

    def timed_factory(*f_args, **f_kwargs):
        core, module = factory(*f_args, **f_kwargs)
        config = f_kwargs.get("config", f_args[0] if f_args else None)
        threads = int(
            getattr(config, "nthreads", _jit_launch.DEFAULT_TRANSPORT_KERNEL_CONFIG.nthreads)
        )

        def timed(*args, **kwargs):
            _, e_count, cap, _ = args[0]
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            core(*args, **kwargs)
            cp.cuda.Device().synchronize()
            launches.append(
                {
                    "wall_s": time.perf_counter() - start,
                    "electrons": int(e_count),
                    "row_capacity": int(cap),
                    "continuation": kwargs.get("continuation") is not None,
                    "threads_per_block": threads,
                }
            )

        return timed, module

    _jit_launch.make_cuda_transport_core = timed_factory
    _jit_launch.transport_kernel_variant = recorded_variant


def cuda_resources(kernels, threads=32):
    """Registers, local memory and occupancy of the launched kernel variants."""
    import cupy as cp
    from cupy_backends.cuda.api import driver

    per_sm = cp.cuda.Device().attributes["MaxThreadsPerMultiProcessor"]
    resources = []
    try:
        for switches, kernel in kernels.items():
            for function, _ in kernel._cache.values():
                blocks = driver.occupancyMaxActiveBlocksPerMultiprocessor(function.ptr, threads, 0)
                resources.append(
                    {
                        "switches": switches,
                        "registers_per_thread": driver.funcGetAttribute(
                            driver.CU_FUNC_ATTRIBUTE_NUM_REGS, function.ptr
                        ),
                        "local_bytes_per_thread": driver.funcGetAttribute(
                            driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function.ptr
                        ),
                        "threads_per_block": threads,
                        "theoretical_occupancy": blocks * threads / per_sm,
                    }
                )
    except (AttributeError, TypeError, ValueError) as error:
        # Resource introspection is a reported limitation, never a run failure.
        return [{"unavailable": repr(error)}]
    return resources or [{"unavailable": "no specialized kernel variant was launched"}]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["exact", "table", "refined"])
    parser.add_argument("audit", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--core", choices=["per-electron", "cuda"], required=True)
    parser.add_argument("--arm", choices=["source"], default="source")
    parser.add_argument("--cutoff-ev", type=float)
    args, adapter_args = parser.parse_known_args()
    adapter_args.extend(["--core", args.core, "--arm", args.arm])
    if args.cutoff_ev is not None:
        adapter_args.extend(["--cutoff-ev", str(args.cutoff_ev)])
    mode = "exact" if args.variant == "exact" else "table"
    adapter_args.extend(["--radiative-moments", mode])
    return args, adapter_args


def source_record(core):
    package = importlib.import_module("pyrite.montecarlo.transport")
    transport_dir = Path(package.__file__).parent
    owner = "_jit_kernel.py" if core == "cuda" else "cores.py"
    sampler = "_jit_radiative_device.py" if core == "cuda" else "_jit_radiative.py"
    lookup = "_jit_moment_lookup_device.py" if core == "cuda" else "_jit_moment_lookup.py"
    return {
        "reference_sha256": _sha((transport_dir / owner).read_bytes()),
        "lookup_sha256": _sha((transport_dir / lookup).read_bytes()),
        "packer_sha256": _sha((transport_dir / "radiative_moment_transport.py").read_bytes()),
        "wrapper_sha256": _sha(Path(__file__).read_bytes()),
        "protocol_sha256": _sha(Path(__file__).with_name("moment-policy-788.json").read_bytes()),
        "exact_sampler_sha256": _sha((transport_dir / sampler).read_bytes()),
        "path": "production",
    }


def main():
    args, adapter_args = parse_args()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    source = source_record(args.core)
    preparations, launches, kernels = [], [], {}
    if args.variant != "exact":
        install_preparation(args.variant, preparations)
    if args.core == "cuda":
        install_cuda_timing(launches, kernels)

    import numpy as np
    from paired_equivalence import coupling_record
    from precision_checks import trajectory_checks

    from pyrite.perf.precision import grouped_scores

    namespace = runpy.run_path(str(Path(__file__).with_name("pyrite_adapter.py")))
    _, scene, _ = namespace["load_scene"](adapter_args[0])
    cutoff = args.cutoff_ev if args.cutoff_ev is not None else scene["hard_photon_cutoff_eV"]
    original_score = namespace["score_segments"]
    blocks, models = [], set()

    def score(result, count):
        extra, checks = trajectory_checks(result, count, scene["energy_keV"], cutoff)
        ids = np.asarray(result["electron_id"])
        photons = np.asarray(result["hard_radiative_k_eV"])
        extra["hard_count"] = grouped_scores(ids, (photons > 0).astype(float), count)
        extra["path_length"] = grouped_scores(ids, np.asarray(result["L_ang"]), count)
        radiative = result.get("radiative", {})
        models.add((radiative.get("moment_model"), radiative.get("moment_table_sha256")))
        digest = hashlib.sha256()
        for key, array in sorted(result.items()):
            if isinstance(array, np.ndarray):
                digest.update(key.encode())
                digest.update(array.tobytes())
        blocks.append(
            {
                "primaries": count,
                "segments": int(ids.size),
                "arrays_sha256": digest.hexdigest(),
                "checks": checks,
                "launch_end": len(launches),
            }
        )
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
        later = blocks[len(prior) :]
        second = [b["arrays_sha256"] for b in later[len(later) // 2 :]]
        if first != second or not first:
            raise ValueError("within-policy same-seed replay changed trajectory arrays")
        replay = {"passed": True, "blocks": len(first), "hashes": first}
    expected = "exact" if args.variant == "exact" else "radiative-moments-v1"
    observed = sorted({str(model) for model, _ in models})
    if len(observed) != 1 or not observed[0].startswith(expected):
        raise ValueError(f"result moment policy {observed} does not match variant {args.variant}")
    metadata = sample.get("provenance", {}) | sample
    sample.update(
        schema="radiative-moment-transport-diagnostic-v1",
        harness=HARNESS,
        moment_variant=args.variant,
        moment_source=source,
        moment_results=sorted([list(m) for m in models], key=str),
        preparations=preparations,
        same_seed_replay=replay,
        primary_coupling=coupling_record(metadata["seed"], metadata["histories"]),
    )
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")
    audit = {
        "harness": HARNESS,
        "variant": args.variant,
        "core": args.core,
        "source": source,
        "preparations": preparations,
        "launches": launches,
        "blocks": blocks,
        "same_seed_replay": replay,
        "scope": "shipped exact/table policies; exact event-row total, conditional sampler, photon debit and RNG unchanged",
        "limitations": (
            "per-call CUDA wall time and launched electrons/row capacity; launched "
            "electrons beyond the primaries are capacity replays (full-history schedule). "
            "Preparations are counted per process; the host cache reuses identical tables."
        ),
    }
    if args.core == "cuda":
        threads = {launch["threads_per_block"] for launch in launches} or {32}
        audit["resources"] = cuda_resources(kernels, max(threads))
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
