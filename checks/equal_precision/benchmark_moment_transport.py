"""Isolated source-specialized CUDA transport pilot; production remains exact.

Only the first four total-moment call sites are replaced. The final event
rate, element choice, conditional energy/angle samplers and RNG code retain
the committed exact source. Refuse a changed call-site inventory.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import runpy
import sys
import time
from dataclasses import replace
from pathlib import Path


def install_kernel(audit):
    package = importlib.import_module("pyrite.montecarlo.transport")
    name = package.__name__ + "._jit_kernel"
    if name in sys.modules:
        raise RuntimeError("install diagnostic kernel before CUDA imports")
    helper_name = package.__name__ + "._moment_transport_device"
    helper_path = Path(__file__).with_name("moment_transport_device.py")
    helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper
    helper_spec.loader.exec_module(helper)
    original = Path(package.__file__).with_name("_jit_kernel.py").read_text()
    if original.count("_rad_layer_moment(") != 5:
        raise ValueError("radiative total-moment call-site inventory changed")
    source = original.replace("_rad_layer_moment(", "lookup_layer_moment(", 4)
    source += "\nfrom ._moment_transport_device import lookup_layer_moment\n"
    destination = audit.with_suffix(".kernel.py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    package._jit_kernel = module
    return {
        "reference_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "frequent_calls_replaced": 4,
        "exact_event_total_calls_retained": 1,
    }


def install_tables(rtol, audit):
    import cupy as cp
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
            counts = np.array([g.size for g in prepared.energy_eV], dtype=np.int32)
            starts = np.r_[0, np.cumsum(counts[:-1])]
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            device = cp.asarray(
                np.r_[
                    packed[-1],
                    starts,
                    counts,
                    *prepared.energy_eV,
                    *prepared.log_energy,
                    np.concatenate(prepared.moments).ravel(),
                ]
            )
            cp.cuda.Device().synchronize()
            audit.append(
                {
                    "digest": prepared.digest,
                    "identity": json.loads(prepared.identity_json),
                    "build_s": build_s,
                    "upload_s": time.perf_counter() - start,
                    "device_bytes": device.nbytes,
                    "layers": len(compositions),
                }
            )
            cache[key] = device
        return (*packed[:-1], cache[key])

    def result_fields(result, photons, z, tables, cutoff, seed, plasma=None):
        # Only true layer plasma entries belong in the physical record.
        layers = audit[-1]["layers"]
        original_result(result, photons, z, tables, cutoff, seed, cp.asnumpy(plasma[:layers]))
        result["radiative"]["diagnostic_moment_table"] = audit[-1]["digest"]

    hard_radiative.radiative_core_args = arguments
    hard_radiative.add_radiative_result_fields = result_fields


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["exact", "table", "refined"])
    parser.add_argument("audit", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--core", choices=["cuda"], required=True)
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
    source = {} if args.variant == "exact" else install_kernel(args.audit)
    package = importlib.import_module("pyrite.montecarlo.transport")
    transport_dir = Path(package.__file__).parent
    source.setdefault(
        "reference_sha256",
        hashlib.sha256((transport_dir / "_jit_kernel.py").read_bytes()).hexdigest(),
    )
    source["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source["protocol_sha256"] = hashlib.sha256(
        Path(__file__).with_name("moment-policy-788.json").read_bytes()
    ).hexdigest()
    source["exact_sampler_sha256"] = hashlib.sha256(
        (transport_dir / "_jit_radiative_device.py").read_bytes()
    ).hexdigest()
    preparations = []
    if args.variant != "exact":
        install_tables(2e-4 if args.variant == "table" else 1e-4, preparations)
    import cupy as cp
    import numpy as np
    from precision_checks import trajectory_checks

    from pyrite.montecarlo import transport
    from pyrite.montecarlo.transport import _jit_launch
    from pyrite.perf.precision import grouped_scores

    # A caller-owned learned capacity, identical policy for baseline/candidate.
    capacity = 512
    simulate = transport.simulate_trajectories

    def simulated(**kwargs):
        kwargs["per_electron_config"] = replace(
            kwargs["per_electron_config"], seg_capacity=capacity
        )
        return simulate(**kwargs)

    transport.simulate_trajectories = simulated
    factory_original = _jit_launch.make_cuda_transport_core
    launches = []

    def factory(*f_args, **f_kwargs):
        f_kwargs["config"] = replace(_jit_launch.DEFAULT_TRANSPORT_KERNEL_CONFIG, nthreads=32)
        original, module = factory_original(*f_args, **f_kwargs)

        def core(*core_args):
            nonlocal capacity
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            original(*core_args)
            cp.cuda.Device().synchronize()
            seconds = time.perf_counter() - start
            counts = cp.asnumpy(core_args[7][0])
            cap = int(core_args[0][2])
            capacity = max(capacity, int(np.ceil(1.25 * counts.max())))
            launches.append(
                {
                    "primaries": int(core_args[0][1]),
                    "capacity": cap,
                    "segments": int(counts.sum()),
                    "max_segments": int(counts.max()),
                    "replayed": bool(counts.max() > cap),
                    "wall_s": seconds,
                    "scratch_bytes": sum(a.nbytes for a in core_args[6]),
                }
            )

        return core, module

    _jit_launch.make_cuda_transport_core = factory
    namespace = runpy.run_path(str(Path(__file__).with_name("pyrite_adapter.py")))
    _, scene, _ = namespace["load_scene"](adapter_args[0])
    resolved_cutoff = (
        args.cutoff_ev if args.cutoff_ev is not None else scene["hard_photon_cutoff_eV"]
    )
    score_original = namespace["score_segments"]
    blocks = []

    def score(result, count):
        extra, checks = trajectory_checks(result, count, scene["energy_keV"], resolved_cutoff)
        ids = np.asarray(result["electron_id"])
        photons = np.asarray(result["hard_radiative_k_eV"])
        extra["hard_count"] = grouped_scores(ids, (photons > 0).astype(float), count)
        extra["path_length"] = grouped_scores(ids, np.asarray(result["L_ang"]), count)
        digest = hashlib.sha256()
        for key, array in sorted(result.items()):
            if isinstance(array, np.ndarray):
                digest.update(key.encode())
                digest.update(array.tobytes())
        blocks.append(
            {
                "primaries": count,
                "arrays_sha256": digest.hexdigest(),
                "checks": checks,
                "launch_end": len(launches),
            }
        )
        return score_original(result, count) | extra

    namespace["main"].__globals__["score_segments"] = score
    sys.argv = [sys.argv[0], *adapter_args]
    namespace["main"]()
    sample_path = Path(adapter_args[adapter_args.index("--output") + 1])
    sample = json.loads(sample_path.read_text())
    replay = None
    if args.replay:
        first_blocks = list(blocks)
        alternate = list(adapter_args)
        alternate[alternate.index("--output") + 1] = str(sample_path.with_suffix(".replay.json"))
        sys.argv = [sys.argv[0], *alternate]
        namespace["main"]()
        second_blocks = blocks[len(first_blocks) :]
        # Each adapter invocation performs one complete warmup, then measurement.
        left = [b["arrays_sha256"] for b in first_blocks[len(first_blocks) // 2 :]]
        right = [b["arrays_sha256"] for b in second_blocks[len(second_blocks) // 2 :]]
        if left != right:
            raise ValueError("within-policy same-seed replay changed trajectory arrays")
        replay = {"passed": True, "blocks": len(left), "hashes": left}
    sample.update(
        schema="radiative-moment-transport-diagnostic-v1",
        moment_variant=args.variant,
        moment_source=source,
        preparations=preparations,
        same_seed_replay=replay,
    )
    sample_path.write_text(json.dumps(sample, indent=2) + "\n")
    resources = [
        {
            "registers_per_thread": cp.cuda.driver.funcGetAttribute(4, k.ptr),
            "local_bytes_per_thread": cp.cuda.driver.funcGetAttribute(3, k.ptr),
        }
        for k, _ in _jit_launch._transport_kernel._cache.values()
    ]
    args.audit.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "source": source,
                "preparations": preparations,
                "launches": launches,
                "blocks": blocks,
                "resources": resources,
                "threads_per_block": 32,
                "same_seed_replay": replay,
                "scope": "diagnostic total-moment replacement only; exact conditional sampler and RNG source retained",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
