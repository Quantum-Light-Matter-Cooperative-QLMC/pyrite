"""Fresh-process, opt-in radiative precision diagnostics; production stays FP64."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path

from precision_source import VARIANTS, precision_source


def install_variant(variant, destination):
    package = importlib.import_module("pyrite.montecarlo.transport")
    name = package.__name__ + "._jit_radiative_device"
    if name in sys.modules:
        raise RuntimeError("precision variant must be installed before CUDA imports")
    original = Path(package.__file__).with_name("_jit_radiative_device.py")
    source = precision_source(original.read_text(encoding="utf-8"), variant)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    package._jit_radiative_device = module
    return {
        "reference_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "variant_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("audit", type=Path)
    parser.add_argument(
        "--precision-protocol",
        type=Path,
        default=Path(__file__).with_name("precision-experiment.json"),
    )
    parser.add_argument("--study-phase", choices=["preflight", "pilot", "heldout"], default="pilot")
    args, adapter_args = parser.parse_known_args()
    if "--core" not in adapter_args or adapter_args[adapter_args.index("--core") + 1] != "cuda":
        parser.error("precision experiment requires explicit --core cuda")
    source = install_variant(args.variant, args.audit.with_suffix(".source.py"))
    import cupy as cp
    import numpy as np
    from precision_checks import trajectory_checks

    from pyrite.montecarlo.transport import _jit_launch

    original_rad_args = _jit_launch._radiative_kernel_args
    table_dtypes = []

    def radiative_args(packed, segments):
        if packed is not None:
            dtype = cp.float64 if args.variant == "fp64" else cp.float32
            if args.variant in {"rad-boundary64-24", "rad-quotient32-24"}:
                packed = (*packed[:9], packed[9].astype(cp.float32, copy=False))
            else:
                packed = tuple(
                    a.astype(dtype, copy=False)
                    if isinstance(a, cp.ndarray) and a.dtype.kind == "f"
                    else a
                    for a in packed
                )
            table_dtypes.append(
                [str(a.dtype) for a in packed if isinstance(a, cp.ndarray) and a.dtype.kind == "f"]
            )
        return original_rad_args(packed, segments)

    _jit_launch._radiative_kernel_args = radiative_args
    factory_original = _jit_launch.make_cuda_transport_core
    launches = []

    def factory(*f_args, **f_kwargs):
        original, module = factory_original(*f_args, **f_kwargs)

        def core(*core_args):
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            original(*core_args)
            cp.cuda.Device().synchronize()
            seconds = time.perf_counter() - start
            counts = cp.asnumpy(core_args[7][0])
            launches.append(
                {
                    "primaries": int(core_args[0][1]),
                    "capacity": int(core_args[0][2]),
                    "max_segments": int(counts.max()),
                    "replayed": bool(counts.max() > core_args[0][2]),
                    "wall_s": seconds,
                }
            )

        return core, module

    _jit_launch.make_cuda_transport_core = factory
    namespace = runpy.run_path(str(Path(__file__).with_name("pyrite_adapter.py")))
    original_score = namespace["score_segments"]
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
    scene = next(s for s in manifest["scenes"] if s["id"] == adapter_args[0])
    cutoff = (
        float(adapter_args[adapter_args.index("--cutoff-ev") + 1])
        if "--cutoff-ev" in adapter_args
        else 1000.0
    )
    blocks = []

    def score(result, count):
        try:
            extra, checks = trajectory_checks(result, count, scene["energy_keV"], cutoff)
        except ValueError as error:
            fields = [
                "electron_id",
                "event_kind",
                "E_start_keV",
                "E_end_keV",
                "hard_radiative_k_eV",
                "v_hat",
                "L_ang",
            ]
            np.savez_compressed(
                args.audit.with_suffix(".failed-block.npz"), **{k: result[k] for k in fields}
            )
            blocks.append({"primaries": count, "failed": str(error)})
            raise
        digest = hashlib.sha256()
        for key, value in sorted(result.items()):
            if isinstance(value, np.ndarray):
                digest.update(key.encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        blocks.append(
            {
                "primaries": count,
                "segments": int(result["electron_id"].size),
                "arrays_sha256": digest.hexdigest(),
                **checks,
            }
        )
        return {**original_score(result, count), **extra}

    namespace["main"].__globals__["score_segments"] = score
    sys.argv = [sys.argv[0], *adapter_args]
    try:
        namespace["main"]()
        sample_path = Path(adapter_args[adapter_args.index("--output") + 1])
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        from paired_equivalence import coupling_record

        sample.update(
            study_phase=args.study_phase,
            primary_coupling=coupling_record(sample["seed"], sample["histories"]),
            schema="pyrite.equal-precision.precision-sample.v1",
            precision_variant=args.variant,
            precision_source=source,
            precision_protocol_sha256=hashlib.sha256(
                args.precision_protocol.read_bytes()
            ).hexdigest(),
        )
        sample_path.write_text(
            json.dumps(sample, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    finally:
        sys.setprofile(None)
        codes = list(_jit_launch._transport_kernel.cached_codes.values())
        for index, code in enumerate(codes):
            args.audit.with_suffix(f".kernel-{index}.cu").write_text(code, encoding="utf-8")
        resources = []
        for kernel, _ in _jit_launch._transport_kernel._cache.values():
            resources.append(
                {
                    name: cp.cuda.driver.funcGetAttribute(code, kernel.ptr)
                    for name, code in [("registers", 4), ("local_bytes", 3)]
                }
            )
        args.audit.write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "source": source,
                    "launches": launches,
                    "blocks": blocks,
                    "radiative_table_dtypes": table_dtypes,
                    "compiled_resources": resources,
                    "generated_sources": len(codes),
                    "interpretation": "Mixed-precision diagnostic. Trajectory state, RNG, stopping and elastic helpers remain FP64. No qualified speedup or accuracy claim.",
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
