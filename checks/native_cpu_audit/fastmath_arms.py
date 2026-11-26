"""Complete-case fastmath / reduced-accuracy arms for #1469 (diagnostic only).

Runs one ``checks/native_cpu_audit.py`` case in one process under one arm:

- ``--fastmath FLAGSET`` injects that Numba fastmath set into EVERY PyRITE
  ``@njit`` (``numba.njit`` is wrapped before PyRITE is imported). This is an
  upper bound on what per-kernel flags could give, not a production variant:
  it also compiles ``vector_sin`` with fastmath, which its contract forbids.
- ``--reducer EDIT[:FLAGSET]`` swaps the node-window reducer for a
  ``fast_variants.build_reducer`` variant (production source unchanged).
- ``--asm`` records, per active dispatcher, packed vs scalar FP64 arithmetic in
  the generated assembly and the math calls in the LLVM IR. It needs a cold
  Numba cache (cache-loaded overloads carry no IR); the sbatch arm gives it one.
- ``--arrays NPZ`` keeps the case's float arrays for spectrum-change bounds.

Every arm needs its own ``NUMBA_CACHE_DIR``: Numba's cache index does not key
on fastmath flags or on replaced globals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAGSETS = {
    "strict": None,
    "contract": {"contract"},
    "afn": {"afn"},
    "reassoc": {"reassoc"},
    "arcp_contract": {"arcp", "contract"},
    "no_nnan_ninf": {"afn", "arcp", "contract", "reassoc", "nsz"},
    "true": True,
}
_PACKED = re.compile(r"\tv(?:add|sub|mul|div|sqrt|max|min|f[n]?m(?:add|sub)\w*)pd\t.*%[yz]mm")
_SCALAR = re.compile(r"\tv(?:add|sub|mul|div|sqrt|max|min|f[n]?m(?:add|sub)\w*)sd\t")
_MATH = re.compile(
    r"call [^@]*@\"?((?:llvm\.)?(?:sin|cos|tan|exp|exp2|expm1|log|log2|log10|log1p|pow|sqrt"
    r"|atan2?|asin|acos|sinh|cosh|tanh|cbrt|hypot|erf|erfc|lgamma|tgamma)(?:\.f64)?"
    r"|__svml_\w+|numba_\w*(?:gamma|erf)\w*)\b"
)


def _patch_njit(flags):
    import numba

    original = numba.njit

    def njit(*args, **kwargs):
        kwargs["fastmath"] = flags
        return original(*args, **kwargs)

    numba.njit = njit


def _asm_stats():
    import gc

    from numba.core.dispatcher import Dispatcher

    rows = []
    for obj in gc.get_objects():
        if not issubclass(type(obj), Dispatcher) or not obj.overloads:
            continue
        f = obj.py_func
        if not str(f.__module__).startswith("pyrite"):
            continue
        for sig in obj.signatures:
            try:
                asm, llvm = obj.inspect_asm(sig), obj.inspect_llvm(sig)
            except Exception as exc:  # cache-loaded overload: no IR
                rows.append({"kernel": f"{f.__module__}.{f.__qualname__}", "error": repr(exc)})
                continue
            packed, scalar = len(_PACKED.findall(asm)), len(_SCALAR.findall(asm))
            rows.append(
                {
                    "kernel": f"{f.__module__}.{f.__qualname__}",
                    "signature": str(sig),
                    "parallel": bool(obj.targetoptions.get("parallel")),
                    "packed_fp64_arith": packed,
                    "scalar_fp64_arith": scalar,
                    "zmm": "%zmm" in asm,
                    "math_calls": sorted(set(_MATH.findall(llvm))),
                }
            )
    return sorted(rows, key=lambda r: r["kernel"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--fastmath", choices=FLAGSETS, default="strict")
    parser.add_argument("--reducer", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, default=None)
    parser.add_argument("--asm", action="store_true")
    args = parser.parse_args()
    if FLAGSETS[args.fastmath] is not None:
        _patch_njit(FLAGSETS[args.fastmath])
    spec = importlib.util.spec_from_file_location("audit", HERE.parent / "native_cpu_audit.py")
    audit = importlib.util.module_from_spec(spec)
    sys.modules["audit"] = audit
    spec.loader.exec_module(audit)
    if args.reducer:
        sys.path.insert(0, str(HERE))
        from fast_variants import FLAGSETS as REDUCER_FLAGS
        from fast_variants import build_reducer

        from pyrite.montecarlo.spectrum.cxr import _node_window

        edit, _, flags = args.reducer.partition(":")
        _node_window._sincsq_window_reduce = build_reducer(edit, REDUCER_FLAGS[flags or "strict"])
    kept = {}
    if args.arrays is not None:
        import numpy as np

        real_digest = audit.digest

        def digest(array):
            a = np.asarray(array)
            if a.dtype.kind in "fc":
                kept[f"a{len(kept):03d}"] = a.copy()
            return real_digest(array)

        audit.digest = digest
    audit_args = argparse.Namespace(case=args.case, repeats=args.repeats)
    report = audit.measure(audit_args)
    report["arm"] = {"fastmath": args.fastmath, "reducer": args.reducer}
    import numba

    report["arm"]["numba"] = numba.__version__
    report["arm"]["using_svml"] = bool(numba.config.USING_SVML)
    if args.asm:
        report["asm"] = _asm_stats()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.arrays is not None:
        import numpy as np

        # The profiled call is the last digest pass; keep only its arrays.
        n = len(kept) // (args.repeats + 2)
        last = dict(list(kept.items())[-n:]) if n else {}
        np.savez_compressed(args.arrays, **last)
    print(
        json.dumps(
            {
                "case": args.case,
                "arm": report["arm"],
                "wall_s": [s["wall_s"] for s in report["samples"]],
            }
        )
    )


if __name__ == "__main__":
    main()
