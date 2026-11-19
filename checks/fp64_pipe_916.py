"""Hot FP64 sources of the exact CUDA transport kernel (#916).

``run OUT ARGS...``
    Runs ``checks/streamed_block_sizing.py ARGS...`` (the #907 scene) with
    every transport-kernel variant recompiled through
    :mod:`pyrite.montecarlo.transport._jit_rawmodule` under
    ``KernelBuild(lineinfo=True)``. ``--generate-line-info`` adds only debug
    sections; the SASS is unchanged, which ``identity`` checks. Writes each
    variant's generated source and NVRTC options to ``OUT``. Run it under
    ``ncu`` (see ``fp64_pipe_916.sbatch``) with ``CUPY_CACHE_SAVE_CUDA_SOURCE=1``
    so the profiler can resolve the generated source.

``scene-identity [N]``
    The #907 scene's joined transport at N and 2N histories: every segment
    array hashed, line-info build against ``cupyx.jit``, plus registers.

``scene-digest [N...] [--observables OUT.json]``
    #907-scene segment digests of this checkout's production kernel (joined
    transport, no scoring), for bitwise comparison of a kernel change across
    checkouts; optionally per-array transport observables.

``compare-observables BASE.json NEW.json``
    Transport-only comparison of two ``scene-digest --observables`` files
    (an output-changing variant): per-array mean change and z-score.

``identity``
    #831 W 1 MeV anchor (SBETHE physics), small population: every returned array of the
    line-info build hashed against the ``cupyx.jit`` kernel, both precisions.

``pipe``
    Microbenchmark of the consumer-card FP64 pipe: a dependent ``fma`` chain
    with 1, 2, 4, 8, 16 or 32 active lanes per warp (CUDA events). Says
    whether a partially active warp instruction costs the pipe a full warp.

``lines SASS.csv DISASM.txt SOURCE.cu``
    Joins an ``ncu --page source --print-source sass --csv`` export of one
    launch with ``nvdisasm -gi`` of the same cubin (line info, inline chains)
    and aggregates per generated-source line (innermost and outermost
    inlined frame) and per enclosing device function: executed
    warp instructions, FP64 instructions (``D*`` opcodes, ``MUFU.*64*``,
    ``F2F.F64``-class conversions), thread instructions (divergence) and
    warp-stall samples. Prints JSON.

Needs a CUDA device for ``run``/``identity``; run through ``pyrite remote``.
No validation claim.
"""

import csv
import hashlib
import json
import os
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# SASS opcodes executed on the FP64 pipe (sm_120): DADD/DMUL/DFMA/DSETP/DMNMX,
# 64-bit MUFU (RCP64H/RSQ64H) and FP64 conversions.
_FP64 = re.compile(r"^(D[A-Z]+\b|MUFU\.\w*64|F2F\.F64|F2F\.F32\.F64|I2F\.F64|F2I\.\w*F64|DSETP)")


def install_lineinfo(out=None):
    """Swap the launcher's variant factory for line-info RawModule builds."""
    from pyrite.montecarlo.transport import _jit_launch
    from pyrite.montecarlo.transport._jit_rawmodule import KernelBuild, RawModuleKernel

    original = _jit_launch.transport_kernel_variant
    kernels = {}
    build = KernelBuild(lineinfo=True)

    def variant(switches, precision="fp64"):
        key = (switches, precision)
        if key not in kernels:
            kernels[key] = _Dumping(RawModuleKernel(original(switches, precision), build), out)
        return kernels[key]

    _jit_launch._SPECIALIZE_TRANSPORT_KERNEL = True
    _jit_launch.transport_kernel_variant = variant
    return kernels


class _Dumping:
    """Writes the compiled source/options once per argument-type tuple."""

    def __init__(self, inner, out):
        self.inner, self.out, self.seen = inner, out, set()

    def __call__(self, grid, block, args, shared_mem=0):
        if self.out is not None:
            from cupy_backends.cuda.api import driver

            function, source, options, _ = self.inner.compiled(args)
            digest = hashlib.sha256(source.encode()).hexdigest()[:12]
            if digest not in self.seen:
                self.seen.add(digest)
                registers = driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_NUM_REGS, function.ptr)
                Path(self.out).mkdir(parents=True, exist_ok=True)
                Path(self.out, f"kernel-{digest}.cu").write_text(source)
                meta = dict(options=list(options), registers=registers, grid=list(grid))
                Path(self.out, f"kernel-{digest}.json").write_text(json.dumps(meta))
        self.inner(grid, block, args, shared_mem)


def run(argv):
    out, rest = argv[0], argv[1:]
    install_lineinfo(out)
    sys.argv = [str(HERE / "streamed_block_sizing.py"), *rest]
    runpy.run_path(sys.argv[0], run_name="__main__")


def _digest(result):
    h = hashlib.sha256()
    for key in sorted(result):
        value = result[key]
        if type(value).__module__.startswith("cupy"):
            value = value.get()
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.dtype == object:
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def identity(argv):
    """Line-info build vs ``cupyx.jit`` on the #831/#875 W 1 MeV anchor."""
    sys.path.insert(0, str(HERE))
    from transport_kernel_builds import _scene_kwargs

    from pyrite.montecarlo.transport import _jit_launch
    from pyrite.montecarlo.transport import _simulate_trajectories as simulate

    histories = int(argv[0]) if argv else 1_000
    kwargs = _scene_kwargs("w-1MeV-100um")
    jit_factory = _jit_launch.transport_kernel_variant
    records = []
    for precision in ("mixed", "fp64"):
        for seed in (1, 2):
            digests = {}
            for arm in ("jit", "lineinfo"):
                _jit_launch.transport_kernel_variant = jit_factory
                if arm == "lineinfo":
                    install_lineinfo()
                result = simulate(**kwargs, Ne=histories, seed=seed, transport_precision=precision)
                digests[arm] = _digest(result)
            record = dict(precision=precision, seed=seed, **digests)
            record["identical"] = digests["jit"] == digests["lineinfo"]
            records.append(record)
            print(json.dumps(record), flush=True)
    print(json.dumps(dict(all_identical=all(r["identical"] for r in records))))


def scene_identity(argv):
    """Line-info build vs ``cupyx.jit`` on the #907 scene (joined transport).

    Hashes every segment array of one ``runner._transport_case`` call per
    arm and population (seed 7), and records each compiled kernel's registers.
    """
    sys.path.insert(0, str(HERE))
    from streamed_block_sizing import scene_for
    from transport_kernel_builds import _function_attributes, _functions

    import pyrite
    from pyrite import api
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.transport_options import transport_tuning
    from pyrite.montecarlo.transport import _jit_launch

    histories = int(argv[0]) if argv else 2_048
    pyrite.fetch_tables()
    jit_factory = _jit_launch.transport_kernel_variant
    used = {"jit": [], "lineinfo": []}

    def capture(arm, factory):
        def variant(switches, precision="fp64"):
            kernel = factory(switches, precision)
            if kernel not in used[arm]:
                used[arm].append(kernel)
            return kernel

        _jit_launch._SPECIALIZE_TRANSPORT_KERNEL = True
        _jit_launch.transport_kernel_variant = variant

    records = []
    for n in (histories, 2 * histories):
        digests = {}
        for arm in ("jit", "lineinfo"):
            _jit_launch.transport_kernel_variant = jit_factory
            if arm == "lineinfo":
                install_lineinfo()
            capture(arm, _jit_launch.transport_kernel_variant)
            scene = scene_for(n, None, 2_048, 32, 0)
            case = api.build_case(scene, scene.numerics)
            with transport_tuning(scene.numerics, streaming=False):
                segs = runner._transport_case(case, transport_core="cuda")["segs"]
            digests[arm] = _digest(segs)
        record = dict(histories=n, **digests)
        record["identical"] = digests["jit"] == digests["lineinfo"]
        records.append(record)
        print(json.dumps(record), flush=True)
    registers = {}
    for arm, kernels in used.items():
        registers[arm] = []
        for kernel in kernels:
            inner = getattr(kernel, "inner", kernel)
            registers[arm] += [_function_attributes(f)["registers"] for f, _ in _functions(inner)]
    print(json.dumps(dict(registers=registers, all_identical=all(r["identical"] for r in records))))


def _observables(segs):
    """Transport-only summary of one population's segment arrays.

    Every scalar count, and for each numeric 1-D array its size, sum, mean,
    variance and the 1/50/99% quantiles (with ``hard_W_keV`` restricted to
    rows with a hard event), so two output-changing arms can be compared
    without scoring spectra.
    """
    out = {}
    for key in sorted(segs):
        value = segs[key]
        if type(value).__module__.startswith("cupy"):
            value = value.get()
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind not in "biuf":
            continue
        if array.ndim == 0:
            out[key] = dict(value=float(array))
            continue
        if array.ndim != 1 or array.size == 0:
            continue
        data = array.astype(float)
        if key == "hard_W_keV":
            data = data[data > 0.0]
        if data.size == 0:
            continue
        q = np.quantile(data, [0.01, 0.5, 0.99])
        out[key] = dict(
            n=int(data.size),
            sum=float(data.sum()),
            mean=float(data.mean()),
            var=float(data.var()),
            q01=float(q[0]),
            q50=float(q[1]),
            q99=float(q[2]),
        )
    return out


def scene_digest(argv):
    """``scene-digest N... [--observables OUT.json]``: #907-scene segments.

    The production ``cupyx.jit`` kernel, joined transport only (no
    scoring). Prints one digest per population; compare across checkouts to
    check that a kernel change keeps transport bitwise. ``--observables``
    also writes :func:`_observables` of each population.
    """
    sys.path.insert(0, str(HERE))
    from streamed_block_sizing import scene_for

    import pyrite
    from pyrite import api
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.transport_options import transport_tuning

    target = None
    if "--observables" in argv:
        index = argv.index("--observables")
        target = Path(argv[index + 1])
        argv = argv[:index] + argv[index + 2 :]
    pyrite.fetch_tables()
    summaries = {}
    for n in [int(a) for a in argv] or [2_048, 4_096]:
        scene = scene_for(n, None, 2_048, 32, 0)
        case = api.build_case(scene, scene.numerics)
        with transport_tuning(scene.numerics, streaming=False):
            segs = runner._transport_case(case, transport_core="cuda")["segs"]
        rows = int(np.asarray(segs["L_ang"]).shape[0])
        print(json.dumps(dict(histories=n, rows=rows, digest=_digest(segs))), flush=True)
        if target is not None:
            summaries[str(n)] = _observables(segs)
    if target is not None:
        target.write_text(json.dumps(summaries, indent=1))


def compare_observables(argv):
    """``compare-observables BASE.json NEW.json``: relative change and z-score.

    Per population and array: relative change of the mean, and its z-score
    against the larger arm's standard error of the mean (unpaired, so
    conservative for same-seed arms).
    """
    base, new = (json.loads(Path(a).read_text()) for a in argv[:2])
    out = {}
    for n, arrays in base.items():
        rows = {}
        for key, a in arrays.items():
            b = new.get(n, {}).get(key)
            if b is None:
                continue
            if "value" in a:
                rows[key] = dict(base=a["value"], new=b["value"])
                continue
            se = np.sqrt(max(a["var"] / a["n"], b["var"] / b["n"]))
            rows[key] = dict(
                n=(a["n"], b["n"]),
                mean_rel=b["mean"] / a["mean"] - 1.0 if a["mean"] else None,
                z=(b["mean"] - a["mean"]) / se if se > 0 else None,
                q50_rel=b["q50"] / a["q50"] - 1.0 if a["q50"] else None,
            )
        out[n] = rows
    print(json.dumps(out, indent=1))


_PIPE_SOURCE = r"""
extern "C" __global__ void fp64_chain(double* out, int active, int n) {
    int lane = threadIdx.x & 31;
    double a = 1.0 + 1e-9 * threadIdx.x, b = 0.999999, c = 1e-7;
    if (lane < active) {
        double x0 = a, x1 = a + 1.0, x2 = a + 2.0, x3 = a + 3.0;
        for (int i = 0; i < n; ++i) {
            x0 = fma(x0, b, c); x1 = fma(x1, b, c); x2 = fma(x2, b, c); x3 = fma(x3, b, c);
        }
        out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3;
    }
}
"""


def pipe(argv):
    """DFMA throughput against active lanes per warp."""
    import cupy

    kernel = cupy.RawKernel(_PIPE_SOURCE, "fp64_chain")
    sms = cupy.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
    blocks, threads, n = sms * 8, 256, 20_000
    out = cupy.zeros(blocks * threads)
    records = []
    for active in (32, 16, 8, 4, 2, 1):
        kernel((blocks,), (threads,), (out, np.int32(active), np.int32(n)))
        times = []
        for _ in range(5):
            start, stop = cupy.cuda.Event(), cupy.cuda.Event()
            start.record()
            kernel((blocks,), (threads,), (out, np.int32(active), np.int32(n)))
            stop.record()
            stop.synchronize()
            times.append(cupy.cuda.get_elapsed_time(start, stop))
        warp_inst = blocks * threads // 32 * n * 4
        ms = float(np.median(times))
        records.append(
            dict(
                active=active,
                ms=ms,
                warp_dfma_per_ns=warp_inst / (ms * 1e6),
                thread_dfma_per_ns=warp_inst * active / (ms * 1e6),
            )
        )
        print(json.dumps(records[-1]), flush=True)


_FUNC = re.compile(r"^(?:__device__|extern \"C\" __global__)[^(]*?\b(\w+)\s*\(")
_LOCATION = re.compile(r'"([^"]+)", line (\d+)')
_INSTRUCTION = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")


def _enclosing_functions(source_lines):
    """Generated-source line number (1-based) -> enclosing function name."""
    names, current = {}, "?"
    for number, text in enumerate(source_lines, 1):
        match = _FUNC.match(text)
        if match:
            current = match[1]
        names[number] = current
    return names


def _number(text):
    text = (text or "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _line_map(disassembly, source_name):
    """SASS offset -> (innermost, outermost) line of the generated source.

    ``disassembly`` is ``nvdisasm -gi`` output: a ``//## File`` comment names
    the innermost location and its ``inlined at`` chain and holds until the
    next one. Frames outside the generated source (CuPy headers) are skipped.
    """
    mapping, frames = {}, []
    for text in disassembly.splitlines():
        if text.lstrip().startswith("//## File"):
            frames = [int(n) for path, n in _LOCATION.findall(text) if path.endswith(source_name)]
            continue
        match = _INSTRUCTION.search(text)
        if match:
            inner = frames[0] if frames else None
            outer = frames[-1] if frames else None
            mapping[int(match[1], 16)] = (inner, outer)
    return mapping


def _opcode(text):
    words = text.strip().split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return words[0] if words else ""


def lines(argv):
    """``pipe``
        Microbenchmark of the consumer-card FP64 pipe: a dependent ``fma`` chain
        with 1, 2, 4, 8, 16 or 32 active lanes per warp (CUDA events). Says
        whether a partially active warp instruction costs the pipe a full warp.

    ``lines SASS.csv DISASM.txt SOURCE.cu``: rank lines and functions."""
    report, disassembly, source = Path(argv[0]), Path(argv[1]), Path(argv[2])
    source_lines = source.read_text().splitlines()
    functions = _enclosing_functions(source_lines)
    mapping = _line_map(disassembly.read_text(), source.name)
    with report.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next(i for i, r in enumerate(rows) if "Address" in r and "Source" in r)
    header = rows[header_index]
    col = {name: i for i, name in enumerate(header)}
    stall_columns = [n for n in header if n.startswith("stall_") and "Not Issued" not in n]
    body = [r for r in rows[header_index + 1 :] if len(r) == len(header)]
    base = int(body[0][col["Address"]], 16)
    keys = ("warp_inst", "thread_inst", "stall_samples", "fp64_warp_inst", "fp64_thread_inst")
    per = {kind: defaultdict(lambda: defaultdict(float)) for kind in ("inner", "outer", "func")}
    totals = defaultdict(float)
    unmapped = 0
    for row in body:
        offset = int(row[col["Address"]], 16) - base
        inner, outer = mapping.get(offset, (None, None))
        unmapped += offset not in mapping
        warp = _number(row[col["Instructions Executed"]])
        values = dict(
            warp_inst=warp,
            thread_inst=_number(row[col["Thread Instructions Executed"]]),
            stall_samples=_number(row[col["Warp Stall Sampling (All Samples)"]]),
        )
        if _FP64.match(_opcode(row[col["Source"]])):
            values["fp64_warp_inst"] = warp
            values["fp64_thread_inst"] = values["thread_inst"]
        for name in stall_columns:
            values[name] = _number(row[col[name]])
        for kind, where in (("inner", inner), ("outer", outer), ("func", functions.get(inner))):
            for key, value in values.items():
                per[kind][where][key] += value
        for key, value in values.items():
            totals[key] += value

    def table(kind, key, n):
        ranked = sorted(per[kind].items(), key=lambda kv: -kv[1].get(key, 0.0))[:n]
        out = []
        for where, value in ranked:
            item = dict(where=where, share=value.get(key, 0.0) / max(totals[key], 1.0))
            if isinstance(where, int) and 0 < where <= len(source_lines):
                item["text"] = source_lines[where - 1].strip()[:160]
                item["function"] = functions.get(where)
            item["active_threads"] = value["thread_inst"] / max(value["warp_inst"], 1.0)
            top = sorted(stall_columns, key=lambda c: -value.get(c, 0.0))[:3]
            item["top_stalls"] = {c: value.get(c, 0.0) for c in top}
            item.update({k: value.get(k, 0.0) for k in keys})
            out.append(item)
        return out

    result = dict(
        instructions=len(body),
        unmapped=unmapped,
        totals={k: totals[k] for k in keys},
        stall_totals={c: totals[c] for c in sorted(stall_columns, key=lambda c: -totals[c])},
        active_threads=totals["thread_inst"] / max(totals["warp_inst"], 1.0),
        fp64_share=totals["fp64_warp_inst"] / max(totals["warp_inst"], 1.0),
    )
    for kind in ("func", "inner", "outer"):
        for key, n in (("fp64_warp_inst", 30), ("stall_samples", 30), ("warp_inst", 20)):
            result[f"{kind}:{key}"] = table(kind, key, n)
    print(json.dumps(result, indent=1))


def main():
    mode, argv = sys.argv[1], sys.argv[2:]
    modes = {"run": run, "identity": identity, "scene-identity": scene_identity}
    modes.update({"scene-digest": scene_digest, "lines": lines, "pipe": pipe})
    modes["compare-observables"] = compare_observables
    if mode not in modes:
        raise SystemExit(__doc__)
    os.environ.setdefault("PYRITE_MC_BACKEND", "cuda")
    modes[mode](argv)


if __name__ == "__main__":
    main()
