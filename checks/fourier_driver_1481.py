"""Device Fourier bin-mean driver: host syncs and launches per flush, base vs candidate (#1481).

Usage:
  fourier_driver_1481.py capture MATERIAL E0_KEV TILT_DEG HISTORIES FLUSH_DIR
  fourier_driver_1481.py replay FLUSH_DIR OUT_JSON [REPEATS]
  fourier_driver_1481.py plan FLUSH_DIR OUT_JSON [REPEATS]
  fourier_driver_1481.py compare OUT_JSON BASE_JSON CAND_JSON

``capture`` resolves one ``PROFILE`` case (default ``cleo_sub``; profile,
beam and detector from the shipped user config) as ``pyrite run`` does, fixes
its population at ``HISTORIES`` and runs it once through ``run_cases``
(production serial path). Every ``cuda_fourier_bin_mean_reduce`` call is
synchronized and timed; every ``CAPTURE_EVERY``-th call (default 16, from the
first) is saved as ``flush-<i>.npz`` (``E_res``, ``a_width``, ``weight``,
``axis``, ``kwargs``) with its call index. Record: ``capture.json``.

``replay`` runs the checkout's ``cuda_fourier_bin_mean_reduce`` on every saved
flush: one warm call, then ``REPEATS`` (default 5) calls, each synchronized
before and after and inside an NVTX range ``replay.reduce`` (run it under
``nsys`` for copies, synchronizations and launches per call). Records wall per
call and every output as FP64 hex, so ``compare`` can test bitwise equality.

``plan`` times cutoff planning only on sorted-line prefixes of the first
flush (2^14 .. all lines): host planning (one copy of the counts) against the
device planner (``_HOST_PLAN_MAX_LINES`` threshold, #1481).

``compare`` reports, per flush, whether base and candidate outputs are
bitwise equal, their largest difference over the spectrum peak, and the same
for each side's repeat-to-repeat spread: the gather spreader and the direct
rectangles accumulate with FP64 atomics, so repeated device calls need not
be bitwise equal even on one checkout.

Engineering evidence; no physics-validation claim.
"""

import dataclasses
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

PROFILE = os.environ.get("PROFILE", "cleo_sub")
CAPTURE_EVERY = int(os.environ.get("CAPTURE_EVERY", "16"))
os.environ["PROFILE"] = PROFILE


def _cp():
    import cupy

    return cupy


def _sync():
    _cp().cuda.Device().synchronize()


def _gpu():
    cp = _cp()
    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def _write(path, record):
    Path(path).write_text(json.dumps(record, indent=1, default=str))


def _base_record(**extra):
    import pyrite

    return dict(
        issue=1481,
        code_sha=os.environ.get("PROBE_SHA"),
        pyrite_file=pyrite.__file__,
        gpu=_gpu(),
        **extra,
    )


def capture(material, e0_kev, tilt_deg, histories, flush_dir):
    from pyrite.montecarlo.runner.scheduling import run_cases
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc
    from pyrite.runs import scan

    helper = runpy.run_path(str(Path(__file__).with_name("adaptive_block_size_1252.py")))
    n = int(histories)
    resolved = scan._resolved_run

    def fixed(args, mat):
        settings, sweep, identity, stem = resolved(args, mat)
        brem = (
            {"brem_min_histories": n, "brem_max_histories": n}
            if getattr(settings.precision, "brem_max_histories", None) is not None
            else {}
        )
        precision = dataclasses.replace(
            settings.precision, min_histories=n, max_histories=n, pilot_histories=None, **brem
        )
        return dataclasses.replace(settings, precision=precision), sweep, identity, stem

    scan._resolved_run = fixed
    case, _, _ = helper["profile_case"](material, float(e0_kev), float(tilt_deg), "auto")
    out = Path(flush_dir)
    out.mkdir(parents=True, exist_ok=True)
    calls, saved = [], []
    real = fc.cuda_fourier_bin_mean_reduce

    def timed(E_res, a_width, weight, axis, **kwargs):
        _sync()
        t0 = time.perf_counter()
        res = real(E_res, a_width, weight, axis, **kwargs)
        _sync()
        calls.append((int(np.size(E_res)), time.perf_counter() - t0))
        if (len(calls) - 1) % CAPTURE_EVERY == 0:
            cp = _cp()
            path = out / f"flush-{len(saved)}.npz"
            np.savez(
                path,
                E_res=cp.asnumpy(cp.asarray(E_res, dtype=cp.float64)),
                a_width=cp.asnumpy(cp.asarray(a_width, dtype=cp.float64)),
                weight=cp.asnumpy(cp.asarray(weight, dtype=cp.float64)),
                axis=np.array(axis, dtype=np.float64),
                kwargs=json.dumps(kwargs),
                call=len(calls) - 1,
            )
            saved.append(path.name)
        return res

    fc.cuda_fourier_bin_mean_reduce = timed
    error = None
    t0 = time.perf_counter()
    try:
        run_cases([case], max_workers=0, progress=False)
    except Exception as exc:  # record failure as evidence
        error = f"{type(exc).__name__}: {exc}"
    record = _base_record(
        profile=PROFILE,
        material=material,
        E0_keV=float(e0_kev),
        tilt_deg=float(tilt_deg),
        histories=n,
        error=error,
        wall_s=time.perf_counter() - t0,
        reduce_calls=len(calls),
        reduce_lines=sum(c[0] for c in calls),
        reduce_s=sum(c[1] for c in calls),
        calls=calls,
        saved=saved,
    )
    _write(out / "capture.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "calls"}))


def _flushes(flush_dir):
    paths = sorted(Path(flush_dir).glob("flush-*.npz"), key=lambda p: int(p.stem.split("-")[1]))
    for path in paths:
        with np.load(path) as z:
            axis = tuple(z["axis"].tolist())
            yield (
                path.name,
                z["E_res"],
                z["a_width"],
                z["weight"],
                axis,
                json.loads(str(z["kwargs"])),
            )


def replay(flush_dir, out_json, repeats=5):
    from pyrite.montecarlo.spectrum.cxr._fourier_cuda import cuda_fourier_bin_mean_reduce

    cp = _cp()
    nvtx = cp.cuda.nvtx
    rows = []
    nvtx.RangePush("gpuhost.run")  # the outer range nsys_gaps.py expects
    for name, E, a, w, axis, kwargs in _flushes(flush_dir):
        c0, h, n_bins = axis
        axis = (c0, h, int(n_bins))
        dE, da, dw = (cp.asarray(x) for x in (E, a, w))
        cuda_fourier_bin_mean_reduce(dE, da, dw, axis, **kwargs)  # warm: JIT, cuFFT plans
        _sync()
        walls, outputs = [], []
        for _ in range(int(repeats)):
            _sync()
            nvtx.RangePush("replay.reduce")
            t0 = time.perf_counter()
            res = cuda_fourier_bin_mean_reduce(dE, da, dw, axis, **kwargs)
            _sync()
            walls.append(time.perf_counter() - t0)
            nvtx.RangePop()
            outputs.append([float(v).hex() for v in cp.asnumpy(res)])
        rows.append(
            dict(flush=name, lines=int(E.size), n_bins=axis[2], walls_s=walls, outputs=outputs)
        )
        print(json.dumps({"flush": name, "lines": int(E.size), "min_wall_s": min(walls)}))
    nvtx.RangePop()
    _write(out_json, _base_record(repeats=int(repeats), flushes=rows))


def plan(flush_dir, out_json, repeats=5):
    from pyrite.montecarlo.spectrum.cxr import _fourier_cuda as fc
    from pyrite.montecarlo.spectrum.cxr._fourier_cutoffs import cutoff_rectangle_array

    cp = _cp()
    name, E, a, w, axis, kwargs = next(iter(_flushes(flush_dir)))
    # Period of the flush as the driver picks it is not needed: the plan cost
    # depends on the sorted counts; a representative period is the widest-line one.
    from pyrite.montecarlo.spectrum.cxr._bin_fourier import fourier_period

    c0, h, n_bins = axis
    keep = w != 0
    E, a = E[keep], a[keep]
    _, P, _ = fourier_period(np.array([E.min(), E.max()]), np.array([a.min()]), c0, h, int(n_bins))
    counts = np.sort(np.floor(2.0 * a / (2.0 * np.pi / P)).astype(np.int64))
    counts = counts[counts <= fc.FOURIER_MAX_LINE_MODES]
    sizes = [1 << k for k in range(14, 40) if (1 << k) < counts.size] + [counts.size]
    rng = np.random.default_rng(1481)
    rows = []
    for size in sizes:
        sub = np.sort(rng.choice(counts, size, replace=False)) if size < counts.size else counts
        d = cp.asarray(sub)
        arms = {
            "host": lambda d=d: cutoff_rectangle_array(cp.asnumpy(d)),
            "device": lambda d=d: cp.asnumpy(cutoff_rectangle_array(d, xp=cp)),
        }
        row = dict(lines=int(size), top=int(sub[-1]))
        ref = None
        for arm, fn in arms.items():
            fn()
            walls = []
            for _ in range(int(repeats)):
                _sync()
                t0 = time.perf_counter()
                rects = fn()
                walls.append(time.perf_counter() - t0)
            ref = rects if ref is None else ref
            row[arm] = dict(min_s=min(walls), median_s=float(np.median(walls)))
            row[f"{arm}_equal"] = bool(np.array_equal(rects, ref))
        row["rects"] = int(len(ref))
        rows.append(row)
        print(json.dumps(row))
    _write(out_json, _base_record(flush=name, threshold=fc._HOST_PLAN_MAX_LINES, rows=rows))


def _decode(hexes):
    return np.array([float.fromhex(v) for v in hexes])


def _spread(outputs):
    first = _decode(outputs[0])
    peak = max(float(np.abs(first).max()), 1e-300)
    diffs = [float(np.abs(_decode(o) - first).max()) / peak for o in outputs[1:]]
    bitwise = all(o == outputs[0] for o in outputs[1:])
    return dict(bitwise=bitwise, max_over_peak=max(diffs, default=0.0))


def compare(out_json, base_json, cand_json):
    base = json.loads(Path(base_json).read_text())
    cand = json.loads(Path(cand_json).read_text())
    rows = []
    for b, c in zip(base["flushes"], cand["flushes"], strict=True):
        if b["flush"] != c["flush"]:
            raise SystemExit("flush lists differ")
        ref = _decode(b["outputs"][0])
        peak = max(float(np.abs(ref).max()), 1e-300)
        diff = max(float(np.abs(_decode(o) - ref).max()) for o in c["outputs"]) / peak
        rows.append(
            dict(
                flush=b["flush"],
                lines=b["lines"],
                bitwise_equal=any(o == b["outputs"][0] for o in c["outputs"]),
                cand_vs_base_max_over_peak=diff,
                base_repeat=_spread(b["outputs"]),
                cand_repeat=_spread(c["outputs"]),
                base_min_wall_s=min(b["walls_s"]),
                cand_min_wall_s=min(c["walls_s"]),
                speedup=min(b["walls_s"]) / min(c["walls_s"]),
            )
        )
    record = dict(
        issue=1481,
        base_sha=base["code_sha"],
        cand_sha=cand["code_sha"],
        gpu=base["gpu"],
        flushes=rows,
        total_base_min_s=sum(r["base_min_wall_s"] for r in rows),
        total_cand_min_s=sum(r["cand_min_wall_s"] for r in rows),
    )
    _write(out_json, record)
    print(json.dumps({k: v for k, v in record.items() if k != "flushes"}))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"capture": capture, "replay": replay, "plan": plan, "compare": compare}[cmd](*args)
