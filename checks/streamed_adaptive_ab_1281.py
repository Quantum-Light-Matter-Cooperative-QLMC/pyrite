"""Streamed vs retained adaptive scoring: memory, wall and parity (#1281, #1280).

Usage:
  streamed_adaptive_ab_1281.py run ROUTE MATERIAL E0_KEV TILT_DEG MAX_HISTORIES OUT_PREFIX
  streamed_adaptive_ab_1281.py compare REF_PREFIX OTHER_PREFIX
  streamed_adaptive_ab_1281.py table OUT_DIR

``run`` resolves one case of the ``PROFILE`` catalog profile (default
``cleo_sub``; the profile, beam and detector come from the user config
directory, so remote jobs set ``XDG_CONFIG_HOME`` to a shipped copy) exactly
as ``pyrite run`` does, changes only ``max_histories``, and runs it once
through ``run_cases`` (production serial path). ROUTE:

- ``retained``: ``PYRITE_STREAMED_ADAPTIVE=off`` (every kept chunk joined).
- ``auto``: the production route; streams only when the #1279 kept-bytes
  projection exceeds host headroom.
- ``streamed``: the production streamed route forced at the first admission
  check, in blocks of ``PROBE_STREAM_BLOCK_MIB`` (default 2048, the
  production ``STREAMED_BLOCK_BYTES``) of observed rows.
- ``measure``: ``auto`` on the #1280 measure-only streamed route
  (``streamed_adaptive.MONITORED_PASS = False``): the same-revision arm of the
  #1397 single-pass / folded-evidence comparison against ``auto``.

``PROBE_FRESH_GRID=1`` removes the line-grid speed cache (``cache_dir() /
"line-grids"``; point ``PYRITE_CACHE_DIR`` at a scratch directory) before the
arm, so its policy grid misses the cache.

One arm per process, so ``ru_maxrss`` is that arm's host peak; device peaks
use the #1252 sampler (``adaptive_block_size_1252.py``); phase splits are
synchronized walls of ``transport`` (retained, or the measure-only pass and
each streamed pass's transport), ``spectrum`` (retained) and ``streamed``
(the whole streamed rerun, transport included).
Every array output is saved to ``OUT_PREFIX.npz``; the record (route taken,
realized pair, reflection selection, memory, wall) to ``OUT_PREFIX.json``,
written as a stub first so a timed-out arm leaves evidence.

``compare`` checks the realized pair, reflection-selection record and every shared array (max relative difference) of two arms.
``table`` summarizes the records under ``OUT_DIR``.

Remote only (GPU, up to hours per arm at 20M). Engineering evidence for
``adaptive-sample-size-stopping`` streamed parity; no accuracy verdict.
"""

import dataclasses
import json
import os
import resource
import runpy
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("PROFILE", "cleo_sub")
PROFILE = os.environ["PROFILE"]
HELPER = runpy.run_path(str(Path(__file__).with_name("adaptive_block_size_1252.py")))
ROUTES = ("retained", "auto", "streamed", "measure")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else f"<array {value.shape}>"
    return str(value)


def _override_max(n):
    from pyrite.runs import scan

    resolved = scan._resolved_run

    def capped(args, material):
        settings, sweep, identity, stem = resolved(args, material)
        overrides = {"max_histories": int(n)}
        if settings.precision.brem_max_histories is not None:
            overrides["brem_max_histories"] = min(int(n), settings.precision.brem_max_histories)
        precision = dataclasses.replace(settings.precision, **overrides)
        return dataclasses.replace(settings, precision=precision), sweep, identity, stem

    scan._resolved_run = capped


def _force_stream(block_bytes):
    """Raise the production streamed route at the first admission check."""
    from pyrite.montecarlo.runner import host_admission as ha

    real = ha.AdaptiveHostAdmission.check

    def check(self, blocks, histories, **kwargs):
        if self.stream and blocks and histories:
            per_history = max(1, ha.block_bytes(blocks) // histories)
            raise ha.StreamedAdaptiveRoute(max(1, block_bytes // per_history), 0, 0)
        return real(self, blocks, histories, **kwargs)

    ha.AdaptiveHostAdmission.check = check


def _selection(out):
    """The reflection-selection record (cutoff, count, tail per radiator), if any."""
    record = (out or {}).get("reflection_selection")
    return record if isinstance(record, dict) else None


def _instrument(timers):
    """Synchronized wall of the transport, retained spectrum and streamed seams."""
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import recovery

    runner._transport_case = timers.wrap("transport", runner._transport_case, "transport")
    runner._spectrum_case = timers.wrap("spectrum", runner._spectrum_case, "spectrum")
    recovery._streamed_adaptive = timers.wrap("streamed", recovery._streamed_adaptive, "streamed")


def run(route, material, e0_kev, tilt_deg, max_histories, out_prefix):
    from pyrite.montecarlo.runner.scheduling import run_cases

    if route not in ROUTES:
        raise SystemExit(f"ROUTE must be one of {ROUTES}")
    if route == "retained":
        os.environ["PYRITE_STREAMED_ADAPTIVE"] = "off"
    elif route == "streamed":
        _force_stream(int(float(os.environ.get("PROBE_STREAM_BLOCK_MIB", "2048")) * 1024**2))
    elif route == "measure":
        from pyrite.montecarlo.runner import streamed_adaptive

        streamed_adaptive.MONITORED_PASS = False
    if os.environ.get("PROBE_FRESH_GRID") == "1":
        import shutil

        from pyrite.paths import cache_dir

        shutil.rmtree(cache_dir() / "line-grids", ignore_errors=True)
    _override_max(max_histories)
    case, _, _ = HELPER["profile_case"](material, float(e0_kev), float(tilt_deg), "auto")
    record = {
        "issue": 1281,
        "stub": True,
        "code_sha": os.environ.get("PROBE_SHA"),
        "profile": PROFILE,
        "route_requested": route,
        "material": material,
        "E0_keV": float(e0_kev),
        "tilt_deg": float(tilt_deg),
        "max_histories": int(max_histories),
        "backend": os.environ.get("PYRITE_MC_BACKEND"),
        "stream_block_MiB": os.environ.get("PROBE_STREAM_BLOCK_MIB"),
    }
    Path(f"{out_prefix}.json").write_text(json.dumps(record, default=_json_default))
    memory = HELPER["Memory"]()
    timers = HELPER["Timers"](memory)
    _instrument(timers)
    timings: list = []
    out, error = None, None
    t0 = time.perf_counter()
    with memory, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = run_cases([case], max_workers=0, progress=False, on_timing=timings.append)[0]
        except Exception as exc:  # record OOM/failure as evidence, never mask it
            error = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-4000:]
    wall = time.perf_counter() - t0
    sampling = (out or {}).get("adaptive_sampling") or {}
    record.update(
        stub=False,
        error=error,
        wall_s=wall,
        split_s=timers.seconds,
        memory=memory.summary(),
        host_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        route_taken=((out or {}).get("_adaptive_route") or {"route": "retained"}),
        realized_electrons=sampling.get("realized_electrons"),
        realized_brem=sampling.get("realized_electrons_brem"),
        stop_reason=sampling.get("stop_reason"),
        relative_se={
            k: v.get("relative_se") for k, v in (sampling.get("statistics") or {}).items()
        },
        n_segments=None if out is None else out.get("n_segments"),
        selection=_selection(out),
        grid_pass=(out or {}).get("_streamed_line_grid_pass"),
        grid_record=(out or {}).get("line_grid_resolved"),
        warnings=[str(w.message)[:400] for w in caught],
        output_meta={} if out is None else HELPER["_scalar_meta"](out),
    )
    if out is not None:
        arrays = {
            k: np.asarray(v)
            for k, v in out.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "fiucb"
        }
        np.savez_compressed(f"{out_prefix}.npz", **arrays)
        sampling_dump = {k: v for k, v in sampling.items() if k not in ("statistics",)}
        record["adaptive_sampling"] = sampling_dump
    Path(f"{out_prefix}.json").write_text(json.dumps(record, default=_json_default))
    print(
        json.dumps({k: v for k, v in record.items() if k != "output_meta"}, default=_json_default)
    )
    return 1 if error else 0


def _grid_record(record):
    """The resolved-grid record without its cache provenance."""
    grid = record.get("grid_record") or {}
    return {k: v for k, v in grid.items() if k not in ("cache", "truncation_audit")}


def compare(ref_prefix, other_prefix):
    ref = json.loads(Path(f"{ref_prefix}.json").read_text())
    other = json.loads(Path(f"{other_prefix}.json").read_text())
    a, b = np.load(f"{ref_prefix}.npz"), np.load(f"{other_prefix}.npz")
    arrays = {}
    for key in sorted(set(a.files) & set(b.files)):
        x, y = a[key].astype(float), b[key].astype(float)
        if x.shape != y.shape:
            arrays[key] = f"shape {x.shape} vs {y.shape}"
            continue
        scale = max(float(np.max(np.abs(x), initial=0.0)), 1e-300)
        arrays[key] = float(np.max(np.abs(x - y), initial=0.0) / scale)
    report = {
        "ref": ref_prefix,
        "other": other_prefix,
        "routes": [ref.get("route_taken"), other.get("route_taken")],
        "realized_equal": (ref.get("realized_electrons"), ref.get("realized_brem"))
        == (other.get("realized_electrons"), other.get("realized_brem")),
        "realized": [ref.get("realized_electrons"), other.get("realized_electrons")],
        "n_segments": [ref.get("n_segments"), other.get("n_segments")],
        "selection_equal": ref.get("selection") == other.get("selection"),
        "grid_record_equal": _grid_record(ref) == _grid_record(other),
        "selection": [ref.get("selection"), other.get("selection")],
        "only_in": [sorted(set(a.files) - set(b.files)), sorted(set(b.files) - set(a.files))],
        "max_rel_diff": arrays,
    }
    print(json.dumps(report, indent=1, default=_json_default))


def table(out_dir):
    records = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]
    records = [r for r in records if r.get("issue") == 1281 and "route_requested" in r]
    print(
        "material E0 tilt max route->taken realized stop wall_s host_MiB "
        "card_peak_MiB pool_res_MiB error"
    )
    for r in records:
        mem = r.get("memory") or {}
        taken = (r.get("route_taken") or {}).get("route")
        print(
            r["material"],
            r["E0_keV"],
            r["tilt_deg"],
            r["max_histories"],
            f"{r['route_requested']}->{taken}",
            r.get("realized_electrons"),
            r.get("stop_reason"),
            None if r.get("wall_s") is None else round(r["wall_s"], 1),
            None if r.get("host_peak_MiB") is None else round(r["host_peak_MiB"]),
            None if not mem else round(mem.get("card_peak_MiB", 0)),
            None if not mem else round(mem.get("pool_reserved_peak_MiB", 0)),
            "stub" if r.get("stub") else r.get("error"),
        )


if __name__ == "__main__":
    command, *argv = sys.argv[1:]
    if command == "run":
        sys.exit(run(*argv))
    {"compare": compare, "table": table}[command](*argv)
