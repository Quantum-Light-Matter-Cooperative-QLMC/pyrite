"""Adaptive-sampling block size: wall time, memory and stopping cost (#1252).

Usage:
  adaptive_block_size_1252.py run MATERIAL E0_KEV TILT_DEG BLOCK OUT_PREFIX
  adaptive_block_size_1252.py replay VALUES_NPZ OUT_JSON BLOCK...
  adaptive_block_size_1252.py table OUT_DIR

``run`` resolves one case of the ``cleo_sub`` catalog profile (``PROFILE``
overrides it; the profile, its beam and detector come from the user config
directory, so remote jobs set ``XDG_CONFIG_HOME`` to a shipped copy) exactly
as ``pyrite run`` does, changes only the adaptive block: ``block_histories =
BLOCK`` with ``min``/``pilot``/``max`` rounded up to multiples of it, and
runs it through ``run_cases`` (one case, the production serial CUDA path,
segments kept on device). One arm per process, so ``ru_maxrss`` is that arm's
host peak.

Recorded: wall; synchronized splits of transport (per block: simulate and the
stopping monitor's measure; the block join; population finalization) and the
spectrum reduction; CUDA-event time of every transport launch with its
population, so per-block fixed costs (launch gaps, sync, Python) are wall minus
kernel; CuPy pool reserved/used peaks and whole-card used memory (``memGetInfo``,
5 ms sampling, by phase); rows per electron; realized electrons; stop reason;
final relative standard errors; and any OOM/recovery fields of the output. The
per-electron stopping values (one per electron and observable, exactly what the
monitor saw) are saved to ``OUT_PREFIX.values.npz`` (after every block with
``PROBE_CAPTURE_EVERY=1``, so a timed-out capture keeps its prefix).

``replay`` runs the stopping rule (``adaptive.run_stopping_rule``) on saved
per-electron values for other block sizes. Block size does not change
transport, so each block's stop is exact while it lies inside the captured
population; a stop beyond it is reported as ``beyond_capture``. Values are
normalized per block in the monitor; the rule's statistics (relative SE, max
share, effective size, stability) are scale invariant, so a uniform factor does
not change decisions.

``PROBE_PROFILE=1`` wraps ``run`` in cProfile (``OUT_PREFIX.prof`` and printed
cumulative/self tables). ``table`` summarizes ``*.json`` arm records under ``OUT_DIR``.

Engineering evidence; no physics-validation claim.
"""

import json
import math
import os
import resource
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROFILE = os.environ.get("PROFILE", "cleo_sub")


def _cupy():
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        return None


CUPY = _cupy()


def _sync():
    if CUPY is not None:
        CUPY.cuda.Device().synchronize()


def rescaled(precision, block):
    """``precision`` with ``block`` electrons per block; counts rounded up to multiples."""

    def up(n):
        return None if n is None else int(math.ceil(n / block) * block)

    return replace(
        precision,
        block_histories=int(block),
        min_histories=up(precision.min_histories),
        pilot_histories=up(precision.pilot_histories),
        max_histories=up(precision.max_histories),
    )


def profile_case(material, e0_kev, tilt_deg, block):
    """One profile case, resolved like ``pyrite run``, with block ``block``."""
    from pyrite.runs import scan

    scan._load_runtime()
    args = SimpleNamespace(catalog_profile=PROFILE, checkpoint_dir="/nonexistent")
    settings, sweep, _identity, _stem = scan._resolved_run(args, material)
    precision = rescaled(settings.precision, block)
    settings = replace(settings, precision=precision)
    cases = [
        c
        for c in scan.build_cases(sweep, settings)
        if math.isclose(c.E0_keV, e0_kev) and math.isclose(c.tilt_deg, tilt_deg)
    ]
    if len(cases) != 1:
        raise SystemExit(f"{len(cases)} cases match {material} {e0_kev} keV {tilt_deg} deg")
    return cases[0], precision


class Timers:
    """Synchronized wall totals by name, and per-block records."""

    def __init__(self, memory):
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.blocks: list[dict] = []
        self.memory = memory

    def wrap(self, name, fn, phase=None):
        depth = [0]

        def timed(*args, **kwargs):
            if depth[0]:  # re-entrant call (adaptive -> runner): time the outer one only
                return fn(*args, **kwargs)
            depth[0] += 1
            _sync()
            previous = self.memory.phase
            if phase is not None:
                self.memory.phase = phase
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                depth[0] -= 1
                _sync()
                self.memory.phase = previous
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0
                self.calls[name] = self.calls.get(name, 0) + 1

        return timed


class Launches:
    """CUDA-event brackets around every transport core launch (both drivers)."""

    def __init__(self):
        self.events = []  # (population, start, stop)
        self.active = []

    def install(self):
        if CUPY is None:
            return
        from pyrite.montecarlo.transport import batching, continuation

        probe = self

        def timed_core(core):
            def timed(launch, *a, **k):
                start, stop = CUPY.cuda.Event(), CUPY.cuda.Event()
                start.record()
                out = core(launch, *a, **k)
                stop.record()
                try:
                    population = int(launch[1])
                except Exception:
                    population = -1
                probe.events.append((population, start, stop))
                return out

            return timed

        for module, name in (
            (batching, "_drive_continuation_rounds"),
            (batching, "_drive_per_electron_batches"),
        ):
            real = getattr(module, name)

            def drive(core, *args, _real=real, **kwargs):
                return _real(timed_core(core), *args, **kwargs)

            setattr(module, name, drive)
        real_pop = continuation.active_population

        def active_population(*args, **kwargs):
            out = real_pop(*args, **kwargs)
            probe.active.append(int(out[0]))
            return out

        continuation.active_population = active_population

    def summary(self):
        if CUPY is None or not self.events:
            return {"launches": 0}
        _sync()
        pops = np.array([p for p, _, _ in self.events], dtype=np.int64)
        ms = np.array([CUPY.cuda.get_elapsed_time(a, b) for _, a, b in self.events])
        return {
            "launches": len(self.events),
            "kernel_s": float(ms.sum() / 1e3),
            "population_max": int(pops.max(initial=0)),
            "population_time_weighted": float((pops * ms).sum() / max(ms.sum(), 1e-30)),
            "continuation_active": sorted(set(self.active)),
            "continuation_passes": len(self.active),
        }


class Memory:
    """Pool reserved/used and whole-card used peaks, sampled every 5 ms, by phase."""

    def __init__(self):
        self.phase = "setup"
        self.by_phase: dict[str, int] = {}
        self._stop = threading.Event()
        if CUPY is None:
            self.pool = None
            return
        self.pool = CUPY.get_default_memory_pool()
        free, self.total = CUPY.cuda.runtime.memGetInfo()
        self.card_base = self.card_peak = self.total - free
        self.reserved_peak = self.used_peak = 0

    def _sample(self):
        free, _ = CUPY.cuda.runtime.memGetInfo()
        card = self.total - free
        self.card_peak = max(self.card_peak, card)
        self.reserved_peak = max(self.reserved_peak, self.pool.total_bytes())
        self.used_peak = max(self.used_peak, self.pool.used_bytes())
        self.by_phase[self.phase] = max(self.by_phase.get(self.phase, 0), card)

    def _run(self):
        while not self._stop.wait(0.005):
            self._sample()

    def __enter__(self):
        if self.pool is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self._stop.set()
            self._thread.join()
            self._sample()

    def summary(self):
        if self.pool is None:
            return {}
        mib = 1024**2
        limit = self.pool.get_limit()
        return {
            "card_total_MiB": self.total / mib,
            "card_base_MiB": self.card_base / mib,
            "card_peak_MiB": self.card_peak / mib,
            "pool_reserved_peak_MiB": self.reserved_peak / mib,
            "pool_used_peak_MiB": self.used_peak / mib,
            "pool_limit_MiB": None if not limit else limit / mib,
            "card_peak_by_phase_MiB": {k: v / mib for k, v in self.by_phase.items()},
        }


def save_values(out_prefix, values, block):
    """Write the captured per-electron values atomically (the replay input)."""
    tmp = f"{out_prefix}.values.tmp.npz"
    np.savez_compressed(tmp, **{k: np.concatenate(v) for k, v in values.items()}, block=block)
    os.replace(tmp, f"{out_prefix}.values.npz")


def instrument(timers, values, on_values=None):
    """Patch the adaptive block seams with timers and a per-electron value capture.

    ``on_values`` (optional) is called after every captured block, so a
    long capture that later times out still leaves its prefix.
    """
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import adaptive, block_transport

    real_init = adaptive.StoppingMonitor.__init__

    def init(self, settings, measure):
        def captured(start, stop, block):
            out = measure(start, stop, block)
            if start == 0:
                values.clear()  # a restarted or replayed transport is measured again
            for name, array in out.items():
                host = array.get() if hasattr(array, "get") else np.asarray(array)
                values.setdefault(name, []).append(np.asarray(host, dtype=np.float64))
            if on_values is not None:
                on_values()
            return out

        real_init(self, settings, captured)

    adaptive.StoppingMonitor.__init__ = init

    real_blocks = block_transport.transport_electron_blocks

    def transport_electron_blocks(simulate_block, *args, on_block=None, **kwargs):
        def simulate(start, stop):
            _sync()
            t0 = time.perf_counter()
            timers.memory.phase = "transport.simulate"
            block = simulate_block(start, stop)
            _sync()
            t1 = time.perf_counter()
            key = next((k for k in ("electron_id", "elec_id", "L_ang") if k in block), None)
            rows = -1 if key is None else int(block[key].shape[0])
            timers.blocks.append(
                {"start": start, "stop": stop, "simulate_s": t1 - t0, "rows": rows}
            )
            return block

        def monitor(start, stop, block):
            timers.memory.phase = "transport.measure"
            t0 = time.perf_counter()
            on_block(start, stop, block)
            _sync()
            timers.blocks[-1]["measure_s"] = time.perf_counter() - t0
            timers.memory.phase = "transport.simulate"

        return real_blocks(
            simulate, *args, on_block=None if on_block is None else monitor, **kwargs
        )

    block_transport.transport_electron_blocks = transport_electron_blocks
    block_transport.merge_electron_blocks = timers.wrap(
        "transport.join", block_transport.merge_electron_blocks, "transport.join"
    )
    block_transport.finalize_population_fields = timers.wrap(
        "transport.finalize", block_transport.finalize_population_fields, "transport.finalize"
    )
    runner._transport_case = timers.wrap("transport", runner._transport_case, "transport")
    runner._spectrum_case = timers.wrap("spectrum", runner._spectrum_case, "spectrum")
    adaptive.batch_means = timers.wrap("batch_means", adaptive.batch_means, "batch_means")


def _scalar_meta(out):
    return {
        k: v
        for k, v in out.items()
        if k.startswith("_")
        and isinstance(v, (int, float, str, bool, list))
        and not (isinstance(v, list) and len(v) > 64)
    }


def run(material, e0_kev, tilt_deg, block, out_prefix):
    from pyrite.montecarlo.runner.scheduling import run_cases

    case, precision = profile_case(material, float(e0_kev), float(tilt_deg), int(block))
    memory = Memory()
    timers = Timers(memory)
    values: dict[str, list] = {}
    incremental = os.environ.get("PROBE_CAPTURE_EVERY") == "1"
    # A stub record first, so ``replay`` can read a capture whose run timed out.
    Path(f"{out_prefix}.json").write_text(
        json.dumps(
            {
                "issue": 1252,
                "stub": True,
                "material": material,
                "E0_keV": float(e0_kev),
                "tilt_deg": float(tilt_deg),
                "block": int(block),
                "precision": precision.to_dict(),
                "realized_electrons": None,
            },
            default=_json_default,
        )
    )
    instrument(
        timers,
        values,
        (lambda: save_values(out_prefix, values, np.int64(block))) if incremental else None,
    )
    launches = Launches()
    launches.install()
    timings = []
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    error = None
    t0 = time.perf_counter()
    with memory:
        try:
            outs = run_cases([case], max_workers=0, progress=False, on_timing=timings.append)
            out = outs[0]
        except Exception as exc:  # record OOM/failure as evidence, never mask it
            out, error = None, f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    sampling = (out or {}).get("adaptive_sampling") or {}
    realized = sampling.get("realized_electrons")
    n_segments = int(out["n_segments"]) if out is not None and "n_segments" in out else None
    blocks = timers.blocks
    record = {
        "issue": 1252,
        "code_sha": os.environ.get("PROBE_SHA"),
        "profile": PROFILE,
        "material": material,
        "E0_keV": float(e0_kev),
        "tilt_deg": float(tilt_deg),
        "block": int(block),
        "precision": precision.to_dict(),
        "backend": os.environ.get("PYRITE_MC_BACKEND"),
        "device": None
        if CUPY is None
        else CUPY.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "error": error,
        "wall_s": wall,
        "split_s": timers.seconds,
        "split_calls": timers.calls,
        "blocks_run": len(blocks),
        "block_simulate_s": float(sum(b["simulate_s"] for b in blocks)),
        "block_measure_s": float(sum(b.get("measure_s", 0.0) for b in blocks)),
        "per_block": blocks,
        "kernel": launches.summary(),
        "memory": memory.summary(),
        "host_peak_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "host_peak_before_MiB": rss_before / 1024,
        "realized_electrons": realized,
        "stop_reason": sampling.get("stop_reason"),
        "checks": sampling.get("checks"),
        "pilot": sampling.get("pilot"),
        "replayed_transport": sampling.get("replayed_transport"),
        "relative_se": {
            k: v.get("relative_se") for k, v in (sampling.get("statistics") or {}).items()
        },
        "guards": {k: v.get("passes") for k, v in (sampling.get("statistics") or {}).items()},
        "n_segments": n_segments,
        "rows_per_electron": None if not realized or n_segments is None else n_segments / realized,
        "runner_timing": timings,
        "output_meta": {} if out is None else _scalar_meta(out),
    }
    if values:
        save_values(out_prefix, values, np.int64(block))
    Path(f"{out_prefix}.json").write_text(json.dumps(record, default=_json_default))
    summary = {k: v for k, v in record.items() if k not in ("per_block", "runner_timing")}
    print(json.dumps(summary, default=_json_default))
    return 1 if error else 0


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def replay(values_npz, out_json, *blocks):
    """Stopping decisions for other block sizes on one captured population."""
    from pyrite._precision import Precision
    from pyrite.montecarlo.runner.adaptive import run_stopping_rule

    stem = Path(values_npz).name.removesuffix(".values.npz")
    source = json.loads(Path(values_npz).with_name(stem + ".json").read_text())
    data = np.load(values_npz)
    names = [k for k in data.files if k != "block"]
    population = {k: data[k] for k in names}
    captured = min(v.size for v in population.values())
    base = Precision.from_dict(source["precision"])
    rows = []
    for block in (int(b) for b in blocks):
        settings = rescaled(base, block)
        beyond = [False]

        def measure(start, stop, _block, beyond=beyond):
            if stop > captured:
                beyond[0] = True
                raise StopIteration
            return {k: v[start:stop] for k, v in population.items()}

        try:
            record = run_stopping_rule(settings, measure)
        except StopIteration:
            record = None
        rows.append(
            {
                "block": block,
                "min": settings.min_histories,
                "pilot": settings.pilot_histories,
                "max": settings.max_histories,
                "realized": None if record is None else record["realized_electrons"],
                "stop_reason": "beyond_capture" if beyond[0] else record["stop_reason"],
                "checks": None if record is None else record["checks"],
                "projected": None
                if record is None
                else (record["pilot"] or {}).get("projected_electrons"),
                "relative_se": None
                if record is None
                else {k: v["relative_se"] for k, v in record["statistics"].items()},
            }
        )
    report = {
        "source": stem,
        "source_block": source["block"],
        "source_realized": source["realized_electrons"],
        "captured_electrons": int(captured),
        "replayed_transport": source.get("replayed_transport"),
        "rows": rows,
    }
    Path(out_json).write_text(json.dumps(report, indent=1, default=_json_default))
    print(json.dumps(report, default=_json_default))


def table(out_dir):
    records = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]
    records = [r for r in records if r.get("issue") == 1252 and "wall_s" in r]
    records.sort(key=lambda r: (r["material"], r["E0_keV"], r["tilt_deg"], r["block"]))
    head = (
        "material E0_keV tilt block realized stop wall_s transport_s spectrum_s kernel_s "
        "blocks rows/e card_peak_MiB pool_res_MiB host_MiB error"
    )
    print(head)
    for r in records:
        mem = r.get("memory") or {}
        split = r.get("split_s") or {}
        rpe = r.get("rows_per_electron")
        print(
            r["material"],
            r["E0_keV"],
            r["tilt_deg"],
            r["block"],
            r.get("realized_electrons"),
            r.get("stop_reason"),
            f"{r['wall_s']:.1f}",
            f"{split.get('transport', float('nan')):.1f}",
            f"{split.get('spectrum', float('nan')):.1f}",
            f"{(r.get('kernel') or {}).get('kernel_s', float('nan')):.1f}",
            r.get("blocks_run"),
            "-" if rpe is None else f"{rpe:.1f}",
            f"{mem.get('card_peak_MiB', float('nan')):.0f}",
            f"{mem.get('pool_reserved_peak_MiB', float('nan')):.0f}",
            f"{r.get('host_peak_MiB', float('nan')):.0f}",
            (r.get("error") or "")[:60],
        )


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    command, rest = argv[1], argv[2:]
    if command == "run" and len(rest) == 5:
        if os.environ.get("PROBE_PROFILE") == "1":
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            rc = profiler.runcall(run, *rest)
            profiler.dump_stats(f"{rest[-1]}.prof")
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(40)
            pstats.Stats(profiler).sort_stats("tottime").print_stats(30)
            return rc
        return run(*rest)
    if command == "replay" and len(rest) >= 3:
        return replay(*rest)
    if command == "table" and len(rest) == 1:
        return table(*rest)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
