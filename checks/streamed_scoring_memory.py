"""Peak memory and wall time of joined vs live streamed scoring (#611).

Usage:
  streamed_scoring_memory.py run MODE NE BLOCK OUT_JSON [MATERIAL [E0_KEV [CORE [THICKNESS_ANG [SEED [TILT_DEG [AZIM_DEG]]]]]]]
  streamed_scoring_memory.py compare REF_JSON OTHER_JSON...
  streamed_scoring_memory.py gpu-busy REPORT.sqlite

MODE: ``joined`` (``run_case``: full trajectory downloaded, then spectra;
BLOCK ignored), ``joined-device`` (the same with ``keep_segments_on_device``,
as ``pyrite.phases`` runs it), ``streamed`` (``run_case_streamed`` with
BLOCK electrons per block), ``streamed-complete`` (the same with each block's
hard-photon completion forced back on, the pre-#864 cost; spectra must match
``streamed`` bitwise). ``PYRITE_MC_TRANSPORT_SCHEDULE`` is recorded.
CORE (default ``per-electron``; ``cuda`` on the GPU box) is pinned for both
arms, because ``auto`` can resolve to the block-incapable lockstep core.
One arm per process, so ``ru_maxrss`` is that arm's host peak. Device peak is
the CuPy pool's reserved and used bytes, sampled every 5 ms above the
pre-run level (reserved bytes are never returned, so they include
fragmentation; used bytes are live arrays).

Phase timing wraps the runner's own seams with a device synchronization on
each side, so asynchronous work is charged to the phase that queued it:
``transport`` and ``spectrum`` for joined arms; for streamed arms per-block
``score`` (upload + scoring + reduction into the running totals), ``release``
(the per-block pool release) and ``transport`` (the rest of the wall). The
synchronizations themselves are part of this measurement, not of the
production path. ``segment_bytes`` sums the host segment arrays handed to
scoring: each crosses the bus once down from transport and once up again for
scoring (``joined-device`` skips both copies).

The case is a production-style coupled BremsLib midpoint run on a slab
(1 um unless THICKNESS_ANG is given) with explicit line and continuum bins (no case-local grid resolution).

``STREAM_GRID`` (#1001) selects the line grid: ``final`` (default, the
explicit bins above), ``policy`` (the scene's automatic ``line_grid_policy``)
or ``measured`` (``resonance-population`` bandwidth, ``resonance-local``
spacing, ``bin-mean`` quadrature). Policy arms disable the line-grid speed
cache, so every arm resolves its grid. They time grid resolution as phase
``grid`` (joined) or ``grid_pass`` (streamed, the whole first transport pass;
its pool releases are ``grid_pass.release``), and record the resolved grid,
the evidence bytes of the streamed grid pass, and its resolution record.
MODE ``grid-pass`` runs only the streamed grid pass (no scoring), for its
memory and wall against NE. ``compare`` requires bitwise-equal ``E_grid``.

``STREAM_SPILL_MIB`` (#1141) sets the grid pass's spill threshold in MiB
(``0`` always spills, ``none`` never; unset keeps the default) and ``STREAM_SPILL_DIR`` its
scratch parent. Every arm also records ``host_anon_peak_MiB``, the sampled
peak of ``RssAnon``: spilled evidence is read through file-backed memory maps,
which ``ru_maxrss`` counts although the kernel can reclaim them.

``STREAM_NVTX=1`` (#862) also opens an NVTX range ``phase.<name>`` around each
timed call and ``phase.run`` around the arm. ``gpu-busy`` reads an
``nsys export --type sqlite`` of such a run and prints, per phase, its wall
time and the union of GPU kernel, copy and memset intervals inside it (any
stream), and the top kernels by GPU time. ``transport`` is ``run`` less the
top-level ``score``, ``select``, ``release`` and ``grid_pass`` ranges
(``select`` is the automatic reflection selection on the held prefix, #1281).
Idle GPU time inside ``transport`` bounds what a transport/scoring overlap
could hide on the device. ``STREAM_PHASES=0`` skips
every phase wrapper (no synchronizations) and records only the production
wall time. ``STREAM_CPROFILE=PATH`` profiles the arm with ``cProfile`` and
writes its stats to ``PATH`` (``python -m pstats PATH``) for host-cost
attribution by function. ``STREAM_CXR=none`` (#862) runs the case with
``Physics(cxr_source="none")`` (#1327): brem-only scoring, no lines,
reflection selection or lattice share.
"""

import contextlib
import json
import os
import resource
import sys
import threading
import time
from dataclasses import replace

import numpy as np

from pyrite.config import FixedSampling

SPECTRA = ("spec", "spec_characteristic", "brem_wide")
GRID = os.environ.get("STREAM_GRID", "final")
NVTX = os.environ.get("STREAM_NVTX") == "1"
CXR_SOURCE = os.environ.get("STREAM_CXR", "mc")


def _case(material, ne, e0_kev, thickness_ang=1.0e4, seed=None, tilt_deg=30.0, azim_deg=0.0):
    import pyrite as pr
    from pyrite import api
    from pyrite.detectors import EnergyBins

    cxr = np.linspace(1000.0, 20000.0, 400) if GRID == "final" else None
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=ne, brem_histories=ne),
        energy_model="midpoint",
        energy_bins=EnergyBins(cxr=cxr, brem=np.arange(1000.0, e0_kev * 1e3, 1000.0)),
    )
    physics = pr.Physics(
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        cxr_source=CXR_SOURCE,
    )
    target = pr.geometry.Slab(
        material,
        thickness_ang=thickness_ang,
        tilt_deg=tilt_deg,
        tilt_azim_deg=azim_deg,
        footprint=pr.geometry.Footprint(5.0, 5.0),
    )
    scene = pr.Scene(
        pr.Beam(energy_keV=e0_kev, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        target,
        pr.Detector(),
        physics=physics,
    )
    case = api.build_case(scene, numerics)
    assert (case.get("line_grid_policy") is None) == (GRID == "final"), GRID
    if GRID == "measured":
        from pyrite._cxr_grid_policy import resolve_cxr_grid_policy

        band = case["line_grid_policy"]["bandwidth"]
        policy = resolve_cxr_grid_policy(
            start_eV=band["start_eV"],
            stop_eV=band["stop_eV"],
            per_call={
                "bandwidth": "resonance-population",
                "resolution": "resonance-local",
                "quadrature": "bin-mean",
            },
        ).payload()
        case = replace(case, line_grid_policy=policy, line_quadrature="bin-mean")
    elif GRID != "policy" and GRID != "final":
        raise SystemExit(f"unknown STREAM_GRID {GRID!r}")
    case = replace(case, E_cut_lines_keV=5.0, E_cut_brem_keV=5.0)
    return case if seed is None else replace(case, seed=int(seed))


@contextlib.contextmanager
def _nvtx(name):
    """An NVTX range ``phase.<name>`` when ``STREAM_NVTX=1`` (#862)."""
    if not NVTX:
        yield
        return
    from cupy.cuda import nvtx

    nvtx.RangePush(f"phase.{name}")
    try:
        yield
    finally:
        nvtx.RangePop()


def _sync():
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except Exception:
        pass


class _Phases:
    """Synchronized wall-clock totals of wrapped callables, by phase name."""

    def __init__(self):
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.segment_bytes = 0
        self.block_rows: list[int] = []
        self.prefix = ""

    def _count(self, segs):
        self.block_rows.append(int(segs["L_ang"].shape[0]))
        self.segment_bytes += sum(
            int(getattr(v, "nbytes", 0)) for v in segs.values() if hasattr(v, "shape")
        )

    def wrap(self, name, fn, *, block_arg=False, result_segs=False):
        def timed(*args, **kwargs):
            if block_arg:
                self._count(args[-1])
            _sync()
            t0 = time.perf_counter()
            try:
                with _nvtx(self.prefix + name):
                    out = fn(*args, **kwargs)
                    _sync()
                if result_segs:
                    self._count(out["segs"])
                return out
            finally:
                _sync()
                key = self.prefix + name
                self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - t0
                self.calls[key] = self.calls.get(key, 0) + 1

        return timed


def _configure_spill():
    """Apply ``STREAM_SPILL_MIB``/``STREAM_SPILL_DIR`` to the grid pass (#1141)."""
    mib, directory = os.environ.get("STREAM_SPILL_MIB"), os.environ.get("STREAM_SPILL_DIR")
    if mib is None and directory is None:
        return
    from functools import partial

    from pyrite.montecarlo.runner import line_grid_spill

    kwargs = {"directory": directory}
    if mib is not None:
        kwargs["threshold_bytes"] = None if mib == "none" else int(float(mib) * 2**20)
    line_grid_spill.GridSpillPolicy = partial(line_grid_spill.GridSpillPolicy, **kwargs)


class _AnonPeak:
    """Sampled peak ``RssAnon`` of this process [bytes]."""

    def __init__(self, interval=0.05):
        self.interval, self.peak = interval, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def sample():
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.sample())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.sample())


def _instrument(mode, phases):
    """Patch the runner seams of ``mode`` with ``phases`` timers."""
    from pyrite import _backend
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_driver, cxr_grid, streamed_line_grid, streamed_scoring

    # Every policy arm resolves its grid (#1001); a warm cache would skip it.
    cxr_grid.cached_coordinates = lambda key: None
    cxr_grid.store_coordinates = lambda key, record: None
    case_driver.resolve_cxr_grid = phases.wrap("grid", case_driver.resolve_cxr_grid)
    real_grid_pass = streamed_line_grid.resolve_streamed_line_grid

    def grid_pass(*args, **kwargs):
        phases.prefix = "grid_pass."
        try:
            return real_grid_pass(*args, **kwargs)
        finally:
            phases.prefix = ""

    streamed_line_grid.resolve_streamed_line_grid = phases.wrap("grid_pass", grid_pass)

    if mode.startswith("joined"):
        last = [None]

        def stage_sink(stage, _done, _total):
            _sync()
            now = time.perf_counter()
            if last[0] is not None:
                key = f"spectrum.{stage}"
                phases.seconds[key] = phases.seconds.get(key, 0.0) + now - last[0]
            last[0] = now

        real_spec = runner._spectrum_case

        def spec_case(*args, **kwargs):
            _sync()
            last[0] = time.perf_counter()
            token = runner._SPECTRUM_PROGRESS.set(stage_sink)
            try:
                return real_spec(*args, **kwargs)
            finally:
                runner._SPECTRUM_PROGRESS.reset(token)

        runner._spectrum_case = spec_case
        runner._transport_case = phases.wrap("transport", runner._transport_case, result_segs=True)
        runner._spectrum_with_recovery = phases.wrap("spectrum", runner._spectrum_with_recovery)
    else:
        scorer = streamed_scoring.BlockSpectrumScorer
        scorer.add = phases.wrap("score", scorer.add, block_arg=True)
        # Per-estimator split inside ``add`` (#872); nested in ``score``.
        runner._segments_on_device = phases.wrap("score.upload", runner._segments_on_device)
        from pyrite.montecarlo.runner.emission_adapters import EmissionAdapter

        real_evaluate = EmissionAdapter.evaluate
        active_slots = set()

        def evaluate(self, batch, slot, **kwargs):
            # The continuum adapter delegates to a native adapter through the
            # same method. Its inclusive outer timer already counts that work.
            if slot in active_slots:
                return real_evaluate(self, batch, slot, **kwargs)
            active_slots.add(slot)
            try:
                timed = phases.wrap(f"score.{slot}", real_evaluate)
                return timed(self, batch, slot, **kwargs)
            finally:
                active_slots.remove(slot)

        EmissionAdapter.evaluate = evaluate
        backend = _backend.BACKEND
        backend.release_memory = phases.wrap("release", backend.release_memory)
        # Automatic reflection selection on the held prefix (#1281, #862): its own
        # top-level phase, otherwise charged to transport.
        from pyrite.montecarlo.runner import streamed_selection

        streamed_selection._select_on_blocks = phases.wrap(
            "select", streamed_selection._select_on_blocks
        )


class _PoolPeak:
    """Sample CuPy pool reserved bytes on a thread; no-op without CuPy."""

    def __init__(self):
        try:
            import cupy

            self.pool = cupy.get_default_memory_pool()
        except Exception:
            self.pool = None
        self.base = self.peak = 0 if self.pool is None else self.pool.total_bytes()
        self.used_base = self.used_peak = 0 if self.pool is None else self.pool.used_bytes()
        self._stop = threading.Event()

    def __enter__(self):
        if self.pool is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(0.005):
            self.peak = max(self.peak, self.pool.total_bytes())
            self.used_peak = max(self.used_peak, self.pool.used_bytes())

    def __exit__(self, *exc):
        self._stop.set()
        if self.pool is not None:
            self._thread.join()
            self.peak = max(self.peak, self.pool.total_bytes())
            self.used_peak = max(self.used_peak, self.pool.used_bytes())

    @property
    def above_base(self):
        return int(self.peak - self.base)


def run(
    mode,
    ne,
    block,
    out_path,
    material="silicon",
    e0_kev=300.0,
    core="per-electron",
    thickness_ang=1.0e4,
    seed=None,
    tilt_deg=30.0,
    azim_deg=0.0,
):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed

    case = _case(
        material,
        int(ne),
        float(e0_kev),
        float(thickness_ang),
        seed,
        float(tilt_deg),
        float(azim_deg),
    )
    phases = _Phases()
    if os.environ.get("STREAM_PHASES", "1") != "0":
        _instrument(mode, phases)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _configure_spill()
    anon_before = _AnonPeak.sample()
    profile_path = os.environ.get("STREAM_CPROFILE")
    profiler = None
    if profile_path:
        import cProfile

        profiler = cProfile.Profile()
    t0 = time.perf_counter()
    with (
        _PoolPeak() as pool,
        _AnonPeak() as anon,
        _nvtx("run"),
        profiler if profiler is not None else contextlib.nullcontext(),
    ):
        if mode == "joined":
            out = runner.run_case(case, transport_core=core)
        elif mode == "joined-device":
            out = runner.run_case(case, transport_core=core, keep_segments_on_device=True)
        elif mode == "streamed":
            out = run_case_streamed(case, int(block), transport_core=core)
        elif mode == "grid-pass":
            from pyrite.montecarlo.runner import streamed_line_grid
            from pyrite.montecarlo.runner.artifacts import _case_geometry

            _, _, _, n_hat, groove = _case_geometry(case)
            grid, record, info = streamed_line_grid.resolve_streamed_line_grid(
                case, int(block), transport_core=core, n_hat=n_hat, groove=groove
            )
            out = {
                "E_grid": grid,
                "line_grid_resolved": record,
                "_streamed_line_grid_pass": info,
                "n_segments": -1,
                "eta": float("nan"),
                **{key: [] for key in SPECTRA},
            }
        elif mode == "streamed-complete":
            from pyrite.montecarlo.transport import hard_radiative

            add_fields = hard_radiative.add_radiative_result_fields
            hard_radiative.add_radiative_result_fields = lambda *a, complete=True, **k: add_fields(
                *a, complete=True, **k
            )
            out = run_case_streamed(case, int(block), transport_core=core)
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    wall = time.perf_counter() - t0
    if profiler is not None:
        profiler.dump_stats(profile_path)
    if mode.startswith("streamed"):
        accounted = (
            phases.seconds.get("score", 0.0)
            + phases.seconds.get("select", 0.0)
            + phases.seconds.get("release", 0.0)
            + phases.seconds.get("grid_pass", 0.0)
        )
        phases.seconds["transport"] = wall - accounted
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "mode": mode,
        "Ne": int(ne),
        "block": int(block) if mode.startswith(("streamed", "grid-pass")) else None,
        "grid": GRID,
        "schedule": os.environ.get("PYRITE_MC_TRANSPORT_SCHEDULE", "default"),
        "material": material,
        "E0_keV": float(e0_kev),
        "core": core,
        "thickness_ang": float(thickness_ang),
        "tilt_deg": float(tilt_deg),
        "azim_deg": float(azim_deg),
        "wall_s": wall,
        "host_peak_MiB": rss_peak / 1024,
        "host_peak_before_run_MiB": rss_before / 1024,
        "host_anon_peak_MiB": anon.peak / 2**20,
        "host_anon_before_run_MiB": anon_before / 2**20,
        "device_pool_peak_above_base_MiB": pool.above_base / 2**20,
        "device_used_peak_above_base_MiB": (pool.used_peak - pool.used_base) / 2**20,
        "fp64": os.environ.get("PYRITE_FP64") == "1",
        "cxr_source": CXR_SOURCE,
        "overlap": os.environ.get("PYRITE_STREAM_OVERLAP") == "1",
        "seed": int(case["seed"]),
        "phase_s": phases.seconds,
        "phase_calls": phases.calls,
        "segment_bytes": phases.segment_bytes,
        "max_block_rows": max(phases.block_rows, default=0),
        "n_segments": int(out["n_segments"]),
        "eta": float(out["eta"]),
        "spectrum_meta": {
            k: v
            for k, v in out.items()
            if k.startswith("_")
            and isinstance(v, (int, float, str, bool, list))
            and not (isinstance(v, list) and len(v) > 64)
        },
        **{key: np.asarray(out[key]).tolist() for key in SPECTRA},
        "E_grid_points": int(np.asarray(out["E_grid"]).size),
        "E_grid": np.asarray(out["E_grid"]).tolist() if GRID != "final" else None,
        "grid_pass": out.get("_streamed_line_grid_pass"),
        "line_grid_resolved": _jsonable(out.get("line_grid_resolved")),
    }
    with open(out_path, "w") as handle:
        json.dump(record, handle)
    summary = {
        k: v for k, v in record.items() if k not in (*SPECTRA, "E_grid", "line_grid_resolved")
    }
    print(json.dumps(summary))


def _jsonable(value):
    """A resolution record as JSON values (numpy scalars and arrays to lists)."""
    return None if value is None else json.loads(json.dumps(value, default=_numpy_default))


def _numpy_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def compare(ref_path, *others):
    ref = json.load(open(ref_path))
    worst = 0.0
    for path in others:
        other = json.load(open(path))
        if other.get("E_grid") != ref.get("E_grid"):
            print(f"{path} E_grid differs from {ref_path}")
            return 1
        same_eta = np.array_equal(other["eta"], ref["eta"], equal_nan=True)  # grid-pass: NaN
        assert other["n_segments"] == ref["n_segments"] and same_eta, path
        for key in SPECTRA:
            a, b = np.asarray(other[key]), np.asarray(ref[key])
            scale = np.maximum(np.abs(b), np.finfo(float).tiny)
            rel = float(np.max(np.abs(a - b) / scale, initial=0.0))
            worst = max(worst, rel)
            print(f"{path} {key}: max rel diff {rel:.3e}")
    print(f"worst max rel diff {worst:.3e}")
    return 0 if worst <= 1e-12 else 1


def _union(intervals):
    """Sorted disjoint union of ``(start, end)`` intervals."""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def _covered(union, start, end):
    """Length of ``[start, end)`` covered by a disjoint sorted ``union``."""
    import bisect

    i = max(bisect.bisect_right([u[0] for u in union], start) - 1, 0)
    total = 0
    for lo, hi in union[i:]:
        if lo >= end:
            break
        total += max(0, min(hi, end) - max(lo, start))
    return total


def gpu_busy(path):
    """Per-phase wall and GPU-busy seconds from an Nsight sqlite export (#862)."""
    import sqlite3

    db = sqlite3.connect(path)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    has_ids = "StringIds" in tables
    text = "COALESCE(e.text, s.value)" if has_ids else "e.text"
    join = "LEFT JOIN StringIds s ON e.textId = s.id" if has_ids else ""
    ranges = {}
    for name, start, end in db.execute(
        f"SELECT {text}, e.start, e.end FROM NVTX_EVENTS e {join} WHERE e.end IS NOT NULL"
    ):
        if name and name.startswith("phase."):
            ranges.setdefault(name[6:], []).append((start, end))
    activity = []
    kinds = ("KERNEL", "MEMCPY", "MEMSET")
    for kind in kinds:
        table = f"CUPTI_ACTIVITY_KIND_{kind}"
        if table in tables:
            activity += db.execute(f"SELECT start, end FROM {table}").fetchall()
    union = _union(activity)
    streams = (
        db.execute("SELECT COUNT(DISTINCT streamId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
        else 0
    )
    out = {"kernel_streams": streams}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        names = dict(db.execute("SELECT id, value FROM StringIds")) if has_ids else {}
        kernels = {}
        for name, start, end in db.execute(
            "SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            calls, total = kernels.get(name, (0, 0))
            kernels[name] = (calls + 1, total + end - start)
        ranked = sorted(kernels.items(), key=lambda item: -item[1][1])[:15]
        out["top_kernels"] = [
            {"name": str(names.get(n, n)), "calls": c, "gpu_s": t / 1e9} for n, (c, t) in ranked
        ]
    for name, spans in sorted(ranges.items()):
        wall = sum(end - start for start, end in spans)
        busy = sum(_covered(union, start, end) for start, end in spans)
        out[name] = {"calls": len(spans), "wall_s": wall / 1e9, "gpu_busy_s": busy / 1e9}
    (run_start, run_end), *_ = ranges.get("run", [(0, 0)])
    top = _union(
        span for key in ("score", "select", "release", "grid_pass") for span in ranges.get(key, ())
    )
    wall = (run_end - run_start) - sum(hi - lo for lo, hi in top)
    busy = _covered(union, run_start, run_end) - sum(_covered(union, lo, hi) for lo, hi in top)
    out["transport"] = {"wall_s": wall / 1e9, "gpu_busy_s": busy / 1e9}
    for record in out.values():
        if isinstance(record, dict) and record.get("wall_s", 0) > 0:
            record["gpu_busy_frac"] = record["gpu_busy_s"] / record["wall_s"]
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif sys.argv[1] == "compare":
        sys.exit(compare(*sys.argv[2:]))
    elif sys.argv[1] == "gpu-busy":
        print(json.dumps(gpu_busy(sys.argv[2]), indent=1))
    else:
        raise SystemExit(__doc__)
