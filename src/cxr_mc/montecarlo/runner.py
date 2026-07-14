"""
montecarlo.runner

Parallel case driver: the per-case transport + spectrum + brem worker
(run_case and its CPU/GPU phase split) and run_cases, which pipelines the
CPU transport across a worker pool behind the single-CUDA-context GPU phase.
The phase functions stay module-level so they pickle into Windows spawn
workers.
"""

import os
import sys
from time import perf_counter
from typing import Any

import numpy as np

from .._energy_grid import decode_energy_grid
from ._backend import _GPU, cp
from .geometry import tilted_geometry
from .spectrum import _segments_in_layer, mc_brem_spectrum, mc_spectrum
from .transport import simulate_trajectories

# Opt-in Gate-0 phase profiling for the sweep-acceleration work (TODO P?/#numba;
# see docs/acceleration-technique-evaluation.md). With CXR_MC_TIMING set (to
# anything but "" / "0"), each phase records its own wall time onto the dict it
# returns under a private "_t_*" key, and run_cases accumulates those (plus the
# GPU-idle wait) and prints a per-phase summary + the pipeline verdict. The keys
# are STRIPPED by _TimingAgg.collect before any result is stored/checkpointed, so
# timing never leaks into the pickle. The flag is read at import so it applies in
# every spawned transport worker too (env is inherited on spawn/forkserver).
_TIMING = os.environ.get("CXR_MC_TIMING", "") not in ("", "0")


def _env_chunk(name, default):
    """Chunk-size default, overridable via env for the A1 sweep-acceleration spike
    (docs/acceleration-technique-evaluation.md, A1: sweep spec/brem chunk on the lab
    box and read the GPU spectrum-phase time). Read once at import so it applies in
    the main GPU process; an explicit per-case ``spec_chunk``/``brem_chunk`` still
    wins. Unset / blank / non-positive / non-integer -> the memory-safe default."""
    try:
        v = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


_SPEC_CHUNK = _env_chunk("CXR_MC_SPEC_CHUNK", 40000)  # segments per line-spectrum GPU matmul
_BREM_CHUNK = _env_chunk("CXR_MC_BREM_CHUNK", 20000)  # segments per brem-spectrum GPU matmul

# A2 (docs/acceleration-technique-evaluation.md): stretch the CuPy memory-pool
# free cadence. free_all_blocks() forces a device sync + full realloc, so paying
# it once per case is the conservative default; the spike frees less often and
# leans on a reserved-pool watermark to stay bounded. Read once at import; the
# GPU free path is driver-process only (workers run transport), so no locking.
_FREE_EVERY = _env_chunk("CXR_MC_FREE_EVERY", 1)  # free the pool every N GPU cases (1 = per-case)
_FREE_WATERMARK_MB = _env_chunk(
    "CXR_MC_FREE_WATERMARK_MB", 0
)  # ...or when reserved pool exceeds this; 0 = off
_cases_since_free = 0  # GPU cases since the last free (module-global: single driver process)
_pool_peak_bytes = 0  # high-water reserved pool size, for the A2 operational watermark check


def _should_free(cases_since, every, reserved_bytes, watermark_mb):
    """A2 free-cadence predicate (pure, no CuPy so it unit-tests without a GPU).

    Free the pool when ``every`` cases have elapsed since the last free, OR when
    the reserved pool has crossed the watermark (``watermark_mb == 0`` -> the
    watermark is off). The default (every=1, watermark off) fires every case."""
    over_watermark = watermark_mb > 0 and reserved_bytes > watermark_mb * (1 << 20)
    return cases_since >= every or over_watermark


def _maybe_free_pool():
    """Return this case's GPU scratch to the device on the A2 cadence.

    ``free_all_blocks()`` releases the CuPy pool's free blocks back to the card so
    a long sweep can't let the reserved pool grow/fragment until it fills VRAM --
    but it forces a device sync + full realloc, so A2 stretches how often it runs
    (see :func:`_should_free`). The trigger reads ``total_bytes()`` (reserved), NOT
    ``used_bytes()``: by the time control reaches this inter-case point the case's
    CuPy temporaries are already dereferenced, so ``used_bytes()`` is ~0 and would
    never trip -- ``total_bytes()`` is the footprint that actually grows. Default
    (1 / off) reproduces the original per-case free exactly. Driver-process only,
    so the module counter needs no lock."""
    global _cases_since_free, _pool_peak_bytes
    pool = cp.get_default_memory_pool()
    reserved = pool.total_bytes()
    if reserved > _pool_peak_bytes:
        _pool_peak_bytes = reserved
    _cases_since_free += 1
    if _should_free(_cases_since_free, _FREE_EVERY, reserved, _FREE_WATERMARK_MB):
        pool.free_all_blocks()
        _cases_since_free = 0


class _TimingAgg:
    """Main-process accumulator for CXR_MC_TIMING phase profiling.

    Lives only in the driver process (never pickled). Per-case transport and
    spectrum deltas ride back on the phase dicts (workers -> main for transport);
    the GPU-idle wait is measured directly in run_cases' pipeline loop. ``collect``
    both records and strips the private keys so stored results stay clean.
    """

    def __init__(self):
        self.transport: list[float] = []  # worker compute time for _transport_case
        self.spectrum: list[float] = []  # _spectrum_case body (GPU work in main proc)
        self.wait: list[float] = []  # driver blocked on the transport future (GPU idle)

    def collect(self, out):
        """Pop the private _t_* deltas a phase dict rode back on and record them."""
        t = out.pop("_t_transport", None)
        if t is not None:
            self.transport.append(t)
        s = out.pop("_t_spectrum", None)
        if s is not None:
            self.spectrum.append(s)

    def report(self, mode, nw):
        """Print the phase split, GPU-idle fraction, and pipeline verdict to stderr."""
        _report_timing(self, mode, nw)


def _fmt_ms(xs):
    """(mean, median, n) formatted in ms, or '(none)' for an empty series."""
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return "        (none)        "
    return f"mean {a.mean() * 1e3:8.2f} ms  median {np.median(a) * 1e3:8.2f} ms  (n={a.size})"


def _report_timing(agg, mode, nw):
    tr = np.asarray(agg.transport, dtype=float)
    sp = np.asarray(agg.spectrum, dtype=float)
    wt = np.asarray(agg.wait, dtype=float)
    lines = [
        "",
        f"[cxr-timing] mode={mode}  cases={max(tr.size, sp.size)}  workers={nw}",
        f"  transport (worker compute) : {_fmt_ms(tr)}",
        f"  spectrum  (GPU/main proc)  : {_fmt_ms(sp)}",
    ]
    if wt.size:
        lines.append(f"  driver wait on transport   : {_fmt_ms(wt)}")
        # GPU-idle fraction: of the driver's serial timeline (spectrum work +
        # blocking on the transport future), the share spent waiting. This is the
        # Gate-0 decision metric -- transport hidden behind spectrum => low.
        denom = wt.sum() + sp.sum()
        idle = wt.sum() / denom if denom else float("nan")
        # Warmup (pool fill + first-touch CUDA alloc/JIT) inflates the first ~nw
        # waits; report steady state too for the real production picture.
        drop = min(nw, max(0, wt.size - 1))
        wt_ss, sp_ss = wt[drop:], sp[drop:]
        denom_ss = wt_ss.sum() + sp_ss.sum()
        idle_ss = wt_ss.sum() / denom_ss if denom_ss else float("nan")
        lines.append(
            f"  GPU-idle fraction          : {idle:6.1%}  (all cases)"
            f"   |   {idle_ss:6.1%}  (steady state, first {drop} dropped)"
        )
        if sp.size and tr.size:
            feed = np.median(tr) / nw  # per-case transport throughput of the pool
            bound = "SPECTRUM-bound" if np.median(sp) > feed else "TRANSPORT-bound"
            lines.append(
                f"  pipeline balance           : {bound}  "
                f"(median spectrum {np.median(sp) * 1e3:.1f} ms vs "
                f"transport/nw {feed * 1e3:.1f} ms)"
            )
        # Gate-0 verdict per docs/acceleration-technique-evaluation.md decision rule.
        ref_idle = idle_ss if np.isfinite(idle_ss) else idle
        if ref_idle < 0.20:
            verdict = (
                "Branch A (accelerate the spectrum phase) is the production path; "
                "transport speedups (Branch B) are capped near the idle fraction."
            )
        elif ref_idle >= 0.25:
            verdict = (
                "transport pool cannot feed the card -- Branch B (faster transport) "
                "helps production too; do it alongside Branch A."
            )
        else:
            verdict = "20-25% idle: borderline -- Branch A first, re-measure before Branch B."
        lines.append(f"  Gate-0 verdict             : {verdict}")
    else:
        lines.append("  (CPU-only mode: no GPU phase. Split sizes the Branch B / mode-2 payoff.)")
    if _GPU and _pool_peak_bytes:
        cadence = f"every {_FREE_EVERY} cases" if _FREE_EVERY > 1 else "per case"
        wm = f", watermark {_FREE_WATERMARK_MB} MB" if _FREE_WATERMARK_MB > 0 else ""
        lines.append(
            f"  CuPy pool peak (reserved)  : {_pool_peak_bytes / (1 << 20):8.1f} MB"
            f"   (free {cadence}{wm})"  # A2 operational watermark: must stay bounded
        )
    lines.append("")
    print("\n".join(lines), file=sys.stderr, flush=True)


def run_case(case):
    """
    Worker for one (crystal, beam energy) Monte Carlo case: transport + line
    spectrum + bremsstrahlung. Module-level so it can be pickled into worker
    processes on Windows (notebook-defined functions cannot).

    case: a plain dict --
        required: crystal, composition, hkl_list, B_ang2, E0_keV, thickness_ang,
                theta_obs_rad, Ne, Ne_brem, seed, and EITHER a single
                E_grid = (start_eV, stop_eV, step_eV) OR the decoupled pair
                E_grid_line / E_grid_brem (legacy uniform triples or exact arrays):
                the lines are evaluated on the fine NARROW E_grid_line, the
                smooth bremsstrahlung on the coarse WIDE E_grid_brem. Sweep-built
                uniform brem triples extend to the beam energy; scalar/nonuniform
                exact arrays retain their specified samples.
        optional: tilt_deg (0), tilt_azim_deg (0), beam_uvw (None),
                azimuth_rad (0), recip_miscut_rad (None; (polar_rad, azim_rad)
                crystal miscut of g relative to n -- None is a strict no-op,
                see montecarlo.geometry._orientation_R), E_cut_lines_keV (5),
                E_cut_brem_keV (1),
                spec_chunk (40000) / brem_chunk (20000): segments per GPU matmul
                -- lower these to cap peak GPU memory on a busy/shared device;
                the per-case default is overridable via the CXR_MC_SPEC_CHUNK /
                CXR_MC_BREM_CHUNK env vars (A1 sweep-accel spike, an explicit
                per-case value still wins),
                sinc_cutoff (None = exact lineshapes; windowing buys nothing
                for bulk targets, where scattering Doppler-spreads the lines
                across the whole grid),
                beam_fwhm_mm (None): transverse electron-beam spot size (Gaussian
                FWHM, mm) -- None is a strict no-op, the point-source beam
                (montecarlo.transport.simulate_trajectories, BIT-FOR-BIT
                unchanged spectrum; see that docstring for why),
                crystal_width_mm / crystal_height_mm (both None): optional full
                rectangular crystal footprint dimensions in mm. They must be
                supplied together and strictly positive to enable finite lateral
                transport and six-face self-absorption; both None retains the
                laterally infinite slab,
                mosaic_mc_fwhm_rad (None) / mosaic_mc_nodes (1): the exact
                Monte-Carlo crystal-mosaicity average (mc_spectrum); None/1 ->
                perfect crystal,
                brem_step_eV (10; legacy single-E_grid fallback only)

    Returns dict(E_grid, spec, brem [on E_grid], E_grid_brem, brem_wide [the
                full-range background], eta, n_segments) plus crystal/E0.
    """
    return _spectrum_case(case, _transport_case(case))


def _transport_case(case):
    """CPU-only phase of run_case: the line + brem trajectory transport (pure
    numpy, never touches the GPU). Returns the segments + geometry + grids the
    spectrum phase consumes. run_cases farms this out to a worker pool so the
    transport of upcoming cases overlaps the GPU work on the current one."""
    t0 = perf_counter() if _TIMING else 0.0
    if "E_grid_line" in case:
        E_grid = decode_energy_grid(case["E_grid_line"])
        E_brem = decode_energy_grid(case["E_grid_brem"])
    else:
        E_grid = decode_energy_grid(case["E_grid"])
        step_b = case.get("brem_step_eV", 10.0)
        E_brem = np.arange(E_grid[0], E_grid[-1] + step_b, step_b)
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    # film-on-substrate stack drives multilayer transport too (substrate
    # backscatter / substrate brem); None -> single-material slab (unchanged).
    layers = case.get("abs_layers")
    beam_fwhm_mm = case.get("beam_fwhm_mm")
    segs = simulate_trajectories(
        case["E0_keV"],
        case["Ne"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_lines_keV", 5.0),
        seed=case["seed"],
        beam_dir=beam,
        layers=layers,
        beam_fwhm_mm=beam_fwhm_mm,
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
    )
    segs_b = simulate_trajectories(
        case["E0_keV"],
        case["Ne_brem"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        seed=case["seed"] + 1,
        beam_dir=beam,
        layers=layers,
        beam_fwhm_mm=beam_fwhm_mm,
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
    )
    tp: dict[str, Any] = dict(E_grid=E_grid, E_brem=E_brem, n_hat=n_hat, segs=segs, segs_b=segs_b)
    if _TIMING:
        tp["_t_transport"] = perf_counter() - t0
    return tp


def _brem_wide_from_segments(segs_b, E_brem, case, n_hat, abs_layers):
    """Bremsstrahlung background on ``E_brem`` from already-transported brem
    segments ``segs_b``. EVERY layer radiates with its OWN composition (each
    Z^2 cross section) and self-absorbs through the WHOLE stack
    (``layers=abs_layers``); the per-layer contributions are summed. A single
    layer (``n_layers == 1``) is exactly the old single-material brem. Honors
    ``brem_chunk`` (segments per GPU matmul). Pure move of _spectrum_case's brem
    block; shared with :func:`_brem_for_case` so a brem-only repair regenerates
    the SAME multilayer background as a live sweep."""
    brem_chunk = case.get("brem_chunk") or _BREM_CHUNK
    n_lay = int(segs_b.get("n_layers", 1))
    if n_lay == 1:
        return mc_brem_spectrum(
            segs_b,
            E_brem,
            composition=case["composition"],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
        )
    brem_wide = np.zeros(E_brem.shape, dtype=float)
    for L in range(n_lay):
        sL = _segments_in_layer(segs_b, L)
        if sL["L_ang"].size == 0:
            continue
        brem_wide = brem_wide + mc_brem_spectrum(
            sL,
            E_brem,
            composition=abs_layers[L][2],
            n_hat=n_hat,
            chunk=brem_chunk,
            layers=abs_layers,
        )
    return brem_wide


def _brem_for_case(case, E_brem):
    """Regenerate a case's bremsstrahlung background on ``E_brem`` from scratch:
    build the tilted geometry, transport ``Ne_brem`` electrons through the stack
    (``layers=abs_layers``, same ``seed + 1`` offset as a live run), and sum brem
    per layer via :func:`_brem_wide_from_segments`. Returns ``brem_wide``.

    This is the brem half of run_case's transport + spectrum phases factored out
    so :func:`cxr_mc.run.repair_brem_wide` reuses the EXACT live-sweep path.
    Previously the repair rebuilt single-slab brem by hand -- ``layers=`` omitted,
    no per-layer sum, ``brem_chunk`` ignored -- silently dropping substrate
    backscatter/brem and cross-stack absorption on stacked/multilayer records."""
    abs_layers = case.get("abs_layers")
    beam, n_hat = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0.0)),
        np.deg2rad(case.get("tilt_azim_deg", 0.0)),
    )
    segs_b = simulate_trajectories(
        case["E0_keV"],
        case["Ne_brem"],
        case["thickness_ang"],
        composition=case["composition"],
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        seed=case["seed"] + 1,
        beam_dir=beam,
        layers=abs_layers,
        beam_fwhm_mm=case.get("beam_fwhm_mm"),
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
    )
    return _brem_wide_from_segments(segs_b, E_brem, case, n_hat, abs_layers)


def _spectrum_case(case, tp):
    """GPU phase of run_case: line spectrum + brem from the already-transported
    segments ``tp`` (from _transport_case). Runs in the main process, so only one
    CUDA context ever touches the device."""
    t0 = perf_counter() if _TIMING else 0.0
    E_grid, E_brem, n_hat = tp["E_grid"], tp["E_brem"], tp["n_hat"]
    segs, segs_b = tp["segs"], tp["segs_b"]
    # optional film-on-substrate stack (None -> single slab, unchanged)
    abs_layers = case.get("abs_layers")

    # LINES: each CRYSTALLINE layer radiates its own PXR/CBS lines, summed
    # INCOHERENTLY (separate crystals -> no cross-layer coherence); every line
    # self-absorbs through the WHOLE stack (layers=abs_layers). `layer_radiators`
    # is a per-layer list aligned with the stack -- a dict of crystal params for a
    # crystalline layer (film or crystalline substrate), None for an amorphous one
    # (no coherent lines). layer_radiators absent -> single slab: the film radiates
    # from ALL its segments via the case's scalar crystal keys (bit-for-bit the
    # pre-multilayer path). See docs/multilayer-materials.md (per-layer radiation).
    radiators = case.get("layer_radiators")
    mosaic_kw = dict(
        mosaic_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),  # None -> perfect crystal
        mosaic_nodes=case.get("mosaic_mc_nodes", 1),
    )
    spec_chunk = case.get("spec_chunk") or _SPEC_CHUNK
    if radiators is None:
        spec = mc_spectrum(
            segs,
            E_grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            n_hat=n_hat,
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case.get("beam_uvw"),
            azimuth_rad=case.get("azimuth_rad", 0.0),
            recip_miscut_rad=case.get("recip_miscut_rad"),
            sinc_cutoff=case.get("sinc_cutoff"),
            chunk=spec_chunk,
            layers=abs_layers,
            **mosaic_kw,
        )
    else:
        spec = np.zeros(E_grid.shape, dtype=float)
        for L, rad in enumerate(radiators):
            if rad is None:  # amorphous layer -> no coherent lines
                continue
            sL = _segments_in_layer(segs, L)
            if sL["L_ang"].size == 0:
                continue
            spec = spec + mc_spectrum(
                sL,
                E_grid,
                crystal=rad["crystal"],
                hkl_list=rad["hkl_list"],
                n_hat=n_hat,
                B_ang2=rad["B_ang2"],
                composition=abs_layers[L][2],
                beam_uvw=rad.get("beam_uvw"),
                # per-layer in-plane orientation (LayerSpec.azimuth_deg); radiators
                # from pre-stack checkpoints lack the key -> case-level fallback
                azimuth_rad=rad.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
                recip_miscut_rad=rad.get("recip_miscut_rad", case.get("recip_miscut_rad")),
                sinc_cutoff=case.get("sinc_cutoff"),
                chunk=spec_chunk,
                layers=abs_layers,
                **mosaic_kw,
            )

    # BREM: EVERY layer radiates with its OWN composition (each Z^2 cross
    # section); each layer's brem self-absorbs through the whole stack, summed
    # over layers (a single layer is exactly the old single-material brem).
    # Factored into _brem_wide_from_segments so run.repair_brem_wide reuses this
    # SAME path (via _brem_for_case) and can't drift back to single-slab brem.
    brem_wide = _brem_wide_from_segments(segs_b, E_brem, case, n_hat, abs_layers)
    brem = np.interp(E_grid, E_brem, brem_wide)  # brem under the lines (line grid)
    # Return this case's GPU scratch on the A2 cadence so the CuPy memory pool
    # can't accumulate (and fragment) across a long sweep until it fills the card.
    if _GPU:
        _maybe_free_pool()
    out = dict(
        E_grid=E_grid,
        spec=spec,
        brem=brem,
        E_grid_brem=E_brem,
        brem_wide=brem_wide,
        eta=segs["n_backscattered"] / segs["Ne"],
        n_segments=int(segs["L_ang"].size),
        crystal=case["crystal"],
        E0_keV=case["E0_keV"],
    )
    if _TIMING:
        # Ride the phase deltas back to the driver on the result dict; run_cases'
        # _TimingAgg.collect strips both keys before the result is stored. Carry
        # _t_transport through so the CPU-pool path (where transport time only
        # exists inside this worker) can report the split too.
        out["_t_spectrum"] = perf_counter() - t0
        if "_t_transport" in tp:
            out["_t_transport"] = tp["_t_transport"]
    return out


def _worker_init():
    """
    Runs once in each worker process: drop to BELOW_NORMAL priority so the
    desktop stays responsive. Workers still use idle CPU at full speed; the
    OS just schedules interactive applications first.
    """
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[reportAttributeAccessIssue]
        # typed signatures matter: the untyped pseudo-handle (-1) gets
        # truncated on 64-bit and the call silently fails
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        try:
            if hasattr(os, "nice"):
                os.nice(10)  # type: ignore[reportAttributeAccessIssue]  # POSIX fallback
        except Exception:
            pass


def run_cases(cases, max_workers=None, progress=True, callback=None):
    """
    Run a list of case dicts through run_case, results in input order.

    GPU present (the usual path): the CPU transport is PIPELINED across a worker
    pool while THIS process drives the spectrum/brem serially on the single CUDA
    context -- the ~40% transport idle overlaps the GPU work, with no device
    contention (multiple CUDA contexts are what crawled the old max_workers>1).
    Workers run ONLY transport (pure CPU/numpy), never the GPU. Callbacks fire in
    input order as each case's GPU phase finishes.

    No GPU: cases run through a worker pool (or serially), completion order.

    max_workers: None -> sized automatically (a few transport workers when a GPU
        is present; ~3/4 of the CPUs otherwise). An integer pins the count; 0
        runs everything serially in this process (debugging / safe fallback).
    progress: tqdm bar over completed cases.
    callback: callable(i, case, out) invoked in THIS process as each case
        finishes; stream/checkpoint/plot without waiting for the batch.
        Exceptions propagate and abort the run.

    Crawl protections: workers run BELOW_NORMAL priority (_worker_init) and get
    single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, inherited) -- N workers
    x M BLAS threads is the classic oversubscription freeze.
    """

    def _maybe_bar(iterable):
        if not progress:
            return iterable
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, total=len(cases), desc="cases")
        except ImportError:
            # tqdm.auto picks the widget bar inside Jupyter, and that bar
            # raises ImportError AT CONSTRUCTION if ipywidgets is missing --
            # fall back to the plain-text console bar before giving up.
            try:
                from tqdm import tqdm

                return tqdm(iterable, total=len(cases), desc="cases")
            except ImportError:
                return iterable

    n = len(cases)
    results: list[Any] = [None] * n
    if n == 0:
        return results

    timing = _TimingAgg() if _TIMING else None

    def _serial():
        for i in _maybe_bar(range(n)):
            out = run_case(cases[i])
            if timing is not None:
                timing.collect(out)
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
        if timing is not None:
            timing.report("serial", nw=1)
        return results

    def _single_thread_blas():
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = "1"

    # ---- GPU: pipeline CPU transport (worker pool) behind the serial GPU ------
    if _GPU:
        if max_workers == 0:
            return _serial()
        if max_workers is None:
            ncpu = os.process_cpu_count() or os.cpu_count() or 8
            nw = max(2, min(n, ncpu // 2))  # ~physical cores; transport is the tail
        else:
            nw = min(max_workers, n)
        if nw < 2:
            return _serial()
        _single_thread_blas()
        from concurrent.futures import ProcessPoolExecutor

        prefetch = nw + 2  # keep the transport pool ahead
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as ex:
            inflight = {i: ex.submit(_transport_case, cases[i]) for i in range(min(prefetch, n))}
            for i in _maybe_bar(range(n)):
                tw0 = perf_counter() if timing is not None else 0.0
                tp = inflight.pop(i).result()  # transport (already overlapped)
                if timing is not None:
                    timing.wait.append(perf_counter() - tw0)  # GPU idle: draining the pool
                j = i + prefetch
                if j < n:
                    inflight[j] = ex.submit(_transport_case, cases[j])
                out = _spectrum_case(cases[i], tp)  # GPU, THIS process only
                if timing is not None:
                    timing.collect(out)
                results[i] = out
                if callback is not None:
                    callback(i, cases[i], out)
        if timing is not None:
            timing.report("GPU-pipeline", nw=nw)
        return results

    # ---- no GPU: serial in-process, or a full-case worker pool ---------------
    if max_workers is None:
        ncpu = os.process_cpu_count() or os.cpu_count() or 8
        max_workers = max(1, min(n, ncpu * 3 // 4))
    if max_workers == 0:
        return _serial()
    _single_thread_blas()
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
        futures = {ex.submit(run_case, c): i for i, c in enumerate(cases)}
        for fut in _maybe_bar(as_completed(futures)):
            i = futures[fut]
            out = fut.result()
            if timing is not None:
                timing.collect(out)
            results[i] = out
            if callback is not None:
                callback(i, cases[i], out)
    if timing is not None:
        timing.report("CPU-pool", nw=max_workers)
    return results
