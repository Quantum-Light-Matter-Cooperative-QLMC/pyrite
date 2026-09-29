"""Driver-side phase timing for ``PYRITE_MC_TIMING`` profiling.

The accumulator lives only in the driver process; per-case transport and
spectrum deltas ride back on the result dicts and are stripped here.
"""

import sys

import numpy as np

from .chunking import _RESOURCE_POLICY


class _TimingAgg:
    """Main-process accumulator for PYRITE_MC_TIMING phase profiling.

    Lives only in the driver process (never pickled). Per-case transport and
    spectrum deltas ride back on the phase dicts (workers -> main for transport);
    the GPU-idle wait is measured directly in run_cases' pipeline loop. ``collect``
    both records and strips the private keys so stored results stay clean.
    """

    def __init__(self):
        self.transport: list[float] = []  # worker compute time for _transport_case
        self.spectrum: list[float] = []  # _spectrum_case body (GPU work in main proc)
        self.wait: list[float] = []  # driver blocked on the transport future (GPU idle)
        self.transport_total = 0.0
        self.spectrum_total = 0.0
        self.wait_total = 0.0

    def collect(self, out, *, wait_seconds=None):
        """Strip private metrics, accumulate them, and return one profile update."""
        t = out.pop("_t_transport", None)
        if t is not None:
            self.transport.append(t)
            self.transport_total += t
        s = out.pop("_t_spectrum", None)
        if s is not None:
            self.spectrum.append(s)
            self.spectrum_total += s
        if wait_seconds is not None:
            self.wait.append(wait_seconds)
            self.wait_total += wait_seconds
        retries = out.pop("_gpu_oom_retries", 0)
        chunk_metrics = {
            key[1:]: out.pop(key)
            for key in (
                "_line_gpu_oom_retries",
                "_brem_gpu_oom_retries",
                "_generic_gpu_oom_retries",
                "_attempted_spec_chunk",
                "_effective_spec_chunk",
                "_attempted_brem_chunk",
                "_effective_brem_chunk",
                "_learned_spec_chunk",
                "_line_axis_nodes",
                "_brem_axis_nodes",
                "_segments",
                "_line_tab_points",
                "_line_table_mib",
                "_host_rss_mib",
                "_host_rss_peak_mib",
            )
            if key in out
        }
        pool = {
            key[1:]: out.pop(key)
            for key in (
                "_cupy_pool_used_mib",
                "_cupy_pool_reserved_mib",
                "_cupy_pool_peak_mib",
                "_allocator_used_mib",
                "_allocator_reserved_mib",
                "_allocator_peak_mib",
                "_backend",
                "_backend_vendor",
                "_backend_device",
            )
            if key in out
        }
        fallback_reason = out.pop("_backend_fallback_reason", None)
        return {
            "timed_case_count": max(len(self.transport), len(self.spectrum)),
            "transport_seconds": t,
            "spectrum_seconds": s,
            "driver_wait_seconds": wait_seconds,
            "transport_seconds_total": self.transport_total,
            "spectrum_seconds_total": self.spectrum_total,
            "driver_wait_seconds_total": self.wait_total,
            "gpu_oom_retry_count": retries,
            **({"backend_fallback_reason": fallback_reason} if fallback_reason is not None else {}),
            **chunk_metrics,
            **pool,
        }

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
        f"[pyrite-timing] mode={mode}  cases={max(tr.size, sp.size)}  workers={nw}",
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
        # Gate-0 verdict per docs/repo-design/compute/compute-performance-optimization.md.
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
    if _RESOURCE_POLICY.gpu and _RESOURCE_POLICY.pool_peak_bytes:
        cadence = (
            f"every {_RESOURCE_POLICY.free_every} cases"
            if _RESOURCE_POLICY.free_every > 1
            else "per case"
        )
        wm = (
            f", watermark {_RESOURCE_POLICY.free_watermark_mb} MB"
            if _RESOURCE_POLICY.free_watermark_mb > 0
            else ""
        )
        lines.append(
            f"  CuPy pool peak (reserved)  : {_RESOURCE_POLICY.pool_peak_bytes / (1 << 20):8.1f} MB"
            f"   (free {cadence}{wm})"  # A2 operational watermark: must stay bounded
        )
    lines.append("")
    print("\n".join(lines), file=sys.stderr, flush=True)
