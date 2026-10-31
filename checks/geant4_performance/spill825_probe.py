"""#825 device-staging spill probe: per-round download timing plus host memory pressure.

Wraps ``precision_variants.py`` (same arguments) and instruments
``continuation._Retained.append``: each call records the round, whether it
spilled, bytes moved to the host, wall time, process RSS, and kernel memory
counters (``/proc/vmstat`` reclaim/compaction/THP plus the job cgroup's
``memory.stat``). Records go to ``$SPILL_PROBE_OUT`` as JSON lines.

Usage: SPILL_PROBE_OUT=x.jsonl python spill825_probe.py <precision_variants args>
"""

import json
import os
import runpy
import sys
import time
from pathlib import Path

_VMSTAT = (
    "pgscan_direct",
    "pgsteal_direct",
    "pgscan_kswapd",
    "allocstall_normal",
    "allocstall_movable",
    "compact_stall",
    "compact_fail",
    "thp_fault_alloc",
    "thp_fault_fallback",
    "pswpout",
    "pgmajfault",
)
_CGSTAT = ("anon", "file", "pgscan", "pgsteal", "workingset_refault_file", "thp_fault_alloc")


def _kv(path):
    try:
        with open(path) as f:
            return dict(line.split()[:2] for line in f if line.strip())
    except OSError:
        return {}


def _cgroup_dir():
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        return Path("/sys/fs/cgroup") / rel.lstrip("/")
    except (OSError, IndexError):
        return None


_CG = _cgroup_dir()


def _snapshot():
    vm = _kv("/proc/vmstat")
    status = _kv("/proc/self/status")
    snap = {k: int(vm.get(k, 0)) for k in _VMSTAT}
    snap["rss_kib"] = int(status.get("VmRSS:", 0))
    if _CG is not None:
        cg = _kv(_CG / "memory.stat")
        snap.update({f"cg_{k}": int(cg.get(k, 0)) for k in _CGSTAT})
        for name in ("memory.current", "memory.max", "memory.peak"):
            try:
                text = (_CG / name).read_text().strip()
                snap["cg_" + name.split(".")[1]] = None if text == "max" else int(text)
            except OSError:
                pass
    return snap


def _install(out):
    from pyrite.montecarlo.transport import continuation as cont

    sink = open(out, "a", buffering=1)  # noqa: SIM115 - process lifetime
    original_append = cont._Retained.append
    original_assemble = cont._assemble
    state = {"round": 0}

    def append(self, gathered):
        was_on_device = self.on_device
        before = _snapshot()
        t0 = time.perf_counter()
        original_append(self, gathered)
        wall = time.perf_counter() - t0
        after = _snapshot()
        spilled = was_on_device and not self.on_device
        moved = (
            0
            if self.on_device
            else (self.staged_bytes if spilled else cont._chunk_bytes(self.chunks[-1]))
        )
        record = {
            "event": "spill" if spilled else ("stage" if self.on_device else "download"),
            "round": state["round"],
            "bytes": int(moved),
            "staged_bytes": int(self.staged_bytes),
            "wall_s": wall,
            "before": before,
            "delta": {k: (after[k] - before[k]) if after[k] is not None else None for k in after},
            "rss_kib": after["rss_kib"],
        }
        sink.write(json.dumps(record) + "\n")
        state["round"] += 1

    def assemble(*args, **kwargs):
        before = _snapshot()
        t0 = time.perf_counter()
        result = original_assemble(*args, **kwargs)
        after = _snapshot()
        sink.write(
            json.dumps(
                {
                    "event": "assemble",
                    "on_device": bool(args[-1]),
                    "wall_s": time.perf_counter() - t0,
                    "delta": {
                        k: (after[k] - before[k]) if after[k] is not None else None for k in after
                    },
                    "rss_kib": after["rss_kib"],
                }
            )
            + "\n"
        )
        state["round"] = 0
        return result

    cont._Retained.append = append
    cont._assemble = assemble

    if os.environ.get("SPILL_PROBE_FREEPOOL"):
        # Arm: return the staged rounds' pool blocks to the driver at the spill.
        import cupy

        probed_append = cont._Retained.append

        def freeing_append(self, gathered):
            was_on_device = self.on_device
            probed_append(self, gathered)
            if was_on_device and not self.on_device:
                cupy.get_default_memory_pool().free_all_blocks()

        cont._Retained.append = freeing_append

    if os.environ.get("SPILL_PROBE_TRIM"):
        # Arm: hand freed round chunks back to the OS after the host scatter.
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        original_scatter = cont._scatter_host

        def trimming_scatter(*args, **kwargs):
            original_scatter(*args, **kwargs)
            t0 = time.perf_counter()
            libc.malloc_trim(0)
            sink.write(
                json.dumps(
                    {
                        "event": "trim",
                        "wall_s": time.perf_counter() - t0,
                        "rss_kib": _snapshot()["rss_kib"],
                        "delta": {},
                    }
                )
                + "\n"
            )

        cont._scatter_host = trimming_scatter


# (module, attribute) of the phases the RSS sampler labels, outermost first.
_PHASES = (
    ("pyrite.montecarlo.transport.batching", "_drive_continuation_rounds"),
    ("pyrite.montecarlo.transport.continuation", "_assemble"),
    ("pyrite.montecarlo.transport.host_pass", "_assemble_result"),
    ("pyrite.montecarlo.transport.host_pass", "_add_optional_fields"),
    ("pyrite.montecarlo.transport.hard_radiative", "complete_hard_radiative_events"),
)


def _install_rss(out):
    """Sample RSS every 5 ms; report the peak inside each labelled phase."""
    import functools
    import importlib
    import threading

    stack, peaks = ["other"], {}
    page_kib = os.sysconf("SC_PAGE_SIZE") // 1024

    def sample():
        with open("/proc/self/statm") as f:
            while True:
                f.seek(0)
                rss = int(f.read().split()[1]) * page_kib
                label = stack[-1]
                if rss > peaks.get(label, 0):
                    peaks[label] = rss
                time.sleep(0.005)

    def wrap(fn, label):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            stack.append(label)
            try:
                return fn(*args, **kwargs)
            finally:
                stack.pop()

        return inner

    for module_name, attr in _PHASES:
        module = importlib.import_module(module_name)
        setattr(module, attr, wrap(getattr(module, attr), attr))
    threading.Thread(target=sample, daemon=True).start()

    import atexit

    atexit.register(
        lambda: Path(out).write_text(
            json.dumps({k: v / 2**20 for k, v in peaks.items()}, indent=1) + "\n"
        )
    )


def main():
    out = os.environ.get("SPILL_PROBE_OUT")
    if out:
        _install(out)
    if os.environ.get("SPILL_PROBE_NUMPY_DST"):
        # Arm: one exact-size np.empty per round column (C heap), no shared blocks.
        from pyrite.montecarlo.transport import continuation as cont

        cont._ColumnArena.BLOCK_BYTES = 0
    rss_out = os.environ.get("SPILL_PROBE_RSS")
    if rss_out:
        _install_rss(rss_out)
    script = Path(__file__).resolve().parents[1] / "fp32_speed_of_light" / "precision_variants.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
