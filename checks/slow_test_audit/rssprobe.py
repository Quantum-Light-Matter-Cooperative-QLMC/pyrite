"""Per-test RSS probe (issue #646 audit; not part of the suite).

Load with ``-p checks.slow_test_audit.rssprobe`` from the checkout. Writes one JSON line per
test to ``$RSSPROBE_DIR/<worker>.jsonl``: RSS at test start, sampled peak
process RSS and process-tree RSS (children included), wall time, ru_maxrss.
"""

import json
import os
import resource
import threading
import time

import psutil
import pytest

_PAGE = os.sysconf("SC_PAGE_SIZE")
_DIR = os.environ.get("RSSPROBE_DIR", ".")
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "main")
_OUT = None


def _self_rss():
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * _PAGE


class _Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak = _self_rss()
        self.tree_peak = self.peak
        self.proc = psutil.Process()

    def run(self):
        n = 0
        while not self.stop.wait(0.01):
            rss = _self_rss()
            self.peak = max(self.peak, rss)
            n += 1
            if n % 10 == 0:
                tree = rss
                try:
                    for child in self.proc.children(recursive=True):
                        try:
                            tree += child.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                self.tree_peak = max(self.tree_peak, tree)
        self.tree_peak = max(self.tree_peak, self.peak)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _OUT
    if _OUT is None:
        os.makedirs(_DIR, exist_ok=True)
        _OUT = open(os.path.join(_DIR, f"{_WORKER}.jsonl"), "a", buffering=1)
    start = _self_rss()
    maxrss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sampler = _Sampler()
    sampler.start()
    t0 = time.perf_counter()
    yield
    wall = time.perf_counter() - t0
    sampler.stop.set()
    sampler.join()
    end = _self_rss()
    maxrss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    _OUT.write(
        json.dumps(
            {
                "nodeid": item.nodeid,
                "worker": _WORKER,
                "pid": os.getpid(),
                "wall_s": round(wall, 4),
                "start_rss": start,
                "end_rss": end,
                "peak_rss": sampler.peak,
                "peak_tree_rss": sampler.tree_peak,
                "ru_maxrss_kb_before": maxrss0,
                "ru_maxrss_kb_after": maxrss1,
                "children_maxrss_kb": ch,
                "t_end": time.time(),
            }
        )
        + "\n"
    )
