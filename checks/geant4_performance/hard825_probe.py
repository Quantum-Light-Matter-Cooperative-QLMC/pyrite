"""#825 hard-photon field probe: per-phase peak RSS and wall time, optional field digest.

Wraps ``precision_variants.py`` (same arguments). A 5 ms sampler records the
peak RSS inside each labelled phase and the phase's summed wall time; both go
to ``$HARD825_OUT`` as JSON at exit. With ``HARD825_DIGEST=1`` every
``complete_hard_radiative_events`` call also records the SHA-256 of its two
per-segment output fields (outside the timed phase), so trees can be compared
bit for bit.

Usage: HARD825_OUT=x.json python hard825_probe.py <precision_variants args>
"""

import atexit
import functools
import hashlib
import importlib
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

# (module, attribute) of the phases the sampler labels, outermost first.
_PHASES = (
    ("pyrite.montecarlo.transport.continuation", "_drive_continuation_rounds"),
    ("pyrite.montecarlo.transport.continuation", "_assemble"),
    ("pyrite.montecarlo.transport.host_pass", "_assemble_result"),
    ("pyrite.montecarlo.transport.host_pass", "_add_optional_fields"),
    ("pyrite.montecarlo.transport.hard_radiative", "complete_hard_radiative_events"),
)
_FIELDS = ("hard_radiative_direction", "hard_radiative_target_momentum_eV_c")


def _digest(segments):
    import numpy as np

    out = {"rows": int(np.asarray(segments["hard_radiative_k_eV"]).size)}
    out["photons"] = int(np.count_nonzero(np.asarray(segments["hard_radiative_k_eV"]) > 0.0))
    for key in _FIELDS:
        value = np.ascontiguousarray(segments[key])
        out[key] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
        }
    return out


def main():
    out = Path(os.environ["HARD825_OUT"])
    digest = bool(os.environ.get("HARD825_DIGEST"))
    stack, peaks, walls, digests = ["other"], {}, {}, []
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
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                walls[label] = walls.get(label, 0.0) + time.perf_counter() - t0
                stack.pop()
                if digest and label == "complete_hard_radiative_events":
                    digests.append(_digest(args[0]))

        return inner

    for module_name, attr in _PHASES:
        module = importlib.import_module(module_name)
        setattr(module, attr, wrap(getattr(module, attr), attr))
    threading.Thread(target=sample, daemon=True).start()
    atexit.register(
        lambda: out.write_text(
            json.dumps(
                {
                    "peak_gib": {k: v / 2**20 for k, v in peaks.items()},
                    "wall_s": walls,
                    "digests": digests,
                },
                indent=1,
            )
            + "\n"
        )
    )
    script = Path(__file__).resolve().parents[1] / "fp32_speed_of_light" / "precision_variants.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
