# TODO / Backlog (branch: bugfix/cli-output-noise)

Scope: P3 #3 -- CLI/remote output noise.

Problem: `src/cxr_mc/montecarlo/_backend.py` printed a GPU/CPU banner
("Using GPU" / "No GPU found, or cupy not installed! ...") at import time on
every `cxr` invocation, and `src/cxr_mc/montecarlo/transport.py`'s
`_sample_cos_theta` printed "no Mott transport table for 'X'" the first time
an element without a NIST table was seen. Both used `print`, so nothing could
gate them, and `montecarlo.runner`'s `ProcessPoolExecutor` worker pool
(`_worker_init`) re-imports `_backend` and re-seeds transport's per-element
`_NO_MOTT` cache in every worker process -- so a `dev/remote.py` sweep with N
workers reprinted both messages N times per run, independent of the
already-existing per-process dedup.

Fix: converted both `print()` calls to `logger.debug(...)` on
per-module `logging.getLogger(__name__)` loggers. `cxr_mc/__init__.py` now
attaches a `NullHandler` to the `"cxr_mc"` package logger (so effective level
stays WARNING and DEBUG records are silently dropped by default -- true in
every worker process too), and opts a caller into visibility with
`CXR_MC_DEBUG=1`, which attaches a `StreamHandler` at DEBUG to just the
`"cxr_mc"` logger without touching the caller's root logging config. Kept the
existing `_NO_MOTT` per-element dedup in transport.py (still useful when
`CXR_MC_DEBUG=1` is set, to avoid repeat lines within one process).

Files touched: `src/cxr_mc/__init__.py`, `src/cxr_mc/montecarlo/_backend.py`,
`src/cxr_mc/montecarlo/transport.py`, `README.md` (GPU banner note),
`tests/test_output_noise.py` (new).
