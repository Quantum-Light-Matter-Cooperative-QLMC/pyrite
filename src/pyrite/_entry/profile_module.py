"""cProfile a ``python -m`` module and keep its exit status.

``python -m cProfile -m <module>`` swallows ``SystemExit`` and always exits 0,
so a usage error in the profiled command looked like a successful profile
(issue #314). This shim runs the module as ``__main__`` under a profiler,
writes the stats, then lets the module's ``SystemExit`` propagate unchanged.

Usage: ``python -m pyrite._entry.profile_module OUTFILE MODULE [ARGS...]``
"""

import cProfile
import runpy
import sys


def main(argv=None):
    """Profile ``MODULE`` into ``OUTFILE``; return or raise the module's exit."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit("usage: python -m pyrite._entry.profile_module OUTFILE MODULE [ARGS...]")
    outfile, module, *module_args = argv
    sys.argv[:] = [module, *module_args]
    profiler = cProfile.Profile()
    try:
        profiler.runcall(runpy.run_module, module, run_name="__main__", alter_sys=True)
    finally:
        profiler.dump_stats(outfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
