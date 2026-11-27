"""Headless CXR scan entry point -- thin ``python -m`` shim over the run command.

The remote box invokes ``python -m pyrite._entry.scan <material>`` inside its
pixi-installed checkout. The scan itself -- and the rationale for the __main__ guard
(spawn/forkserver re-import the entry module per worker) -- lives in
``pyrite.runs.scan``; the Click surface lives in ``pyrite.cli.commands.scan``.
Entry shims are drivers, so binding the two together here is what keeps
``runs`` from importing ``cli`` (issue #64, finding 2).
Prefer the installed CLI: ``pyrite run [PROFILE] -m <material>``.
"""

from pyrite._jit_cache import apply_managed_cache_env

# Numba reads NUMBA_CACHE_DIR once on import; set it before kernel modules load.
apply_managed_cache_env()

from pyrite.materials import MaterialConfigError  # noqa: E402

try:
    from pyrite.cli.commands.scan import _command as command
    from pyrite.console.output import run
except MaterialConfigError as exc:
    raise SystemExit(str(exc)) from None


def enable_stack_dump_signal():
    """Let ``SIGUSR1`` print every thread's Python stack to stderr.

    Remote scans write stderr to the job log, so ``pyrite remote job stack`` can ask a
    live scan what it is doing without ptrace or stopping it. The handler is a
    C-level ``faulthandler`` hook: it also answers while Python is blocked in a
    long NumPy or CuPy call. A no-op where ``SIGUSR1`` does not exist.
    """
    import faulthandler
    import signal

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def enable_graceful_termination():
    """Turn ``SIGTERM`` into ``SystemExit(143)`` so a cancelled scan unwinds.

    ``pyrite remote stop`` scancels the job; without a handler the scan dies
    before publishing finished cases of a partly run config (see
    ``run_sweep``). The exit status stays 143, which the job script reads as a
    cancellation; a second ``SIGTERM`` falls back to the default kill.
    """
    import signal

    def terminate(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)


def main(argv=None):
    """Run the scan command under the pyrite exit contract."""
    enable_stack_dump_signal()
    enable_graceful_termination()
    return run(command, argv, prog_name="pyrite run")


if __name__ == "__main__":
    raise SystemExit(main())
