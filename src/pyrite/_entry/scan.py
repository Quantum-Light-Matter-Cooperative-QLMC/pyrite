"""Headless CXR scan entry point -- thin ``python -m`` shim over the run command.

The remote box invokes ``python -m pyrite._entry.scan <material>`` inside its
uv-synced checkout. The scan itself -- and the rationale for the __main__ guard
(spawn/forkserver re-import the entry module per worker) -- lives in
``pyrite.runs.scan``; the Click surface lives in ``pyrite.cli.commands.scan``.
Entry shims are drivers, so binding the two together here is what keeps
``runs`` from importing ``cli`` (issue #64, finding 2).
Prefer the installed CLI: ``pyrite run [PROFILE] -m <material>``.
"""

from pyrite.materials import MaterialConfigError

try:
    from pyrite.cli.commands.scan import _command as command
    from pyrite.console.output import run
except MaterialConfigError as exc:
    raise SystemExit(str(exc)) from None


def enable_stack_dump_signal():
    """Let ``SIGUSR1`` print every thread's Python stack to stderr.

    Remote scans write stderr to the job log, so ``pyrite job stack`` can ask a
    live scan what it is doing without ptrace or stopping it. The handler is a
    C-level ``faulthandler`` hook: it also answers while Python is blocked in a
    long NumPy or CuPy call. A no-op where ``SIGUSR1`` does not exist.
    """
    import faulthandler
    import signal

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def main(argv=None):
    """Run the scan command under the pyrite exit contract."""
    enable_stack_dump_signal()
    return run(command, argv, prog_name="pyrite run")


if __name__ == "__main__":
    raise SystemExit(main())
