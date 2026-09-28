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


def main(argv=None):
    """Run the scan command under the pyrite exit contract."""
    return run(command, argv, prog_name="pyrite run")


if __name__ == "__main__":
    raise SystemExit(main())
