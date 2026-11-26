"""Click wiring and launcher for ``pyrite app viewer``.

Starts the marimo 3D visualization app (``src/pyrite/apps/trace_app.py``) with a
chosen initial material, or renders it to static HTML.

The "which material does the dropdown start on" logic is a pure helper in
:mod:`pyrite.apps.viewer`, so the notebook and the tests can call it without
importing any CLI machinery. Precedence for the initial material: an explicit
CLI material always wins; else the persisted default
(:func:`pyrite._app_defaults.get_viewer_default`, written by ``-d/--save-default``);
else ``"hopg"``.

Two transports carry the resolved material into the marimo subprocess, since we
can't verify in this environment whether ``mo.cli_args()`` reaches ``marimo edit``
(only confirmed for ``marimo run``): ``-- --material <X>`` app args (marimo's own
mechanism) AND a ``PYRITE_VIEWER_INITIAL`` environment variable set on the
subprocess, as a belt-and-suspenders fallback.
:func:`pyrite.apps.viewer.initial_material` checks both.

    pyrite app viewer                    # persisted default (fallback hopg)
    pyrite app viewer wse2               # transient: this run only, doesn't persist
    pyrite app viewer -d wse2            # persist wse2 as the new default, and launch it
    pyrite app viewer --watch            # add marimo's --watch (combinable with either)
    pyrite app viewer --smoke            # execute the app once without a browser
    pyrite app viewer --edit             # `marimo edit` instead of `marimo run`
    pyrite app viewer --no-token         # pass marimo's --no-token (disable auth token)
"""

from functools import partial

import click

from ..._app_defaults import get_viewer_default, set_viewer_default
from ...console import output as _cli_core
from ._app_launcher import (
    MATERIAL_ARGUMENT,
    MarimoApp,
    export,
    launch,
    marimo_command,
    persisted_launch,
    persisted_launch_command,
    smoke_command,
)

TUNNEL_PORT = 2719
APP = MarimoApp(
    "viewer", "trace_app.py", TUNNEL_PORT, "3D viewer", initial_env="PYRITE_VIEWER_INITIAL"
)
NOTEBOOK = APP.notebook
_command = partial(marimo_command, APP)
_smoke_command = partial(smoke_command, APP)
_launch = partial(launch, APP)


def _export(stem: str | None, material: str | None) -> None:
    """Export viewer HTML without starting a marimo server or browser."""
    export(APP, stem, material or get_viewer_default() or "hopg")


def _cli(args):
    persisted_launch(args, APP, get_viewer_default, set_viewer_default, _launch)


command = persisted_launch_command(
    "viewer",
    help=(
        "Launch the interactive 3D viewer with marimo run or edit.\n\n"
        "3D trajectory and crystal structure visualization.\n\n"
        "MATERIAL overrides the persisted default for this run. --save-default "
        "stores it for later no-argument launches."
    ),
    cli=lambda args: _cli(args),  # late-bound: tests patch the module _cli/_launch
)


@click.command("export", help="Render the viewer as static HTML without starting marimo.")
@MATERIAL_ARGUMENT
@click.option("--stem", help="Output stem under pyrite-output/results/ (without .html).")
def export_command(material, stem):
    _export(stem, material)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="pyrite-viewer")


if __name__ == "__main__":
    raise SystemExit(main())
