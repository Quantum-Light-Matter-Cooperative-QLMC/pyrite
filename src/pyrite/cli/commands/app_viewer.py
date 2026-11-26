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

import subprocess

import click

from ..._app_defaults import get_viewer_default, set_viewer_default
from ...console import output as _cli_core
from ...console.outputs import output_dir
from ...paths import app_dir
from .. import _completion as _cli_completion
from .._extras import require_cli_extra
from ._app_launcher import MarimoApp, app_command

TUNNEL_PORT = 2719
APP = MarimoApp(
    name="viewer",
    notebook=str(app_dir() / "trace_app.py"),
    tunnel_port=TUNNEL_PORT,
    initial_env="PYRITE_VIEWER_INITIAL",
    get_default=get_viewer_default,
    set_default=set_viewer_default,
)
NOTEBOOK = APP.notebook
_command = APP.command
_smoke_command = APP.smoke_command
_launch = APP.launch


def _export(stem: str | None, material: str | None) -> None:
    """Export viewer HTML without starting a marimo server or browser."""
    require_cli_extra("apps", feature="pyrite app viewer export")
    resolved = material or get_viewer_default() or "hopg"
    stem = stem or f"pyrite_viewer_{resolved}"
    output = output_dir("results") / f"{stem}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"exporting {NOTEBOOK} -> {output}")
    subprocess.run(_smoke_command(resolved, output), check=True)


def _cli(args):
    APP.run_cli(args, _launch)


command = app_command(
    "viewer",
    help=(
        "Launch the interactive 3D viewer with marimo run or edit.\n\n"
        "3D trajectory and crystal structure visualization.\n\n"
        "MATERIAL overrides the persisted default for this run. --save-default "
        "stores it for later no-argument launches."
    ),
    cli=lambda args: _cli(args),  # late-bound: tests patch the module _cli
)


@click.command("export", help="Render the viewer as static HTML without starting marimo.")
@click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)
@click.option("--stem", help="Output stem under pyrite-output/results/ (without .html).")
def export_command(material, stem):
    _export(stem, material)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="pyrite-viewer")


if __name__ == "__main__":
    raise SystemExit(main())
