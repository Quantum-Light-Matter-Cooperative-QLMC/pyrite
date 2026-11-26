"""Click wiring and launcher for ``pyrite app analysis``.

Starts the marimo analysis app (``src/pyrite/apps/analysis_app.py``) with a
chosen initial material.

The "which material does the dropdown start on" logic is a pure helper in
:mod:`pyrite.apps.analyze`, so the notebook and the tests can call it without
importing any CLI machinery. Precedence for the initial material: an explicit
CLI material always wins; else the persisted default
(:func:`pyrite._app_defaults.get_analysis_default`, written by ``-d/--save-default``);
else ``"hopg"``.

Two transports carry the resolved material into the marimo subprocess, since we
can't verify in this environment whether ``mo.cli_args()`` reaches ``marimo edit``
(only confirmed for ``marimo run``): ``-- --material <X>`` app args (marimo's own
mechanism) AND a ``PYRITE_ANALYZE_INITIAL`` environment variable set on the
subprocess, as a belt-and-suspenders fallback.
:func:`pyrite.apps.analyze.initial_material` checks both.

    pyrite app analysis                    # persisted default (fallback hopg)
    pyrite app analysis wse2               # transient: this run only, doesn't persist
    pyrite app analysis -d wse2            # persist wse2 as the new default, and launch it
    pyrite app analysis --watch            # add marimo's --watch (combinable with either)
    pyrite app analysis --smoke            # execute the app once without a browser
    pyrite app analysis --edit             # `marimo edit` instead of `marimo run`
    pyrite app analysis --no-token         # pass marimo's --no-token (disable auth token)
"""

from ..._app_defaults import get_analysis_default, set_analysis_default
from ...console import output as _cli_core
from ...paths import app_dir
from ._app_launcher import MarimoApp, app_command

TUNNEL_PORT = 2718
# Spectra at the High/Publication/Full plot resolutions exceed marimo's 8 MB
# default cell-output cap (17 tilts at Full: ~115 MB). A caller's own
# MARIMO_RUNTIME_OUTPUT_MAX_BYTES wins.
OUTPUT_MAX_BYTES = 512_000_000
APP = MarimoApp(
    name="analysis",
    notebook=str(app_dir() / "analysis_app.py"),
    tunnel_port=TUNNEL_PORT,
    initial_env="PYRITE_ANALYZE_INITIAL",
    get_default=get_analysis_default,
    set_default=set_analysis_default,
    base_env={"MARIMO_RUNTIME_OUTPUT_MAX_BYTES": str(OUTPUT_MAX_BYTES)},
)
NOTEBOOK = APP.notebook
_command = APP.command
_smoke_command = APP.smoke_command
_launch = APP.launch


def _cli(args):
    APP.run_cli(args, _launch)


command = app_command(
    "analyze",
    help=(
        "Launch the interactive analysis application with marimo run or edit.\n\n"
        "MATERIAL overrides the persisted default for this run. --save-default "
        "stores it for later no-argument launches."
    ),
    cli=lambda args: _cli(args),  # late-bound: tests patch the module _cli
)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="pyrite-analyze")


if __name__ == "__main__":
    raise SystemExit(main())
