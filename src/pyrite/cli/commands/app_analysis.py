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

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from ..._acp import running_acp
from ..._app_defaults import get_analysis_default, set_analysis_default
from ...paths import app_dir
from .. import _completion as _cli_completion
from .. import _core as _cli_core
from .._deprecations import canonical_option

NOTEBOOK = str(app_dir() / "analysis_app.py")
TUNNEL_PORT = 2718


def _command(material, *, edit=False, watch=False, tunnel=False, no_token=False):
    """The marimo argv for one launch (module-run through the current
    interpreter so the venv's marimo is the one that runs). Marimo's own flags
    go before the notebook path; app args go after ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        *(["--port", str(TUNNEL_PORT)] if tunnel else []),
        *(["--no-token"] if no_token else []),
        NOTEBOOK,
        "--",
        "--material",
        material,
    ]


def _smoke_command(material, output):
    """The one-shot, headless command used to execute the analysis app in CI-like checks."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        NOTEBOOK,
        "--output",
        str(output),
        "--force",
        "--",
        "--material",
        material,
    ]


def _launch(
    material, *, edit=False, watch=False, smoke=False, acp=False, tunnel=False, no_token=False
):
    cmd = _command(material, edit=edit, watch=watch, tunnel=tunnel, no_token=no_token)
    print(f"launching {NOTEBOOK} ({'edit' if edit else 'run'}) with material={material}")
    if tunnel:
        print(f"ssh -L {TUNNEL_PORT}:127.0.0.1:{TUNNEL_PORT} <your-pi-ssh-host>")
        print(f"http://127.0.0.1:{TUNNEL_PORT}")
    env = {**os.environ, "PYRITE_ANALYZE_INITIAL": material}
    if smoke:
        with tempfile.TemporaryDirectory(prefix="pyrite-analysis-") as tmpdir:
            subprocess.run(
                _smoke_command(material, Path(tmpdir) / "analysis.html"), check=True, env=env
            )
        return
    if acp:
        with running_acp():
            subprocess.run(cmd, check=True, env=env)
    else:
        subprocess.run(cmd, check=True, env=env)


def _cli(args):
    if args.default and args.material is None:
        raise SystemExit("pyrite app analysis -d/--save-default: no material given to persist")

    if args.default:
        set_analysis_default(args.material)

    material = args.material or get_analysis_default() or "hopg"
    launch_args = {"edit": args.edit, "watch": args.watch}
    if args.smoke:
        launch_args["smoke"] = True
    if args.acp:
        launch_args["acp"] = True
    if args.tunnel:
        launch_args["tunnel"] = True
    if args.no_token:
        launch_args["no_token"] = True
    _launch(material, **launch_args)


@click.command(
    "analyze",
    help=(
        "Launch the interactive analysis application with marimo run or edit.\n\n"
        "MATERIAL overrides the persisted default for this run. --save-default "
        "stores it for later no-argument launches."
    ),
)
@click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)
@canonical_option(
    "-d",
    "--save-default",
    "persist_default",
    is_flag=True,
    help="Persist MATERIAL as default for future no-argument runs.",
)
@click.option("--watch", is_flag=True, help="Reload app when source files change.")
@click.option("--smoke", is_flag=True, help="Execute app once headlessly and exit.")
@click.option("--edit", is_flag=True, help="Use `marimo edit` instead of `marimo run`.")
@click.option("--acp", is_flag=True, help="Start local Claude and Codex ACP bridges.")
@click.option("--tunnel", is_flag=True, help="Bind fixed port for SSH tunneling.")
@click.option("--no-token", is_flag=True, help="Disable marimo auth token.")
def command(material, persist_default, watch, smoke, edit, acp, tunnel, no_token):
    if persist_default and material is None:
        raise click.UsageError("--save-default requires MATERIAL")
    return _cli_core.invoke_legacy(
        _cli,
        material=material,
        default=persist_default,
        watch=watch,
        smoke=smoke,
        edit=edit,
        acp=acp,
        tunnel=tunnel,
        no_token=no_token,
    )


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="pyrite-analyze")


if __name__ == "__main__":
    raise SystemExit(main())
