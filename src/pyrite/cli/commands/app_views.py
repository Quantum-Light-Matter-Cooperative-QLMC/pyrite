"""Click wiring and launchers for ``pyrite app pixels`` and ``pyrite app compare``.

Both apps split out of the analysis app and share one launcher shape:

- ``pixels`` starts ``src/pyrite/apps/pixel_app.py`` (stored pixel-detector
  observations, discovered per checkpoint stem);
- ``compare`` starts ``src/pyrite/apps/compare_app.py`` (case basket across
  checkpoints and cross-material comparison).

Each app's material picker starts on MATERIAL when given, else on the analysis
app's persisted default (:func:`pyrite._app_defaults.get_analysis_default`,
written by ``pyrite app analysis launch --save-default``), else ``"hopg"``.
These launchers read that default but never write it. The resolved material
reaches the marimo subprocess through ``-- --material <X>`` and the
``PYRITE_ANALYZE_INITIAL`` environment variable, which
:func:`pyrite.apps.analyze.initial_material` checks.

    pyrite app pixels launch               # persisted analysis default (fallback hopg)
    pyrite app pixels launch wse2          # start the picker on wse2
    pyrite app compare launch --smoke      # execute the app once without a browser
    pyrite app compare export --stem cmp   # write results/cmp.html
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click

from ..._acp import running_acp
from ..._app_defaults import get_analysis_default
from ...paths import app_dir
from .. import _completion as _cli_completion


@dataclass(frozen=True)
class MarimoApp:
    """One launchable marimo app: its command name, notebook, and tunnel port."""

    name: str
    notebook_name: str
    tunnel_port: int
    title: str

    @property
    def notebook(self) -> str:
        return str(app_dir() / self.notebook_name)


PIXELS = MarimoApp("pixels", "pixel_app.py", 2720, "pixel-detector observation")
COMPARE = MarimoApp("compare", "compare_app.py", 2721, "case and cross-material comparison")


def resolve_material(material: str | None) -> str:
    """Explicit MATERIAL, else the analysis app's persisted default, else hopg."""
    return material or get_analysis_default() or "hopg"


def _command(app, material, *, edit=False, watch=False, tunnel=False, no_token=False):
    """The marimo argv for one launch. Marimo's flags precede the notebook; app args follow ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        *(["--port", str(app.tunnel_port)] if tunnel else []),
        *(["--no-token"] if no_token else []),
        app.notebook,
        "--",
        "--material",
        material,
    ]


def _smoke_command(app, material, output):
    """The one-shot, headless ``marimo export html`` argv for one app."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        app.notebook,
        "--output",
        str(output),
        "--force",
        "--",
        "--material",
        material,
    ]


def _launch(
    app,
    material,
    *,
    edit=False,
    watch=False,
    smoke=False,
    acp=False,
    tunnel=False,
    no_token=False,
):
    cmd = _command(app, material, edit=edit, watch=watch, tunnel=tunnel, no_token=no_token)
    click.echo(f"launching {app.notebook} ({'edit' if edit else 'run'}) with material={material}")
    if tunnel:
        click.echo(f"ssh -L {app.tunnel_port}:127.0.0.1:{app.tunnel_port} <your-pi-ssh-host>")
        click.echo(f"http://127.0.0.1:{app.tunnel_port}")
    env = {**os.environ, "PYRITE_ANALYZE_INITIAL": material}
    if smoke:
        with tempfile.TemporaryDirectory(prefix=f"pyrite-{app.name}-") as tmpdir:
            output = Path(tmpdir) / f"{app.name}.html"
            subprocess.run(_smoke_command(app, material, output), check=True, env=env)
        return
    if acp:
        with running_acp():
            subprocess.run(cmd, check=True, env=env)
    else:
        subprocess.run(cmd, check=True, env=env)


def _export(app, stem: str | None, material: str | None) -> None:
    """Export the app's HTML without starting a marimo server or browser."""
    resolved = resolve_material(material)
    stem = stem or f"pyrite_{app.name}_{resolved}"
    output = Path("results") / f"{stem}.html"
    click.echo(f"exporting {app.notebook} -> {output}")
    env = {**os.environ, "PYRITE_ANALYZE_INITIAL": resolved}
    subprocess.run(_smoke_command(app, resolved, output), check=True, env=env)


_MATERIAL_ARGUMENT = click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)


def _launch_command(app: MarimoApp) -> click.Command:
    @click.command(
        app.name,
        help=(
            f"Launch the interactive {app.title} app with marimo run or edit.\n\n"
            "MATERIAL starts the checkpoint picker for this run. Without it, the picker "
            "starts on the analysis app's saved default material, else hopg; this command "
            "never changes that default."
        ),
    )
    @_MATERIAL_ARGUMENT
    @click.option("--watch", is_flag=True, help="Reload app when source files change.")
    @click.option("--smoke", is_flag=True, help="Execute app once headlessly and exit.")
    @click.option("--edit", is_flag=True, help="Use `marimo edit` instead of `marimo run`.")
    @click.option("--acp", is_flag=True, help="Start local Claude and Codex ACP bridges.")
    @click.option("--tunnel", is_flag=True, help="Bind fixed port for SSH tunneling.")
    @click.option("--no-token", is_flag=True, help="Disable marimo auth token.")
    def command(material, watch, smoke, edit, acp, tunnel, no_token):
        _launch(
            app,
            resolve_material(material),
            edit=edit,
            watch=watch,
            smoke=smoke,
            acp=acp,
            tunnel=tunnel,
            no_token=no_token,
        )

    return command


def _export_command(app: MarimoApp) -> click.Command:
    @click.command(
        "export",
        help=(
            f"Render the {app.title} app as static HTML without starting marimo.\n\n"
            "Writes results/<stem>.html; the stem defaults to pyrite_"
            f"{app.name}_<material>."
        ),
    )
    @_MATERIAL_ARGUMENT
    @click.option("--stem", help="Output stem under results/ (without .html).")
    def command(material, stem):
        _export(app, stem, material)

    return command


pixels_command = _launch_command(PIXELS)
pixels_export_command = _export_command(PIXELS)
compare_command = _launch_command(COMPARE)
compare_export_command = _export_command(COMPARE)
