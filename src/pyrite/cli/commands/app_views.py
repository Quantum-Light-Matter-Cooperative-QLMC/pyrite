"""Shared launchers for pixel, comparison and planar channeling apps.

The checkpoint apps split out of the analysis app and share one launcher shape:

- ``pixels`` starts ``src/pyrite/apps/pixel_app.py`` (stored pixel-detector
  observations, discovered per checkpoint stem);
- ``compare`` starts ``src/pyrite/apps/compare_app.py`` (case basket across
  checkpoints and cross-material comparison).
- ``channeling`` starts ``src/pyrite/apps/channeling_app.py`` (standalone planar
  quantum demo, with controls inside the app and no material argument).
- ``transition-radiation`` starts ``src/pyrite/apps/transition_radiation_app.py``
  (standalone TR/CTR demo, with no material argument).

Each checkpoint app's material picker starts on MATERIAL when given, else on the analysis
app's persisted default (:func:`pyrite._app_defaults.get_analysis_default`,
written by ``pyrite app analysis launch --save-default``), else ``"hopg"``.
These launchers read that default but never write it. The resolved material
reaches the marimo subprocess through ``-- --material <X>`` and the
``PYRITE_ANALYZE_INITIAL`` environment variable, which
:func:`pyrite.apps.analyze.initial_material` checks.

    pyrite app pixels launch               # persisted analysis default (fallback hopg)
    pyrite app pixels launch wse2          # start the picker on wse2
    pyrite app compare launch --smoke      # execute the app once without a browser
    pyrite app compare export --stem cmp   # write pyrite-output/results/cmp.html
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
from ...console.outputs import output_dir
from ...paths import app_dir
from .. import _completion as _cli_completion
from .._extras import require_cli_extra


@dataclass(frozen=True)
class MarimoApp:
    """One launchable marimo app: its command name, notebook, and tunnel port."""

    name: str
    notebook_name: str
    tunnel_port: int
    title: str
    material_picker: bool = True

    @property
    def notebook(self) -> str:
        return str(app_dir() / self.notebook_name)


PIXELS = MarimoApp("pixels", "pixel_app.py", 2720, "pixel-detector observation")
CONFIGURATION = MarimoApp("configuration", "configuration_app.py", 2726, "configuration", False)
COMPARE = MarimoApp("compare", "compare_app.py", 2721, "case and cross-material comparison")
CHANNELING = MarimoApp("channeling", "channeling_app.py", 2722, "planar channeling", False)
ANGULAR_HEATMAP = MarimoApp(
    "angular-heatmap", "angular_heatmap_app.py", 2724, "directional source heatmap", False
)
TRANSITION_RADIATION = MarimoApp(
    "transition-radiation", "transition_radiation_app.py", 2723, "TR/CTR", False
)


#: ``PYRITE_MC_BACKEND`` values (mirrors ``pyrite._backend``, which probes devices on import).
BACKEND_CHOICES = ("auto", "cpu", "cuda", "rocm", "sycl")


def resolve_material(material: str | None) -> str:
    """Explicit MATERIAL, else the analysis app's persisted default, else hopg."""
    return material or get_analysis_default() or "hopg"


def _command(app, material=None, *, edit=False, watch=False, tunnel=False, no_token=False):
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
        *(["--", "--material", material] if material is not None else []),
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
        *(["--", "--material", material] if material is not None else []),
    ]


def _launch(
    app,
    material=None,
    *,
    edit=False,
    watch=False,
    smoke=False,
    acp=False,
    tunnel=False,
    no_token=False,
    backend=None,
):
    require_cli_extra("apps", feature=f"pyrite app {app.name} launch")
    cmd = _command(app, material, edit=edit, watch=watch, tunnel=tunnel, no_token=no_token)
    label = f" with material={material}" if material is not None else ""
    click.echo(f"launching {app.notebook} ({'edit' if edit else 'run'}){label}")
    if tunnel:
        click.echo(f"ssh -L {app.tunnel_port}:127.0.0.1:{app.tunnel_port} <your-pi-ssh-host>")
        click.echo(f"http://127.0.0.1:{app.tunnel_port}")
    env = {**os.environ, **({"PYRITE_ANALYZE_INITIAL": material} if material is not None else {})}
    if backend is not None:
        # The array backend is chosen once at import; the app process inherits it.
        env["PYRITE_MC_BACKEND"] = backend
        click.echo(f"array backend: {backend}")
    if smoke:
        with tempfile.TemporaryDirectory(prefix=f"pyrite-{app.name}-") as tmpdir:
            output = Path(tmpdir) / f"{app.name}.html"
            _run(_smoke_command(app, material, output), env)
        return
    if acp:
        with running_acp():
            _run(cmd, env)
    else:
        _run(cmd, env)


def _run(command: list[str], env: dict[str, str]) -> None:
    """Keep child failures nonzero and report them through Click diagnostics."""
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"marimo exited with status {exc.returncode}") from exc
    except OSError as exc:
        raise click.ClickException(f"could not start marimo: {exc}") from exc


def _export(app, stem: str | None, material: str | None) -> None:
    """Export the app's HTML without starting a marimo server or browser."""
    require_cli_extra("apps", feature=f"pyrite app {app.name} export")
    resolved = resolve_material(material) if app.material_picker else None
    suffix = f"_{resolved}" if resolved is not None else ""
    stem = stem or f"pyrite_{app.name}{suffix}"
    output = output_dir("results") / f"{stem}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"exporting {app.notebook} -> {output}")
    env = {**os.environ, **({"PYRITE_ANALYZE_INITIAL": resolved} if resolved is not None else {})}
    _run(_smoke_command(app, resolved, output), env)


_MATERIAL_ARGUMENT = click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)


def _launch_command(app: MarimoApp) -> click.Command:
    material_argument = _MATERIAL_ARGUMENT if app.material_picker else lambda command: command

    @click.command(
        app.name,
        help=(
            f"Launch the interactive {app.title} app with marimo run or edit.\n\n"
            + (
                "MATERIAL starts the checkpoint picker for this run. Without it, the picker "
                "starts on the analysis app's saved default material, else hopg; this command "
                "never changes that default."
                if app.material_picker
                else "Choose the crystal and beam settings inside the app; no checkpoint is needed."
            )
        ),
    )
    @material_argument
    @click.option("--watch", is_flag=True, help="Reload app when source files change.")
    @click.option("--smoke", is_flag=True, help="Execute app once headlessly and exit.")
    @click.option("--edit", is_flag=True, help="Use `marimo edit` instead of `marimo run`.")
    @click.option("--acp", is_flag=True, help="Start local Claude and Codex ACP bridges.")
    @click.option("--tunnel", is_flag=True, help="Bind fixed port for SSH tunneling.")
    @click.option("--no-token", is_flag=True, help="Disable marimo auth token.")
    @click.option(
        "--backend",
        type=click.Choice(BACKEND_CHOICES),
        default=None,
        help=(
            "Array backend for the app process (sets PYRITE_MC_BACKEND). "
            "Default: inherit the environment. An explicit accelerator fails if unavailable."
        ),
    )
    def command(watch, smoke, edit, acp, tunnel, no_token, backend, material=None):
        _launch(
            app,
            resolve_material(material) if app.material_picker else None,
            edit=edit,
            watch=watch,
            smoke=smoke,
            acp=acp,
            tunnel=tunnel,
            no_token=no_token,
            backend=backend,
        )

    return command


def _export_command(app: MarimoApp) -> click.Command:
    material_argument = _MATERIAL_ARGUMENT if app.material_picker else lambda command: command

    @click.command(
        "export",
        help=(
            f"Render the {app.title} app as static HTML without starting marimo.\n\n"
            "Writes pyrite-output/results/<stem>.html; the stem defaults to pyrite_"
            f"{app.name}" + ("_<material>." if app.material_picker else ".")
        ),
    )
    @material_argument
    @click.option("--stem", help="Output stem under pyrite-output/results/ (without .html).")
    def command(stem, material=None):
        _export(app, stem, material)

    return command


pixels_command = _launch_command(PIXELS)
configuration_command = _launch_command(CONFIGURATION)
pixels_export_command = _export_command(PIXELS)
compare_command = _launch_command(COMPARE)
compare_export_command = _export_command(COMPARE)
channeling_command = _launch_command(CHANNELING)
channeling_export_command = _export_command(CHANNELING)
transition_radiation_command = _launch_command(TRANSITION_RADIATION)
transition_radiation_export_command = _export_command(TRANSITION_RADIATION)

angular_heatmap_command = _launch_command(ANGULAR_HEATMAP)
angular_heatmap_export_command = _export_command(ANGULAR_HEATMAP)


OPTICAL_DIAGNOSTICS = MarimoApp(
    "optical-diagnostics",
    "optical_diagnostics_app.py",
    2725,
    "optical radiation diagnostics",
    False,
)
optical_diagnostics_command = _launch_command(OPTICAL_DIAGNOSTICS)
optical_diagnostics_export_command = _export_command(OPTICAL_DIAGNOSTICS)
