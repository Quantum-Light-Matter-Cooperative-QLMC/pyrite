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

import click

from ..._app_defaults import get_analysis_default
from ._app_launcher import (
    BACKEND_CHOICES,
    MATERIAL_ARGUMENT,
    MarimoApp,
    export,
    launch,
    marimo_command,
    smoke_command,
)

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


_command = marimo_command
_smoke_command = smoke_command
_launch = launch


def resolve_material(material: str | None) -> str:
    """Explicit MATERIAL, else the analysis app's persisted default, else hopg."""
    return material or get_analysis_default() or "hopg"


def _export(app, stem: str | None, material: str | None) -> None:
    """Export the app's HTML without starting a marimo server or browser."""
    export(app, stem, resolve_material(material) if app.material_picker else None)


def _launch_command(app: MarimoApp) -> click.Command:
    material_argument = MATERIAL_ARGUMENT if app.material_picker else lambda command: command

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
    material_argument = MATERIAL_ARGUMENT if app.material_picker else lambda command: command

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
