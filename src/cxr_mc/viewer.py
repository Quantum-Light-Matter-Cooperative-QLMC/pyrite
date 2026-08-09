"""``cxr app viewer`` -- launch the marimo 3D visualization app (``notebooks/trace_app.py``)
with a chosen initial material.

Marimo apps in this environment can't be driven live (no browser/kernel access;
see the repo's marimo-live-driving notes), so the "which material does the
dropdown start on" logic lives here as a pure, unit-testable helper
(:func:`initial_material`) rather than inline in the notebook. The notebook just
calls it in a cell that runs before the dropdown cell.

Precedence for the initial material: an explicit CLI material always wins; else
the persisted default (:func:`get_default_material`, written by
``-d/--save-default``);
else ``"hopg"``.

Two transports carry the resolved material into the marimo subprocess, since we
can't verify in this environment whether ``mo.cli_args()`` reaches ``marimo edit``
(only confirmed for ``marimo run``): ``-- --material <X>`` app args (marimo's own
mechanism) AND a ``CXR_VIEWER_INITIAL`` environment variable set on the
subprocess, as a belt-and-suspenders fallback. :func:`initial_material` checks
both.

    cxr app viewer                    # persisted default (fallback hopg)
    cxr app viewer wse2               # transient: this run only, doesn't persist
    cxr app viewer -d wse2            # persist wse2 as the new default, and launch it
    cxr app viewer --watch            # add marimo's --watch (combinable with either)
    cxr app viewer --smoke            # execute the app once without a browser
    cxr app viewer --edit             # `marimo edit` instead of `marimo run`
    cxr app viewer --no-token         # pass marimo's --no-token (disable auth token)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from ._acp import running_acp
from .cli import _completion as _cli_completion
from .cli import _core as _cli_core
from .cli._deprecations import canonical_option

NOTEBOOK = "notebooks/trace_app.py"
TUNNEL_PORT = 2719

# Anchored to the repo root (src/cxr_mc/viewer.py -> parents[2] = repo root),
# the same convention run.py uses for _DEFAULT_CHECKPOINT_DIR, so the persisted
# default is found regardless of the caller's cwd.
_DEFAULT_FILE = Path(__file__).resolve().parents[2] / ".cxr-viewer-default"


def get_default_material():
    """The persisted default material, or None if never set / empty."""
    try:
        text = _DEFAULT_FILE.read_text().strip()
    except FileNotFoundError:
        return None
    return text or None


def set_default_material(material):
    """Persist ``material`` as the default for future no-argument runs."""
    _DEFAULT_FILE.write_text(material)


def initial_material(cli_args, persisted_default):
    """Resolve the dropdown's initial value. Pure function -- called both from
    the notebook cell (with ``mo.cli_args()``) and directly from tests.

    Precedence: cli-arg material -> ``CXR_VIEWER_INITIAL`` env var (fallback
    transport, in case ``mo.cli_args()`` doesn't reach ``marimo edit``) ->
    ``persisted_default`` -> ``"hopg"``.
    """
    cli_material = cli_args.get("material")
    if cli_material:
        return str(cli_material)
    env_material = os.environ.get("CXR_VIEWER_INITIAL")
    if env_material:
        return env_material
    if persisted_default:
        return persisted_default
    return "hopg"


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
    """The one-shot, headless command used to execute the viewer app in CI-like checks."""
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


def _export(stem: str | None, material: str | None) -> None:
    """Export viewer HTML without starting a marimo server or browser."""
    resolved = material or get_default_material() or "hopg"
    stem = stem or f"cxr_viewer_{resolved}"
    output = Path("results") / f"{stem}.html"
    click.echo(f"exporting {NOTEBOOK} -> {output}")
    subprocess.run(_smoke_command(resolved, output), check=True)


def _launch(
    material, *, edit=False, watch=False, smoke=False, acp=False, tunnel=False, no_token=False
):
    cmd = _command(material, edit=edit, watch=watch, tunnel=tunnel, no_token=no_token)
    print(f"launching {NOTEBOOK} ({'edit' if edit else 'run'}) with material={material}")
    if tunnel:
        print(f"ssh -L {TUNNEL_PORT}:127.0.0.1:{TUNNEL_PORT} <your-pi-ssh-host>")
        print(f"http://127.0.0.1:{TUNNEL_PORT}")
    env = {**os.environ, "CXR_VIEWER_INITIAL": material}
    if smoke:
        with tempfile.TemporaryDirectory(prefix="cxr-mc-viewer-") as tmpdir:
            subprocess.run(
                _smoke_command(material, Path(tmpdir) / "viewer.html"), check=True, env=env
            )
        return
    if acp:
        with running_acp():
            subprocess.run(cmd, check=True, env=env)
    else:
        subprocess.run(cmd, check=True, env=env)


def _cli(args):
    if args.default and args.material is None:
        raise SystemExit("cxr app viewer -d/--save-default: no material given to persist")

    if args.default:
        set_default_material(args.material)

    material = args.material or get_default_material() or "hopg"
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
    "viewer",
    help=(
        "Launch the interactive 3D viewer with marimo run or edit.\n\n"
        "3D trajectory and crystal structure visualization.\n\n"
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
    retired=["--default"],
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


@click.command("export", help="Render the viewer as static HTML without starting marimo.")
@click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)
@click.option("--stem", help="Output stem under results/ (without .html).")
def export_command(material, stem):
    _export(stem, material)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="cxr-viewer")


if __name__ == "__main__":
    raise SystemExit(main())
