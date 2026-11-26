"""Shared marimo launcher behind every ``pyrite app <name> launch|export`` command.

Each app declares one :class:`MarimoApp` (command name, notebook, tunnel port,
title, initial-material environment variable). Argv construction, headless
smoke/export, ACP bridges, environment precedence and child-failure reporting
live here once. ``app_views`` builds the material-picker and standalone apps;
``app_analysis`` and ``app_viewer`` add a persisted default material
(``-d/--save-default``) through :func:`persisted_launch_command`.
"""

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import click

from ..._acp import running_acp
from ...console import output as _cli_core
from ...console.outputs import output_dir
from ...paths import app_dir
from .. import _completion as _cli_completion
from .._deprecations import canonical_option
from .._extras import require_cli_extra


@dataclass(frozen=True)
class MarimoApp:
    """One launchable marimo app.

    ``initial_env`` carries the resolved material to the app process and always
    wins; ``base_env`` entries sit below the caller's environment, so a caller's
    own value wins.
    """

    name: str
    notebook_name: str
    tunnel_port: int
    title: str
    material_picker: bool = True
    initial_env: str = "PYRITE_ANALYZE_INITIAL"
    base_env: Mapping[str, str] = field(default_factory=dict, hash=False, compare=False)

    @property
    def notebook(self) -> str:
        return str(app_dir() / self.notebook_name)


#: ``PYRITE_MC_BACKEND`` values (mirrors ``pyrite._backend``, which probes devices on import).
BACKEND_CHOICES = ("auto", "cpu", "cuda", "rocm", "sycl")

MATERIAL_ARGUMENT = click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)


def _material_args(material: str | None) -> list[str]:
    return ["--", "--material", material] if material is not None else []


def marimo_command(app, material=None, *, edit=False, watch=False, tunnel=False, no_token=False):
    """The marimo argv for one launch, module-run through the current interpreter
    so the environment's marimo runs. Marimo's flags precede the notebook; app
    args follow ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        *(["--port", str(app.tunnel_port)] if tunnel else []),
        *(["--no-token"] if no_token else []),
        app.notebook,
        *_material_args(material),
    ]


def smoke_command(app, material, output):
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
        *_material_args(material),
    ]


def app_env(app, material=None, backend=None) -> dict[str, str]:
    """Child environment: app base entries < caller environment < material/backend."""
    env = {**app.base_env, **os.environ}
    if material is not None:
        env[app.initial_env] = material
    if backend is not None:
        # The array backend is chosen once at import; the app process inherits it.
        env["PYRITE_MC_BACKEND"] = backend
    return env


def run_marimo(command: list[str], env: dict[str, str]) -> None:
    """Keep child failures nonzero and report them through Click diagnostics."""
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"marimo exited with status {exc.returncode}") from exc
    except OSError as exc:
        raise click.ClickException(f"could not start marimo: {exc}") from exc


def launch(
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
    cmd = marimo_command(app, material, edit=edit, watch=watch, tunnel=tunnel, no_token=no_token)
    label = f" with material={material}" if material is not None else ""
    click.echo(f"launching {app.notebook} ({'edit' if edit else 'run'}){label}")
    if tunnel:
        click.echo(f"ssh -L {app.tunnel_port}:127.0.0.1:{app.tunnel_port} <your-pi-ssh-host>")
        click.echo(f"http://127.0.0.1:{app.tunnel_port}")
    env = app_env(app, material, backend)
    if backend is not None:
        click.echo(f"array backend: {backend}")
    if smoke:
        with tempfile.TemporaryDirectory(prefix=f"pyrite-{app.name}-") as tmpdir:
            output = Path(tmpdir) / f"{app.name}.html"
            run_marimo(smoke_command(app, material, output), env)
        return
    if acp:
        with running_acp():
            run_marimo(cmd, env)
    else:
        run_marimo(cmd, env)


def export(app, stem: str | None, material: str | None) -> None:
    """Export the app's HTML (``material`` already resolved) without a server or browser."""
    require_cli_extra("apps", feature=f"pyrite app {app.name} export")
    suffix = f"_{material}" if material is not None else ""
    stem = stem or f"pyrite_{app.name}{suffix}"
    output = output_dir("results") / f"{stem}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"exporting {app.notebook} -> {output}")
    run_marimo(smoke_command(app, material, output), app_env(app, material))


def persisted_launch(args, app, get_default, set_default, launch_fn):
    """Resolve MATERIAL > persisted default > ``hopg``; ``-d`` persists MATERIAL first."""
    if args.default and args.material is None:
        raise SystemExit(f"pyrite app {app.name} -d/--save-default: no material given to persist")
    if args.default:
        set_default(args.material)
    material = args.material or get_default() or "hopg"
    launch_args = {"edit": args.edit, "watch": args.watch}
    for flag in ("smoke", "acp", "tunnel", "no_token"):
        if getattr(args, flag):
            launch_args[flag] = True
    launch_fn(material, **launch_args)


def persisted_launch_command(name: str, help: str, cli: Callable) -> click.Command:
    """Click launch command with ``-d/--save-default``; ``cli`` receives the parsed args."""

    @click.command(name, help=help)
    @MATERIAL_ARGUMENT
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
            cli,
            material=material,
            default=persist_default,
            watch=watch,
            smoke=smoke,
            edit=edit,
            acp=acp,
            tunnel=tunnel,
            no_token=no_token,
        )

    return command
