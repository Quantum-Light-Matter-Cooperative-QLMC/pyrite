"""Shared marimo launcher behind ``pyrite app analysis`` and ``pyrite app viewer``.

Each app module declares one :class:`MarimoApp` (notebook, tunnel port, initial
material environment variable, persisted-default accessors) and builds its
Click command with :func:`app_command`; argv construction, launch and the
``-d/--save-default`` precedence live here once.
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
from .. import _completion as _cli_completion
from .._deprecations import canonical_option
from .._extras import require_cli_extra


@dataclass(frozen=True)
class MarimoApp:
    """One marimo app launched with an initial material.

    ``base_env`` entries sit below the caller's environment, so a caller's own
    value wins; ``initial_env`` carries the resolved material and always wins.
    """

    name: str
    notebook: str
    tunnel_port: int
    initial_env: str
    get_default: Callable[[], str | None]
    set_default: Callable[[str], None]
    base_env: Mapping[str, str] = field(default_factory=dict)

    def command(self, material, *, edit=False, watch=False, tunnel=False, no_token=False):
        """The marimo argv for one launch (module-run through the current
        interpreter so the venv's marimo is the one that runs). Marimo's own flags
        go before the notebook path; app args go after ``--``."""
        return [
            sys.executable,
            "-m",
            "marimo",
            "edit" if edit else "run",
            *(["--watch"] if watch else []),
            *(["--port", str(self.tunnel_port)] if tunnel else []),
            *(["--no-token"] if no_token else []),
            self.notebook,
            "--",
            "--material",
            material,
        ]

    def smoke_command(self, material, output):
        """The one-shot, headless command used to execute the app in CI-like checks."""
        return [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            self.notebook,
            "--output",
            str(output),
            "--force",
            "--",
            "--material",
            material,
        ]

    def launch(
        self,
        material,
        *,
        edit=False,
        watch=False,
        smoke=False,
        acp=False,
        tunnel=False,
        no_token=False,
    ):
        require_cli_extra("apps", feature=f"pyrite app {self.name} launch")
        cmd = self.command(material, edit=edit, watch=watch, tunnel=tunnel, no_token=no_token)
        print(f"launching {self.notebook} ({'edit' if edit else 'run'}) with material={material}")
        if tunnel:
            print(f"ssh -L {self.tunnel_port}:127.0.0.1:{self.tunnel_port} <your-pi-ssh-host>")
            print(f"http://127.0.0.1:{self.tunnel_port}")
        env = {**self.base_env, **os.environ, self.initial_env: material}
        if smoke:
            with tempfile.TemporaryDirectory(prefix=f"pyrite-{self.name}-") as tmpdir:
                output = Path(tmpdir) / f"{self.name}.html"
                subprocess.run(self.smoke_command(material, output), check=True, env=env)
            return
        if acp:
            with running_acp():
                subprocess.run(cmd, check=True, env=env)
        else:
            subprocess.run(cmd, check=True, env=env)

    def run_cli(self, args, launch):
        """Resolve the initial material from ``args`` and call ``launch``."""
        if args.default and args.material is None:
            raise SystemExit(
                f"pyrite app {self.name} -d/--save-default: no material given to persist"
            )

        if args.default:
            self.set_default(args.material)

        material = args.material or self.get_default() or "hopg"
        launch_args = {"edit": args.edit, "watch": args.watch}
        for flag in ("smoke", "acp", "tunnel", "no_token"):
            if getattr(args, flag):
                launch_args[flag] = True
        launch(material, **launch_args)


def app_command(name, help, cli):
    """Click command for one app launcher; ``cli`` receives the legacy args."""

    @click.command(name, help=help)
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
