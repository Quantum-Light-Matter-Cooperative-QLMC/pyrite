"""``cxr analyze`` -- launch the marimo analysis app (``notebooks/analysis_app.py``)
with a chosen initial material.

Marimo apps in this environment can't be driven live (no browser/kernel access;
see the repo's marimo-live-driving notes), so the "which material does the
dropdown start on" logic lives here as a pure, unit-testable helper
(:func:`initial_material`) rather than inline in the notebook. The notebook just
calls it in a cell that runs before the dropdown cell.

Precedence for the initial material: an explicit CLI material always wins; else
the persisted default (:func:`get_default_material`, written by ``-d/--default``);
else ``"hopg"``.

Two transports carry the resolved material into the marimo subprocess, since we
can't verify in this environment whether ``mo.cli_args()`` reaches ``marimo edit``
(only confirmed for ``marimo run``): ``-- --material <X>`` app args (marimo's own
mechanism) AND a ``CXR_ANALYZE_INITIAL`` environment variable set on the
subprocess, as a belt-and-suspenders fallback. :func:`initial_material` checks
both.

    cxr analyze                    # persisted default (fallback hopg)
    cxr analyze wse2               # transient: this run only, doesn't persist
    cxr analyze -d wse2            # persist wse2 as the new default, and launch it
    cxr analyze --watch            # add marimo's --watch (combinable with either)
    cxr analyze --headless         # start without opening a browser
    cxr analyze --smoke            # execute the app once without a browser
    cxr analyze --edit             # `marimo edit` instead of `marimo run`
"""

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from ._acp import running_acp
from .materials import CATALOG

NOTEBOOK = "notebooks/analysis_app.py"
TUNNEL_PORT = 2718

# Anchored to the repo root (src/cxr_mc/analyze.py -> parents[2] = repo root),
# the same convention run.py uses for _DEFAULT_CHECKPOINT_DIR, so the persisted
# default is found regardless of the caller's cwd.
_DEFAULT_FILE = Path(__file__).resolve().parents[2] / ".cxr-analyze-default"


class MaterialMenuRow(TypedDict):
    """One configured material and whether its analysis checkpoint exists."""

    value: str
    label: str
    disabled: bool


def material_menu(
    checkpoint_dir: Path | str,
    materials: tuple[str, ...] | None = None,
    labels: Mapping[str, str] | None = None,
) -> tuple[MaterialMenuRow, ...]:
    """Return configured analysis materials, disabling ones without a checkpoint.

    Only direct ``<material>.pkl`` children of ``checkpoint_dir`` count. This
    deliberately excludes archived/reproduction caches and unknown pickle stems.
    """
    materials = CATALOG.material_keys if materials is None else materials
    labels = (
        {material: CATALOG.material(material).label for material in materials}
        if labels is None
        else labels
    )
    available = {path.stem for path in Path(checkpoint_dir).glob("*.pkl") if path.stem in materials}
    return tuple(
        {
            "value": material,
            "label": labels.get(material, material),
            "disabled": material not in available,
        }
        for material in materials
    )


def select_initial_material(requested: str | None, menu: tuple[MaterialMenuRow, ...]) -> str | None:
    """Keep an available requested material, otherwise use the first available one."""
    selectable = [row["value"] for row in menu if not row["disabled"]]
    return requested if requested in selectable else next(iter(selectable), None)


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

    Precedence: cli-arg material -> ``CXR_ANALYZE_INITIAL`` env var (fallback
    transport, in case ``mo.cli_args()`` doesn't reach ``marimo edit``) ->
    ``persisted_default`` -> ``"hopg"``.
    """
    cli_material = cli_args.get("material")
    if cli_material:
        return str(cli_material)
    env_material = os.environ.get("CXR_ANALYZE_INITIAL")
    if env_material:
        return env_material
    if persisted_default:
        return persisted_default
    return "hopg"


def _command(material, *, edit=False, watch=False, headless=False, tunnel=False):
    """The marimo argv for one launch (module-run through the current
    interpreter so the venv's marimo is the one that runs). Marimo's own flags
    go before the notebook path; app args go after ``--``."""
    return [
        sys.executable,
        "-m",
        "marimo",
        "edit" if edit else "run",
        *(["--watch"] if watch else []),
        *(["--headless"] if headless else []),
        *(["--port", str(TUNNEL_PORT)] if tunnel else []),
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


def _launch(material, *, edit=False, watch=False, headless=False, smoke=False, acp=False, tunnel=False):
    cmd = _command(material, edit=edit, watch=watch, headless=headless, tunnel=tunnel)
    print(f"launching {NOTEBOOK} ({'edit' if edit else 'run'}) with material={material}")
    if tunnel:
        print(f"ssh -L {TUNNEL_PORT}:127.0.0.1:{TUNNEL_PORT} <your-pi-ssh-host>")
        print(f"http://127.0.0.1:{TUNNEL_PORT}")
    env = {**os.environ, "CXR_ANALYZE_INITIAL": material}
    if smoke:
        with tempfile.TemporaryDirectory(prefix="cxr-mc-analysis-") as tmpdir:
            subprocess.run(_smoke_command(material, Path(tmpdir) / "analysis.html"), check=True, env=env)
        return
    if acp:
        with running_acp():
            subprocess.run(cmd, check=True, env=env)
    else:
        subprocess.run(cmd, check=True, env=env)


def _cli(args):
    if args.default and args.material is None:
        raise SystemExit("cxr analyze -d/--default: no material given to persist")
    if args.material is not None and args.material not in CATALOG.materials:
        valid = ", ".join(CATALOG.material_keys)
        raise SystemExit(f"cxr analyze: unknown material {args.material!r}; valid: {valid}")

    if args.default:
        set_default_material(args.material)

    material = args.material or get_default_material() or "hopg"
    launch_args = {"edit": args.edit, "watch": args.watch}
    if args.headless:
        launch_args["headless"] = True
    if args.smoke:
        launch_args["smoke"] = True
    if args.acp:
        launch_args["acp"] = True
    if args.tunnel:
        launch_args["tunnel"] = True
    _launch(material, **launch_args)


def add_subparser(sub):
    """Register the ``analyze`` subcommand on an argparse subparsers object."""
    ap = sub.add_parser("analyze", help=f"launch {NOTEBOOK} (marimo run/edit)")
    ap.add_argument(
        "material",
        nargs="?",
        default=None,
        help="initial dropdown material (transient; default: persisted default, else hopg)",
    )
    ap.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="also persist <material> as the new default for future no-argument runs",
    )
    ap.add_argument("--watch", action="store_true", help="pass marimo's --watch")
    ap.add_argument("--headless", action="store_true", help="start marimo without opening a browser")
    ap.add_argument("--smoke", action="store_true", help="execute the app once headlessly and exit")
    ap.add_argument("--edit", action="store_true", help="use `marimo edit` instead of `marimo run`")
    ap.add_argument("--acp", action="store_true", help="start local Claude and Codex ACP bridges")
    ap.add_argument("--tunnel", action="store_true", help="use a fixed port for SSH tunneling")
    ap.set_defaults(func=_cli)
    return ap


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cxr-analyze", description="launch the analysis app")
    add_subparser(ap.add_subparsers(dest="command", required=True))
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
