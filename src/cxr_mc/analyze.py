"""``cxr app analysis`` -- launch the marimo analysis app (``notebooks/analysis_app.py``)
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

    cxr app analysis                    # persisted default (fallback hopg)
    cxr app analysis wse2               # transient: this run only, doesn't persist
    cxr app analysis -d wse2            # persist wse2 as the new default, and launch it
    cxr app analysis --watch            # add marimo's --watch (combinable with either)
    cxr app analysis --smoke            # execute the app once without a browser
    cxr app analysis --edit             # `marimo edit` instead of `marimo run`
    cxr app analysis --no-token         # pass marimo's --no-token (disable auth token)
"""

import gzip
import os
import pickle
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import click

from . import _checkpoint_store
from ._acp import running_acp
from .cli import _completion as _cli_completion
from .cli import _core as _cli_core

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
    """Return configured analysis materials, checkpoint-available ones first.

    Only direct ``<material>.pkl`` children of ``checkpoint_dir`` count. This
    deliberately excludes archived/reproduction caches and unknown pickle stems.
    Rows are grouped by availability -- materials with a checkpoint before
    those without -- and each group is sorted alphabetically by label
    (case-insensitive), so the dropdown surfaces ready-to-browse materials
    first instead of raw catalog order.
    """
    if materials is None or labels is None:
        from .materials import CATALOG

    materials = CATALOG.material_keys if materials is None else materials
    labels = (
        {material: CATALOG.material(material).label for material in materials}
        if labels is None
        else labels
    )
    available = set(_checkpoint_store.discover(checkpoint_dir)) & set(materials)
    rows: list[MaterialMenuRow] = [
        {
            "value": material,
            "label": labels.get(material, material),
            "disabled": material not in available,
        }
        for material in materials
    ]
    rows.sort(key=lambda row: (row["disabled"], row["label"].casefold()))
    return tuple(rows)


def select_initial_material(requested: str | None, menu: tuple[MaterialMenuRow, ...]) -> str | None:
    """Keep an available requested material, otherwise use the first available one."""
    selectable = [row["value"] for row in menu if not row["disabled"]]
    return requested if requested in selectable else next(iter(selectable), None)


def checkpoint_stem(material: str, face: str) -> str:
    """Checkpoint stem for a material's face variant.

    ``material`` itself for the flat (``cxr scan``) face; ``f"{material}_blazed"``
    for the blazed (sawtooth entrance-face) checkpoint written by
    ``cxr material blaze``.
    Passing this stem to :func:`~cxr_mc.run.load_checkpoint` loads the matching
    ``.pkl`` -- no change to ``load_checkpoint`` / ``checkpoint_path_for`` needed.
    """
    return material if face == "flat" else f"{material}_blazed"


def _read_checkpoint_or_none(material: str, read):
    """Return one analysis read, or ``None`` while its pickle is incomplete.

    A pull that writes into an active checkpoint path can briefly expose a
    truncated gzip/pickle. Analysis is read-only, so treating that interval as
    unavailable is safe; simulation resume paths continue to fail closed.
    """
    try:
        return read()
    except (
        EOFError,
        FileNotFoundError,
        gzip.BadGzipFile,
        pickle.UnpicklingError,
        zlib.error,
    ) as exc:
        print(
            f"checkpoint {material!r} is temporarily unreadable; "
            f"skipping it; retry after the pull finishes ({exc})",
            file=sys.stderr,
        )
        return None


def load_analysis_checkpoint(material: str, checkpoint_dir: Path | str | None = None):
    """Load a checkpoint for analysis, tolerating an in-progress transfer."""
    from .run import _DEFAULT_CHECKPOINT_DIR, load_checkpoint

    root = _DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return _read_checkpoint_or_none(material, lambda: load_checkpoint(material, root))


def analysis_checkpoint_manifest(material: str, checkpoint_dir: Path | str | None = None):
    """Read/backfill an analysis manifest, skipping an in-progress transfer."""
    from .run import _DEFAULT_CHECKPOINT_DIR, checkpoint_manifest

    root = _DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return _read_checkpoint_or_none(material, lambda: checkpoint_manifest(material, root))


def cached_analysis(material: str, analyze, key, checkpoint_dir: Path | str | None = None):
    """Run cached cross-material analysis unless its checkpoint is mid-transfer."""
    from .run import _DEFAULT_CHECKPOINT_DIR, cached_material_analysis

    root = _DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return _read_checkpoint_or_none(
        material,
        lambda: cached_material_analysis(material, analyze, key, root),
    )


def face_menu(material: str, checkpoint_dir: Path | str) -> tuple[MaterialMenuRow, ...]:
    """Two face rows (flat, blazed) for ``material``, each disabled when absent.

    Mirrors :func:`material_menu`'s ``Path(checkpoint_dir).glob("*.pkl")``
    existence convention so the face menu's ``disabled`` flags stay consistent
    with the material menu: a face is enabled iff its ``checkpoint_stem`` has a
    direct ``<stem>.pkl`` child of ``checkpoint_dir``. Flat is listed first
    (preferred default via :func:`select_initial_material` semantics).
    """
    available = set(_checkpoint_store.discover(checkpoint_dir))
    return tuple(
        {
            "value": face,
            "label": label,
            "disabled": checkpoint_stem(material, face) not in available,
        }
        for face, label in (("flat", "Flat"), ("blazed", "Blazed"))
    )


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
    env = {**os.environ, "CXR_ANALYZE_INITIAL": material}
    if smoke:
        with tempfile.TemporaryDirectory(prefix="cxr-mc-analysis-") as tmpdir:
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
        raise SystemExit("cxr app analysis -d/--default: no material given to persist")

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
    "analyze",
    help=(
        f"Launch {NOTEBOOK} with marimo run or edit.\n\n"
        "MATERIAL overrides the persisted default for this run. --persist-default "
        "stores it for later no-argument launches."
    ),
)
@click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)
@click.option(
    "-d",
    "--default",
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
        raise click.UsageError("--default requires MATERIAL")
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
    return _cli_core.run(command, argv, prog_name="cxr-analyze")


if __name__ == "__main__":
    raise SystemExit(main())
