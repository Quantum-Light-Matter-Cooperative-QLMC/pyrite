"""Checkpoint loading, menu building and emission selection for the marimo
analysis app (``src/pyrite/apps/analysis_app.py``).

Marimo apps in this environment can't be driven live (no browser/kernel access;
see the repo's marimo-live-driving notes), so the app's data plumbing and the
"which material does the dropdown start on" logic live here as pure,
unit-testable helpers rather than inline in the notebook. The notebook calls
them from cells that run before the widget cells.

The launcher that resolves and transports the material into the marimo
subprocess is :mod:`pyrite.cli.commands.app_analysis`; this module stays free of
CLI imports so the notebook can load it on its own.
"""

import gzip
import json
import pickle
import sys
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from .._app_defaults import get_analysis_default, set_analysis_default
from .._env import env_value
from ..checkpoints import _checkpoint_store


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

    A material counts as available if it has a direct component checkpoint or
    legacy ``<material>.pkl`` OR at least one other stem whose ``meta.json`` sidecar
    identifies it as that material (a named ``catalog_profile`` run,
    ``--quick``, etc. -- see :func:`profile_menu`, which lists them). An
    unrecognized stem (no sidecar, or a sidecar for a material outside
    ``materials``) stays excluded. Rows are grouped by availability --
    materials with a checkpoint before those without -- and each group is
    sorted alphabetically by label (case-insensitive), so the dropdown
    surfaces ready-to-browse materials first instead of raw catalog order.
    """
    if materials is None or labels is None:
        from ..materials import CATALOG

    materials = CATALOG.material_keys if materials is None else materials
    labels = (
        {material: CATALOG.material(material).label for material in materials}
        if labels is None
        else labels
    )
    material_set = set(materials)
    stems = _checkpoint_store.discover(checkpoint_dir)
    available = set(stems) & material_set
    for stem in stems:
        if stem in available:
            continue
        identity = _stem_dataset_identity(stem, checkpoint_dir)
        material_value = None if identity is None else identity.get("material")
        if isinstance(material_value, str) and material_value in material_set:
            available.add(material_value)
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

    ``material`` itself for the flat (``pyrite run``) face; ``f"{material}_blazed"``
    for the blazed (sawtooth entrance-face) checkpoint written by
    ``pyrite material blaze``.
    Passing this stem to :func:`~pyrite.runs.run.load_checkpoint` loads the matching
    component directory or legacy ``.pkl``.
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
        OSError,
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
    from ..runs.run import DEFAULT_CHECKPOINT_DIR, load_checkpoint

    root = DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return _read_checkpoint_or_none(material, lambda: load_checkpoint(material, root))


def analysis_checkpoint_manifest(material: str, checkpoint_dir: Path | str | None = None):
    """Read/backfill an analysis manifest, skipping an in-progress transfer."""
    from ..runs.run import DEFAULT_CHECKPOINT_DIR, checkpoint_manifest

    root = DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return _read_checkpoint_or_none(material, lambda: checkpoint_manifest(material, root))


def cached_analysis(material: str, analyze, key, checkpoint_dir: Path | str | None = None):
    """Run cached cross-material analysis unless its checkpoint is mid-transfer."""
    from ..runs.run import DEFAULT_CHECKPOINT_DIR, cached_material_analysis

    root = DEFAULT_CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    return _read_checkpoint_or_none(
        material,
        lambda: cached_material_analysis(material, analyze, key, root),
    )


def face_menu(material: str, checkpoint_dir: Path | str) -> tuple[MaterialMenuRow, ...]:
    """Two face rows (flat, blazed) for ``material``, each disabled when absent.

    Uses the same component-or-legacy discovery convention as
    :func:`material_menu`, so the face menu's ``disabled`` flags stay consistent:
    a face is enabled iff its ``checkpoint_stem`` is discoverable. Flat is listed first
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


def _stem_manifest_path(stem: str, checkpoint_dir: Path | str) -> Path | None:
    """The ``meta.json``/``.meta.json`` sidecar for ``stem``, or ``None``.

    Checks the component-store location (``<stem>/meta.json``) then the
    legacy flat-pickle sidecar (``<stem>.meta.json``), matching ``run.py``'s
    ``_manifest_path_for`` convention without importing ``run`` (avoids a
    heavier import for a menu-building helper).
    """
    root = Path(checkpoint_dir)
    dir_manifest = _checkpoint_store.manifest_path(stem, root)
    if dir_manifest.is_file():
        return dir_manifest
    flat_manifest = root / f"{stem}.meta.json"
    return flat_manifest if flat_manifest.is_file() else None


def _stem_dataset_identity(stem: str, checkpoint_dir: Path | str) -> dict[str, object] | None:
    """Read ``stem``'s recorded ``dataset_identity`` straight from its sidecar.

    Deliberately does not recompute/re-hash (see ``profiles.identity_from_stem``,
    which does and can silently stop matching once a named ``catalog_profile``
    is edited after the run that produced the stem). The sidecar is written at
    save time (``run._manifest_save``) and never changes after, so reading it
    back is the only source that stays correct regardless of later catalog edits.
    """
    manifest = _stem_manifest_path(stem, checkpoint_dir)
    if manifest is None:
        return None
    try:
        with manifest.open() as handle:
            return json.load(handle).get("dataset_identity")
    except OSError, ValueError, TypeError:
        return None


def profile_menu(material: str, checkpoint_dir: Path | str) -> tuple[MaterialMenuRow, ...]:
    """Selectable checkpoint variants for ``material``: the canonical run
    (labeled ``Standard``) plus any other stem whose sidecar identifies it as
    the same material -- named ``catalog_profile`` runs, ``--quick``, etc.

    Each non-canonical row is labeled by its resolved ``catalog_profile`` (or
    ``variant``/``fidelity`` when there is no named profile), suffixed with a
    short digest so two runs under the same profile stay distinguishable.
    Rows are canonical-first, then newest-first, so an unmodified
    :func:`select_initial_material` default keeps existing single-checkpoint
    materials behaving exactly as before.
    """
    root = Path(checkpoint_dir)
    if not material or not root.is_dir():
        return ()
    rows: list[MaterialMenuRow] = []
    if _checkpoint_store.checkpoint_exists(material, root):
        rows.append({"value": material, "label": "Standard", "disabled": False})
    variants: list[tuple[float, MaterialMenuRow]] = []
    for stem in _checkpoint_store.discover(root):
        if stem == material or stem == f"{material}_blazed":
            continue
        identity = _stem_dataset_identity(stem, root)
        if identity is None or identity.get("material") != material:
            continue
        label = identity.get("catalog_profile")
        if not label or label == "standard":
            label = identity.get("variant") or identity.get("fidelity") or "variant"
        digest = str(identity.get("parameter_sha256", ""))[:6]
        manifest = _stem_manifest_path(stem, root)
        mtime = manifest.stat().st_mtime if manifest is not None else 0.0
        variants.append(
            (-mtime, {"value": stem, "label": f"{label} ({digest})", "disabled": False})
        )
    variants.sort(key=lambda item: item[0])
    rows.extend(row for _, row in variants)
    return tuple(rows)


def comparison_stem(material: str, checkpoint_dir: Path | str) -> str | None:
    """The checkpoint stem cross-material comparison should read for ``material``.

    :func:`profile_menu`'s canonical-first, newest-first ordering already picks
    the right stem for a single-material dropdown; reuse it here so a material
    with only a named ``catalog_profile`` checkpoint (no direct ``<material>.pkl``)
    still counts as available -- :func:`analyze.analysis_checkpoint_manifest`
    only ever checks the direct stem and would otherwise miss it. Returns
    ``None`` if ``material`` has no checkpoint at all.
    """
    rows = profile_menu(material, checkpoint_dir)
    return rows[0]["value"] if rows else None


def emission_menu(results) -> tuple[MaterialMenuRow, ...]:
    """Emission-view rows for a LOADED checkpoint's records: ``Incoherent`` (the
    always-stored ``spec``), ``Coherent`` (the ``spec_coherent`` a
    ``coherent``/``both`` run also stored), and ``Both`` (overlay both traces on
    one chart). ``Coherent``/``Both`` are ``disabled`` unless at least one loaded
    record carries a ``spec_coherent`` array, so an incoherent checkpoint offers
    only the incoherent view. Mirrors :func:`face_menu`'s ``disabled`` convention;
    gated on the stored spectra rather than the sidecar so it reflects exactly
    what can be drawn."""
    has_incoherent = False
    has_coherent = False
    for by_energy in (results or {}).values():
        for record in by_energy.values():
            if record.get("spec") is not None:
                has_incoherent = True
            if record.get("spec_coherent") is not None:
                has_coherent = True
    return (
        {"value": "incoherent", "label": "Incoherent", "disabled": not has_incoherent},
        {"value": "coherent", "label": "Coherent", "disabled": not has_coherent},
        {"value": "both", "label": "Both", "disabled": not (has_incoherent and has_coherent)},
    )


def pick_spectrum(record, emission):
    """Central incoherent-vs-coherent spectrum picker: the coherent
    ``spec_coherent`` when ``emission == "coherent"`` and the record stored one,
    else the incoherent ``spec``. Every analysis-app spectrum read routes through
    this (via :func:`apply_emission`) so a ``both`` checkpoint can toggle emission
    without any downstream plot knowing the difference."""
    if emission == "coherent":
        coherent = record.get("spec_coherent")
        if coherent is not None:
            return coherent
    return record.get("spec")


def apply_emission(results, emission):
    """Return a ``results`` view whose every record's ``spec`` is the
    :func:`pick_spectrum` choice for ``emission``. ``incoherent`` (the default,
    and any record lacking ``spec_coherent``) returns ``results`` unchanged;
    ``coherent`` returns a shallow-copied nested dict where each record with a
    ``spec_coherent`` has its ``spec`` pointed at that array. Records are
    shallow-copied (array refs shared, never mutated), so the loaded checkpoint is
    left intact while every ``r["spec"]`` reader sees the selected emission."""
    if emission != "coherent":
        return results
    picked: dict = {}
    for name, by_energy in (results or {}).items():
        picked[name] = {}
        for energy, record in by_energy.items():
            chosen = pick_spectrum(record, emission)
            if chosen is record.get("spec"):
                picked[name][energy] = record
            else:
                new_record = dict(record)
                new_record["spec"] = chosen
                picked[name][energy] = new_record
    return picked


def apply_characteristic(results, *, include: bool):
    """Return a view with characteristic radiation included or removed.

    Records keep ``spec_characteristic`` separate from the line arrays, and
    consumers add it through :func:`pyrite._spectral_components.line_spectrum`.
    When ``include`` is false, return shallow record copies without that
    component. Source records are never mutated.
    """
    if include or not any(
        record.get("spec_characteristic") is not None
        for by_energy in (results or {}).values()
        for record in by_energy.values()
    ):
        return results
    picked: dict = {}
    for name, by_energy in (results or {}).items():
        picked[name] = {}
        for energy, record in by_energy.items():
            if "spec_characteristic" not in record:
                picked[name][energy] = record
                continue
            picked[name][energy] = {
                key: value for key, value in record.items() if key != "spec_characteristic"
            }
    return picked


def get_default_material():
    """The persisted default material, or None if never set / empty."""
    return get_analysis_default()


def set_default_material(material):
    """Persist ``material`` as the default for future no-argument runs."""
    set_analysis_default(material)


def initial_material(cli_args, persisted_default):
    """Resolve the dropdown's initial value. Pure function -- called both from
    the notebook cell (with ``mo.cli_args()``) and directly from tests.

    Precedence: cli-arg material -> ``PYRITE_ANALYZE_INITIAL`` env var (fallback
    transport, in case ``mo.cli_args()`` doesn't reach ``marimo edit``) ->
    ``persisted_default`` -> ``"hopg"``.
    """
    cli_material = cli_args.get("material")
    if cli_material:
        return str(cli_material)
    env_material = env_value("PYRITE_ANALYZE_INITIAL")
    if env_material:
        return env_material
    if persisted_default:
        return persisted_default
    return "hopg"
