"""Checkpoint reclamation: obsolete-record pruning and dataset deletion.

``prune_checkpoints`` reclaims records no longer reproducible under the current
catalog scan profiles; ``clear_checkpoints`` deletes explicitly selected
datasets together with the CAS blobs they alone kept reachable. Click wiring for
both lives in :mod:`pyrite.cli.commands.cleanup`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from ..campaign.profiles import _jsonable, named_profile_identity, named_profile_stem
from ..cli import _core as _cli_core
from ..materials import CATALOG
from ..paths import workspace_root
from . import _checkpoint_store

_DEFAULT_CHECKPOINT_DIR = str(workspace_root() / "checkpoints")
_FIDELITIES = ("full", "survey")


@dataclass(frozen=True)
class _Target:
    stem: str
    material: str
    fidelity: str
    catalog_profile: str
    identity: dict[str, Any]


def _profile_materials(catalog_profile: str) -> tuple[str, ...]:
    if catalog_profile == "standard":
        return CATALOG.material_keys
    try:
        membership = CATALOG.profile_materials(catalog_profile)
    except KeyError as exc:
        raise click.UsageError(str(exc)) from None
    return CATALOG.material_keys if membership is None else membership


def _selected_profiles(all_profiles: bool, catalog_profile: str | None) -> tuple[str, ...]:
    if all_profiles and catalog_profile is not None:
        raise click.UsageError("gc --all cannot be combined with --profile")
    if all_profiles:
        return ("standard", *CATALOG.profile_names)
    return (catalog_profile or "standard",)


def _targets(all_profiles: bool, catalog_profile: str | None) -> list[_Target]:
    targets: dict[str, _Target] = {}
    for profile in _selected_profiles(all_profiles, catalog_profile):
        for material in _profile_materials(profile):
            for fidelity in _FIDELITIES:
                identity = named_profile_identity(
                    material,
                    fidelity,
                    catalog_profile=profile,
                )
                stem = named_profile_stem(
                    material,
                    fidelity,
                    catalog_profile=profile,
                )
                targets.setdefault(
                    stem,
                    _Target(stem, material, fidelity, profile, identity),
                )
    return list(targets.values())


def _case_key(case: dict[str, Any]) -> str:
    """Stable exact comparison covering grids, geometry, statistics, and seeds."""
    return json.dumps(
        _jsonable(case),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )


def _current_case_keys(target: _Target) -> dict[tuple[str, float], str]:
    from ..campaign.config import default_settings, material_sweep
    from ..campaign.sweep import build_cases

    settings = default_settings(target.fidelity)
    sweep = material_sweep(
        target.material,
        fidelity=target.fidelity,
        catalog_profile=target.catalog_profile,
    )
    return {
        (case["name"], float(case["E0_keV"])): _case_key(case)
        for case in build_cases(
            sweep,
            settings.n_electrons,
            settings.n_electrons_brem,
            coherent_emission=settings.coherent_emission,
            xray_dispersion=settings.xray_dispersion,
        )
    }


def _partition(results: dict, target: _Target) -> tuple[dict, int]:
    current = _current_case_keys(target)
    kept: dict = {}
    stale = 0
    for name, by_energy in results.items():
        selected = {}
        for energy, record in by_energy.items():
            expected = current.get((name, float(energy)))
            case = record.get("case")
            if expected is None or not isinstance(case, dict) or _case_key(case) != expected:
                stale += 1
                continue
            selected[energy] = record
        if selected:
            kept[name] = selected
    return kept, stale


def _write_target(target: _Target, root: Path, results: dict) -> None:
    """Atomically rewrite current storage form and refresh dataset identity."""
    from ..runs.run import _checkpoint_save, _manifest_save

    component = _checkpoint_store.component_path(target.stem, "line", root)
    if component.is_file():
        checkpoint = _checkpoint_store.checkpoint_dir(target.stem, root)
        _checkpoint_store.save(target.stem, root, results)
        _manifest_save(str(checkpoint), results, target.identity)
        return
    legacy = _checkpoint_store.legacy_path(target.stem, root)
    _checkpoint_save(str(legacy), results)
    _manifest_save(str(legacy), results, target.identity)


def prune_checkpoints(
    *,
    all_profiles: bool = False,
    catalog_profile: str | None = None,
    yes: bool = False,
    checkpoint_dir: str | Path = _DEFAULT_CHECKPOINT_DIR,
) -> int:
    """Preview or remove records not exactly reproducible by current profiles.

    Returns number of stale records found. Only current named-profile stems are
    eligible; custom, quick, high-energy-floor, archived, and unrecognized
    checkpoints remain untouched.
    """
    root = Path(checkpoint_dir)
    selected = [
        target
        for target in _targets(all_profiles, catalog_profile)
        if _checkpoint_store.checkpoint_exists(target.stem, root)
    ]
    label = "all profiles" if all_profiles else f"profile={catalog_profile or 'standard'}"
    if not selected:
        click.echo(f"(nothing to prune for {label}: no current checkpoints)")
        return 0

    changes: list[tuple[_Target, dict, int, int]] = []
    for target in selected:
        results = _checkpoint_store.load(target.stem, root)
        kept, stale = _partition(results, target)
        if stale:
            changes.append(
                (target, kept, stale, sum(len(by_energy) for by_energy in results.values()))
            )
    if not changes:
        click.echo(f"(nothing stale for {label})")
        return 0

    root_label = root.name or str(root)
    click.echo("would prune:")
    for target, kept, stale, before in changes:
        after = sum(len(by_energy) for by_energy in kept.values())
        click.echo(f"  {root_label}/{target.stem}/: {stale} stale record(s) ({before} -> {after})")
    if not _cli_core.confirm_destructive(yes, "Delete these stale checkpoint records?"):
        return sum(change[2] for change in changes)

    for target, kept, stale, before in changes:
        _write_target(target, root, kept)
        after = before - stale
        click.echo(
            f"pruned {root_label}/{target.stem}/: removed {stale} stale record(s) "
            f"({before} -> {after})"
        )
    return sum(change[2] for change in changes)


@dataclass(frozen=True)
class _CaseManifest:
    path: Path
    digest: str
    material: str
    catalog_profile: str | None
    keys: frozenset[str]


def _read_case_manifest(path: Path) -> _CaseManifest:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise _cli_core.CLIError(f"cannot safely clear: invalid manifest {path}: {exc}") from None
    if not isinstance(payload, dict) or payload.get("schema") != "cxr.case-manifest.v1":
        raise _cli_core.CLIError(
            f"cannot safely clear: invalid manifest {path}: expected cxr.case-manifest.v1"
        )
    material = payload.get("material")
    cases = payload.get("cases")
    profile = payload.get("catalog_profile")
    if not isinstance(material, str) or not material:
        raise _cli_core.CLIError(f"cannot safely clear: invalid manifest {path}: missing material")
    if profile is not None and (not isinstance(profile, str) or not profile):
        raise _cli_core.CLIError(
            f"cannot safely clear: invalid manifest {path}: invalid catalog_profile"
        )
    if not isinstance(cases, list):
        raise _cli_core.CLIError(
            f"cannot safely clear: invalid manifest {path}: cases must be a list"
        )
    keys: set[str] = set()
    for index, case in enumerate(cases):
        key = case.get("content_key") if isinstance(case, dict) else None
        if not isinstance(key, str):
            raise _cli_core.CLIError(
                f"cannot safely clear: invalid manifest {path}: "
                f"cases[{index}].content_key is not a SHA-256 digest"
            )
        try:
            keys.add(_checkpoint_store._validate_content_key(key))
        except (TypeError, ValueError):
            raise _cli_core.CLIError(
                f"cannot safely clear: invalid manifest {path}: "
                f"cases[{index}].content_key is not a SHA-256 digest"
            ) from None
    return _CaseManifest(
        path=path,
        digest=hashlib.sha256(raw).hexdigest(),
        material=material,
        catalog_profile=profile,
        keys=frozenset(keys),
    )


def _active_manifest_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for stem in _checkpoint_store.discover(root):
        directory = root / stem / "cases.json"
        legacy = root / f"{stem}.cases.json"
        if directory.is_file():
            paths[stem] = directory
        elif legacy.is_file():
            paths[stem] = legacy
    return paths


def _archive_manifest_paths(root: Path) -> list[Path]:
    archive = root / "archive"
    if not archive.is_dir():
        return []
    return sorted(
        {
            *archive.glob("*.cases.json"),
            *archive.glob("*/cases.json"),
        }
    )


def _legacy_material_match(stem: str, material: str) -> bool:
    return stem == material or stem in {f"{material}_quick", f"{material}_blazed"}


def _select_stems(
    root: Path,
    manifests: dict[str, _CaseManifest],
    *,
    materials: tuple[str, ...],
    catalog_profile: str | None,
    all_datasets: bool,
) -> list[str]:
    stems = _checkpoint_store.discover(root)
    if all_datasets:
        return stems
    if catalog_profile is not None:
        return sorted(
            stem
            for stem, manifest in manifests.items()
            if manifest.catalog_profile == catalog_profile
        )
    wanted = set(materials)
    return sorted(
        stem
        for stem in stems
        if (stem in manifests and manifests[stem].material in wanted)
        or any(_legacy_material_match(stem, material) for material in wanted)
    )


def _dataset_artifacts(root: Path, stem: str) -> list[Path]:
    directory = root / stem
    artifacts = [
        directory / "line.pkl",
        directory / "brem.pkl",
        directory / "meta.json",
        directory / "cases.json",
        directory / "parts",
        root / f"{stem}.pkl",
        root / f"{stem}.meta.json",
        root / f"{stem}.cases.json",
    ]
    return [path for path in artifacts if path.exists()]


def _revalidate(manifests: list[_CaseManifest]) -> None:
    for manifest in manifests:
        current = _read_case_manifest(manifest.path)
        if current.digest != manifest.digest:
            raise _cli_core.CLIError(
                f"cannot safely clear: manifest changed during preview: {manifest.path}"
            )


def clear_checkpoints(
    *,
    materials: tuple[str, ...] = (),
    catalog_profile: str | None = None,
    all_datasets: bool = False,
    yes: bool = False,
    checkpoint_dir: str | os.PathLike[str] = "checkpoints",
) -> int:
    """Preview or delete selected active datasets and newly unreachable CAS blobs."""
    root = Path(checkpoint_dir)
    active_paths = _active_manifest_paths(root)
    active = {stem: _read_case_manifest(path) for stem, path in active_paths.items()}
    archived = [_read_case_manifest(path) for path in _archive_manifest_paths(root)]
    selected = _select_stems(
        root,
        active,
        materials=materials,
        catalog_profile=catalog_profile,
        all_datasets=all_datasets,
    )
    if not selected:
        _cli_core.emit_result("(nothing to clear)")
        return 0

    selected_set = set(selected)
    selected_refs = {
        (manifest.material, key)
        for stem, manifest in active.items()
        if stem in selected_set
        for key in manifest.keys
    }
    retained_refs = {
        (manifest.material, key)
        for stem, manifest in active.items()
        if stem not in selected_set
        for key in manifest.keys
    }
    retained_refs.update((manifest.material, key) for manifest in archived for key in manifest.keys)
    unreachable = sorted(selected_refs - retained_refs)
    blobs = [
        _checkpoint_store.cas_blob_path(material, key, root)
        for material, key in unreachable
        if _checkpoint_store.cas_blob_path(material, key, root).is_file()
    ]

    _cli_core.emit_result("would delete local checkpoint datasets:")
    for stem in selected:
        _cli_core.emit_result(f"  {root / stem}")
    _cli_core.emit_result(f"newly unreachable CAS blobs: {len(blobs)}")
    for blob in blobs:
        _cli_core.emit_result(f"  {blob}")
    if not _cli_core.confirm_destructive(
        yes,
        "Delete these checkpoint datasets and newly unreachable CAS blobs?",
    ):
        return 0

    _revalidate([*active.values(), *archived])
    for stem in selected:
        for artifact in _dataset_artifacts(root, stem):
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
        directory = root / stem
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass
    for blob in blobs:
        blob.unlink()
        try:
            blob.parent.rmdir()
        except OSError:
            pass
    _cli_core.emit_result(
        f"deleted {len(selected)} dataset(s) and {len(blobs)} unreachable CAS blob(s)"
    )
    return 0
