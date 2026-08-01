"""Reachability-safe local checkpoint dataset deletion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import click

from . import _checkpoint_store
from .cli import _completion as _cli_completion
from .cli import _core as _cli_core


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
        raise _cli_core.CLIError(
            f"cannot safely clear: invalid manifest {path}: missing material"
        )
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
    retained_refs.update(
        (manifest.material, key) for manifest in archived for key in manifest.keys
    )
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
    if not yes:
        _cli_core.emit_result("preview only; re-run with --yes to delete")
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


@click.command("clear")
@click.argument(
    "materials",
    nargs=-1,
    shell_complete=_cli_completion.complete_material,
)
@click.option(
    "--profile",
    "catalog_profile",
    shell_complete=_cli_completion.complete_profile,
    metavar="NAME",
    help="Delete active datasets owned by catalog profile NAME.",
)
@click.option("--all", "all_datasets", is_flag=True, help="Delete every active dataset.")
@click.option("--yes", is_flag=True, help="Delete exact previewed targets.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    show_default=True,
    metavar="DIR",
    help="Checkpoint root containing active datasets, archives, and shared CAS blobs.",
)
def command(materials, catalog_profile, all_datasets, yes, checkpoint_dir):
    """Delete local checkpoint datasets; preview unless --yes.

    Archived and retained active manifests remain CAS reachability roots.
    """
    selectors = int(bool(materials)) + int(catalog_profile is not None) + int(all_datasets)
    if selectors != 1:
        raise click.UsageError("clear needs exactly one of MATERIAL..., --profile NAME, or --all")
    return clear_checkpoints(
        materials=tuple(materials),
        catalog_profile=catalog_profile,
        all_datasets=all_datasets,
        yes=yes,
        checkpoint_dir=checkpoint_dir,
    )
