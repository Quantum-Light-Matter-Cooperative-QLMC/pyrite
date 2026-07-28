"""Remove checkpoint records obsolete under current catalog scan profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from . import _checkpoint_store
from .cli import _completion as _cli_completion
from .materials import CATALOG
from .profiles import _jsonable, named_profile_identity, named_profile_stem

_DEFAULT_CHECKPOINT_DIR = str(Path(__file__).resolve().parents[2] / "checkpoints")
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
        raise click.UsageError("prune --all cannot be combined with --profile")
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
    from .config import default_settings, material_sweep
    from .sweep import build_cases

    settings = default_settings(target.fidelity)
    sweep = material_sweep(
        target.material,
        fidelity=target.fidelity,
        catalog_profile=target.catalog_profile,
    )
    return {
        (case["name"], float(case["E0_keV"])): _case_key(case)
        for case in build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
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
    from .run import _checkpoint_save, _manifest_save

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
    if not yes:
        click.echo("would prune (re-run with --yes to delete):")
        for target, kept, stale, before in changes:
            after = sum(len(by_energy) for by_energy in kept.values())
            click.echo(
                f"  {root_label}/{target.stem}/: {stale} stale record(s) ({before} -> {after})"
            )
        return sum(change[2] for change in changes)

    for target, kept, stale, before in changes:
        _write_target(target, root, kept)
        after = before - stale
        click.echo(
            f"pruned {root_label}/{target.stem}/: removed {stale} stale record(s) "
            f"({before} -> {after})"
        )
    return sum(change[2] for change in changes)


@click.command(
    help=(
        "Drop records obsolete under current scan profiles; preview unless --yes. "
        "With neither selector, prune profile=standard."
    )
)
@click.option(
    "--all",
    "all_profiles",
    is_flag=True,
    help="Prune current checkpoints for standard and every named catalog profile.",
)
@click.option(
    "--profile",
    "catalog_profile",
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Prune current full and survey checkpoints for catalog profile NAME.",
)
@click.option("--yes", is_flag=True, help="Delete exact previewed stale records.")
def command(all_profiles: bool, catalog_profile: str | None, yes: bool) -> None:
    if all_profiles and catalog_profile is not None:
        raise click.UsageError("prune --all cannot be combined with --profile")
    prune_checkpoints(
        all_profiles=all_profiles,
        catalog_profile=catalog_profile,
        yes=yes,
    )


def main(argv=None):
    from .cli._core import run

    return run(command, argv, prog_name="cxr prune")


if __name__ == "__main__":
    raise SystemExit(main())
