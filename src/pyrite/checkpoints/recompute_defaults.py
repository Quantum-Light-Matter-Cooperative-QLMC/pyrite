"""Profile-aware defaults shared by line and bremsstrahlung recomputes."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..campaign.profiles import FIDELITY_NAMES


@dataclass(frozen=True)
class DatasetContext:
    """Resolved identity needed by component recompute."""

    material: str
    fidelity: str
    catalog_profile: str
    identified: bool


def _identity_files(checkpoint_path) -> tuple[Path, Path]:
    path = Path(checkpoint_path)
    if path.suffix == ".pkl":
        return path.with_suffix(".meta.json"), path.with_suffix(".cases.json")
    return path / "meta.json", path / "cases.json"


def _json_object(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid checkpoint metadata {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"invalid checkpoint metadata {path}: expected an object")
    return value


def dataset_context(
    checkpoint_path,
    *,
    catalog_profile: str | None = None,
    fidelity: str | None = None,
    require_identity: bool = False,
) -> DatasetContext:
    """Resolve material/profile/fidelity from sidecars, with safe legacy fallback."""
    meta_path, cases_path = _identity_files(checkpoint_path)
    meta = _json_object(meta_path)
    cases = _json_object(cases_path)
    identity = meta.get("dataset_identity") if meta is not None else None
    if identity is not None and not isinstance(identity, dict):
        raise ValueError(
            f"invalid checkpoint metadata {meta_path}: dataset_identity is not an object"
        )
    if identity is not None:
        from ..campaign.profiles import normalize_dataset_identity

        assert meta is not None
        identity = normalize_dataset_identity(identity)
        recorded_version = meta.get("identity_version", identity["identity_version"])
        if recorded_version != identity["identity_version"]:
            raise ValueError(f"invalid checkpoint metadata {meta_path}: identity versions disagree")

    material = identity.get("material") if identity else None
    stored_fidelity = identity.get("fidelity") if identity else None
    stored_profile = identity.get("catalog_profile") if identity else None
    if cases is not None:
        if cases.get("schema") != "cxr.case-manifest.v1":
            raise ValueError(
                f"invalid checkpoint metadata {cases_path}: expected cxr.case-manifest.v1"
            )
        case_material = cases.get("material")
        case_profile = cases.get("catalog_profile")
        from ..campaign.profiles import normalize_dataset_identity

        case_version = normalize_dataset_identity(
            {"identity_version": cases.get("identity_version", 1)}
        )["identity_version"]
        if identity is not None and case_version != identity["identity_version"]:
            raise ValueError("checkpoint metadata disagrees on identity version")
        if material is not None and case_material != material:
            raise ValueError(
                f"checkpoint metadata disagrees on material: {material!r} != {case_material!r}"
            )
        material = material or case_material
        if stored_profile is not None and case_profile not in (None, stored_profile):
            raise ValueError(
                "checkpoint metadata disagrees on catalog profile: "
                f"{stored_profile!r} != {case_profile!r}"
            )
        stored_profile = stored_profile or case_profile

    if catalog_profile is not None and stored_profile not in (None, catalog_profile):
        raise ValueError(
            f"checkpoint belongs to profile {stored_profile!r}, not requested {catalog_profile!r}"
        )
    resolved_profile = catalog_profile or stored_profile
    identified = identity is not None or cases is not None
    if material is None:
        stem = (
            Path(checkpoint_path).stem
            if str(checkpoint_path).endswith(".pkl")
            else Path(checkpoint_path).name
        )
        from .._catalog_keys import material_keys

        if stem in material_keys():
            material = stem
            resolved_profile = resolved_profile or "standard"
        elif catalog_profile is not None or not require_identity:
            material = stem
            resolved_profile = resolved_profile or "standard"
        else:
            raise ValueError(
                f"checkpoint {stem!r} has no usable dataset identity; pass --profile NAME "
                "only with an unambiguous material checkpoint"
            )
    if resolved_profile is None:
        raise ValueError(
            f"checkpoint for {material!r} has no catalog-profile identity; pass --profile NAME"
        )
    resolved_fidelity = fidelity or stored_fidelity or "full"
    _validate_fidelity(resolved_fidelity)
    return DatasetContext(
        str(material),
        resolved_fidelity,
        str(resolved_profile),
        identified,
    )


def _validate_fidelity(fidelity: str) -> str:
    if fidelity not in FIDELITY_NAMES:
        raise ValueError(f"fidelity must be one of {', '.join(FIDELITY_NAMES)}")
    return fidelity


def settings(fidelity: str):
    """Return settings for ``fidelity``, including compatibility before presets land."""
    from ..campaign.config import default_settings

    fidelity = _validate_fidelity(fidelity)
    try:
        return cast(Any, default_settings)(fidelity=fidelity)
    except TypeError:
        current = default_settings()
        if fidelity == "full":
            return current
        return replace(current, n_electrons=60, n_electrons_brem=30)


def sweep(material: str, fidelity: str, *, catalog_profile: str = "standard"):
    """Return material sweep for ``fidelity``, with a standalone survey fallback."""
    from ..campaign.config import material_sweep

    fidelity = _validate_fidelity(fidelity)
    try:
        return cast(Any, material_sweep)(
            material,
            fidelity=fidelity,
            catalog_profile=catalog_profile,
        )
    except TypeError:
        current = material_sweep(material)
        if fidelity == "full":
            return current

        def reduced(values, limit):
            array = np.atleast_1d(np.asarray(values, dtype=float))
            if array.size <= limit:
                return array
            indices = np.linspace(0, array.size - 1, limit, dtype=int)
            return array[indices]

        bins = current.detector.energy_bins
        line_by_energy = bins.line_by_energy
        if line_by_energy is not None:
            line_by_energy = {
                energy: np.asarray(grid, dtype=float)[::5]
                for energy, grid in line_by_energy.items()
            }
        brem = bins.brem
        if brem is not None:
            brem = np.asarray(brem, dtype=float)[::5]
        return replace(
            current,
            energy_keV=reduced(current.energy_keV, 2),
            thickness_ang=reduced(current.thickness_ang, 3),
            tilt_deg=reduced(current.tilt_deg, 5),
            tilt_azim_deg=reduced(current.tilt_azim_deg, 2),
            detector=replace(
                current.detector,
                energy_bins=replace(bins, line_by_energy=line_by_energy, brem=brem),
            ),
            n_families=2,
        )


def uniform_bounds(grid) -> tuple[float, float, float]:
    """Return NumPy-arange ``start, stop, step`` for a regular profile grid."""
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("profile grid must contain at least two values")
    steps = np.diff(values)
    step = float(steps[0])
    if step <= 0 or not np.allclose(steps, step):
        raise ValueError("profile grid must be strictly increasing and uniform")
    return float(values[0]), float(values[-1] + step), step
