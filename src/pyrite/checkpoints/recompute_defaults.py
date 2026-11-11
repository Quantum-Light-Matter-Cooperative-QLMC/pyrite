"""Profile-aware defaults shared by line and bremsstrahlung recomputes."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..campaign.profiles import RETIRED_SURVEY_TIER, legacy_tier


@dataclass(frozen=True)
class DatasetContext:
    """Resolved identity needed by component recompute."""

    material: str
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
    require_identity: bool = False,
) -> DatasetContext:
    """Resolve material/profile from sidecars, with safe legacy fallback.

    A checkpoint whose identity records the retired ``survey`` tier is refused:
    its reduced grid and history counts no longer exist, so a profile-aware
    recompute would silently retarget it onto the full grid.
    """
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
    if identity and legacy_tier(identity) == RETIRED_SURVEY_TIER:
        raise ValueError(
            f"checkpoint {Path(checkpoint_path).name!r} was produced by the retired survey "
            "tier; it still loads, lists, pulls and slims, but cannot be recomputed. "
            "Re-run the material under its profile instead"
        )
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
    return DatasetContext(
        str(material),
        str(resolved_profile),
        identified,
    )


def settings():
    """Return the built-in run settings profile-aware recomputes start from."""
    from ..campaign.config import default_settings

    return default_settings()


def sweep(material: str, *, catalog_profile: str = "standard"):
    """Return the current catalog sweep for ``material`` under ``catalog_profile``."""
    from ..campaign.config import material_sweep

    return material_sweep(material, catalog_profile=catalog_profile)


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
