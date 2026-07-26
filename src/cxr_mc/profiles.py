"""Named sweep profiles and reproducible dataset identities.

Profiles are policies applied after catalog material lookup.  ``full`` is an
exact compatibility policy: it leaves catalog grids unchanged and retains the
production electron counts.  ``survey`` is provisional and intentionally
smaller.  Every resolved run can be serialized into a deterministic identity,
so differently resolved variants never silently resume into one dataset.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .results import Settings
from .sweep import Sweep, crystal_params

PROFILE_NAMES = ("full", "survey")
DATASET_IDENTITY_SCHEMA = "cxr.dataset-identity.v1"


def _centered_sample(values: Any, limit: int | None) -> Any:
    """Keep at most ``limit`` values from a centered, narrower input span."""
    if limit is None:
        return values
    array = np.atleast_1d(np.asarray(values))
    if array.size <= limit:
        return values
    indices = np.linspace(0.2 * (array.size - 1), 0.8 * (array.size - 1), limit)
    indices = np.unique(np.rint(indices).astype(int))
    return array[indices]


def _coarsen_grid(values: Any, factor: int, span_fraction: float) -> np.ndarray:
    """Crop a uniform/nonuniform grid around its center, then decimate it."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.size <= 2:
        return array.copy()
    keep = max(2, int(np.ceil(array.size * span_fraction)))
    start = (array.size - keep) // 2
    cropped = array[start : start + keep]
    coarse = cropped[::factor]
    if coarse[-1] != cropped[-1]:
        coarse = np.append(coarse, cropped[-1])
    return coarse


@dataclass(frozen=True)
class SweepProfile:
    """Independent settings and grid-reduction policy for one named profile."""

    name: str
    n_electrons: int
    n_electrons_brem: int
    max_energies: int | None = None
    max_thicknesses: int | None = None
    max_tilts: int | None = None
    max_azimuths: int | None = None
    n_families: int | None = None
    max_reflections: int | None = None
    photon_grid_stride: int = 1
    photon_grid_span_fraction: float = 1.0
    provisional: bool = False

    def apply_settings(self, settings: Settings) -> Settings:
        """Return settings with this profile's transport counts resolved."""
        return replace(
            settings,
            n_electrons=self.n_electrons,
            n_electrons_brem=self.n_electrons_brem,
        )

    def apply_sweep(self, sweep: Sweep) -> Sweep:
        """Return catalog sweep reduced according to this profile."""
        if self.name == "full":
            return sweep
        energies = _centered_sample(sweep.energy_keV, self.max_energies)
        energy_values = {float(value) for value in np.atleast_1d(energies)}
        by_energy = sweep.E_grid_line_by_energy
        if by_energy is not None:
            by_energy = {
                float(energy): _coarsen_grid(
                    grid, self.photon_grid_stride, self.photon_grid_span_fraction
                )
                for energy, grid in by_energy.items()
                if float(energy) in energy_values
            }
        line_grid = (
            None
            if sweep.E_grid_line is None
            else _coarsen_grid(
                sweep.E_grid_line,
                self.photon_grid_stride,
                self.photon_grid_span_fraction,
            )
        )
        brem_grid = (
            None
            if sweep.E_grid_brem is None
            else _coarsen_grid(
                sweep.E_grid_brem,
                self.photon_grid_stride,
                self.photon_grid_span_fraction,
            )
        )
        return replace(
            sweep,
            energy_keV=energies,
            thickness_ang=_centered_sample(sweep.thickness_ang, self.max_thicknesses),
            tilt_deg=_centered_sample(sweep.tilt_deg, self.max_tilts),
            tilt_azim_deg=_centered_sample(sweep.tilt_azim_deg, self.max_azimuths),
            n_families=self.n_families or sweep.n_families,
            max_reflections=self.max_reflections,
            E_grid_line=line_grid,
            E_grid_line_by_energy=by_energy,
            E_grid_brem=brem_grid,
        )


_PROFILES = {
    "full": SweepProfile("full", n_electrons=300, n_electrons_brem=150),
    "survey": SweepProfile(
        "survey",
        n_electrons=60,
        n_electrons_brem=30,
        max_energies=2,
        max_thicknesses=3,
        max_tilts=5,
        max_azimuths=2,
        n_families=2,
        max_reflections=4,
        photon_grid_stride=4,
        photon_grid_span_fraction=0.7,
        provisional=True,
    ),
}


def get_profile(name: str = "full") -> SweepProfile:
    """Return named profile or raise a boundary-friendly ``ValueError``."""
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown sweep profile {name!r} (choose from {PROFILE_NAMES})") from None


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def dataset_identity(
    material: str,
    profile: str,
    settings: Settings,
    sweep: Sweep,
    *,
    variant: str | None = None,
) -> dict[str, Any]:
    """Return profile plus exact resolved parameters and stable SHA-256 digest."""
    crystallography = crystal_params(sweep.material, sweep.n_families)
    reflections = crystallography["hkl_list"]
    if sweep.max_reflections is not None:
        reflections = reflections[: sweep.max_reflections]
    resolved = {
        "material": material,
        "profile": profile,
        "variant": variant,
        "settings": _jsonable(settings),
        "sweep": _jsonable(sweep),
        "crystallography": {
            "crystal": crystallography["crystal"],
            "hkl_list": _jsonable(reflections),
            "beam_uvw": _jsonable(
                sweep.beam_uvw if sweep.beam_uvw is not None else crystallography["beam_uvw"]
            ),
            "surface_hkl": _jsonable(
                None if sweep.beam_uvw is not None else crystallography["surface_hkl"]
            ),
        },
    }
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": DATASET_IDENTITY_SCHEMA,
        "material": material,
        "profile": profile,
        "variant": variant,
        "parameter_sha256": hashlib.sha256(encoded).hexdigest(),
        "resolved_parameters": resolved,
    }


def variant_stem(identity: Mapping[str, Any], *, canonical_full: bool = False) -> str:
    """Map identity to active checkpoint stem.

    Canonical ``full`` retains historical ``<material>`` storage. Other
    resolved variants use a readable profile plus digest suffix.
    """
    material = str(identity["material"])
    if canonical_full and identity["profile"] == "full" and identity.get("variant") is None:
        return material
    variant = identity.get("variant")
    if variant == "quick":
        return f"{material}_quick"
    label = str(variant or identity["profile"])
    return f"{material}--{label}-{str(identity['parameter_sha256'])[:12]}"


def named_profile_identity(material: str, profile: str = "full") -> dict[str, Any]:
    """Resolve identity for an unmodified named material/profile pair."""
    # Local import avoids config -> profiles -> config import cycle.
    from .config import default_settings, material_sweep

    return dataset_identity(
        material,
        profile,
        default_settings(profile),
        material_sweep(material, profile=profile),
    )


def named_profile_stem(material: str, profile: str = "full") -> str:
    """Checkpoint stem for an unmodified named material/profile pair."""
    return variant_stem(
        named_profile_identity(material, profile),
        canonical_full=profile == "full",
    )


def high_energy_floor_identity(
    material: str, floor_kev: float, profile: str = "full"
) -> dict[str, Any]:
    """Resolve identity for a material/profile pair with its energy_keV grid
    floored at ``floor_kev`` (mats_to_sim.toml's ``high_energy_materials``
    convention -- see ``scan._resolved_run``). Mirrors ``named_profile_identity``
    but for the filtered grid, so remote job orchestration (``_remote/scripts.py``
    ``_stems``) can predict the same non-canonical stem the runner will write."""
    # Local import avoids config -> profiles -> config import cycle.
    from .config import default_settings, material_sweep

    sweep = material_sweep(material, profile=profile)
    energies = np.asarray(sweep.energy_keV, dtype=float)
    kept = energies[energies >= floor_kev]
    if kept.size == 0:
        raise SystemExit(
            f"{material}: no energies >= {floor_kev} keV in its grid (high-energy "
            "floor); lower --high-energy-min-kev or drop this material"
        )
    sweep = replace(sweep, energy_keV=kept)
    return dataset_identity(material, profile, default_settings(profile), sweep)


def high_energy_floor_stem(material: str, floor_kev: float, profile: str = "full") -> str:
    """Checkpoint stem for a high-energy-floored material/profile pair."""
    return variant_stem(
        high_energy_floor_identity(material, floor_kev, profile),
        canonical_full=False,
    )


_VARIANT_STEM_RE = re.compile(
    r"^(?P<material>.+)--(?P<profile>full|survey)-(?P<digest>[0-9a-f]{12})$"
)


def identity_from_stem(stem: str) -> dict[str, Any] | None:
    """Reconstruct unmodified named-profile identity from a checkpoint stem."""
    match = _VARIANT_STEM_RE.fullmatch(stem)
    if match is not None:
        identity = named_profile_identity(match["material"], match["profile"])
        if identity["parameter_sha256"].startswith(match["digest"]):
            return identity
        return None
    try:
        identity = named_profile_identity(stem, "full")
    except ValueError:
        return None
    return identity if named_profile_stem(stem, "full") == stem else None
