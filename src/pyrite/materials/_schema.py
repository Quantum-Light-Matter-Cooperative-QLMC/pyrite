"""Immutable types for the material catalog schema."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

from .._numerics import CONVERGENCE_KEYS, TRANSPORT_KEYS
from .._precision import Precision
from ._catalog_decode import LineGridByEnergy, _number
from ._identity import CutFrame, MaterialIdentity, hexagonal_setting, reduce_indices

_SCAN_KEYS = (
    "thickness_ang",
    "thickness_layers",
    "energy_keV",
    "tilt_deg",
    "tilt_azim_deg",
    "E_grid_line",
    "E_grid_brem",
    "n_electrons",
    "n_electrons_brem",
)

#: Valid ``[profiles.NAME].emission`` values. A plain ``str`` type (not
#: ``results.store.EmissionMode``): importing ``results`` from ``materials``
#: would cycle back through ``results/tables.py``'s ``from ..materials import
#: CATALOG``.
_EMISSION_VALUES = ("incoherent", "coherent", "both")
_PROFILE_SCALAR_NUMERICS_KEYS = (*CONVERGENCE_KEYS, *TRANSPORT_KEYS)
_DEFAULT_PROFILE_DETECTOR_SPEC: Mapping[str, object] = MappingProxyType({})


def _negative(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    return cast(tuple[int, int, int], tuple(-value for value in hkl))


def _grouped_error_lines(errors: Sequence[str], *, max_paths: int = 3) -> list[str]:
    """Collapse identical-message errors across many material paths into one
    summary line each. A profile-wide setting invalid for every material
    otherwise repeats the same message once per material -- unreadable at
    catalog scale."""
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for entry in errors:
        path, _, message = entry.partition(": ")
        grouped.setdefault(message, []).append(path)
        if message not in order:
            order.append(message)
    lines = []
    for message in order:
        paths = grouped[message]
        shown = paths[:max_paths]
        remainder = len(paths) - max_paths
        truncated = remainder > 0
        if len(paths) == 1:
            line = f"{paths[0]}: {message}"
        else:
            shown_text = ", ".join(shown)
            suffix = f", +{remainder} more" if truncated else ""
            line = f"{len(paths)} paths ({shown_text}{suffix}): {message}"
        lines.append(line)
    return lines


class MaterialConfigError(ValueError):
    """Report one or more path-qualified material-catalog errors.

    Parameters
    ----------
    errors
        Individual validation messages, normally prefixed by catalog paths.
    profile
        Optional selected profile, retained for caller compatibility and
        reporting context.

    Attributes
    ----------
    errors
        Immutable tuple of the original ungrouped messages.
    """

    def __init__(self, errors: Sequence[str], *, profile: str | None = None):
        self.errors = tuple(errors)
        super().__init__(
            "invalid material catalog:\n"
            + "\n".join(f"- {e}" for e in _grouped_error_lines(self.errors))
        )


@dataclass(frozen=True)
class CrystalInfo:
    """Store CIF-derived crystallographic data in PyRITE units.

    Parameters
    ----------
    lattice
        Lattice-system metadata and cell lengths in angstroms/angles in degrees.
    basis
        Expanded ``(element, fractional_xyz)`` unit-cell sites.
    V_cell
        Unit-cell volume in cubic angstroms.
    composition
        ``(element, number_density)`` pairs in atoms per cubic angstrom.
    mosaic_fwhm_deg
        Optional configured c-axis mosaic FWHM in degrees.
    """

    lattice: Mapping[str, str | float]
    basis: tuple[tuple[str, np.ndarray], ...]
    V_cell: float
    composition: tuple[tuple[str, float], ...]
    mosaic_fwhm_deg: float | None


@dataclass(frozen=True)
class CrystalSpec:
    """Describe one configured crystal and its CIF-derived structure.

    Parameters
    ----------
    key
        Stable catalog key.
    cif
        Resolved path to the packaged phase-specific CIF.
    validation_id
        Validation-ledger identifier for the crystal structure.
    formula
        Chemical formula in ASCII, e.g. ``"MoS2"``.
    full_name, phase
        Optional display name and phase label.
    cod_id, mp_id
        Optional Crystallography Open Database and Materials Project identifiers.
    B_ang2
        Isotropic Debye--Waller ``B`` factor in square angstroms.
    beam_uvw
        Optional default crystal direction parallel to the incident beam.
    E_grid
        Optional legacy photon-energy grid in eV.
    hkl_families
        Pinned representative Miller-index families.
    hkl_reason
        Optional provenance for the pinned reflection selection.
    layers_per_cell
        Optional layer-count conversion for thickness grids.
    info
        Parsed crystallographic lattice, basis, composition, and volume.
    surface_hkl
        Optional Miller indices of the surface normal.
    """

    key: str
    cif: Path
    validation_id: str
    formula: str
    full_name: str | None
    phase: str | None
    cod_id: int | None
    mp_id: str | None
    B_ang2: float
    beam_uvw: tuple[int, int, int] | None
    E_grid: np.ndarray | None
    hkl_families: tuple[tuple[int, int, int], ...]
    hkl_reason: str | None
    layers_per_cell: int | None
    info: CrystalInfo
    surface_hkl: tuple[int, int, int] | None

    @property
    def hkl_list(self) -> tuple[tuple[int, int, int], ...]:
        """Pinned representatives expanded to both reciprocal directions."""
        return tuple(hkl for family in self.hkl_families for hkl in (family, _negative(family)))

    @property
    def cut_frame(self) -> CutFrame | None:
        """Which space the declared slab normal lives in, or ``None`` if unset.

        ``beam_uvw`` and ``surface_hkl`` are mutually exclusive spellings of the
        same axis, so at most one of them decides this.
        """
        if self.surface_hkl is not None:
            return "plane"
        if self.beam_uvw is not None:
            return "direction"
        return None

    @property
    def cut(self) -> tuple[int, int, int] | None:
        """Declared slab normal reduced to its primitive representative.

        This is the crystal cut, and it is deliberately unrelated to
        :attr:`hkl_families`, which pins diffracting reflections.
        """
        indices = self.surface_hkl if self.surface_hkl is not None else self.beam_uvw
        return reduce_indices(indices) if indices is not None else None

    @property
    def hexagonal(self) -> bool:
        """Whether the CIF-derived cell is given on hexagonal axes.

        Notation only -- see :func:`._identity.hexagonal_setting`. No geometry
        path consults this; the reciprocal normal is already exact in every
        setting.
        """
        return hexagonal_setting(self.lattice)

    @property
    def lattice(self) -> Mapping[str, str | float]:
        """CIF-derived lattice parameters."""
        return self.info.lattice

    @property
    def basis(self) -> tuple[tuple[str, np.ndarray], ...]:
        """CIF-derived expanded fractional basis."""
        return self.info.basis

    @property
    def V_cell(self) -> float:
        """Unit-cell volume in Angstrom cubed."""
        return self.info.V_cell

    @property
    def composition(self) -> tuple[tuple[str, float], ...]:
        """Element number densities in atoms per cubic Angstrom."""
        return self.info.composition

    @property
    def mosaic_fwhm_deg(self) -> float | None:
        """Configured c-axis mosaic FWHM, if present."""
        return self.info.mosaic_fwhm_deg


@dataclass(frozen=True)
class MediumSpec:
    """Describe an amorphous medium by elemental number density.

    Parameters
    ----------
    key
        Stable catalog key.
    composition
        ``(element, number_density)`` pairs in atoms per cubic angstrom.
    """

    key: str
    composition: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ScanSpec:
    """Store resolved, read-only one-dimensional scan grids.

    Parameters
    ----------
    thickness_ang
        Film-thickness grid in angstroms.
    energy_keV
        Electron kinetic-energy grid in keV.
    tilt_deg, tilt_azim_deg
        Target polar and azimuthal tilt grids in degrees.
    E_grid_line
        Shared line photon-energy grid in eV, or ``None`` when energy-specific
        grids are used.
    E_grid_line_by_energy
        Mapping from electron energy in keV to this material's stored line grids
        in eV. Missing energies, including an empty mapping, use automatic
        case-local policy. ``None`` accompanies an explicit fixed line grid.
    E_grid_brem
        Bremsstrahlung photon-energy grid in eV.
    thickness_layers
        Optional film-thickness grid in crystal layers.
    n_electrons, n_electrons_brem
        Optional macro-electron count grids overriding fidelity defaults.
    """

    thickness_ang: np.ndarray
    energy_keV: np.ndarray
    tilt_deg: np.ndarray
    tilt_azim_deg: np.ndarray
    E_grid_line: np.ndarray | None
    E_grid_line_by_energy: LineGridByEnergy | None
    E_grid_brem: np.ndarray
    thickness_layers: np.ndarray | None = None
    #: Optional electron-count grids (profile settings): None -> the runner's
    #: settings-level counts (fidelity policy) apply. Single values typical;
    #: multiple values sweep transport statistics (see ``Sweep.n_electrons``).
    n_electrons: np.ndarray | None = None
    n_electrons_brem: np.ndarray | None = None


@dataclass(frozen=True)
class LayerSpec:
    """Describe one fixed substrate-side layer.

    Parameters
    ----------
    material
        Catalog crystal or medium key.
    thickness_ang
        Positive layer thickness in angstroms.
    beam_uvw
        Optional crystal direction aligned with the incident beam.
    azimuth_deg
        In-plane crystal rotation in degrees relative to the film.
    """

    material: str
    thickness_ang: float
    beam_uvw: tuple[int, int, int] | None = None
    azimuth_deg: float = 0.0


@dataclass(frozen=True)
class MaterialValidationSpec:
    """Store independent validation state attached to one material.

    Parameters
    ----------
    crystal_database_match
        ``"verified"``, ``"unverified"``, or ``None`` when not applicable.
    """

    crystal_database_match: Literal["verified", "unverified"] | None = None


@dataclass(frozen=True)
class MaterialSpec:
    """Describe one runnable catalog scan target.

    Parameters
    ----------
    key
        Stable machine key.
    identity
        Structured display identity — formula, phase, cut — from which
        :attr:`label` is derived rather than authored.
    profile
        Profile from which this resolved material inherited scan defaults.
    crystal_key
        Entrance-film crystal key.
    scan
        Resolved scan and photon-energy grids.
    substrate
        Optional legacy substrate material key.
    stack
        Film-excluded substrate-side layers in beam-entrance order.
    validation
        Independent material-validation state.
    """

    key: str
    identity: MaterialIdentity
    #: Name of the ``[profiles.*]`` campaign row this material resolved its
    #: scan defaults from (plus that profile's per-material override, if any).
    profile: str
    crystal_key: str
    scan: ScanSpec
    substrate: str | None = None
    stack: tuple[LayerSpec, ...] = ()
    validation: MaterialValidationSpec = MaterialValidationSpec()

    @property
    def crystal(self) -> str:
        """Compatibility spelling for the film crystal key."""
        return self.crystal_key

    @property
    def label(self) -> str:
        """Display label derived from :attr:`identity`."""
        return self.identity.label

    @property
    def formula(self) -> str:
        """ASCII chemical formula of the entrance-film crystal."""
        return self.identity.formula

    @property
    def phase(self) -> str | None:
        """Polytype or structural phase, or ``None`` when the crystal declares none."""
        return self.identity.phase

    @property
    def full_name(self) -> str | None:
        """English name of the entrance-film crystal."""
        return self.identity.full_name

    @property
    def cut(self) -> tuple[int, int, int] | None:
        """Reduced slab-normal indices, or ``None`` when no orientation is declared."""
        return self.identity.cut

    @property
    def cut_frame(self) -> CutFrame | None:
        """Which space :attr:`cut` lives in; ``None`` exactly when :attr:`cut` is."""
        return self.identity.cut_frame

    @property
    def hexagonal(self) -> bool:
        """Whether the entrance-film cell is given on hexagonal axes."""
        return self.identity.hexagonal


@dataclass(frozen=True)
class MaterialCatalog:
    """Expose deeply immutable material and profile registries.

    Parameters
    ----------
    schema_version
        Parsed catalog schema version.
    crystals, media, materials
        Keyed immutable registries of crystal, amorphous medium, and runnable
        material specifications.
    material_keys
        Runnable material keys in source declaration order.
    profile_names, profile_memberships
        Profile names and optional explicit material membership lists.
    profile_beams, profile_detectors, profile_emissions
        Resolved per-profile beam, scalar-detector, and emission overrides.
    profile_temporal_profiles
        Profiles that opt in to the line temporal intensity profile ``I(t)``.
    profile_precisions
        Per-profile adaptive electron-count policies.
    profile_transport_numerics
        Per-profile result-affecting scalar numerical controls.
    profile_line_grid_policies
        Per-profile named line-grid bandwidth and resolution policies.
    profile_filters, profile_physical_detectors
        Declarative finite-filter and planar-detector profile data.
    profile_energy_grid_refs, resolved_energy_grid_refs
        Explicit artifact references and verified references used by this load.
    beams, beam_keys
        Named beam definitions and their source declaration order.
    detectors, detector_labels, detector_keys
        Named scalar detectors, display labels, and source declaration order.

    Notes
    -----
    Instances are produced by :func:`load_material_catalog`; callers should not
    mutate nested arrays or mappings.
    """

    schema_version: int
    crystals: Mapping[str, CrystalSpec]
    media: Mapping[str, MediumSpec]
    materials: Mapping[str, MaterialSpec]
    material_keys: tuple[str, ...]
    #: Every ``[profiles.*]`` name defined by the source TOML.
    profile_names: tuple[str, ...] = ()
    #: Explicit ``profiles.NAME.materials`` membership lists, keyed by profile;
    #: profiles without a membership row are absent (all materials allowed).
    profile_memberships: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    #: Decoded ``[profiles.NAME.beam]`` distribution blocks (transverse size,
    #: bunch length/shape/offsets, rep-rate, charge, reserved divergence/spread),
    #: keyed by profile; profiles with no beam block are absent. Energy is NOT
    #: here -- it stays the per-material ``ScanSpec.energy_keV`` scan grid.
    profile_beams: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Explicit profile detector blocks, as validated acceptance fields rather
    #: than built detectors -- ``campaign.config`` applies them onto its default
    #: response-free geometry. Missing selected-profile blocks inherit
    #: ``standard``; missing standard falls back to
    #: :data:`_DEFAULT_PROFILE_DETECTOR_SPEC`, which
    #: leaves the response-free 90-degree scalar geometry in place.
    profile_detectors: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Explicit ``profiles.NAME.emission`` overrides ("incoherent"/"coherent"/
    #: "both"), keyed by profile; profiles with no emission key are absent (the
    #: active fidelity preset's emission stands unmodified).
    profile_emissions: Mapping[str, str] = MappingProxyType({})
    #: ``profiles.NAME.temporal_profile = true`` opt-ins (#292); absent is off.
    profile_temporal_profiles: Mapping[str, bool] = MappingProxyType({})
    #: Validated ``[profiles.NAME.precision]`` adaptive-sampling policies, as
    #: plain field mappings; absent means fixed electron counts.
    profile_precisions: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Explicit scalar result-affecting numerics, keyed by profile. Electron
    #: count grids remain on each resolved :class:`ScanSpec`.
    profile_transport_numerics: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Explicit ``profiles.NAME.line_grid_policy`` tables (named bandwidth and
    #: resolution policies), keyed by profile; absent means the automatic policy.
    profile_line_grid_policies: Mapping[str, Mapping[str, str]] = MappingProxyType({})
    #: Declarative finite filter plates, resolved by the single-scene CLI path.
    #: They intentionally remain plain schema data here: importing instrument
    #: objects would invert the materials -> instrument dependency boundary.
    profile_filters: Mapping[str, tuple[Mapping[str, object], ...]] = MappingProxyType({})
    #: Declarative planar-pixel detector geometry for ``material simulate``.
    profile_physical_detectors: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Named per-profile detector collection rows. Each mapping value combines
    #: optional scalar acceptance and optional pixelated geometry settings.
    profile_detector_sets: Mapping[str, Mapping[str, Mapping[str, object]]] = MappingProxyType({})
    #: Explicit immutable energy-grid artifact refs, keyed first by profile and
    #: then material. Legacy ``[energy_grids.*]`` fallback rows are deliberately
    #: absent: callers can distinguish migrated refs from compatibility input.
    profile_energy_grid_refs: Mapping[str, Mapping[str, str]] = MappingProxyType({})
    #: Verified refs used to resolve this catalog instance's selected profile.
    resolved_energy_grid_refs: Mapping[str, str] = MappingProxyType({})
    #: Named ``[beams.*]`` catalog objects (distribution fields plus an optional
    #: ``label``), keyed by beam name. A profile attaches one by name
    #: (``profiles.NAME.beam = "beam-key"``); ``profile_beams`` below holds the
    #: already-resolved, name-stripped payload every consumer actually reads.
    beams: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Every ``[beams.*]`` name defined by the source TOML.
    beam_keys: tuple[str, ...] = ()
    #: Named ``[detectors.*]`` acceptance specs, keyed by detector name. Labels
    #: are stored separately so display metadata cannot enter simulation state.
    detectors: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    detector_labels: Mapping[str, str] = MappingProxyType({})
    #: Every ``[detectors.*]`` name defined by the source TOML.
    detector_keys: tuple[str, ...] = ()

    def profile_beam(self, name: str) -> Mapping[str, object] | None:
        """Decoded ``[profiles.NAME.beam]`` distribution overrides, or ``None``
        when the profile carries no beam block (the ``standard`` beam default
        applies). Consumed by :func:`config.material_sweep` via ``beam_replace``."""
        return self.profile_beams.get(name)

    def profile_emission(self, name: str) -> str | None:
        """Explicit ``profiles.NAME.emission`` override, or ``None`` when the
        profile carries no emission key. Consumed by :func:`scan._resolved_run`
        to override the active fidelity preset's emission."""
        return self.profile_emissions.get(name)

    def profile_temporal_profile(self, name: str) -> bool:
        """Whether ``profiles.NAME`` opts in to the line ``I(t)`` profile (#292)."""
        return bool(self.profile_temporal_profiles.get(name, False))

    def profile_precision(self, name: str) -> Precision | None:
        """Adaptive electron-count policy for ``name``, or ``None`` for fixed counts."""
        payload = self.profile_precisions.get(name)
        return None if payload is None else Precision.from_dict(payload)

    def profile_numerics(self, name: str) -> Mapping[str, object] | None:
        """Explicit result-affecting transport numerics for ``name``."""
        return self.profile_transport_numerics.get(name)

    def profile_line_grid_policy(self, name: str) -> Mapping[str, str] | None:
        """Named line-grid policies for ``name``, or ``None`` for the automatic one.
        Consumed by :func:`config.material_sweep` as ``Sweep.line_grid_policy``."""
        return self.profile_line_grid_policies.get(name)

    def profile_detector(self, name: str) -> Mapping[str, object]:
        """Acceptance fields belonging to ``name``, or code defaults.

        An empty mapping means "every detector default stands". Build the
        detector with :func:`config.catalog_detector`.
        """
        return self.profile_detectors.get(name, _DEFAULT_PROFILE_DETECTOR_SPEC)

    def profile_detector_set(self, name: str) -> Mapping[str, Mapping[str, object]]:
        """Return named detectors, including legacy and implicit defaults."""
        if name not in self.profile_names:
            raise KeyError(f"unknown profile {name!r}; have {list(self.profile_names)}")
        declared = self.profile_detector_sets.get(name)
        if declared is not None:
            return declared
        legacy: dict[str, Mapping[str, object]] = {}
        if name in self.profile_physical_detectors:
            legacy["physical"] = self.profile_physical_detectors[name]
        if name in self.profile_detectors:
            legacy["default"] = self.profile_detectors[name]
        if not legacy:
            legacy["default"] = MappingProxyType({})
        return MappingProxyType(legacy)

    def profile_materials(self, name: str) -> tuple[str, ...] | None:
        """Explicit ``profiles.NAME.materials`` membership, or ``None`` when the
        profile has no membership row (every catalog material is allowed)."""
        if name not in self.profile_names:
            raise KeyError(f"unknown profile {name!r}; have {list(self.profile_names)}")
        return self.profile_memberships.get(name)

    def profile_energy_grid_ref(self, name: str, material: str) -> str | None:
        """Return explicit immutable grid ref for ``name``/``material``.

        ``None`` means the profile still resolves through the read-only legacy
        catalog compatibility path; reads never materialize or repoint refs.
        """
        if name not in self.profile_names:
            raise KeyError(f"unknown profile {name!r}; have {list(self.profile_names)}")
        return self.profile_energy_grid_refs.get(name, {}).get(material)

    def crystal(self, key: str) -> CrystalSpec:
        """Return a crystal by key."""
        try:
            return self.crystals[key]
        except KeyError:
            raise KeyError(f"unknown crystal {key!r}; have {list(self.crystals)}") from None

    def material(self, key: str) -> MaterialSpec:
        """Return a runnable material by key."""
        try:
            return self.materials[key]
        except KeyError:
            raise KeyError(f"unknown material {key!r}; have {list(self.materials)}") from None

    def resolve_stack(
        self, key: str, film_thickness_ang: float
    ) -> tuple[Mapping[str, object], ...]:
        """Resolve a film and its explicit stack into ordered physical layers.

        Boundaries are cumulative from the beam-entrance face. Composition is
        resolved from the catalog's CIF-derived crystal density or named
        medium number density. A crystalline layer inherits its configured
        direct-axis or reciprocal-surface orientation unless the layer supplies
        a direct-axis override; amorphous media have no default orientation.
        """
        thickness = _number(film_thickness_ang)
        if thickness is None or thickness <= 0:
            raise ValueError("film_thickness_ang must be finite and positive")
        material = self.material(key)
        film = self.crystal(material.crystal_key)
        physical: list[Mapping[str, object]] = []
        z_top = 0.0

        def append_layer(
            layer_key: str,
            layer_thickness: float,
            composition: tuple[tuple[str, float], ...],
            beam_uvw: tuple[int, int, int] | None,
            surface_hkl: tuple[int, int, int] | None,
            azimuth_deg: float,
        ) -> None:
            nonlocal z_top
            z_bottom = z_top + layer_thickness
            physical.append(
                MappingProxyType(
                    {
                        "key": layer_key,
                        "z_top_ang": z_top,
                        "z_bottom_ang": z_bottom,
                        "composition": composition,
                        "beam_uvw": beam_uvw,
                        "surface_hkl": surface_hkl,
                        "azimuth_deg": azimuth_deg,
                    }
                )
            )
            z_top = z_bottom

        append_layer(
            film.key,
            thickness,
            film.composition,
            film.beam_uvw,
            film.surface_hkl,
            0.0,
        )
        for layer in material.stack:
            crystal = self.crystals.get(layer.material)
            if crystal is not None:
                composition = crystal.composition
                if layer.beam_uvw is None:
                    beam_uvw = crystal.beam_uvw
                    surface_hkl = crystal.surface_hkl
                else:
                    beam_uvw = layer.beam_uvw
                    surface_hkl = None
            else:
                composition = self.media[layer.material].composition
                beam_uvw = layer.beam_uvw
                surface_hkl = None
            append_layer(
                layer.material,
                layer.thickness_ang,
                composition,
                beam_uvw,
                surface_hkl,
                layer.azimuth_deg,
            )
        return tuple(physical)


# Preserve the public pickle identity used before the schema split. These
# classes remain compatibility exports of the public catalog module.
for _catalog_type in (
    MaterialConfigError,
    CrystalInfo,
    CrystalSpec,
    MediumSpec,
    ScanSpec,
    LayerSpec,
    MaterialValidationSpec,
    MaterialSpec,
    MaterialCatalog,
):
    _catalog_type.__module__ = "pyrite.materials.catalog"
