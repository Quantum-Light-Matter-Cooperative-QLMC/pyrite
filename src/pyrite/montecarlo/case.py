"""Typed, validated input record for one Monte Carlo simulation case."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal, cast

import numpy as np

type Composition = list[tuple[str, float]]
type MillerIndex = tuple[int, int, int]
type EnergyGrid = tuple[float, float, float] | np.ndarray
type AbsorberLayer = tuple[float, float, Composition]
type Radiator = dict[str, object]


class _Absent:
    """Sentinel for legacy keys whose absence is identity-significant."""

    __slots__ = ()

    def __reduce__(self):
        return (_restore_absent, ())


_ABSENT = _Absent()


def _restore_absent() -> _Absent:
    """Return the process-wide absence sentinel when unpickling a Case."""
    return _ABSENT


_CASE_KEY_ORDER = (
    "name",
    "crystal",
    "composition",
    "hkl_list",
    "B_ang2",
    "E0_keV",
    "thickness_ang",
    "crystal_width_mm",
    "crystal_height_mm",
    "beam_fwhm_mm",
    "bunch_charge_pc",
    "rep_rate_hz",
    "beam_fwhm_y_mm",
    "energy_spread_frac",
    "long_shape",
    "bunch_length_fs",
    "long_offsets_fs",
    "longitudinal_distribution",
    "transverse_distribution",
    "E_grid",
    "E_grid_line",
    "E_grid_brem",
    "line_grid_policy",
    "line_quadrature",
    "theta_obs_rad",
    "tilt_deg",
    "tilt_azim_deg",
    "groove_spacing_ang",
    "coherent_emission",
    "straggling",
    "energy_model",
    "max_dE_frac",
    "inelastic_model",
    "inelastic_cutoff_eV",
    "elastic_model",
    "beam_uvw",
    "surface_hkl",
    "mosaic_fwhm_rad",
    "mosaic_mc_fwhm_rad",
    "mosaic_mc_nodes",
    "abs_layers",
    "layer_radiators",
    "brem_file",
    "Ne",
    "Ne_brem",
    "seed",
    "spec_chunk",
    "brem_chunk",
    "dtheta_obs_rad",
    "domega_sr",
    # Accepted legacy/manual-only runner controls. ``build_cases`` never emits
    # them, so they serialize after the producer-owned schema when present.
    "azimuth_rad",
    "recip_miscut_rad",
    "E_cut_lines_keV",
    "E_cut_brem_keV",
    "sinc_cutoff",
    "brem_step_eV",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Case(Mapping[str, Any]):
    """Frozen input schema for one transport and spectrum calculation.

    Length fields use angstrom unless suffixed ``_mm``; electron and cutoff
    energies use keV, photon grids use eV, angles use their suffix (rad/deg),
    compositions use atoms/angstrom^3, and detector acceptance uses steradian.
    Optional divergence-only fields retain an internal absence sentinel so
    :meth:`to_dict` exactly reproduces the historical mapping payload.

    Parameters
    ----------
    name, crystal
        Human-readable case name and catalog crystal key.
    composition
        ``(element, number_density)`` pairs in atoms per cubic angstrom.
    hkl_list, B_ang2
        Reflections and isotropic Debye--Waller ``B`` in square angstroms.
    E0_keV, thickness_ang
        Incident energy in keV and film thickness in angstroms.
    crystal_width_mm, crystal_height_mm
        Paired full transverse dimensions in mm, or both ``None``.
    beam_fwhm_mm, beam_fwhm_y_mm
        Entrance-beam Gaussian FWHM values in mm.
    E_grid, E_grid_line, E_grid_brem
        Compatibility, line, and continuum photon-energy grids in eV.
    theta_obs_rad, dtheta_obs_rad, domega_sr
        Observation direction, full polar acceptance, and solid angle.
    tilt_deg, tilt_azim_deg, azimuth_rad, recip_miscut_rad
        Target and crystal-orientation controls in suffix-named units.
    beam_uvw, surface_hkl
        Optional direct-axis or reciprocal-surface orientation.
    mosaic_fwhm_rad, mosaic_mc_fwhm_rad, mosaic_mc_nodes
        Analytic and quadrature mosaic controls.
    abs_layers, layer_radiators
        Optional film-first absorber and coherent-radiator payloads.
    brem_file
        Optional external bremsstrahlung spectrum path.
    Ne, Ne_brem, seed
        Line/background macro-electron counts and random seed.
    spec_chunk, brem_chunk
        Optional spectrum-kernel chunk sizes.
    bunch_charge_pc, rep_rate_hz
        Optional source-normalization values.
    energy_spread_frac, long_shape, bunch_length_fs, long_offsets_fs
        Legacy beam energy and longitudinal-distribution fields.
    longitudinal_distribution, transverse_distribution
        Resolved declarative phase-space policies.
    groove_spacing_ang
        Optional blazed-groove period in angstroms.
    line_quadrature
        Optional ``"bin-mean"`` closed-form bin integration of the incoherent
        ``sinc^2`` lines; absent samples nodes.
    coherent_emission, straggling, energy_model, max_dE_frac
        Result-affecting opt-in transport and radiation policies.
    inelastic_model, inelastic_cutoff_eV
        Opt-in ``"shell-soft-hard"`` collision-loss scheme and its cutoff in
        eV; both absent is continuous stopping. Requires ``energy_model``.
    elastic_model
        Opt-in ``"elsepa"`` tabulated elastic scattering; absent is the
        historical ``"mott"`` model.
    E_cut_lines_keV, E_cut_brem_keV, sinc_cutoff, brem_step_eV
        Legacy/manual cutoff, truncation, and grid controls.
    """

    # Producer-required fields. Their declaration order need not match the
    # legacy mapping: to_dict follows _CASE_KEY_ORDER explicitly.
    name: str
    crystal: str
    composition: Composition
    hkl_list: list[MillerIndex]
    B_ang2: float
    E0_keV: float
    thickness_ang: float
    crystal_width_mm: float | None
    crystal_height_mm: float | None
    beam_fwhm_mm: float | None
    E_grid: EnergyGrid
    E_grid_line: EnergyGrid
    E_grid_brem: EnergyGrid
    theta_obs_rad: float
    tilt_deg: float
    tilt_azim_deg: float
    beam_uvw: MillerIndex | None
    surface_hkl: MillerIndex | None
    mosaic_fwhm_rad: float | None
    mosaic_mc_fwhm_rad: float | None
    mosaic_mc_nodes: int
    abs_layers: list[AbsorberLayer] | None
    layer_radiators: list[Radiator | None] | None
    brem_file: str | PathLike[str] | None
    Ne: int
    Ne_brem: int
    seed: int
    spec_chunk: int | None
    brem_chunk: int | None
    dtheta_obs_rad: float
    domega_sr: float

    # Producer-conditional fields. `_Absent` is internal and skipped by
    # serialization; explicit None remains a real legacy payload value.
    bunch_charge_pc: float | _Absent = _ABSENT
    rep_rate_hz: float | _Absent = _ABSENT
    beam_fwhm_y_mm: float | None | _Absent = _ABSENT
    energy_spread_frac: float | _Absent = _ABSENT
    long_shape: str | _Absent = _ABSENT
    bunch_length_fs: float | _Absent = _ABSENT
    long_offsets_fs: tuple[float, ...] | _Absent = _ABSENT
    longitudinal_distribution: dict[str, object] | _Absent = _ABSENT
    transverse_distribution: dict[str, object] | _Absent = _ABSENT
    # Automatic case-local line-grid policy (issue #101). Absent means the
    # coordinates in ``E_grid_line`` are final -- an explicit user grid or a
    # stored catalog row. Present means ``E_grid_line`` is the coarsest
    # admissible grid and the runner refines it under this policy from the
    # case's own trajectories; the policy, not the refined coordinates, is what
    # identity hashes, because the refinement is a deterministic function of it.
    line_grid_policy: dict[str, object] | _Absent = _ABSENT
    # Line quadrature (issue #116), divergence-only: absent is node sampling.
    # Its own key rather than only a policy field, so a recompute that pins
    # explicit coordinates and drops the policy keeps the quadrature it used.
    line_quadrature: Literal["bin-mean"] | _Absent = _ABSENT
    groove_spacing_ang: float | _Absent = _ABSENT
    coherent_emission: Literal[True] | _Absent = _ABSENT
    straggling: Literal[True] | _Absent = _ABSENT
    energy_model: Literal["midpoint"] | _Absent = _ABSENT
    max_dE_frac: float | _Absent = _ABSENT
    inelastic_model: Literal["shell-soft-hard"] | _Absent = _ABSENT
    inelastic_cutoff_eV: float | _Absent = _ABSENT
    elastic_model: Literal["elsepa"] | _Absent = _ABSENT

    # Legacy/manual-only controls accepted during the Mapping support window.
    azimuth_rad: float | _Absent = _ABSENT
    recip_miscut_rad: tuple[float, float] | None | _Absent = _ABSENT
    E_cut_lines_keV: float | _Absent = _ABSENT
    E_cut_brem_keV: float | _Absent = _ABSENT
    sinc_cutoff: float | None | _Absent = _ABSENT
    brem_step_eV: float | _Absent = _ABSENT

    def __post_init__(self) -> None:
        """Reject malformed values before they reach transport or identity."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string")
        if not self.crystal or not isinstance(self.crystal, str):
            raise ValueError("crystal must be a non-empty string")
        _positive("B_ang2", self.B_ang2, allow_zero=True)
        _positive("E0_keV", self.E0_keV)
        _positive("thickness_ang", self.thickness_ang)
        _paired_positive_optional_dimensions(
            self.crystal_width_mm,
            self.crystal_height_mm,
        )
        _positive_optional("beam_fwhm_mm", self.beam_fwhm_mm)
        if not isinstance(self.beam_fwhm_y_mm, _Absent):
            _positive_optional("beam_fwhm_y_mm", self.beam_fwhm_y_mm)
        for name in ("E_grid", "E_grid_line", "E_grid_brem"):
            _energy_grid(name, getattr(self, name))
        for name in ("theta_obs_rad", "tilt_deg", "tilt_azim_deg", "dtheta_obs_rad"):
            _finite(name, getattr(self, name))
        _positive("domega_sr", self.domega_sr)
        _positive_optional("mosaic_fwhm_rad", self.mosaic_fwhm_rad, allow_zero=True)
        _positive_optional("mosaic_mc_fwhm_rad", self.mosaic_mc_fwhm_rad, allow_zero=True)
        _positive_int("mosaic_mc_nodes", self.mosaic_mc_nodes)
        _positive_int("Ne", self.Ne)
        _positive_int("Ne_brem", self.Ne_brem)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("spec_chunk", "brem_chunk"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(name, value)
        if self.coherent_emission is not _ABSENT and self.coherent_emission is not True:
            raise ValueError("coherent_emission must be absent or True")
        if self.straggling is not _ABSENT and self.straggling is not True:
            raise ValueError("straggling must be absent or True")
        if self.energy_model is not _ABSENT and self.energy_model != "midpoint":
            raise ValueError("energy_model must be absent or 'midpoint'")
        if self.max_dE_frac is not _ABSENT:
            _positive("max_dE_frac", self.max_dE_frac)
            if self.energy_model != "midpoint":
                raise ValueError("max_dE_frac requires energy_model='midpoint'")
        if (self.inelastic_model is _ABSENT) != (self.inelastic_cutoff_eV is _ABSENT):
            raise ValueError("inelastic_model and inelastic_cutoff_eV are set together")
        if self.inelastic_model is not _ABSENT:
            if self.inelastic_model != "shell-soft-hard":
                raise ValueError("inelastic_model must be absent or 'shell-soft-hard'")
            _positive("inelastic_cutoff_eV", self.inelastic_cutoff_eV)
            if self.energy_model != "midpoint":
                raise ValueError("inelastic_model requires energy_model='midpoint'")
        if self.elastic_model is not _ABSENT and self.elastic_model != "elsepa":
            raise ValueError("elastic_model must be absent or 'elsepa'")
        if self.line_quadrature is not _ABSENT:
            if self.line_quadrature != "bin-mean":
                raise ValueError("line_quadrature must be absent or 'bin-mean'")
            # Validation: sinc-bin-integration
            if self.coherent_emission is not _ABSENT:
                raise ValueError(
                    "line_quadrature='bin-mean' is incoherent-only; coherent_emission "
                    "squares a sum of amplitudes (#117)"
                )
            if self.max_dE_frac is not _ABSENT:
                raise ValueError(
                    "line_quadrature='bin-mean' is incompatible with max_dE_frac: "
                    "numerical substeps add amplitudes before squaring"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact legacy mapping shape and insertion order.

        Returns
        -------
        dict
            Present fields in canonical order. Internal absence sentinels are
            omitted while explicit ``None`` values are retained.
        """
        payload: dict[str, Any] = {}
        for key in _CASE_KEY_ORDER:
            value = getattr(self, key)
            if value is not _ABSENT:
                payload[key] = value
        return payload

    def __getitem__(self, key: str) -> Any:
        value = getattr(self, key, _ABSENT)
        if value is _ABSENT or key not in _CASE_KEY_ORDER:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return (key for key in _CASE_KEY_ORDER if getattr(self, key) is not _ABSENT)

    def __len__(self) -> int:
        return sum(getattr(self, key) is not _ABSENT for key in _CASE_KEY_ORDER)


def _finite(name: str, value: object) -> None:
    try:
        finite = math.isfinite(float(cast(Any, value)))
    except TypeError, ValueError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


def _positive(name: str, value: object, *, allow_zero: bool = False) -> None:
    _finite(name, value)
    numeric = float(cast(Any, value))
    if numeric < 0.0 if allow_zero else numeric <= 0.0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _positive_optional(
    name: str,
    value: object | None,
    *,
    allow_zero: bool = False,
) -> None:
    if value is not None:
        _positive(name, value, allow_zero=allow_zero)


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _paired_positive_optional_dimensions(width: float | None, height: float | None) -> None:
    if (width is None) != (height is None):
        raise ValueError("crystal_width_mm and crystal_height_mm must be supplied together")
    _positive_optional("crystal_width_mm", width)
    _positive_optional("crystal_height_mm", height)


def _energy_grid(name: str, value: object) -> None:
    grid = np.asarray(value, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional energy grid")
    if isinstance(value, tuple) and len(value) != 3:
        raise ValueError(f"{name} uniform encoding must be a three-item tuple")
