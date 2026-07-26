"""Optional third-party validation oracles for crystallography checks.

These helpers keep validation comparisons separate from production physics.
The implemented comparison target is ``Dans_Diffraction`` because it provides
an independent crystallography/scattering implementation with Waasmaier-Kirfel
X-ray factors. The cxr_mc side uses

    F_hkl = sum_j f_j(g, E) exp(+i 2 pi hkl.r_j) exp(-B (|g| / 4 pi)^2)

where ``|g| = 2 pi / d_hkl``. ``Dans_Diffraction`` documents the opposite phase
sign for structure factors, so this module compares lattice geometry, ``|g|``,
and ``|F_hkl|^2`` by default; complex phase comparisons should be a separate
convention audit.

Assumptions: full-occupancy structures, isotropic displacement only, and
non-magnetic X-ray scattering. Limiting case: a missing or optional oracle must
not affect runtime imports or production calculations.

Validation: dans-diffraction-oracle
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cxr_mc.materials.crystal import CRYSTALS, reciprocal_g_vector, structure_factor

_Lattice6 = tuple[float, float, float, float, float, float]


class DansDiffractionUnavailableError(ImportError):
    """Raised when an optional Dans_Diffraction validation check is requested."""


class _DansCellLike(Protocol):
    def lp(self) -> Sequence[float]: ...

    def volume(self) -> float: ...


class _DansScatterLike(Protocol):
    def setup_scatter(self, **kwargs: Any) -> Any: ...

    def structure_factor(self, hkl: Any = None, **kwargs: Any) -> Any: ...


class _DansCrystalLike(Protocol):
    Cell: _DansCellLike
    Scatter: _DansScatterLike

    def new_atoms(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class LatticeComparison:
    """Cell-parameter and volume deltas between cxr_mc and a validation oracle."""

    crystal: str
    cxr_lattice: _Lattice6
    oracle_lattice: _Lattice6
    lattice_delta: _Lattice6
    cxr_volume_ang3: float
    oracle_volume_ang3: float
    volume_delta_ang3: float


@dataclass(frozen=True)
class ReflectionGeometryComparison:
    """Reciprocal-vector magnitude comparison for one reflection."""

    crystal: str
    hkl: tuple[int, int, int]
    cxr_g_inv_ang: float
    oracle_q_inv_ang: float
    absolute_delta_inv_ang: float
    relative_delta: float


@dataclass(frozen=True)
class StructureFactorMagnitudeComparison:
    """Structure-factor magnitude comparison for one reflection."""

    crystal: str
    hkl: tuple[int, int, int]
    photon_E_eV: float
    cxr_abs_f_sq: float
    oracle_abs_f_sq: float
    absolute_delta: float
    relative_delta: float


@dataclass(frozen=True)
class DansOracleTolerances:
    """Numerical acceptance limits for the pinned ``Dans_Diffraction`` check.

    Geometry limits cover roundoff in two implementations of the same lattice
    algebra. The non-resonant limit compares the same Waasmaier--Kirfel tables.
    The dispersive limit is intentionally wider because cxr-mc uses
    Chantler/FFAST while ``Dans_Diffraction`` uses independent Henke/CXRO data.
    """

    lattice_length_abs_ang: float = 1e-10
    lattice_angle_abs_deg: float = 1e-10
    volume_abs_ang3: float = 1e-9
    geometry_relative: float = 1e-12
    nonresonant_structure_factor_relative: float = 1e-12
    dispersive_structure_factor_relative: float = 0.10


DEFAULT_DANS_TOLERANCES = DansOracleTolerances()


@dataclass(frozen=True)
class DansOracleValidationReport:
    """Thresholded result for one crystal and one scattering-factor policy."""

    crystal: str
    use_henke: bool
    lattice: LatticeComparison
    geometry: tuple[ReflectionGeometryComparison, ...]
    structure_factors: tuple[StructureFactorMagnitudeComparison, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether every comparison is finite and within its stated limit."""

        return not self.failures


def load_dans_crystal_from_cif(cif_path: str | Path) -> _DansCrystalLike:
    """Load a ``Dans_Diffraction.Crystal`` from a CIF path.

    ``Dans_Diffraction`` is intentionally imported only inside this function so
    the production package has no runtime dependency on the validation oracle.
    """

    return _dans_crystal_class()(str(cif_path))


def build_dans_crystal_from_cxr(crystal: str, *, uiso: float = 0.0) -> _DansCrystalLike:
    """Build a P1 ``Dans_Diffraction`` crystal from an internal ``CRYSTALS`` entry.

    This is for validation only. It transfers the already-expanded cxr_mc basis
    as full-occupancy P1 atoms, with ``uiso=0`` by default to neutralize the
    oracle's Debye-Waller term when comparing to ``B_ang2=0``.
    """

    xtl = _dans_crystal_class()()
    info = CRYSTALS[crystal]
    lattice = _cxr_lattice_tuple(crystal)
    _set_dans_cell(xtl, lattice)

    basis = info["basis"]
    positions = np.asarray([pos for _, pos in basis], dtype=float)
    elements = [str(element) for element, _ in basis]
    xtl.new_atoms(
        u=positions[:, 0].tolist(),
        v=positions[:, 1].tolist(),
        w=positions[:, 2].tolist(),
        type=elements,
        occupancy=[1.0] * len(elements),
        uiso=[float(uiso)] * len(elements),
    )
    return xtl


def compare_lattice(crystal: str, oracle_crystal: _DansCrystalLike) -> LatticeComparison:
    """Compare lattice lengths, angles, and unit-cell volume."""

    cxr_lattice = _cxr_lattice_tuple(crystal)
    oracle_lattice = _float_tuple(oracle_crystal.Cell.lp(), "oracle Cell.lp()", size=6)
    cxr_volume = float(CRYSTALS[crystal]["V_cell"])
    oracle_volume = float(oracle_crystal.Cell.volume())
    return LatticeComparison(
        crystal=crystal,
        cxr_lattice=cxr_lattice,
        oracle_lattice=oracle_lattice,
        lattice_delta=_float_tuple(
            [o - c for c, o in zip(cxr_lattice, oracle_lattice, strict=True)],
            "lattice delta",
            size=6,
        ),
        cxr_volume_ang3=cxr_volume,
        oracle_volume_ang3=oracle_volume,
        volume_delta_ang3=oracle_volume - cxr_volume,
    )


def compare_reflection_geometry(
    crystal: str, oracle_crystal: _DansCrystalLike, hkls: Sequence[Sequence[int]]
) -> list[ReflectionGeometryComparison]:
    """Compare ``|g| = 2*pi/d`` against ``Dans_Diffraction.Cell.Qmag``."""

    qmag = getattr(oracle_crystal.Cell, "Qmag", None)
    if not callable(qmag):
        raise AttributeError("Dans_Diffraction Cell.Qmag is required for geometry comparisons")

    hkl_array = _hkl_array(hkls)
    oracle_q = np.asarray(qmag(hkl_array), dtype=float).reshape(-1)
    if oracle_q.size != len(hkl_array):
        raise ValueError("oracle Cell.Qmag returned an unexpected number of values")

    results: list[ReflectionGeometryComparison] = []
    lattice = CRYSTALS[crystal]["lattice"]
    for hkl, q_oracle in zip(hkl_array, oracle_q, strict=True):
        _, g_cxr = reciprocal_g_vector(hkl, lattice)
        delta = float(q_oracle - g_cxr)
        results.append(
            ReflectionGeometryComparison(
                crystal=crystal,
                hkl=_hkl_tuple(hkl),
                cxr_g_inv_ang=float(g_cxr),
                oracle_q_inv_ang=float(q_oracle),
                absolute_delta_inv_ang=abs(delta),
                relative_delta=_relative_delta(float(g_cxr), float(q_oracle)),
            )
        )
    return results


def compare_structure_factor_magnitudes(
    crystal: str,
    oracle_crystal: _DansCrystalLike,
    hkls: Sequence[Sequence[int]],
    photon_E_eV: float,
    *,
    B_ang2: float = 0.0,
    use_henke: bool = False,
    scattering_type: str | None = None,
) -> list[StructureFactorMagnitudeComparison]:
    """Compare ``|F_hkl|^2`` against a ``Dans_Diffraction`` structure factor.

    ``scattering_type`` defaults to ``"xray dispersion"`` when ``use_henke`` is
    true, otherwise ``"xray"``. The default comparison avoids phase sign
    ambiguity and normalizes no scale factors; large relative deltas should be
    inspected before any physics claim is promoted in the validation ledger.
    """

    scatter = oracle_crystal.Scatter
    hkl_array = _hkl_array(hkls)
    scattering_type = scattering_type or ("xray dispersion" if use_henke else "xray")
    energy_kev = float(photon_E_eV) / 1000.0
    scatter.setup_scatter(
        scattering_type=scattering_type,
        energy_kev=energy_kev,
        int_hkl=True,
        use_waaskirf=True,
        output=False,
    )
    kwargs: dict[str, Any] = {"scattering_type": scattering_type, "int_hkl": True}
    if "dispersion" in scattering_type.lower():
        kwargs["energy_kev"] = energy_kev
    oracle_sf = np.asarray(scatter.structure_factor(hkl_array, **kwargs), dtype=complex).reshape(-1)
    if oracle_sf.size != len(hkl_array):
        raise ValueError("oracle structure_factor returned an unexpected number of values")

    results: list[StructureFactorMagnitudeComparison] = []
    for hkl, oracle_value in zip(hkl_array, oracle_sf, strict=True):
        cxr_value, _ = structure_factor(
            crystal,
            hkl,
            float(photon_E_eV),
            B_ang2=B_ang2,
            use_henke=use_henke,
        )
        cxr_abs_sq = float(abs(cxr_value) ** 2)
        oracle_abs_sq = float(abs(oracle_value) ** 2)
        results.append(
            StructureFactorMagnitudeComparison(
                crystal=crystal,
                hkl=_hkl_tuple(hkl),
                photon_E_eV=float(photon_E_eV),
                cxr_abs_f_sq=cxr_abs_sq,
                oracle_abs_f_sq=oracle_abs_sq,
                absolute_delta=abs(oracle_abs_sq - cxr_abs_sq),
                relative_delta=_relative_delta(cxr_abs_sq, oracle_abs_sq),
            )
        )
    return results


def validate_dans_crystal(
    crystal: str,
    hkls: Sequence[Sequence[int]],
    photon_energies_eV: Sequence[float],
    *,
    use_henke: bool,
    tolerances: DansOracleTolerances | None = None,
) -> DansOracleValidationReport:
    """Run thresholded lattice, reciprocal-geometry, and ``|F_hkl|²`` checks.

    ``use_henke=False`` compares non-resonant Waasmaier--Kirfel factors.
    ``use_henke=True`` compares cxr-mc's Chantler/FFAST corrections with
    ``Dans_Diffraction``'s independent Henke/CXRO corrections. Consequently the
    two modes use different structure-factor tolerances.

    Assumptions and units match this module's top-level validation contract.
    The missing-backend limit raises :class:`DansDiffractionUnavailableError`
    rather than silently passing.

    Validation: dans-diffraction-oracle
    """

    limits = tolerances or DEFAULT_DANS_TOLERANCES
    oracle = build_dans_crystal_from_cxr(crystal)
    lattice = compare_lattice(crystal, oracle)
    geometry = tuple(compare_reflection_geometry(crystal, oracle, hkls))
    structure_factors = tuple(
        comparison
        for photon_E_eV in photon_energies_eV
        for comparison in compare_structure_factor_magnitudes(
            crystal,
            oracle,
            hkls,
            photon_E_eV,
            use_henke=use_henke,
        )
    )

    failures: list[str] = []
    for name, delta, limit in (
        *(
            (
                f"lattice length {axis}",
                abs(lattice.lattice_delta[index]),
                limits.lattice_length_abs_ang,
            )
            for index, axis in enumerate(("a", "b", "c"))
        ),
        *(
            (
                f"lattice angle {axis}",
                abs(lattice.lattice_delta[index + 3]),
                limits.lattice_angle_abs_deg,
            )
            for index, axis in enumerate(("alpha", "beta", "gamma"))
        ),
        ("cell volume", abs(lattice.volume_delta_ang3), limits.volume_abs_ang3),
    ):
        _append_failure(failures, name, delta, limit)

    for comparison in geometry:
        _append_failure(
            failures,
            f"{crystal} {comparison.hkl} reciprocal geometry",
            comparison.relative_delta,
            limits.geometry_relative,
        )

    sf_limit = (
        limits.dispersive_structure_factor_relative
        if use_henke
        else limits.nonresonant_structure_factor_relative
    )
    for comparison in structure_factors:
        _append_failure(
            failures,
            f"{crystal} {comparison.hkl} |F|^2 at {comparison.photon_E_eV:g} eV",
            comparison.relative_delta,
            sf_limit,
        )

    return DansOracleValidationReport(
        crystal=crystal,
        use_henke=use_henke,
        lattice=lattice,
        geometry=geometry,
        structure_factors=structure_factors,
        failures=tuple(failures),
    )


def _dans_crystal_class() -> Any:
    try:
        module = importlib.import_module("Dans_Diffraction")
    except ModuleNotFoundError as exc:
        raise DansDiffractionUnavailableError(
            "Install the optional 'Dans-Diffraction' package to run this validation check."
        ) from exc
    return module.Crystal


def _set_dans_cell(xtl: Any, lattice: _Lattice6) -> None:
    setter = getattr(xtl, "new_cell", None) or getattr(xtl, "new_latt", None)
    if not callable(setter):
        raise AttributeError("Dans_Diffraction Crystal.new_cell/new_latt is required")
    try:
        setter(list(lattice))
    except TypeError:
        setter(*lattice)


def _cxr_lattice_tuple(crystal: str) -> _Lattice6:
    lattice = CRYSTALS[crystal]["lattice"]
    system = lattice["system"]
    if system == "cubic":
        a = float(lattice["a"])
        return (a, a, a, 90.0, 90.0, 90.0)
    if system == "tetragonal":
        a = float(lattice["a"])
        return (a, a, float(lattice["c"]), 90.0, 90.0, 90.0)
    if system == "orthorhombic":
        return (
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            90.0,
            90.0,
            90.0,
        )
    if system == "hexagonal":
        a = float(lattice["a"])
        return (a, a, float(lattice["c"]), 90.0, 90.0, 120.0)
    if system == "general":
        return (
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            float(lattice["alpha"]),
            float(lattice["beta"]),
            float(lattice["gamma"]),
        )
    raise ValueError(f"unknown crystal system '{system}'")


def _float_tuple(values: Sequence[float], label: str, *, size: int) -> _Lattice6:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != size:
        raise ValueError(f"{label} must contain {size} values, got {len(parsed)}")
    return (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], parsed[5])


def _hkl_array(hkls: Sequence[Sequence[int]]) -> np.ndarray:
    hkl_array = np.asarray(hkls, dtype=int)
    if hkl_array.ndim == 1:
        hkl_array = hkl_array.reshape(1, 3)
    if hkl_array.ndim != 2 or hkl_array.shape[1] != 3:
        raise ValueError("hkls must have shape (n, 3)")
    return hkl_array


def _hkl_tuple(hkl: np.ndarray) -> tuple[int, int, int]:
    return (int(hkl[0]), int(hkl[1]), int(hkl[2]))


def _relative_delta(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), np.finfo(float).tiny)
    return abs(a - b) / scale


def _append_failure(failures: list[str], name: str, value: float, limit: float) -> None:
    if not np.isfinite(value) or value > limit:
        failures.append(f"{name}: {value:.6e} > {limit:.6e}")
