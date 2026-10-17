"""Temperature-dependent thermal displacement for channeling profiles.

The continuum potential of a crystal plane is the static lattice potential
convolved with each site's thermal displacement distribution. Channeling
needs that displacement at the requested crystal temperature, per site, so
this module interpolates source-backed phonon-DOS rows per site and
extrapolates them with the Debye model ``B(T)`` (Willis & Pryor 1975; Peng et
al. 1996, Eq. 8). A single-parameter Debye fit is a poor global model for Si
(about 11% low at 180 K when fitted at 280 K), so it is used only outside the
tabulated range.

Data live in the packaged ``channeling_thermal.toml``, keyed by catalog crystal
and element. Crystals without an entry are ineligible for channeling profiles
(ADR-0018); the catalog scalar ``B_ang2`` used by PXR/CBS is not consulted.

Layered hexagonal crystals carry :class:`UniaxialThermalSite` rows instead: a
single-temperature measurement of the mean-square displacement in the basal
plane and along ``c``. They have no isotropic ``B`` and no temperature model;
the static potential uses their plane-normal projection and thermal diffuse
scattering their full covariance (#535).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import constants
from scipy.optimize import brentq

from ... import DATA_DIR

THERMAL_DATA_PATH = DATA_DIR / "channeling_thermal.toml"

# 6 h^2 / (m_u k_B) in angstrom^2 kelvin; the prefactor of the Debye B(T).
_DEBYE_PREFACTOR_ANG2_K = (
    6.0
    * constants.h**2
    / (constants.physical_constants["atomic mass constant"][0] * constants.k)
    * 1e20
)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


class ChannelingDataError(ValueError):
    """Raised when a crystal lacks the data a channeling profile requires."""


@dataclass(frozen=True)
class ThermalSite:
    """Source-backed thermal displacement data for one element in one crystal.

    Attributes
    ----------
    crystal, element
        Catalog crystal key and element symbol.
    atomic_weight
        Atomic weight on the 12C = 12 scale.
    temperatures_K, B_ang2
        Increasing temperatures [K] and phonon-DOS Debye--Waller
        ``B = 8 pi^2 <u_x^2>`` [angstrom^2] tabulated at them.
    source, doi
        Provenance of the tabulated values.
    """

    crystal: str
    element: str
    atomic_weight: float
    temperatures_K: tuple[float, ...]
    B_ang2: tuple[float, ...]
    source: str
    doi: str

    def __post_init__(self) -> None:
        t = np.asarray(self.temperatures_K, dtype=float)
        b = np.asarray(self.B_ang2, dtype=float)
        if t.ndim != 1 or t.shape != b.shape or t.size < 2:
            raise ChannelingDataError(
                f"{self.crystal}.{self.element}: need matching 1-D temperature/B rows"
            )
        if np.any(np.diff(t) <= 0.0) or np.any(t <= 0.0) or np.any(b <= 0.0):
            raise ChannelingDataError(
                f"{self.crystal}.{self.element}: temperatures must increase and B be positive"
            )

    def b_factor(self, temperature_K: float) -> float:
        """Return ``B(T)`` [angstrom^2].

        Inside the tabulated range the phonon-DOS rows are interpolated
        linearly. Outside it, the Debye model is anchored to the nearest
        tabulated end: its Debye temperature reproduces that end's ``B``.

        Validation: cr-debye-b-of-t
        """
        t = float(temperature_K)
        if t <= 0.0:
            raise ValueError("temperature_K must be positive")
        lo, hi = self.temperatures_K[0], self.temperatures_K[-1]
        if lo <= t <= hi:
            return float(np.interp(t, self.temperatures_K, self.B_ang2))
        anchor_t, anchor_b = (lo, self.B_ang2[0]) if t < lo else (hi, self.B_ang2[-1])
        theta = fit_debye_temperature(anchor_b, anchor_t, self.atomic_weight)
        return debye_b_factor(t, self.atomic_weight, theta)


@dataclass(frozen=True)
class UniaxialThermalSite:
    """Single-temperature uniaxial thermal displacement of one element.

    The Cartesian covariance is ``diag(u_basal, u_basal, u_c)`` with ``c``
    along the hexagonal axis, as site symmetry requires for an element whose
    sites are all symmetry-equivalent on a 3-, 4- or 6-fold axis. The row
    holds at ``temperature_K +- tolerance_K`` only; it is never extrapolated
    (ADR-0018: no Debye curve from a single refinement).

    Attributes
    ----------
    crystal, element
        Catalog crystal key and element symbol.
    atomic_weight
        Atomic weight on the 12C = 12 scale.
    temperature_K, tolerance_K
        Measurement temperature [K] and the admissible half-window [K].
    u_basal_ang2, u_c_ang2
        Mean-square displacement ``<u^2>`` along one basal axis and along
        ``c`` [angstrom^2].
    source, doi
        Provenance of the values.
    """

    crystal: str
    element: str
    atomic_weight: float
    temperature_K: float
    tolerance_K: float
    u_basal_ang2: float
    u_c_ang2: float
    source: str
    doi: str

    def __post_init__(self) -> None:
        values = (self.temperature_K, self.u_basal_ang2, self.u_c_ang2)
        if min(values) <= 0.0 or self.tolerance_K < 0.0:
            raise ChannelingDataError(
                f"{self.crystal}.{self.element}: temperature and displacements must be "
                "positive and tolerance non-negative"
            )

    def _check_temperature(self, temperature_K: float) -> None:
        t = float(temperature_K)
        if abs(t - self.temperature_K) > self.tolerance_K:
            raise ChannelingDataError(
                f"{self.crystal}.{self.element}: thermal row holds only at "
                f"{self.temperature_K:g} +- {self.tolerance_K:g} K, not {t:g} K"
            )

    def b_factor_along(self, cos2_c: float, temperature_K: float) -> float:
        """Return ``B = 8 pi^2 n.U.n`` [angstrom^2] along a unit direction ``n``.

        ``cos2_c`` is the squared cosine between ``n`` and ``c``, so
        ``n.U.n = u_basal (1 - cos2_c) + u_c cos2_c``.

        Validation: cr-uniaxial-thermal-projection
        """
        self._check_temperature(temperature_K)
        if not 0.0 <= cos2_c <= 1.0 + 1e-12:
            raise ValueError("cos2_c must lie in [0, 1]")
        c2 = min(float(cos2_c), 1.0)
        return 8.0 * np.pi**2 * (self.u_basal_ang2 * (1.0 - c2) + self.u_c_ang2 * c2)

    def covariance(self, c_axis, temperature_K: float) -> np.ndarray:
        """Return the Cartesian covariance ``U = u_b I + (u_c - u_b) c c^T`` [angstrom^2].

        ``c_axis`` is the hexagonal axis in the Cartesian frame of the result;
        it is normalized here.

        Validation: cr-uniaxial-thermal-projection
        """
        self._check_temperature(temperature_K)
        c = np.asarray(c_axis, dtype=float)
        c = c / np.linalg.norm(c)
        return self.u_basal_ang2 * np.eye(3) + (self.u_c_ang2 - self.u_basal_ang2) * np.outer(c, c)


def debye_integral(y):
    """Return the Debye integral ``phi(y) = (1/y) * int_0^y x / (exp(x) - 1) dx``.

    Parameters
    ----------
    y
        Scalar or array ``Theta_D / T``, positive and dimensionless.

    Returns
    -------
    numpy.ndarray or float
        ``phi(y)``, shaped like ``y``. ``phi -> 1`` as ``y -> 0`` and
        ``phi -> pi^2 / (6 y)`` as ``y -> inf``.

    Validation: cr-debye-b-of-t
    """
    y_arr = np.asarray(y, dtype=float)
    if np.any(y_arr <= 0.0):
        raise ValueError("Debye integral requires y > 0")
    half = 0.5 * y_arr[..., None]
    x = half * (_GL_NODES + 1.0)
    # Gauss--Legendre nodes are interior, so x > 0; overflow -> integrand 0.
    with np.errstate(over="ignore"):
        integrand = x / np.expm1(x)
    result = (integrand * _GL_WEIGHTS).sum(axis=-1) * half[..., 0] / y_arr
    return result if np.ndim(y) else float(result)


def debye_b_factor(temperature_K, atomic_weight, debye_temperature_K):
    """Return the Debye-model isotropic Debye--Waller ``B(T)``.

    Implements ``B = (6 h^2 / (m_u A k_B)) (T / Theta^2) [phi(Theta/T) +
    Theta / (4 T)]`` (Willis & Pryor 1975; Peng et al. 1996, Eq. 8), with
    ``B = 8 pi^2 <u_x^2>``. The bracket's second term is the zero-point motion.

    Parameters
    ----------
    temperature_K
        Scalar or array crystal temperature [K], positive.
    atomic_weight
        Atomic weight ``A`` on the 12C = 12 scale.
    debye_temperature_K
        Debye temperature ``Theta`` [K].

    Returns
    -------
    numpy.ndarray or float
        ``B`` in square angstroms, shaped like ``temperature_K``.

    Validation: cr-debye-b-of-t
    """
    t = np.asarray(temperature_K, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("temperature_K must be positive")
    y = debye_temperature_K / t
    b = (
        _DEBYE_PREFACTOR_ANG2_K
        / atomic_weight
        * t
        / debye_temperature_K**2
        * (debye_integral(y) + 0.25 * y)
    )
    return b if np.ndim(temperature_K) else float(b)


def fit_debye_temperature(B_ang2, temperature_K, atomic_weight):
    """Return the Debye temperature that reproduces ``B_ang2`` at ``temperature_K``.

    ``debye_b_factor`` decreases monotonically with ``Theta`` at fixed ``T``,
    so the root is unique.

    Validation: cr-debye-b-of-t
    """
    if B_ang2 <= 0.0:
        raise ValueError("B_ang2 must be positive")
    return float(
        brentq(
            lambda theta: debye_b_factor(temperature_K, atomic_weight, theta) - B_ang2,
            1.0,
            1.0e5,
            xtol=1e-10,
        )
    )


AnyThermalSite = ThermalSite | UniaxialThermalSite


@lru_cache(maxsize=4)
def _load(path: Path) -> dict[tuple[str, str], AnyThermalSite]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    sites: dict[tuple[str, str], AnyThermalSite] = {}
    for crystal, elements in raw.items():
        for element, row in elements.items():
            if row.get("model") == "uniaxial_c":
                sites[crystal, element] = UniaxialThermalSite(
                    crystal=crystal,
                    element=element,
                    atomic_weight=float(row["atomic_weight"]),
                    temperature_K=float(row["temperature_K"]),
                    tolerance_K=float(row["tolerance_K"]),
                    u_basal_ang2=float(row["u_basal_ang2"]),
                    u_c_ang2=float(row["u_c_ang2"]),
                    source=str(row["source"]),
                    doi=str(row["doi"]),
                )
                continue
            sites[crystal, element] = ThermalSite(
                crystal=crystal,
                element=element,
                atomic_weight=float(row["atomic_weight"]),
                temperatures_K=tuple(float(t) for t in row["temperatures_K"]),
                B_ang2=tuple(float(b) for b in row["B_ang2"]),
                source=str(row["source"]),
                doi=str(row["doi"]),
            )
    return sites


def load_thermal_sites(path: Path | None = None) -> dict[tuple[str, str], AnyThermalSite]:
    """Return packaged thermal sites keyed by ``(crystal, element)``."""
    return dict(_load(Path(path) if path is not None else THERMAL_DATA_PATH))


def thermal_site(crystal: str, element: str, path: Path | None = None) -> AnyThermalSite:
    """Return one crystal/element thermal site or raise ``ChannelingDataError``."""
    sites = load_thermal_sites(path)
    try:
        return sites[crystal, element]
    except KeyError:
        eligible = sorted({key for key, _ in sites})
        raise ChannelingDataError(
            f"crystal {crystal!r} has no channeling thermal data for {element!r}; "
            f"eligible crystals: {', '.join(eligible)}"
        ) from None


def thermal_b_factor(crystal: str, element: str, temperature_K: float) -> float:
    """Return the Debye-model ``B(T)`` [angstrom^2] for one site of a crystal.

    Raises ``ChannelingDataError`` for a uniaxial row, which has no isotropic
    ``B``.

    Validation: cr-debye-b-of-t
    """
    site = thermal_site(crystal, element)
    if isinstance(site, UniaxialThermalSite):
        raise ChannelingDataError(
            f"{crystal}.{element}: anisotropic thermal row has no isotropic B; "
            "only the planar-potential projection supports it (#535)"
        )
    return site.b_factor(temperature_K)
