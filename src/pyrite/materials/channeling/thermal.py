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


@lru_cache(maxsize=4)
def _load(path: Path) -> dict[tuple[str, str], ThermalSite]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    sites = {}
    for crystal, elements in raw.items():
        for element, row in elements.items():
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


def load_thermal_sites(path: Path | None = None) -> dict[tuple[str, str], ThermalSite]:
    """Return packaged thermal sites keyed by ``(crystal, element)``."""
    return dict(_load(Path(path) if path is not None else THERMAL_DATA_PATH))


def thermal_site(crystal: str, element: str, path: Path | None = None) -> ThermalSite:
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

    Validation: cr-debye-b-of-t
    """
    return thermal_site(crystal, element).b_factor(temperature_K)
