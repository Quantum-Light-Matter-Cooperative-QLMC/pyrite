"""Declarative transverse phase-space policies and their per-case resolution.

Policies are physical inputs independent of the simulated macro-particle count.
Resolved records hold only deterministic, case-specific second moments; the
sampled per-electron coordinates remain runtime data.

The stored input is the *normalized* emittance, because ``energy_keV`` is the
primary swept axis and geometric emittance is not invariant across it. See
``docs/physics/beam-transport/beam-phase-space.md``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import physical_constants

_ELECTRON_REST_KEV = physical_constants["electron mass energy equivalent in MeV"][0] * 1.0e3

# Input units are the accelerator-conventional mm*mrad and metres; the
# diagnostics in beam_metrics.py work in mm*rad and mm/rad.
_MM_MRAD_TO_MM_RAD = 1.0e-3
_M_TO_MM_PER_RAD = 1.0e3


@dataclass(frozen=True)
class TransverseDistribution:
    """Serializable transverse phase-space policy, Courant-Snyder per plane.

    ``normalized_emittance_x_mm_mrad`` and ``beta_twiss_x_m`` are required;
    ``alpha_twiss_x`` defaults to the waist. The ``_y`` fields default to
    ``None``, which mirrors the x-plane and gives a round beam -- the same
    shorthand :meth:`BeamSpec.isotropic` provides for the legacy spot spelling.

    ``alpha_twiss`` is signed: negative describes a diverging beam past its
    waist, so these keys must never join the positive-only profile key set.
    """

    normalized_emittance_x_mm_mrad: float
    beta_twiss_x_m: float
    alpha_twiss_x: float = 0.0
    normalized_emittance_y_mm_mrad: float | None = None
    beta_twiss_y_m: float | None = None
    alpha_twiss_y: float | None = None

    def __post_init__(self) -> None:
        for name in ("normalized_emittance_x_mm_mrad", "beta_twiss_x_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.alpha_twiss_x):
            raise ValueError("alpha_twiss_x must be finite")
        for name in ("normalized_emittance_y_mm_mrad", "beta_twiss_y_m"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when given")
        if self.alpha_twiss_y is not None and not np.isfinite(self.alpha_twiss_y):
            raise ValueError("alpha_twiss_y must be finite when given")

    @property
    def plane_y(self) -> tuple[float, float, float]:
        """The y-plane triplet, mirroring x wherever a ``_y`` field is unset."""
        return (
            self.normalized_emittance_x_mm_mrad
            if self.normalized_emittance_y_mm_mrad is None
            else self.normalized_emittance_y_mm_mrad,
            self.beta_twiss_x_m if self.beta_twiss_y_m is None else self.beta_twiss_y_m,
            self.alpha_twiss_x if self.alpha_twiss_y is None else self.alpha_twiss_y,
        )


@dataclass(frozen=True)
class ResolvedTransversePlane:
    """One plane's second moments at a single case energy.

    Units match ``beam_metrics.PlaneMetrics`` so a resolved policy and a
    measured sample are directly comparable: mm for positions, dimensionless
    radians for slopes, mm*rad for emittance, mm/rad for beta.
    """

    geometric_emittance_mm_rad: float
    beta_twiss_mm_per_rad: float
    alpha_twiss: float
    sigma_position_mm: float
    sigma_slope_rad: float


@dataclass(frozen=True)
class ResolvedTransverseDistribution:
    """Deterministic transverse policy resolved for one simulation case."""

    beta_gamma: float
    x: ResolvedTransversePlane
    y: ResolvedTransversePlane
    provenance: str


def _resolve_plane(
    normalized_emittance_mm_mrad: float,
    beta_twiss_m: float,
    alpha_twiss: float,
    beta_gamma: float,
) -> ResolvedTransversePlane:
    geometric = normalized_emittance_mm_mrad * _MM_MRAD_TO_MM_RAD / beta_gamma
    beta_twiss = beta_twiss_m * _M_TO_MM_PER_RAD
    gamma_twiss = (1.0 + alpha_twiss**2) / beta_twiss
    return ResolvedTransversePlane(
        geometric_emittance_mm_rad=float(geometric),
        beta_twiss_mm_per_rad=float(beta_twiss),
        alpha_twiss=float(alpha_twiss),
        sigma_position_mm=float(np.sqrt(geometric * beta_twiss)),
        sigma_slope_rad=float(np.sqrt(geometric * gamma_twiss)),
    )


def resolve_transverse_distribution(
    policy: TransverseDistribution,
    *,
    energy_keV: float,
) -> ResolvedTransverseDistribution:
    """Convert a normalized-emittance policy into case-energy second moments.

    Geometric emittance is not invariant under acceleration, so the stored
    input is normalized and the geometric value is derived per case from the
    relativistic factor ``beta*gamma = sqrt(gamma^2 - 1)`` with
    ``gamma = 1 + T/m_e c^2``:

    ``eps_geom = eps_n / (beta*gamma)``

    The plane's second moments then follow from Courant-Snyder,
    ``sigma_x = sqrt(eps_geom * beta)`` and
    ``sigma_x' = sqrt(eps_geom * gamma_twiss)`` with
    ``gamma_twiss = (1 + alpha^2) / beta``, and the correlation
    ``<x x'> = -eps_geom * alpha``. This inverts what
    ``beam_metrics._plane_metrics`` reports, so a sampled bunch round-trips.

    Assumptions: the two planes are uncoupled (no ``<x y>`` term), and the
    parameters describe the beam at the crystal entrance face -- no space
    charge and no beamline transport carry it there.

    Limiting cases: ``eps_n -> 0`` gives a collimated point beam;
    ``alpha = 0`` is the waist, where the correlation vanishes and
    ``beta = sigma_x^2 / eps_geom`` recovers the legacy spot-FWHM spelling.
    Two energies differing in ``beta*gamma`` at fixed ``eps_n`` give geometric
    emittances in the inverse ratio of ``beta*gamma``.

    Validation: beam-phase-space-injection
    """
    energy = float(energy_keV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_keV must be finite and positive")
    gamma_rel = 1.0 + energy / _ELECTRON_REST_KEV
    beta_gamma = float(np.sqrt(gamma_rel**2 - 1.0))
    emittance_y, beta_y, alpha_y = policy.plane_y
    return ResolvedTransverseDistribution(
        beta_gamma=beta_gamma,
        x=_resolve_plane(
            policy.normalized_emittance_x_mm_mrad,
            policy.beta_twiss_x_m,
            policy.alpha_twiss_x,
            beta_gamma,
        ),
        y=_resolve_plane(emittance_y, beta_y, alpha_y, beta_gamma),
        provenance=f"normalized emittance at {energy:g} keV (beta*gamma={beta_gamma:g})",
    )


def _sample_plane(
    plane: ResolvedTransversePlane,
    n_electrons: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u1 = rng.standard_normal(n_electrons)
    u2 = rng.standard_normal(n_electrons)
    emittance = plane.geometric_emittance_mm_rad
    beta_twiss = plane.beta_twiss_mm_per_rad
    position = np.sqrt(emittance * beta_twiss) * u1
    slope = np.sqrt(emittance / beta_twiss) * (u2 - plane.alpha_twiss * u1)
    return position, slope


def resolved_from_mapping(payload: Mapping[str, Any]) -> ResolvedTransverseDistribution:
    """Rebuild a resolved distribution from its ``dataclasses.asdict`` form.

    A case dict carries the resolution as plain data (it has to survive JSON
    round-trips through the checkpoint store), but the sampler is written
    against the dataclass so there is exactly one copy of the Courant-Snyder
    algebra. This is the seam between the two.
    """
    return ResolvedTransverseDistribution(
        beta_gamma=float(payload["beta_gamma"]),
        x=ResolvedTransversePlane(**{k: float(v) for k, v in payload["x"].items()}),
        y=ResolvedTransversePlane(**{k: float(v) for k, v in payload["y"].items()}),
        provenance=str(payload.get("provenance", "")),
    )


def sample_transverse(
    resolved: ResolvedTransverseDistribution,
    n_electrons: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(x_mm, x', y_mm, y')`` reproducing the resolved second moments.

    With ``u1, u2`` independent standard normals, ``x = sqrt(eps*beta) u1`` and
    ``x' = sqrt(eps/beta) (u2 - alpha u1)`` give ``<x^2> = eps*beta``,
    ``<x'^2> = eps*gamma_twiss`` and ``<x x'> = -eps*alpha`` exactly, so the
    sample's emittance is ``eps`` up to Monte Carlo error. Each plane consumes
    its own two draws; the planes are independent.

    Validation: beam-phase-space-injection
    """
    x_mm, x_prime = _sample_plane(resolved.x, n_electrons, rng)
    y_mm, y_prime = _sample_plane(resolved.y, n_electrons, rng)
    return x_mm, x_prime, y_mm, y_prime
