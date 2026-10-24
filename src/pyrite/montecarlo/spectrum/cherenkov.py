r"""Bulk Frank--Tamm Cherenkov source in a homogeneous weakly absorbing dielectric (#587).

Internal source yield of one charged particle moving at constant speed
``beta`` along a straight path through an unbounded, isotropic, nonmagnetic
medium whose relative permittivity ``epsilon(E)`` is given explicitly:

.. math::

    \frac{d^2N}{dx\,dE} = \frac{\alpha z^2}{\hbar c}
    \left(1 - \frac{1}{\beta^2 n^2(E)}\right)\Theta(\beta n - 1),
    \qquad \cos\theta_C = \frac{1}{\beta n(E)},

with ``n = Re sqrt(epsilon)`` (principal root, passive ``exp(-i omega t)``
convention, ``Im epsilon >= 0``). Emission is a delta function on the cone
``theta_C(E)`` about the velocity; ``alpha / hbar c = 369.81 eV^-1 cm^-1``.

Units: photons per metre of path per eV of photon energy, per particle.

Validity: weak loss only (``Im epsilon / Re epsilon <= max_loss_tangent``;
larger loss tangents, ``Re epsilon <= 0`` and active media are refused),
infinite straight path at fixed ``beta`` and no recoil (``E << beta gamma m
c^2``). Interface/transition radiation, finite-slab escape, strong absorption
and absorption-edge (resonant X-ray) Cherenkov emission are out of scope; no
escaped or detected yield is implied.

Validation: frank-tamm-homogeneous-dielectric
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG

__all__ = [
    "CHERENKOV_MODEL",
    "FRANK_TAMM_PER_EV_M",
    "BulkEmissionMechanism",
    "BulkEmissionYield",
    "ChargedPassage",
    "ConstantIndex",
    "DielectricResponse",
    "cherenkov_threshold_beta",
    "frank_tamm_yield",
]

#: Identity marker of the bulk Frank--Tamm source.
CHERENKOV_MODEL = "frank-tamm-homogeneous-v1"
#: ``alpha / (hbar c)`` [eV^-1 m^-1]; 369.81 eV^-1 cm^-1.
FRANK_TAMM_PER_EV_M = ALPHA_FS / (HBARC_EV_ANG * 1e-10)
#: Default largest accepted loss tangent ``Im epsilon / Re epsilon``.
DEFAULT_MAX_LOSS_TANGENT = 1e-3


@runtime_checkable
class DielectricResponse(Protocol):
    """Explicit scalar permittivity, e.g. :class:`~pyrite.materials.optical.OpticalResponse`."""

    def epsilon(self, energy_eV: Any) -> Any:
        """Relative permittivity at photon energies [eV]."""
        ...


@dataclass(frozen=True)
class ConstantIndex:
    """Lossless nondispersive medium, ``epsilon = n**2`` at every photon energy.

    A user-declared limiting case of the explicit-permittivity input (no
    material data): valid only inside a transparent band where the index is
    nearly constant, e.g. n = 1.46 for fused silica in the visible.
    """

    n: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.n) and self.n > 0.0):
            raise ValueError("refractive index n must be finite and positive")

    def epsilon(self, energy_eV: Any) -> np.ndarray:
        return np.full(np.shape(energy_eV), complex(self.n**2))


@dataclass(frozen=True)
class ChargedPassage:
    """Straight passage of one charged particle at constant speed.

    ``beta = v/c`` in ``[0, 1)``; ``charge`` is the charge number ``z`` (its
    sign does not enter the yield); ``path_length_m`` is the traversed length
    used by :meth:`BulkEmissionYield.photons_per_eV`, or None for per-metre
    yields only. Slowing down within the path is the caller's responsibility
    (split the path into segments of nearly constant ``beta``).

    Validation: frank-tamm-homogeneous-dielectric
    """

    beta: float
    charge: float = -1.0
    path_length_m: float | None = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.beta) and 0.0 <= self.beta < 1.0):
            raise ValueError("beta must be finite and in [0, 1)")
        if not np.isfinite(self.charge):
            raise ValueError("charge must be finite")
        if self.path_length_m is not None and not (
            np.isfinite(self.path_length_m) and self.path_length_m >= 0.0
        ):
            raise ValueError("path_length_m must be finite and nonnegative or None")


@dataclass(frozen=True, eq=False)
class BulkEmissionYield:
    """Internal bulk source yield on a photon-energy grid.

    ``yield_per_m_per_eV`` is ``d^2N/(dx dE)`` [photons m^-1 eV^-1].
    ``cos_theta`` is the emission-cone cosine about the velocity, NaN where
    no photon is emitted (``above_threshold`` False).

    Validation: frank-tamm-homogeneous-dielectric
    """

    mechanism: str
    passage: ChargedPassage
    energy_eV: np.ndarray
    refractive_index: np.ndarray
    yield_per_m_per_eV: np.ndarray
    cos_theta: np.ndarray
    above_threshold: np.ndarray

    @property
    def theta_rad(self) -> np.ndarray:
        """Cone half-angle [rad]; NaN below threshold."""
        return np.arccos(self.cos_theta)

    def photons_per_eV(self) -> np.ndarray:
        """``dN/dE`` over the passage path [photons eV^-1]."""
        if self.passage.path_length_m is None:
            raise ValueError("passage has no path_length_m")
        return self.yield_per_m_per_eV * self.passage.path_length_m

    def angular_histogram(self, theta_edges_rad: Any) -> np.ndarray:
        """Bin the cone yield into polar-angle bins [photons m^-1 eV^-1 per bin].

        The angular density is ``yield * delta(theta - theta_C)``, so each
        energy deposits its whole yield in the bin holding ``theta_C``
        (half-open ``[lo, hi)``). Shape ``(*energy.shape, n_bins)``; yield
        outside the edges is dropped.
        """
        edges = np.asarray(theta_edges_rad, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("theta_edges_rad must be a strictly increasing 1-D array")
        theta = np.where(self.above_threshold, self.theta_rad, -np.inf)
        index = np.searchsorted(edges, theta, side="right") - 1
        inside = self.above_threshold & (index >= 0) & (index < edges.size - 1)
        out = np.zeros((*self.energy_eV.shape, edges.size - 1))
        np.put_along_axis(
            out,
            np.where(inside, index, 0)[..., None],
            np.where(inside, self.yield_per_m_per_eV, 0.0)[..., None],
            axis=-1,
        )
        return out


class BulkEmissionMechanism(Protocol):
    """Extension point for bulk (interface-free) emission sources.

    A mechanism maps a passage, photon energies [eV] and an explicit
    dielectric response to a :class:`BulkEmissionYield`. Interface terms do
    not belong here.
    """

    def __call__(
        self,
        passage: ChargedPassage,
        energy_eV: Any,
        epsilon: Any,
        /,
    ) -> BulkEmissionYield: ...


def _permittivity(energy: np.ndarray, epsilon: Any) -> np.ndarray:
    if isinstance(epsilon, DielectricResponse):
        epsilon = epsilon.epsilon(energy)
    eps = np.broadcast_to(np.asarray(epsilon, dtype=complex), energy.shape)
    if not np.all(np.isfinite(eps)):
        raise ValueError("epsilon must be finite")
    return eps


def cherenkov_threshold_beta(epsilon: Any) -> np.ndarray | np.float64:
    """Threshold speed ``1/n`` with ``n = Re sqrt(epsilon)``; inf where ``n <= 1``.

    Validation: frank-tamm-homogeneous-dielectric
    """
    n = np.sqrt(np.asarray(epsilon, dtype=complex)).real
    with np.errstate(divide="ignore"):
        return np.where(n > 1.0, 1.0 / n, np.inf)


def frank_tamm_yield(
    passage: ChargedPassage,
    energy_eV: Any,
    epsilon: Any,
    /,
    *,
    max_loss_tangent: float = DEFAULT_MAX_LOSS_TANGENT,
) -> BulkEmissionYield:
    """Return the bulk Frank--Tamm yield ``d^2N/(dx dE)`` and cone angle.

    ``epsilon`` is an array broadcastable to ``energy_eV`` or a
    :class:`DielectricResponse`. The refractive index is ``Re sqrt(epsilon)``;
    absorption enters only through the weak-loss gate, which refuses
    ``Im epsilon < 0`` (active), ``Re epsilon <= 0`` and loss tangents above
    ``max_loss_tangent``. Away from threshold the neglected loss correction
    is ``O(tan delta)``; the sharp lossless threshold is kept.
    Exactly at threshold (``beta n = 1``) the yield is zero.

    Source: Frank and Tamm, Dokl. Akad. Nauk SSSR 14, 109 (1937); Jackson,
    Classical Electrodynamics 3rd ed., Sec. 13.5; PDG Review "Passage of
    particles through matter", Cherenkov radiation section.
    Validation: frank-tamm-homogeneous-dielectric
    """
    if not (np.isfinite(max_loss_tangent) and max_loss_tangent >= 0.0):
        raise ValueError("max_loss_tangent must be finite and nonnegative")
    energy = np.asarray(energy_eV, dtype=float)
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_eV must be finite and positive")
    eps = _permittivity(energy, epsilon)
    if np.any(eps.imag < 0.0):
        raise ValueError("epsilon must be passive (Im epsilon >= 0, exp(-i omega t))")
    if np.any(eps.real <= 0.0):
        raise ValueError("Re epsilon <= 0: strong-loss/metallic response is out of scope")
    if np.any(eps.imag > max_loss_tangent * eps.real):
        raise ValueError(
            f"loss tangent Im/Re epsilon exceeds {max_loss_tangent:g}: "
            "strong absorption is out of scope"
        )
    n = np.sqrt(eps).real
    beta_n = passage.beta * n
    above = beta_n > 1.0
    sin2 = np.where(above, 1.0 - 1.0 / np.where(above, beta_n, 1.0) ** 2, 0.0)
    return BulkEmissionYield(
        mechanism=CHERENKOV_MODEL,
        passage=passage,
        energy_eV=energy,
        refractive_index=n,
        yield_per_m_per_eV=FRANK_TAMM_PER_EV_M * passage.charge**2 * sin2,
        cos_theta=np.where(above, 1.0 / np.where(above, beta_n, 1.0), np.nan),
        above_threshold=above,
    )
