"""Research-only carbon-plate first-interaction diagnostic for issue #411.

This is NOT a PXR/CBS source, a scattering simulation, a detector estimator, or
an admission to PhotonTransport. Only explicitly supplied *escaped* boundary
cells may be projected through the downstream geometry. No electron segment or
CXR angular marginal is interpreted as a photon birth position.

    uv run python checks/cxr_filter_first_collision_411.py

Requires the separately fetched, SHA-pinned EPDL2025 table for real coefficients.
The pure geometry/probability functions can be tested without that dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Mapping

import numpy as np

# Analysis-only, explicitly chosen benchmark parameters. Not experimental fits.
CARBON_DENSITY_G_CM3 = 2.0
CARBON_MOLAR_MASS_G_MOL = 12.011
AVOGADRO_PER_MOL = 6.02214076e23
CHANNELS = ("photoelectric", "coherent", "incoherent", "pair_nuclear", "pair_electron")
ANGSTROM_TO_MM = 1.0e-7


def _vector3(value: object, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector")
    return value.copy()


def _unit3(value: object, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"{name} must have nonzero finite norm")
    return vector / norm


@dataclass(frozen=True, slots=True)
class CircularPlate:
    """Homogeneous finite circular cylinder (mm), with plane-parallel faces.

    The plate is centered at ``center_mm`` and its normal is ``axis``. The
    finite radius and thickness are treated exactly by ray/cylinder clipping.
    No infinite-plane ``thickness / cos(theta)`` shortcut at the sidewall.
    """

    center_mm: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius_mm: float
    thickness_mm: float

    def __post_init__(self) -> None:
        _vector3(self.center_mm, "plate center")
        _unit3(self.axis, "plate axis")
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError("plate radius must be positive and finite")
        if not np.isfinite(self.thickness_mm) or self.thickness_mm <= 0:
            raise ValueError("plate thickness must be positive and finite")


@dataclass(frozen=True, slots=True)
class IdealDisk:
    """One-sided ideal detector disk: only front-crossing rays count."""

    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius_mm: float

    def __post_init__(self) -> None:
        _vector3(self.center_mm, "detector center")
        _unit3(self.normal, "detector normal")
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError("detector radius must be positive and finite")


def detector_distance_mm(origin_mm: object, direction: object, disk: IdealDisk) -> float:
    """Return forward front-side intersection distance or ``inf`` on a miss."""
    origin = _vector3(origin_mm, "ray origin")
    direction = _unit3(direction, "ray direction")
    normal = _unit3(disk.normal, "detector normal")
    toward = float(np.dot(direction, normal))
    if toward <= 1e-14:
        return float("inf")
    distance = float(np.dot(_vector3(disk.center_mm, "detector center") - origin, normal))
    distance /= toward
    if not np.isfinite(distance) or distance <= 0:
        return float("inf")
    hit = origin + distance * direction
    radial = hit - np.asarray(disk.center_mm)
    radial -= np.dot(radial, normal) * normal
    if np.dot(radial, radial) > disk.radius_mm**2 * (1.0 + 1e-12):
        return float("inf")
    return distance


def plate_path_mm(
    origin_mm: object, direction: object, plate: CircularPlate, *, limit_mm: float = np.inf
) -> float:
    """Length inside a *finite* cylinder on a forward, unit-speed ray.

    ``limit_mm`` can clip the ray at a detector plane (before a later plate).
    A ray tangent to a face/cylinder contributes zero path. No origin position
    is inferred: callers must provide it. Positive ``limit_mm`` or +inf only.
    """
    origin = _vector3(origin_mm, "ray origin")
    direction = _unit3(direction, "ray direction")
    if np.isnan(limit_mm) or limit_mm < 0:
        raise ValueError("limit_mm must be nonnegative or infinite")
    axis = _unit3(plate.axis, "plate axis")
    relative = origin - np.asarray(plate.center_mm)
    z0 = float(relative @ axis)
    dz = float(direction @ axis)
    half = plate.thickness_mm / 2.0
    # Intersection with the two axial planes.
    if abs(dz) < 1e-14:
        if abs(z0) >= half:
            return 0.0
        lo, hi = -np.inf, np.inf
    else:
        a = (-half - z0) / dz
        b = (half - z0) / dz
        lo, hi = min(a, b), max(a, b)
    # Intersection with infinite cylinder of finite radius.
    perp_o = relative - z0 * axis
    perp_d = direction - dz * axis
    a = float(perp_d @ perp_d)
    b = float(perp_o @ perp_d)
    c = float(perp_o @ perp_o) - plate.radius_mm**2
    if a < 1e-24:
        if c >= 0:
            return 0.0
    else:
        disc = b * b - a * c
        if disc <= 0:
            return 0.0
        root = np.sqrt(disc)
        lo = max(lo, (-b - root) / a)
        hi = min(hi, (-b + root) / a)
    return float(max(0.0, min(hi, limit_mm) - max(lo, 0.0)))


def carbon_epdl_coefficients_inv_mm(
    energy_eV: object, *, density_g_cm3: float = CARBON_DENSITY_G_CM3
) -> dict[str, np.ndarray]:
    """Actual EPDL2025 per-channel linear removal rates at chosen C density.

    The density is a *benchmark choice*, not an unstated catalog/material fit.
    1 Angstrom^2 * 1 atom/Angstrom^3 = 1/Angstrom = 1e7 /mm.
    Undefined data fail closed. These are integrated channel rates, NOT
    angular-scattering PDFs or fluorescence yields.
    """
    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or not energy.size or not np.all(np.isfinite(energy)):
        raise ValueError("energies must be a nonempty finite 1D array")
    if np.any(energy < 1.0) or np.any(energy > 1.0e11):
        raise ValueError("EPDL2025 energy support is 1 eV to 100 GeV")
    if not np.isfinite(density_g_cm3) or density_g_cm3 <= 0:
        raise ValueError("carbon density must be positive and finite")
    from pyrite.materials.photon_cross_sections import photon_cross_sections_ang2

    n_ang3 = density_g_cm3 * AVOGADRO_PER_MOL / (CARBON_MOLAR_MASS_G_MOL * 1e24)
    sigma = photon_cross_sections_ang2("C", energy)
    coefficients = {name: np.asarray(sigma[name], dtype=float) * n_ang3 * 1e7 for name in CHANNELS}
    if any(np.any(~np.isfinite(mu) | (mu < 0)) for mu in coefficients.values()):
        raise ValueError("EPDL coefficients undefined or negative at requested energies")
    return coefficients


@dataclass(frozen=True, slots=True)
class FirstCollision:
    uncollided: np.ndarray
    by_channel: dict[str, np.ndarray]


def first_collision_probabilities(
    path_mm: object, coefficients_inv_mm: Mapping[str, object]
) -> FirstCollision:
    """Exact first-event partition in a homogeneous plate, no descendants.

    ``P0=exp(-mu_total L)`` and ``P(first channel k) =
    -expm1(-mu_total L) mu_k / mu_total``. Rayleigh and Compton are *first
    interactions*, not detector scatter-in events; photoelectric interaction
    is not a prediction of fluorescent photon production.
    """
    path = np.asarray(path_mm, dtype=float)
    if path.ndim != 1 or not np.all(np.isfinite(path)) or np.any(path < 0):
        raise ValueError("path_mm must be a finite nonnegative 1D array")
    if set(coefficients_inv_mm) != set(CHANNELS):
        raise ValueError("must supply all five EPDL interaction channels")
    mu = {name: np.asarray(coefficients_inv_mm[name], dtype=float) for name in CHANNELS}
    if any(
        v.shape != path.shape or not np.all(np.isfinite(v)) or np.any(v < 0)
        for v in mu.values()
    ):
        raise ValueError("coefficients must be nonnegative finite arrays matching paths")
    total = sum(mu.values())
    optical_depth = total * path
    uncollided = np.exp(-optical_depth)
    events = -np.expm1(-optical_depth)
    by_channel = {}
    for name in CHANNELS:
        fraction = np.divide(mu[name], total, out=np.zeros_like(total), where=total > 0)
        by_channel[name] = events * fraction
    return FirstCollision(uncollided, by_channel)


@dataclass(frozen=True, slots=True)
class BoundaryControl:
    vacuum_detector_per_primary: float
    collisions_disabled_detector_per_primary: float
    uncollided_detector_per_primary: float
    first_collision_per_primary: dict[str, float]


def boundary_controls(cells: object, plate: CircularPlate, detector: IdealDisk) -> BoundaryControl:
    """Apply geometry and *first-event accounting* to supplied escaped cells.

    ``cells`` must be an already validated ``BoundaryCells`` (#411 first
    slice). Its joint cell positions, energy, directions and weights are used
    together. This does NOT synthesize boundary cells or run scattered photons.
    Direct/first-interacted are disjoint; only direct *uncollided* photons are
    counted at the detector. First-interaction totals include off-cone rays and
    are deliberately loose upper bounds on possible downstream redistribution.
    """
    origins = np.asarray(cells.position_ang) * ANGSTROM_TO_MM
    directions = np.asarray(cells.direction)
    energy = np.asarray(cells.energy_eV)
    weights = np.asarray(cells.weight_per_primary)
    n = len(energy)
    if (
        origins.shape != (n, 3)
        or directions.shape != (n, 3)
        or weights.shape != (n,)
        or not all(np.all(np.isfinite(v)) for v in (origins, directions, energy, weights))
        or np.any(weights < 0)
        or np.any(energy <= 0)
    ):
        raise ValueError("expected validated, nonnegative, finite escaped boundary cells")
    if n and not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("boundary directions must be unit vectors")
    if n == 0:
        return BoundaryControl(0.0, 0.0, 0.0, {name: 0.0 for name in CHANNELS})
    detector_dist = np.array(
        [detector_distance_mm(p, d, detector) for p, d in zip(origins, directions, strict=True)]
    )
    hits = np.isfinite(detector_dist)
    # The *full* plate first-event upper bound includes rays initially missing
    # the detector. Clip the ray at the detector plane whenever it lies ahead.
    detector_normal = _unit3(detector.normal, "detector normal")
    delta_to_plane = np.asarray(detector.center_mm) - origins
    toward = directions @ detector_normal
    plane_dist = np.divide(
        delta_to_plane @ detector_normal,
        toward,
        out=np.full(n, np.inf),
        where=toward > 1e-14,
    )
    limits = np.where(plane_dist > 0, plane_dist, np.inf)
    path = np.array(
        [
            plate_path_mm(p, d, plate, limit_mm=t)
            for p, d, t in zip(origins, directions, limits, strict=True)
        ]
    )
    event = first_collision_probabilities(path, carbon_epdl_coefficients_inv_mm(energy))
    vacuum = float(np.sum(weights[hits]))
    return BoundaryControl(
        vacuum_detector_per_primary=vacuum,
        collisions_disabled_detector_per_primary=vacuum,
        uncollided_detector_per_primary=float(np.sum(weights[hits] * event.uncollided[hits])),
        first_collision_per_primary={
            name: float(weights @ event.by_channel[name]) for name in CHANNELS
        },
    )


def disk_solid_angle_sr(radius_mm: float, distance_mm: float) -> float:
    """Exact on-axis solid angle of a perpendicular circular disk."""
    if (
        not np.isfinite(radius_mm)
        or not np.isfinite(distance_mm)
        or min(radius_mm, distance_mm) <= 0
    ):
        raise ValueError("disk dimensions must be finite and positive")
    return 2 * pi * (1 - distance_mm / np.hypot(radius_mm, distance_mm))


def main() -> None:
    """Compute the plate's actual EPDL rates, without a fabricated CXR source."""
    energies = np.array([250.0, 280.0, 290.0, 500.0, 1000.0, 1500.0])
    plate = CircularPlate((0.0, 0.0, 10.0), (0.0, 0.0, 1.0), 4.0, 0.0005)
    detector = IdealDisk((0.0, 0.0, 20.0), (0.0, 0.0, 1.0), 2.922)
    path_mm = plate_path_mm((0, 0, 0), (0, 0, 1), plate)
    rates = carbon_epdl_coefficients_inv_mm(energies)
    event = first_collision_probabilities(np.full(len(energies), path_mm), rates)
    print("#411 conditional carbon plate / ideal detector diagnostic (not CXR transport)")
    print(
        f"rho_C={CARBON_DENSITY_G_CM3:g} g/cm^3; "
        f"thickness={plate.thickness_mm * 1e3:g} um; "
        f"plate_radius={plate.radius_mm:g} mm at 10 mm (chosen); detector 20 mm, "
        f"radius {detector.radius_mm:g} mm; "
        f"solid angle={disk_solid_angle_sr(detector.radius_mm, 20):.6f} sr"
    )
    print(
        "Origin: sample center, *geometry-only* axial probe, "
        "NOT a derived CXR surface emission point"
    )
    print("eV    tau_total   P_no_collision   P_Rayleigh_first   P_Compton_first   P_photo_first")
    for i, energy in enumerate(energies):
        tau = sum(rates[name][i] for name in CHANNELS) * path_mm
        print(f"{energy:4.0f}   {tau:10.5g}     {event.uncollided[i]:12.6g}  "
              f"{event.by_channel['coherent'][i]:15.6g}    "
              f"{event.by_channel['incoherent'][i]:15.6g}    "
              f"{event.by_channel['photoelectric'][i]:12.6g}")
    print("First-event probabilities are not photon scatter-in or fluorescence detector yields.")


if __name__ == "__main__":
    main()
