"""Declarative longitudinal bunch policies and target-line timing.

Policies are physical inputs and therefore independent of simulated
macro-particle count. Resolved records contain only deterministic,
case-specific timing/provenance; sampled electron offsets remain runtime data.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .materials.crystal import (
    HBARC_EV_ANG,
    HC_EV_ANG,
    _direct_lattice_vectors,
    beta_from_Ee,
    reciprocal_g_vector,
)

LongitudinalKind = Literal["gaussian", "microtrain", "compressed"]

# h = hc/c in the units used by BeamSpec's transport clock.
C_ANG_PER_FS = 2997.924580
H_EV_FS = HC_EV_ANG / C_ANG_PER_FS


@dataclass(frozen=True)
class LongitudinalDistribution:
    """Serializable longitudinal bunch policy.

    ``gaussian`` uses ``envelope_rms_fs`` as its bunch RMS duration.
    ``microtrain`` uses it as the Gaussian train-envelope RMS duration and
    derives the width/spacing of its microbunches from the target line.
    ``compressed`` is one Gaussian whose RMS duration is the same derived
    microbunch width. Targeted modes use the catalog-pinned dominant basal
    reflection unless ``target_reflection`` pins one explicitly.
    """

    kind: LongitudinalKind
    envelope_rms_fs: float | None = None
    retained_coherence: float = 0.9
    target_reflection: tuple[int, int, int] | None = None
    spacing_periods: int = 1
    modulation_depth: float = 1.0
    timing_jitter_fs: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("gaussian", "microtrain", "compressed"):
            raise ValueError(f"unknown longitudinal kind {self.kind!r}")
        if self.kind in ("gaussian", "microtrain"):
            if self.envelope_rms_fs is None:
                raise ValueError(f"{self.kind} requires envelope_rms_fs")
            if not np.isfinite(self.envelope_rms_fs) or self.envelope_rms_fs <= 0.0:
                raise ValueError("envelope_rms_fs must be finite and positive")
        elif self.envelope_rms_fs is not None:
            raise ValueError("compressed derives its RMS duration; omit envelope_rms_fs")
        if not 0.0 < self.retained_coherence <= 1.0:
            raise ValueError("retained_coherence must satisfy 0 < eta <= 1")
        if isinstance(self.spacing_periods, bool) or self.spacing_periods < 1:
            raise ValueError("spacing_periods must be a positive integer")
        if int(self.spacing_periods) != self.spacing_periods:
            raise ValueError("spacing_periods must be a positive integer")
        if not 0.0 <= self.modulation_depth <= 1.0:
            raise ValueError("modulation_depth must satisfy 0 <= depth <= 1")
        if not np.isfinite(self.timing_jitter_fs) or self.timing_jitter_fs < 0.0:
            raise ValueError("timing_jitter_fs must be finite and non-negative")
        if self.target_reflection is not None:
            if len(self.target_reflection) != 3 or not any(self.target_reflection):
                raise ValueError("target_reflection must be a nonzero Miller-index triple")
        if self.kind == "gaussian":
            targeted = (
                self.target_reflection is not None
                or self.retained_coherence != 0.9
                or self.spacing_periods != 1
                or self.modulation_depth != 1.0
                or self.timing_jitter_fs != 0.0
            )
            if targeted:
                raise ValueError("gaussian does not accept target-line or modulation controls")


@dataclass(frozen=True)
class ResolvedLongitudinalDistribution:
    """Deterministic longitudinal policy resolved for one simulation case."""

    kind: LongitudinalKind
    envelope_rms_fs: float | None
    rms_duration_fs: float | None
    microbunch_rms_fs: float | None
    retained_coherence: float | None
    spacing_fs: float | None
    modulation_depth: float
    timing_jitter_fs: float
    target_material: str | None
    target_crystal: str | None
    target_reflection: tuple[int, int, int] | None
    target_energy_eV: float | None
    target_wavelength_ang: float | None
    target_period_fs: float | None
    provenance: str


def _dominant_basal_reflection(
    hkl_list: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
    surface_axis: np.ndarray,
    lattice: Mapping[str, float | str],
) -> tuple[int, int, int]:
    """Select the first positive catalog-pinned reflection parallel to the surface normal."""
    surface_unit = surface_axis / np.linalg.norm(surface_axis)
    for raw_hkl in hkl_list:
        hkl = (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
        g_vec, g_norm = reciprocal_g_vector(hkl, lattice)
        cosine = float(g_vec @ surface_unit / g_norm)
        if np.isclose(cosine, 1.0, rtol=0.0, atol=1e-12):
            return hkl
    raise ValueError(
        "catalog-pinned reflections contain no positive basal reflection "
        "parallel to the catalog orientation axis"
    )


def resolve_longitudinal_distribution(
    policy: LongitudinalDistribution,
    *,
    material: str,
    crystal: str,
    lattice: Mapping[str, float | str],
    hkl_list: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
    surface_hkl: tuple[int, int, int] | None,
    beam_uvw: tuple[int, int, int] | None,
    energy_keV: float,
    theta_obs_deg: float,
    tilt_deg: float,
) -> ResolvedLongitudinalDistribution:
    """Resolve target-line timing for one material/energy/geometry case.

    For a basal reflection parallel to the catalog surface normal, the photon
    wavenumber is
    ``k_gamma = beta |g| cos(tilt) / (1 - beta cos(theta_obs))`` [1/Angstrom].
    The target photon energy is ``E = hbar*c*k_gamma``. Its optical period is
    ``T = h/E`` and temporal angular frequency is
    ``Omega = E/hbar = 2*pi/T`` [rad/fs]. A Gaussian time distribution has
    intensity form factor ``|F(Omega)|^2 = exp[-(Omega*sigma_t)^2]``, hence
    ``sigma_t = sqrt(-ln(eta))/Omega``.

    Assumptions: target ``g`` is parallel to the catalog surface normal,
    detector polar angle is measured from the lab beam, and the catalog's
    pinned reflection order defines dominance. Limiting cases:
    ``eta -> 1`` gives zero microbunch width; one spacing period gives
    ``Omega*T = 2*pi``.

    Validation: longitudinal-target-timing
    """
    if policy.kind == "gaussian":
        return ResolvedLongitudinalDistribution(
            kind=policy.kind,
            envelope_rms_fs=policy.envelope_rms_fs,
            rms_duration_fs=policy.envelope_rms_fs,
            microbunch_rms_fs=None,
            retained_coherence=None,
            spacing_fs=None,
            modulation_depth=1.0,
            timing_jitter_fs=0.0,
            target_material=None,
            target_crystal=None,
            target_reflection=None,
            target_energy_eV=None,
            target_wavelength_ang=None,
            target_period_fs=None,
            provenance="explicit Gaussian RMS duration",
        )
    if surface_hkl is not None:
        surface_axis, _ = reciprocal_g_vector(surface_hkl, lattice)
        orientation_provenance = f"surface_hkl={surface_hkl}"
    elif beam_uvw is not None:
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        u, v, w = beam_uvw
        surface_axis = u * a1 + v * a2 + w * a3
        orientation_provenance = f"beam_uvw={beam_uvw}"
    else:
        raise ValueError("targeted longitudinal modes require catalog orientation provenance")

    hkl = policy.target_reflection or _dominant_basal_reflection(
        hkl_list, surface_axis, lattice
    )
    g_vec, g_norm = reciprocal_g_vector(hkl, lattice)
    surface_norm = float(np.linalg.norm(surface_axis))
    alignment = abs(float(g_vec @ surface_axis / (g_norm * surface_norm)))
    if not np.isclose(alignment, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"target_reflection={hkl} is not basal/parallel to {orientation_provenance}"
        )
    # Orient the explicitly pinned reflection toward the beam-facing positive
    # surface normal; a negative member of the same family cannot radiate in
    # this geometry.
    if float(g_vec @ surface_axis) <= 0.0:
        raise ValueError("target_reflection must point along the positive surface normal")

    beta = float(beta_from_Ee(float(energy_keV) * 1e3))
    numerator = beta * g_norm * np.cos(np.deg2rad(float(tilt_deg)))
    denominator = 1.0 - beta * np.cos(np.deg2rad(float(theta_obs_deg)))
    k_gamma_inv_ang = numerator / denominator
    if not np.isfinite(k_gamma_inv_ang) or k_gamma_inv_ang <= 0.0:
        raise ValueError("target photon wavenumber must be finite and positive")
    target_energy_eV = float(HBARC_EV_ANG * k_gamma_inv_ang)
    period_fs = float(H_EV_FS / target_energy_eV)
    omega_rad_fs = 2.0 * np.pi / period_fs
    microbunch_rms_fs = float(
        np.sqrt(-np.log(policy.retained_coherence)) / omega_rad_fs
    )
    rms_duration_fs = microbunch_rms_fs if policy.kind == "compressed" else None

    return ResolvedLongitudinalDistribution(
        kind=policy.kind,
        envelope_rms_fs=policy.envelope_rms_fs,
        rms_duration_fs=rms_duration_fs,
        microbunch_rms_fs=microbunch_rms_fs,
        retained_coherence=policy.retained_coherence,
        spacing_fs=float(policy.spacing_periods * period_fs),
        modulation_depth=policy.modulation_depth,
        timing_jitter_fs=policy.timing_jitter_fs,
        target_material=material,
        target_crystal=crystal,
        target_reflection=hkl,
        target_energy_eV=target_energy_eV,
        target_wavelength_ang=float(HC_EV_ANG / target_energy_eV),
        target_period_fs=period_fs,
        provenance=(
            "catalog-pinned dominant positive basal reflection; "
            f"{orientation_provenance}"
        ),
    )
