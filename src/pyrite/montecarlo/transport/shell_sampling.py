"""Host-side hard loss sampling from the closed shell GOS partition."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c, e, m_e
from scipy.optimize import brentq

from .cores import _rotate_direction_scalar
from .inelastic import _qmin_ev
from .shell_gos import _moller_integrals, _triangle_moments
from .shell_oscillators import MaterialShellOscillators
from .shell_partition import ShellSoftHardPartition

BRANCHES = ("distant_longitudinal", "distant_transverse", "close")
_MC2_EV = m_e * c * c / e


@dataclass(frozen=True, slots=True)
class ShellHardLoss:
    """A hard transfer before recoil and flight-scheduler integration.

    ``secondary_energy_eV`` is None below the production threshold. A
    substituted inner shell reserves binding for relaxation. Outer shells
    use a free-electron secondary proxy with kinetic energy ``W`` and no
    vacancy. Their oscillator labels are not relaxation vacancies.
    """

    oscillator_index: int
    atomic_number: int
    shell_label: str
    branch: str
    transfer_eV: float
    binding_eV: float
    secondary_energy_eV: float | None
    local_deposit_eV: float
    binding_reserve_eV: float
    vacancy: tuple[int, str] | None


@dataclass(frozen=True, slots=True)
class ShellHardCollision:
    """Hard loss and emitted-electron angles in the incoming flight frame."""

    loss: ShellHardLoss
    recoil_energy_eV: float | None
    cos_primary: float
    azimuth_rad: float
    cos_secondary: float | None
    secondary_azimuth_rad: float | None


@dataclass(frozen=True, slots=True)
class ShellWorldDirections:
    """Primary and optional secondary unit directions in laboratory coordinates."""

    primary: tuple[float, float, float]
    secondary: tuple[float, float, float] | None


def shell_collision_world_directions(
    collision: ShellHardCollision, incoming: tuple[float, float, float]
) -> ShellWorldDirections:
    """Rotate both collision directions about one incoming-flight basis.

    Source: PENELOPE-2024 §3.2.5.4 uses opposite primary/secondary azimuths;
    ``cores._rotate_direction_scalar`` defines the transport frame. The same
    frame is used for both particles, preserving their relative azimuth.
    Assumption: ``incoming`` is a laboratory-frame unit direction.
    Limit: a forward primary retains ``incoming``; a suppressed secondary has
    no direction. Validation: penelope-shell-secondary-direction.
    """
    direction = np.asarray(incoming, dtype=float)
    if (
        direction.shape != (3,)
        or not np.all(np.isfinite(direction))
        or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("incoming flight direction must be a finite unit vector")
    d = tuple(float(x) for x in direction)
    primary = _rotate_direction_scalar(*d, collision.cos_primary, collision.azimuth_rad)
    secondary = None
    if collision.cos_secondary is not None:
        if collision.secondary_azimuth_rad is None:
            raise ValueError("secondary polar angle needs an azimuth")
        secondary = _rotate_direction_scalar(
            *d, collision.cos_secondary, collision.secondary_azimuth_rad
        )
    return ShellWorldDirections(primary, secondary)


def _loss_bounds(
    material: MaterialShellOscillators, partition: ShellSoftHardPartition, index: int, branch: int
) -> tuple[float, float, float]:
    osc = material.oscillators[index]
    energy = partition.closure.raw.energy_eV
    u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
    lower = partition.cutoff_eV
    if branch == 2:
        return max(u if u > 0.0 else w, lower), (energy + u) / 2.0, 0.0
    if u == 0.0:
        return w, w, 0.0
    w_dis = 3.0 * w - 2.0 * u
    if energy <= w_dis:
        w_dis = energy
    return max(u, lower), min(w_dis, (energy + u) / 2.0), w_dis


def sample_shell_hard_loss(
    material: MaterialShellOscillators,
    partition: ShellSoftHardPartition,
    u_channel: float,
    u_loss: float,
    *,
    production_threshold_eV: float = 0.0,
) -> ShellHardLoss:
    """Draw a channel and loss from the same DCS used by the partition.

    Source: PENELOPE-2024 Eqs. 3.76, 3.87, 3.94, 3.96, 3.104 and 3.124. The
    bound-shell distant conditional density is ``p_dis(W)/W``; the close
    conditional density is ``F^(-)(E+U,W)/W^2``. A conduction-band distant
    loss is a delta at its resonance. All shells use the hard interval
    ``W > W_c`` used by the partition.
    The manual's Eq. 3.125 instead samples ``p_dis(W)`` without ``1/W``;
    this sampler follows Eq. 3.104 so its samples and partition moments
    describe one differential cross section.

    Assumptions: a sampled bound-shell transfer ionizes its oscillator;
    ``W-U`` is an inner-shell secondary's kinetic energy. Outer shells use
    the ``W`` proxy of PENELOPE §3.2.5 with no residual vacancy. Inner-shell
    binding is reserved for #94's relaxation handoff, even when the
    secondary is below the separate production threshold.
    Limits: a delta branch always returns its resonance; a sampled continuous
    loss lies inside its channel's hard interval; local plus emitted plus
    reserved energy equals the primary loss. Recoil and azimuth are not
    returned here.
    Validation: penelope-shell-hard-loss-sampling
    """
    if partition.closure.raw.oscillators != material.oscillators:
        raise ValueError("partition was not built from these oscillators")
    if not all(np.isfinite(u) and 0.0 <= u < 1.0 for u in (u_channel, u_loss)):
        raise ValueError("sample uniforms must be finite and in [0, 1)")
    if not np.isfinite(production_threshold_eV) or production_threshold_eV < 0.0:
        raise ValueError("production threshold must be finite and non-negative")
    probabilities = partition.hard_channel_probabilities
    if partition.hard_cross_section_cm2 <= 0.0:
        raise ValueError("partition has no hard collisions")
    flat = probabilities.ravel()
    cumulative = np.cumsum(flat)
    selected = min(
        int(np.searchsorted(cumulative, u_channel * cumulative[-1], side="right")), flat.size - 1
    )
    index, branch = divmod(selected, 3)
    osc = material.oscillators[index]
    lower, upper, peak_end = _loss_bounds(material, partition, index, branch)
    if branch != 2 and osc.ionization_energy_eV == 0.0:
        transfer = lower
    else:
        if branch == 2:
            prime = partition.closure.raw.energy_eV + osc.ionization_energy_eV

            def integral(w: float) -> float:
                return float(
                    (
                        _moller_integrals(partition.closure.raw.energy_eV, prime, w)
                        - _moller_integrals(partition.closure.raw.energy_eV, prime, lower)
                    )[0]
                )
        else:

            def integral(w: float) -> float:
                return float(_triangle_moments(osc.ionization_energy_eV, peak_end, lower, w)[0])

        total = integral(upper)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("selected hard channel has no positive loss integral")
        target = u_loss * total
        transfer = (
            lower
            if target == 0.0
            else brentq(lambda w: integral(w) - target, lower, upper, xtol=1e-12, rtol=1e-14)
        )
    binding = osc.ionization_energy_eV
    vacancy = (osc.atomic_number, osc.label) if partition.closure.inner[index] else None
    secondary = max(0.0, transfer - binding) if vacancy else transfer
    emitted = secondary if secondary > 0.0 and secondary >= production_threshold_eV else None
    reserve = binding if vacancy is not None else 0.0
    deposit = 0.0 if emitted is not None else secondary
    return ShellHardLoss(
        index,
        osc.atomic_number,
        osc.label,
        BRANCHES[branch],
        float(transfer),
        binding,
        emitted,
        float(deposit),
        reserve,
        vacancy,
    )


def sample_shell_hard_collision(
    material: MaterialShellOscillators,
    partition: ShellSoftHardPartition,
    u_channel: float,
    u_loss: float,
    u_recoil: float,
    u_azimuth: float,
    *,
    production_threshold_eV: float = 0.0,
) -> ShellHardCollision:
    """Sample a hard loss and primary recoil from the shell GOS branches.

    Source: PENELOPE-2024 Eqs. 3.126–3.129, 3.134 and 3.137–3.138. Longitudinal ``Q``
    has density proportional to ``1/[Q(1+Q/(2mc²))]`` between ``Q_-`` and
    ``Q'_k``. For a broadened bound shell, the angular model uses the modified
    resonance ``W'_k`` in both ``Q_-`` and the polar-angle formula, as in
    Eq. 3.129; sampled loss ``W`` still controls primary energy and binding
    accounting. The transverse branch has no primary deflection. A close
    event has ``Q=W`` and uses Eq. 3.134. An emitted secondary follows the
    momentum-transfer direction, with opposite azimuth. The transverse
    secondary uses the Penelope implementation's fixed cosine of 0.5, since
    that branch neglects the primary recoil. Every primary azimuth is uniform.

    Assumptions: primary recoil is expressed relative to the incoming
    direction; the caller rotates both directions into laboratory coordinates.
    A secondary below production threshold has no direction. Limits: ``u_recoil=0`` gives
    the minimum longitudinal ``Q`` and zero polar deflection; close recoil
    tends to zero deflection as ``W/E -> 0``.
    Validation: penelope-shell-hard-recoil, penelope-shell-secondary-direction
    """
    if not all(np.isfinite(u) and 0.0 <= u < 1.0 for u in (u_recoil, u_azimuth)):
        raise ValueError("recoil uniforms must be finite and in [0, 1)")
    loss = sample_shell_hard_loss(
        material,
        partition,
        u_channel,
        u_loss,
        production_threshold_eV=production_threshold_eV,
    )
    energy = partition.closure.raw.energy_eV
    osc = material.oscillators[loss.oscillator_index]
    secondary_cosine: float | None = None
    if loss.branch == "distant_transverse":
        recoil, cosine = None, 1.0
        if loss.secondary_energy_eV is not None:
            secondary_cosine = 0.5
    elif loss.branch == "close":
        recoil = loss.transfer_eV
        remaining = energy - recoil
        cosine = float(
            np.sqrt(remaining / energy * (energy + 2.0 * _MC2_EV) / (remaining + 2.0 * _MC2_EV))
        )
        if loss.secondary_energy_eV is not None:
            secondary_cosine = float(
                np.sqrt(recoil / energy * (energy + 2.0 * _MC2_EV) / (recoil + 2.0 * _MC2_EV))
            )
    else:
        u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
        if u > 0.0:
            full_width = 3.0 * w - 2.0 * u
            if energy > full_width:
                resonance, q_upper = w, u
            else:
                resonance, q_upper = (energy + 2.0 * u) / 3.0, u * energy / full_width
        else:
            resonance, q_upper = w, w
        q_lower = float(_qmin_ev(energy, resonance))
        if not 0.0 < q_lower < q_upper:
            raise ValueError("longitudinal channel has invalid recoil bounds")
        log_lower = np.log(q_lower / (q_lower + 2.0 * _MC2_EV))
        log_upper = np.log(q_upper / (q_upper + 2.0 * _MC2_EV))
        log_ratio = (1.0 - u_recoil) * log_lower + u_recoil * log_upper
        recoil = float(2.0 * _MC2_EV / np.expm1(-log_ratio))
        p0_sq = energy * (energy + 2.0 * _MC2_EV)
        remaining = energy - resonance
        p1_sq = remaining * (remaining + 2.0 * _MC2_EV)
        cosine = float(
            (p0_sq + p1_sq - recoil * (recoil + 2.0 * _MC2_EV)) / (2.0 * np.sqrt(p0_sq * p1_sq))
        )
        cosine = float(np.clip(cosine, -1.0, 1.0))
        if loss.secondary_energy_eV is not None:
            q_sq = recoil * (recoil + 2.0 * _MC2_EV)
            secondary_cosine = float(
                np.clip((p0_sq + q_sq - p1_sq) / (2.0 * np.sqrt(p0_sq * q_sq)), -1.0, 1.0)
            )
    azimuth = float(2.0 * np.pi * u_azimuth)
    secondary_azimuth = (
        float((azimuth + np.pi) % (2.0 * np.pi)) if secondary_cosine is not None else None
    )
    return ShellHardCollision(loss, recoil, cosine, azimuth, secondary_cosine, secondary_azimuth)
