"""Host-side soft/hard partition of the closed PENELOPE shell GOS moments.

Splits the inner-shell substituted, stopping-closed moments of
:mod:`.shell_rates` at an energy-loss cutoff ``W_c`` into soft moments for a
continuous-loss step and hard moments for discrete events. Nothing here is
sampled or used by a transport mode.
"""

from dataclasses import dataclass

import numpy as np

from .shell_gos import ShellGOSMoments, windowed_shell_gos_moments
from .shell_oscillators import MaterialShellOscillators
from .shell_rates import (
    DEFAULT_INNER_SHELL_THRESHOLD_EV,
    ShellRateClosure,
    catalog_shell_oscillators,
    catalog_shell_rate_closure,
)


@dataclass(frozen=True, slots=True)
class ShellSoftHardPartition:
    """Soft (``W <= W_c``) and hard moments of one closed shell GOS.

    ``soft`` and ``hard`` have the layout of ``closure.moments`` and sum to it.
    Inner shells (``closure.inner``) are hard at every ``W_c``. Units per
    formula unit, as in :class:`~.shell_gos.ShellGOSMoments`.
    """

    closure: ShellRateClosure
    cutoff_eV: float
    soft: ShellGOSMoments
    hard: ShellGOSMoments

    @property
    def soft_stopping_eV_cm2(self) -> float:
        """Soft stopping cross section ``sigma_s^(1)`` (Eq. 4.46 per N)."""
        return float(self.soft.total[1])

    @property
    def soft_straggling_eV2_cm2(self) -> float:
        """Soft straggling cross section ``sigma_s^(2)`` (Eq. 4.47 per N)."""
        return float(self.soft.total[2])

    @property
    def hard_cross_section_cm2(self) -> float:
        """Hard-event cross section ``sigma(W_c)`` (Eqs. 3.124, 4.44 per N)."""
        return float(self.hard.total[0])

    @property
    def hard_stopping_eV_cm2(self) -> float:
        """Mean loss per unit path of hard events, per formula unit."""
        return float(self.hard.total[1])

    @property
    def hard_channel_probabilities(self) -> np.ndarray:
        """Point probabilities of (oscillator, branch) for a hard event.

        Shape ``(n_oscillators, 3)`` over distant longitudinal, distant
        transverse and close: PENELOPE's ``p_k = sigma_k(W_c)/sigma(W_c)``
        (Eq. 3.124) refined by branch. All zero when no hard event is possible.
        """
        channels = np.stack(
            [self.hard.distant_longitudinal, self.hard.distant_transverse, self.hard.close],
            axis=1,
        )[:, :, 0]
        total = channels.sum()
        return channels / total if total > 0.0 else np.zeros_like(channels)

    @property
    def vacancy_cross_sections_cm2(self) -> np.ndarray:
        """Hard ``sigma^(0)`` of each oscillator, zero for outer shells.

        Equals ``closure.adopted_inner_cm2``: every inner-shell event is hard.
        """
        return np.where(self.closure.inner, self.hard.per_shell[:, 0], 0.0)


def partition_shell_rates(
    material: MaterialShellOscillators, closure: ShellRateClosure, cutoff_eV: float
) -> ShellSoftHardPartition:
    """Split closed shell GOS moments at the energy-loss cutoff ``W_c``.

    Source: PENELOPE-2024 Eqs. 3.124 and 4.44–4.47. Soft moments integrate
    the closed DCS over ``W <= W_c`` and hard moments over ``W > W_c``; for
    outer shells and the conduction band each window of
    :func:`~.shell_gos.windowed_shell_gos_moments` is multiplied by the
    closure factor ``N(E)``. Every inner shell stays hard with its full
    closed moments, so its hard rate is the substituted EEDL rate
    ``sigma_si,i rho_i`` for every ``W_c``.

    Deviation: PENELOPE's soft DCS (Eq. 4.113) includes inner shells with
    ``U_i < W_cc``, whose ionisations then create no vacancy. Keeping inner
    shells hard preserves the EEDL vacancy count; the two agree whenever
    ``W_c <= min U_i`` (above 50 eV for the catalog materials).
    Assumptions: a delta loss at exactly ``W_c`` is soft (``W > W_c`` is
    hard), matching the OOS-bin partition. Limits: ``W_c = 0`` leaves no
    soft moment; ``W_c >= (E + U_k)/2`` for every outer oscillator leaves
    only inner-shell hard events; soft plus hard first moments equal the
    adopted stopping for every ``W_c``.

    Units: ``W_c`` in eV; moments per formula unit as in
    :class:`~.shell_gos.ShellGOSMoments`.
    Validation: penelope-shell-soft-hard-partition
    """
    if not np.isfinite(cutoff_eV) or cutoff_eV < 0.0:
        raise ValueError("energy-loss cutoff must be finite and non-negative")
    raw = closure.raw
    if raw.oscillators != material.oscillators:
        raise ValueError("closure was not built from these oscillators")
    energy = raw.energy_eV
    below = windowed_shell_gos_moments(material, energy, 0.0, cutoff_eV)
    above = windowed_shell_gos_moments(material, energy, cutoff_eV, np.inf)
    inner = closure.inner[:, None]
    outer_scale = np.where(inner, 0.0, closure.scale[:, None])

    def split(name: str) -> tuple[np.ndarray, np.ndarray]:
        soft = getattr(below, name) * outer_scale
        hard = np.where(inner, getattr(closure.moments, name), getattr(above, name) * outer_scale)
        return soft, hard

    channels = ("distant_longitudinal", "distant_transverse", "close")
    soft, hard = zip(*(split(name) for name in channels), strict=True)
    return ShellSoftHardPartition(
        closure,
        float(cutoff_eV),
        ShellGOSMoments(energy, raw.oscillators, raw.density_effect, *soft),
        ShellGOSMoments(energy, raw.oscillators, raw.density_effect, *hard),
    )


def catalog_shell_partition(
    key: str,
    energy_eV: float,
    cutoff_eV: float,
    *,
    inner_threshold_eV: float = DEFAULT_INNER_SHELL_THRESHOLD_EV,
) -> ShellSoftHardPartition:
    """Soft/hard partition of :func:`~.shell_rates.catalog_shell_rate_closure`.

    Validation: penelope-shell-soft-hard-partition
    """
    material = catalog_shell_oscillators(key)
    closure = catalog_shell_rate_closure(key, energy_eV, inner_threshold_eV=inner_threshold_eV)
    return partition_shell_rates(material, closure, cutoff_eV)
