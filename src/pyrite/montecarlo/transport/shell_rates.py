"""Host-side PENELOPE-2024 inner-shell rate substitution and stopping closure.

Rescales the raw shell GOS moments of :mod:`.shell_gos` so inner shells carry
EEDL ionization cross sections and the material first moment equals the
adopted corrected SBETHE stopping. Nothing here is sampled or used by a
transport mode.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import xraydb

from ...materials.attenuation import plasma_energy_eV
from ...xsgen.sbethe.catalog import catalog_material, resolve_catalog_table
from ..eedl_ionization import EEDL_SUBSHELL_LABELS
from ..shell_configuration import AtomicShell, load_atomic_shells, match_eedl_shells
from ..shell_ionization import material_shell_ionization_rates
from .shell_gos import ShellGOSMoments, shell_gos_moments
from .shell_oscillators import (
    MaterialShellOscillators,
    ShellOscillator,
    build_shell_oscillators,
    load_conduction_bands,
)

# Equals spectrum.characteristic._MIN_RELAXATION_CUTOFF_EV (50 eV), the
# PENELOPE lower bound of Eq. 2.112; a test keeps the two in step.
DEFAULT_INNER_SHELL_THRESHOLD_EV = 50.0
_INNER_SHELL_SERIES = frozenset("KLMN")
_OUTER_SHELL_SERIES = frozenset("OPQ")


@dataclass(frozen=True, slots=True)
class ShellRateClosure:
    """Inner-shell substituted, stopping-closed moments at one energy.

    ``moments`` has the layout of ``raw``, with every channel of oscillator
    ``k`` multiplied by ``scale[k]``: the inner-shell factor where ``inner``
    is true, ``outer_scale`` (``N(E)``) elsewhere. ``density_ratio`` is the
    GOS ``sigma^(0)`` ratio with/without ``delta_F`` (1 for outer shells);
    ``adopted_inner_cm2`` is the EEDL cross section times that ratio (0 for
    outer shells). Units per formula unit, as in :class:`ShellGOSMoments`.
    """

    raw: ShellGOSMoments
    moments: ShellGOSMoments
    inner: np.ndarray
    density_ratio: np.ndarray
    adopted_inner_cm2: np.ndarray
    scale: np.ndarray
    outer_scale: float
    stopping_eV_cm2: float


def inner_shell_cutoff_eV(
    composition: Mapping[int, float],
    shells: Mapping[int, tuple[AtomicShell, ...]],
    threshold_eV: float = DEFAULT_INNER_SHELL_THRESHOLD_EV,
) -> float:
    """PENELOPE inner-shell cutoff ``E_c = max(threshold, U_max,out(Z_m))``.

    Source: PENELOPE-2024 Eq. 2.112, where ``threshold`` is 50 eV and
    ``U_max,out`` is the largest O/P/Q ionization energy of the heaviest
    element; inner shells are K–N shells with ``U > E_c`` (§2.6, §7.1
    footnote). Limit: light elements have no O/P/Q shell, so
    ``E_c = threshold``.

    Validation: penelope-shell-rate-closure
    """
    if not np.isfinite(threshold_eV) or threshold_eV <= 0.0:
        raise ValueError("inner-shell threshold must be finite and positive")
    heaviest = max(composition)
    if heaviest not in shells:
        raise ValueError(f"no atomic shells for Z={heaviest}")
    outer = [s.ionization_energy_eV for s in shells[heaviest] if s.label[0] in _OUTER_SHELL_SERIES]
    return float(max([threshold_eV, *outer]))


def is_inner_shell(oscillator: ShellOscillator, cutoff_eV: float) -> bool:
    """Whether a bound K–N oscillator lies above the inner-shell cutoff.

    Source: PENELOPE-2024 §2.6 (before Eq. 2.112) and the §7.1 footnote:
    inner shells are K to N shells with ``U > E_c``; the conduction band is
    never inner.

    Validation: penelope-shell-rate-closure
    """
    return (
        oscillator.atomic_number > 0
        and oscillator.label[0] in _INNER_SHELL_SERIES
        and oscillator.ionization_energy_eV > cutoff_eV
    )


def eedl_inner_cross_sections(
    composition: Mapping[int, float],
    shells: Mapping[int, tuple[AtomicShell, ...]],
    material: MaterialShellOscillators,
    energy_eV: float,
    cutoff_eV: float,
) -> dict[tuple[int, str], float]:
    """EEDL ionization cross sections of the inner oscillators, cm² per formula unit.

    Each inner oscillator ``(Z, label)`` takes ``n_Z sum_s sigma_s(E)`` over the
    EEDL subshells joined to its SBETHE shell by
    :func:`~pyrite.montecarlo.shell_configuration.match_eedl_shells`
    (spin-orbit partners that SBETHE leaves empty are summed). Interpolation
    and the no-extrapolation rule are those of
    :func:`~pyrite.montecarlo.shell_ionization.material_shell_ionization_rates`.
    An inner oscillator without an EEDL channel raises.

    Validation: penelope-shell-rate-closure
    """
    inner = [o for o in material.oscillators if is_inner_shell(o, cutoff_eV)]
    adopted: dict[tuple[int, str], float] = {}
    for z in sorted({o.atomic_number for o in inner}):
        symbol = xraydb.atomic_symbol(z)
        rates = material_shell_ionization_rates([(symbol, 1.0)], energy_eV)
        sigma = {
            EEDL_SUBSHELL_LABELS[c.shell_designator]: c.cross_section_cm2 for c in rates.channels
        }
        joined = {m.shell.label: m.eedl_labels for m in match_eedl_shells(symbol, z, shells[z])}
        for osc in (o for o in inner if o.atomic_number == z):
            if osc.label not in joined:
                raise ValueError(f"{symbol} {osc.label}: inner shell has no EEDL channel")
            total = sum(sigma[label] for label in joined[osc.label])
            adopted[(z, osc.label)] = float(composition[z] * total)
    return adopted


def close_shell_rates(
    material: MaterialShellOscillators,
    energy_eV: float,
    stopping_eV_cm2: float,
    inner_cross_sections_cm2: Mapping[tuple[int, str], float],
) -> ShellRateClosure:
    """Substitute inner-shell cross sections and close outer shells to stopping.

    Source: PENELOPE-2024 §3.2.6.1, Eqs. 3.141–3.142. Each inner oscillator
    ``i`` (keys of ``inner_cross_sections_cm2``) takes
    ``sigma_i = sigma_si,i sigma_GOS,i(delta_F)/sigma_GOS,i(0)``; its
    ``sigma^(0..2)`` are multiplied by ``sigma_i/sigma_GOS,i(delta_F)``, so
    its GOS energy-loss PDF is unchanged. All other oscillators are
    multiplied by one ``N(E)`` with
    ``sum_i s_i sigma_i^(1) + N sum_j sigma_j^(1) = S_adopted``.

    Deviation: PENELOPE adopts its own GOS stopping; here ``S_adopted`` is
    the caller's (corrected SBETHE ``stp.dat`` for catalog materials).
    Assumptions: ``delta_F`` enters the shell GOS only via the transverse
    term, so ``sigma_GOS,i(0)`` is evaluated with ``Omega_p = 0`` in
    ``delta_F`` alone (checked below). Limit: no inner shells gives
    ``N = S_adopted/sigma^(1)_GOS``; ``delta_F = 0`` gives a ratio of 1.
    Raises for a non-positive or non-finite ``N``, a non-finite or negative
    inner cross section, an unknown or non-inner key, or a positive
    ``sigma_si`` where the GOS shell cross section vanishes.

    Units: E in eV; stopping in eV cm²; cross sections in cm² per formula unit.
    Validation: penelope-shell-rate-closure
    """
    if not np.isfinite(stopping_eV_cm2) or stopping_eV_cm2 <= 0.0:
        raise ValueError("adopted stopping cross section must be finite and positive")
    index = {
        (o.atomic_number, o.label): k
        for k, o in enumerate(material.oscillators)
        if o.atomic_number > 0
    }
    if len(index) != sum(o.atomic_number > 0 for o in material.oscillators):
        raise ValueError("bound oscillators must have unique (Z, label) keys")
    raw = shell_gos_moments(material, energy_eV)
    bare = shell_gos_moments(replace(material, plasma_energy_eV=0.0), energy_eV)
    if (
        bare.density_effect != 0.0
        or not np.array_equal(bare.distant_longitudinal, raw.distant_longitudinal)
        or not np.array_equal(bare.close, raw.close)
    ):
        raise RuntimeError("shell GOS depends on the plasma energy beyond delta_F")
    sigma0, sigma0_bare = raw.per_shell[:, 0], bare.per_shell[:, 0]
    n = len(material.oscillators)
    inner = np.zeros(n, dtype=bool)
    ratio, adopted, scale = np.ones(n), np.zeros(n), np.zeros(n)
    for key, value in inner_cross_sections_cm2.items():
        if key not in index:
            raise ValueError(f"inner shell {key} is not a bound oscillator")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"inner shell {key}: cross section must be finite and >= 0")
        k = index[key]
        inner[k] = True
        ratio[k] = sigma0[k] / sigma0_bare[k] if sigma0_bare[k] > 0.0 else 1.0
        adopted[k] = value * ratio[k]
        if sigma0[k] > 0.0:
            scale[k] = adopted[k] / sigma0[k]
        elif adopted[k] > 0.0:
            raise ValueError(f"inner shell {key}: GOS has no loss PDF where sigma_si > 0")
    first = raw.per_shell[:, 1]
    outer_stopping = float(np.sum(first[~inner]))
    inner_stopping = float(np.sum(scale[inner] * first[inner]))
    with np.errstate(divide="ignore", invalid="ignore"):
        outer_scale = float(np.divide(stopping_eV_cm2 - inner_stopping, outer_stopping))
    if not np.isfinite(outer_scale) or outer_scale <= 0.0:
        raise ValueError(
            f"outer-shell scale N(E) = {outer_scale} is not finite and positive: "
            "inner-shell stopping alone reaches the adopted stopping"
        )
    scale[~inner] = outer_scale
    factor = scale[:, None]
    moments = ShellGOSMoments(
        raw.energy_eV,
        raw.oscillators,
        raw.density_effect,
        raw.distant_longitudinal * factor,
        raw.distant_transverse * factor,
        raw.close * factor,
    )
    return ShellRateClosure(
        raw, moments, inner, ratio, adopted, scale, outer_scale, float(stopping_eV_cm2)
    )


def adopted_stopping_cs(key: str, energy_eV: float) -> float:
    """Corrected SBETHE ``stp.dat`` stopping, eV cm² per formula unit.

    Log-log interpolation of the catalog table; energies outside its grid
    raise. Owner-adopted stopping for #93 (not PENELOPE's GOS stopping).

    Validation: penelope-shell-rate-closure
    """
    arrays = resolve_catalog_table(key).arrays()
    grid = np.asarray(arrays["stopping_energy_eV"], dtype=np.float64)
    stopping = np.asarray(arrays["stopping_cs_eV_cm2"], dtype=np.float64)
    if not np.isfinite(energy_eV) or energy_eV < grid[0] or energy_eV > grid[-1]:
        raise ValueError(f"{key}: energy is outside the SBETHE stopping table")
    return float(np.exp(np.interp(np.log(energy_eV), np.log(grid), np.log(stopping))))


def catalog_shell_oscillators(key: str) -> MaterialShellOscillators:
    """Measured-conduction-band shell oscillators of a catalog material.

    Catalog composition and ``I``, all-electron ``Omega_p`` and the packaged
    conduction band, as :func:`~.shell_oscillators.build_shell_oscillators`.

    Validation: penelope-shell-oscillators
    """
    inputs = catalog_material(key)
    return build_shell_oscillators(
        inputs.composition,
        inputs.mean_excitation_eV,
        plasma_energy_eV(key),
        load_atomic_shells(),
        load_conduction_bands()[key],
    )


def catalog_shell_rate_closure(
    key: str,
    energy_eV: float,
    *,
    inner_threshold_eV: float = DEFAULT_INNER_SHELL_THRESHOLD_EV,
) -> ShellRateClosure:
    """EEDL-substituted, ``stp.dat``-closed shell moments for a catalog material.

    Builds the measured-conduction-band oscillators, selects inner shells with
    :func:`inner_shell_cutoff_eV`, and applies :func:`close_shell_rates`.

    Validation: penelope-shell-rate-closure
    """
    inputs = catalog_material(key)
    shells = load_atomic_shells()
    material = catalog_shell_oscillators(key)
    stopping = adopted_stopping_cs(key, energy_eV)
    cutoff = inner_shell_cutoff_eV(inputs.composition, shells, inner_threshold_eV)
    sigma = eedl_inner_cross_sections(inputs.composition, shells, material, energy_eV, cutoff)
    return close_shell_rates(material, energy_eV, stopping, sigma)
