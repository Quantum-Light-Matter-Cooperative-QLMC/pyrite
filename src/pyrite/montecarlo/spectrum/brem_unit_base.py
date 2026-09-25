"""EEDL bremsstrahlung photon spectra refined by unit-base interpolation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .brem import BremsstrahlungCrossSectionTable

# Unit-base sub-panels inserted per decade of incident energy between native
# EEDL spectrum panels; see ``_unit_base_panels``.
_EEDL_PANELS_PER_DECADE = 32


@dataclass(frozen=True, slots=True)
class _UnitBasePanels:
    """EEDL photon spectra on a refined incident grid, in unit-base form."""

    incident_energy_eV: np.ndarray
    x: np.ndarray
    photon_min_eV: np.ndarray
    photon_max_eV: np.ndarray
    scaled_density: np.ndarray
    scaled_cumulative: np.ndarray


def _unit_base_panels(table: BremsstrahlungCrossSectionTable) -> _UnitBasePanels:
    """Refine the EEDL incident grid by ENDF unit-base interpolation.

    EEDL tabulates each element's photon spectrum at only 8--10 decade-spaced
    incident energies. Interpolating those panels at fixed photon energy (the
    declared Cartesian law) gives every ``k`` above the lower panel's endpoint
    only the upper panel's share, collapsing the spectrum towards ``k -> T``.
    Unit-base interpolation (ENDF-6 Formats Manual, section 0.5.2.2) instead
    maps panel ``i`` onto ``x = (k - a_i)/(b_i - a_i)`` in ``[0, 1]``, where
    ``[a_i, b_i]`` is its photon range and its density becomes the unit-area
    ``q_i(x) = (b_i - a_i) p_i(k)``. Sub-panels at geometrically spaced ``T``
    take ``q = (1 - w) q_i + w q_{i+1}`` with ``w`` linear in ``ln T``, the
    spacing of the native panels, while ``a, b`` are interpolated linearly in
    ``T`` (INT=2), so ``b = T`` wherever the native panels end at their
    incident energy. Against the Seltzer--Berger tables the ``ln T`` shape
    weight is several times more accurate between panels than a weight
    linear in ``T``. Every panel is held on the union of
    the native ``x`` nodes, which represents each native panel exactly and
    keeps every mixture piecewise linear with unit area. Native panels are
    reproduced unchanged; runtime interpolation between the closely spaced
    sub-panels stays Cartesian, with each sub-panel held at its endpoint
    density up to the next sub-panel's endpoint (see ``_prepare_eedl_grid``).

    Validation: brem-spectrum
    """
    lows = np.array([energy[0] for energy in table.photon_energy_eV_by_incident])
    highs = np.array([energy[-1] for energy in table.photon_energy_eV_by_incident])
    native_x = [
        (energy - low) / (high - low)
        for energy, low, high in zip(table.photon_energy_eV_by_incident, lows, highs, strict=True)
    ]
    x = np.unique(np.concatenate(native_x))  # exactly 0 and 1 at the ends
    native = np.stack(
        [
            np.interp(x, panel_x, density * (high - low))
            for panel_x, density, low, high in zip(
                native_x,
                table.photon_probability_density_per_eV_by_incident,
                lows,
                highs,
                strict=True,
            )
        ]
    )

    incident = table.distribution_incident_energy_eV
    energy_parts, weight_parts, range_parts, lower_parts = [], [], [], []
    for panel in range(incident.size - 1):
        decades = np.log10(incident[panel + 1] / incident[panel])
        count = max(1, int(np.ceil(_EEDL_PANELS_PER_DECADE * decades)))
        energy = incident[panel] * (incident[panel + 1] / incident[panel]) ** (
            np.arange(count) / count
        )
        energy_parts.append(energy)
        weight_parts.append(np.arange(count) / count)
        range_parts.append((energy - incident[panel]) / (incident[panel + 1] - incident[panel]))
        lower_parts.append(np.full(count, panel))
    energy = np.concatenate([*energy_parts, incident[-1:]])
    weight = np.concatenate([*weight_parts, [0.0]])
    range_weight = np.concatenate([*range_parts, [0.0]])
    lower = np.concatenate([*lower_parts, [incident.size - 1]])
    upper = np.minimum(lower + 1, incident.size - 1)
    scaled = (1.0 - weight)[:, None] * native[lower] + weight[:, None] * native[upper]
    cumulative = np.concatenate(
        (
            np.zeros((scaled.shape[0], 1)),
            np.cumsum(0.5 * (scaled[:, :-1] + scaled[:, 1:]) * np.diff(x), axis=1),
        ),
        axis=1,
    )
    return _UnitBasePanels(
        incident_energy_eV=energy,
        x=x,
        photon_min_eV=(1.0 - range_weight) * lows[lower] + range_weight * lows[upper],
        photon_max_eV=(1.0 - range_weight) * highs[lower] + range_weight * highs[upper],
        scaled_density=scaled,
        scaled_cumulative=cumulative,
    )
