"""Exact per-point escape attenuation for the batched layered/grooved route.

Layered and grooved escape evaluates ``mu`` exactly at each line energy (the
single-slab route's log-log elemental tabulation, ``line-absorption-tabulation``,
is not applied there). The compatibility loop gets that value from
:func:`~pyrite.materials.attenuation._mu_total_inv_ang`, a host call per
(reflection, orientation) row and layer. Here the same function is folded once
per composition into a device-resident piecewise-linear table, so a whole
``(n_block, N_g)`` block of line energies is evaluated in a few launches.

Each EPDL2025 channel is lin-lin between its knots, zero below its first knot,
right-continuous at a repeated (edge) knot, and continues its last interval
above its last knot. A weighted sum of such functions is again lin-lin on the
union of every channel's knots: on each union interval every channel is a
single line. The table stores, per union knot ``u_j``, the right-continuous
value ``F(u_j)`` and the slope of the interval starting there, both formed
in float64 from each channel's own interval, so evaluation reproduces
``sum_i n_i sigma_i(E)`` to float64 rounding before the cast to ``REAL``.
Outside ``[EPDL_E_MIN_EV, EPDL_E_MAX_EV]`` the value is NaN, as on the host.

Validation: narrow-beam-total-attenuation
Validation: segment-escape-average
Validation: line-hkl-batch
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...._backend import REAL, xp
from ....materials.atomic import Z_TABLE
from ....materials.photon_cross_sections import (
    EPDL_E_MAX_EV,
    EPDL_E_MIN_EV,
    PHOTON_CHANNELS,
    _table,
)


@dataclass(frozen=True)
class ExactMuTable:
    """Device piecewise-linear ``mu(E)`` [1/Angstrom] of one composition."""

    knots: Any  # (n,) float64 union knots [eV], ascending, distinct
    value: Any  # (n,) float64 right-continuous mu at each knot
    slope: Any  # (n,) float64 d mu / dE on [knot_j, knot_{j+1})

    def __call__(self, energy_eV):
        """``mu`` at ``energy_eV`` (any shape), as ``REAL`` on the table's device."""
        E = energy_eV.astype(np.float64)
        j = xp.clip(xp.searchsorted(self.knots, E, side="right") - 1, 0, self.knots.size - 1)
        mu = self.value[j] + (E - self.knots[j]) * self.slope[j]
        mu = xp.where(E < self.knots[0], 0.0, mu)
        mu = xp.where((E >= EPDL_E_MIN_EV) & (E <= EPDL_E_MAX_EV), mu, np.nan)
        return mu.astype(REAL)


def exact_mu_table(composition):
    """Fold ``composition`` [(element, n_i [1/Angstrom^3])] into an :class:`ExactMuTable`.

    Validation: narrow-beam-total-attenuation
    """
    tables = _table()
    channels = [
        (float(n_i), tables[Z_TABLE[element]][mt])
        for element, n_i in composition
        for mt in PHOTON_CHANNELS.values()
    ]
    knots = np.unique(np.concatenate([e for _, (e, _s) in channels]))
    # A point inside each union interval selects every channel's interval
    # there; past the last knot each channel continues its last interval.
    probe = np.append(0.5 * (knots[:-1] + knots[1:]), 2.0 * knots[-1])
    value = np.zeros_like(knots)
    slope = np.zeros_like(knots)
    for n_i, (e, s) in channels:
        i = np.clip(np.searchsorted(e, probe, side="right") - 1, 0, e.size - 2)
        live = probe >= e[0]
        # A threshold edge pair sits at e[0:2]; dead rows discard its 0/0.
        with np.errstate(divide="ignore", invalid="ignore"):
            ds = np.where(live, (s[i + 1] - s[i]) / (e[i + 1] - e[i]), 0.0)
        value += n_i * np.where(live, s[i] + (knots - e[i]) * ds, 0.0)
        slope += n_i * ds
    return ExactMuTable(
        knots=xp.asarray(knots, dtype=np.float64),
        value=xp.asarray(value, dtype=np.float64),
        slope=xp.asarray(slope, dtype=np.float64),
    )


def escape_mu_tables(request, abs_comp):
    """Per-layer exact tables for a layered or grooved request, else ``None``.

    A groove has one absorber (the crystal); a stack has one table per layer,
    in the layer order of ``segment_escape_pieces``.
    """
    if request.layers is not None:
        return tuple(exact_mu_table(comp) for _, _, comp in request.layers)
    if request.groove is not None:
        return (exact_mu_table(abs_comp),)
    return None


def escape_mu(tables, energy_eV):
    """``(..., K)`` per-layer ``mu`` at ``energy_eV`` for ``piece_mean_transmission``."""
    return xp.stack([table(energy_eV) for table in tables], axis=-1)
