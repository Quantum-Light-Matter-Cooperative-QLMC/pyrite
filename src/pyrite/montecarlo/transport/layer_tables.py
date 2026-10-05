"""Per-layer element tables the transport cores consume.

Host-side only: turns a ``(z_top, z_bottom, composition)`` layer stack into
the ragged per-layer arrays and the grouped tabulated-elastic argument, with
the hot-loop coefficients hoisted out of the cores.
"""

import logging

import numpy as np

from ...materials._transport_data import TRANSPORT_ELEMENTS
from .atomic_electrons import (
    atomic_electron_xi,
    elastic_rate_scale,
    validate_atomic_electron_deflection,
)
from .scattering import _flatten_mott_tables, _mott_alpha_table, pack_elsepa_tables
from .stopping import _element_crossover_keV

logger = logging.getLogger(__name__)


def build_layer_tables(
    layers, elastic_model, elastic_tables, atomic_electron_deflection=None, hard_cutoff_eV=None
):
    """Return the per-layer element arrays and the grouped elastic tables.

    ``mott_group`` carries both tabulated elastic models: the NIST-Mott
    screening tables, then the packed ELSEPA group, which the cores read only
    when ``elastic_model == "elsepa"``.

    ``atomic_electron_deflection="kawrakow"`` folds atomic-electron
    deflection into every element's elastic rate as ``1 + xi/Z``
    (``atomic_electrons.py``); every core, the LUT and the element choice read
    the scaled coefficients. ``hard_cutoff_eV`` is the shell-mode ``W_c``
    (``None`` under continuous stopping). There ``xi`` depends on energy,
    which the analytic ``"sr"``/``"mott"`` rate coefficients cannot carry, so
    that combination is rejected. ``None`` leaves every coefficient bit for bit.

    Validation: inelastic-angular-deflection
    """
    atomic_electrons = validate_atomic_electron_deflection(atomic_electron_deflection)
    if atomic_electrons and hard_cutoff_eV is not None and elastic_model != "elsepa":
        raise ValueError(
            "atomic_electron_deflection with inelastic_model='shell-soft-hard' needs "
            "elastic_model='elsepa'; pass atomic_electron_deflection=None to opt out"
        )
    L_Zs = []
    L_Js = []
    L_ncm3 = []
    L_ks = []
    L_coeffs = []
    L_E_cross = []
    mott_tables = []

    for _, _, lc in layers:
        elements = []
        ncm3_arr = []
        Z_arr = []
        J_arr = []
        k_arr = []
        coeff_arr = []
        E_cross_arr = []
        layer_mott_tables = []

        for el, n_i in lc:
            elements.append(el)
            params = TRANSPORT_ELEMENTS[el]
            Z_i = float(params["Z"])
            A_i = float(params["A"])
            J_i = float(params["J_keV"])
            k_i = 0.731 + 0.0688 * np.log10(Z_i)
            coeff_i = (n_i / 0.602214076) * Z_i

            ncm3_arr.append(n_i * 1e24)
            Z_arr.append(Z_i)
            J_arr.append(J_i)
            k_arr.append(k_i)
            coeff_arr.append(coeff_i)
            E_cross_arr.append(_element_crossover_keV(el, Z_i, A_i, J_i))

            # A missing SRD 64 table raises MottTableUnavailableError here,
            # before any transport runs, instead of degrading to SR angles.
            table = _mott_alpha_table(el, Z_i) if elastic_model == "mott" else None
            layer_mott_tables.append(table)

        L_Zs.append(np.asarray(Z_arr, dtype=float))
        L_Js.append(np.asarray(J_arr, dtype=float))
        L_ncm3.append(np.asarray(ncm3_arr, dtype=float))
        L_ks.append(np.asarray(k_arr, dtype=float))
        L_coeffs.append(np.asarray(coeff_arr, dtype=float))
        L_E_cross.append(np.asarray(E_cross_arr, dtype=float))
        mott_tables.append(layer_mott_tables)

    L_sr_rate_numer = []
    L_mott_numer = []
    L_mott_denom1 = []
    L_mott_denom2 = []
    L_sr_joy_numer = []

    for i, Z_i in enumerate(L_Zs):
        n_cm3_i = L_ncm3[i]
        # Rutherford Scattering coefficient hoisted out of hot loop
        L_sr_rate_numer.append(5.21e-21 * Z_i * Z_i * np.float64(4.0) * np.float64(np.pi) * n_cm3_i)

        # Browning fit coefficients to Mott scattering hoisted out of hot loop
        z17 = Z_i ** np.float64(1.7)
        L_mott_numer.append(np.float64(3.0e-18) * z17 * n_cm3_i)
        L_mott_denom1.append(np.float64(0.005) * z17)
        L_mott_denom2.append(np.float64(0.0007) * Z_i * Z_i)

        # Joy-Luo
        L_sr_joy_numer.append(np.float64(3.4e-3) * Z_i ** np.float64(0.67))

    elsepa_scales = None
    if atomic_electrons:
        # The analytic rates only ever see continuous mode (checked above),
        # where xi = xi_0 is constant, so scaling the coefficient is exact.
        for L, Z_i in enumerate(L_Zs):
            scale = elastic_rate_scale(Z_i, atomic_electron_xi(0.0, layers[L][2]))
            L_sr_rate_numer[L] = L_sr_rate_numer[L] * scale
            L_mott_numer[L] = L_mott_numer[L] * scale
        if elastic_tables is not None:
            elsepa_scales = [
                [
                    elastic_rate_scale(
                        Z_i,
                        atomic_electron_xi(table["energy_eV"], layers[L][2], hard_cutoff_eV),
                    )
                    for Z_i, table in zip(L_Zs[L], layer, strict=True)
                ]
                for L, layer in enumerate(elastic_tables)
            ]

    elsepa_group = pack_elsepa_tables(elastic_tables, L_ncm3, elsepa_scales)
    # One grouped argument carries both tabulated elastic models; the cores
    # read the ELSEPA half only when elastic_model_code == 2.
    mott_group = _flatten_mott_tables(mott_tables) + elsepa_group
    return (
        L_Zs,
        L_Js,
        L_ncm3,
        L_ks,
        L_coeffs,
        L_E_cross,
        mott_tables,
        L_sr_rate_numer,
        L_mott_numer,
        L_mott_denom1,
        L_mott_denom2,
        L_sr_joy_numer,
        mott_group,
    )
