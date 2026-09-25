"""Per-layer element tables the transport cores consume.

Host-side only: turns a ``(z_top, z_bottom, composition)`` layer stack into
the ragged per-layer arrays and the grouped tabulated-elastic argument, with
the hot-loop coefficients hoisted out of the cores.
"""

import logging

import numpy as np

from ...materials._transport_data import TRANSPORT_ELEMENTS
from .scattering import _NO_MOTT, _flatten_mott_tables, _mott_alpha_table, pack_elsepa_tables
from .stopping import _element_crossover_keV

logger = logging.getLogger(__name__)


def build_layer_tables(layers, elastic_model, elastic_tables):
    """Return the per-layer element arrays and the grouped elastic tables.

    ``mott_group`` carries both tabulated elastic models: the NIST-Mott
    screening tables, then the packed ELSEPA group, which the cores read only
    when ``elastic_model == "elsepa"``.
    """
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

            table = None
            if elastic_model == "mott" and el not in _NO_MOTT:
                try:
                    table = _mott_alpha_table(el, Z_i)
                except FileNotFoundError:
                    logger.debug("No NIST Mott table for %s; using analytic SR angles", el)
                    _NO_MOTT.add(el)
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

    elsepa_group = pack_elsepa_tables(elastic_tables, L_ncm3)
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
