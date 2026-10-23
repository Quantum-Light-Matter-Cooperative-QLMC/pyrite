"""Fill BremsLib's holes between its complete grid and 100 MeV.

Above 30 MeV BremsLib v2.0.8 tabulates the scaled SDCS ``chi(T, x)`` only for
``x = k/T <= 0.95`` (at 50, 60, 80 and 100 MeV) and the DDCS only at
``k = 0``. :func:`extend_above_complete` fills the rest of those *extension
rows*, anchored on library data at every step:

**SDCS tip** (``x = 0.975`` and the top node). The ratio to the last library
column is held at its value on the last complete row ``T_c``:

    chi(T, x_m) = chi(T, 0.95) * chi(T_c, x_m) / chi(T_c, 0.95).

An analytic high-energy (Bethe--Heitler, Thomas--Fermi screened, Coulomb
corrected) form cannot supply this ratio: it is a Born result, invalid for
the slow outgoing electron at the tip, and turns negative there for heavy
elements (W at ``x = 0.9999`` from ~10 MeV). The two tip cells carry ~2 % of
the radiative first moment, so a 30 % error in their height moves the
radiative stopping by well under 1 %.

**Angular shape for x > 0** (method A in the #439 back-test). With ``S_BL``
the library's unit-normalized shape and ``S_2BS`` the Schiff (Koch--Motz 2BS)
density,

    S(T, x, theta) ∝ S_BL(T, 0, theta) * [S_2BS(T, x, theta) / S_2BS(T, 0, theta)]
                     * C(x, theta),
    C(x, theta) = S_BL(T_c, x, theta) / (S_BL(T_c, 0, theta)
                  * S_2BS(T_c, x, theta) / S_2BS(T_c, 0, theta)),

renormalized to unit solid-angle integral. It is the library's ``k = 0``
shape at ``T`` exactly, takes its ``x``-dependence from Schiff, and equals the
library shape at ``T_c`` (``C`` is the residual between the two there).

Assumptions: the residual ``C`` is independent of ``T`` above ``T_c``; the
Schiff ratio carries the energy dependence. Limiting cases: ``x -> 0``
recovers the ``k = 0`` library node; ``T -> T_c`` recovers the complete row.
Predicting 30 MeV from 15 MeV this way reproduces the library's half-flux
and 90 % angles within 1--3 % (5--9 % at ``x = 0.975``) for C and W; the
large-angle (> 1 rad) tail is within 10--55 %. Photons in the keV band
(``x ~ 1e-4``) take essentially the exact ``k = 0`` shape.

Validation: bremslib-high-energy-extension
"""

import numpy as np

ELECTRON_REST_ENERGY_MEV = 0.51099895

#: Leading ``k/T1`` columns the library fills at an extension energy.
_LIBRARY_COLUMNS = 11


def schiff_density(
    atomic_number: int, incident_energy_MeV: float, kappa: float, theta_rad: np.ndarray
) -> np.ndarray:
    """Unnormalized Schiff photon density per steradian (Koch-Motz 2BS).

    Koch and Motz, Rev. Mod. Phys. 31, 920 (1959), Formula 2BS; Schiff, Phys.
    Rev. 70, 87 (1946)::

        dsigma/(dk dOmega) ~ 16 y^2 E / ((y^2+1)^4 E0) - (E0+E)^2 / ((y^2+1)^2 E0^2)
            + [(E0^2+E^2) / ((y^2+1)^2 E0^2) - 4 y^2 E / ((y^2+1)^4 E0)] ln M(y),
        y = E0 theta,   1/M = (k/(2 E0 E))^2 + (Z^(1/3) / (111 (y^2+1)))^2,

    with total energies ``E0 = 1 + T/mc^2`` and ``E = E0 - k/mc^2`` and ``k``
    in units of ``mc^2``. Born approximation, Thomas--Fermi screening, no
    electron-electron term, small angles; ``Z -> 0`` in ``M`` gives 2BN(a).

    Validation: bremslib-angular-schiff, bremslib-high-energy-extension
    """
    E0 = 1.0 + incident_energy_MeV / ELECTRON_REST_ENERGY_MEV
    k = kappa * incident_energy_MeV / ELECTRON_REST_ENERGY_MEV
    E = E0 - k
    y2 = (E0 * np.asarray(theta_rad, dtype=float)) ** 2
    d = y2 + 1.0
    inv_M = (k / (2.0 * E0 * E)) ** 2 + (atomic_number ** (1.0 / 3.0) / (111.0 * d)) ** 2
    return (
        16.0 * y2 * E / (d**4 * E0)
        - (E0 + E) ** 2 / (d**2 * E0**2)
        + ((E0**2 + E**2) / (d**2 * E0**2) - 4.0 * y2 * E / (d**4 * E0)) * -np.log(inv_M)
    )


def extend_above_complete(
    t1_MeV: np.ndarray,
    reduced_energy: np.ndarray,
    sdcs: np.ndarray,
    shape: np.ndarray,
    theta_rad: np.ndarray,
    complete_rows: int,
    atomic_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the extension rows' SDCS tip and ``x > 0`` angular shapes.

    ``reduced_energy`` is ``(n_T1, 13)``: the actual ``k/T1`` of every node,
    the top one slightly below 1. ``shape`` holds unit-normalized shapes,
    ``(n_T1, 13, n_theta)``; only column 0 is read on an extension row.
    Returns new ``sdcs`` and ``shape`` arrays; complete rows are unchanged
    and the filled ``x > 0`` shapes are left for the caller to normalize.

    Validation: bremslib-high-energy-extension

    Raises
    ------
    ValueError
        If there is no complete row to anchor on, a library value on an
        extension row is missing, or the Schiff density is not positive.
    """
    c = int(complete_rows) - 1
    if c < 0:
        raise ValueError("a BremsLib extension needs a complete row to anchor on")
    sdcs = np.array(sdcs, dtype=np.float64)
    shape = np.array(shape, dtype=np.float64)
    library = sdcs[c + 1 :, :_LIBRARY_COLUMNS]
    if not np.all(np.isfinite(library)) or np.any(library <= 0.0):
        raise ValueError("a BremsLib extension row lacks its k/T1 <= 0.95 SDCS")
    tip_ratio = sdcs[c, _LIBRARY_COLUMNS:] / sdcs[c, _LIBRARY_COLUMNS - 1]
    sdcs[c + 1 :, _LIBRARY_COLUMNS:] = sdcs[c + 1 :, _LIBRARY_COLUMNS - 1, None] * tip_ratio

    def schiff_ratios(row: int) -> np.ndarray:
        energy = float(t1_MeV[row])
        base = schiff_density(atomic_number, energy, 0.0, theta_rad)
        ratios = np.stack(
            [
                schiff_density(atomic_number, energy, float(x), theta_rad) / base
                for x in reduced_energy[row, 1:]
            ]
        )
        if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
            raise ValueError(f"Schiff density is not positive at T1 = {energy:g} MeV")
        return ratios

    anchor = shape[c, 0] * schiff_ratios(c)
    residual = np.divide(shape[c, 1:], anchor, out=np.zeros_like(anchor), where=anchor > 0.0)
    for row in range(c + 1, t1_MeV.size):
        shape[row, 1:] = shape[row, 0] * schiff_ratios(row) * residual
    return sdcs, shape


__all__ = ["ELECTRON_REST_ENERGY_MEV", "extend_above_complete", "schiff_density"]
