"""The positive low-energy floor of a photon-continuum grid.

A logarithmic or geometric grid cannot contain zero, so it needs a strictly
positive lower node. That bound is a physics choice: the bremsstrahlung
infrared rise makes the integrated background depend on it, so an arbitrarily
small epsilon chosen to keep ``log`` finite is an unexamined infrared cutoff
that silently sets a result. See
[Energy-grid semantics](../../docs/physics/radiation-physics/energy-grid-semantics.md).

The floor here is the larger of two independently derived bounds:

* the **modelled band** -- the medium's bulk plasma energy ``hbar*omega_p``,
  below which this repository's own transparent-medium X-ray optics has
  collapsed (``materials/attenuation.py::plasma_energy_eV`` carries the
  derivation, the assumptions, and the limiting case); and
* **data support** -- the lowest energy at which every table the continuum
  pipeline evaluates carries a real tabulated value rather than an
  extrapolation, measured from the packaged data and recorded in
  :data:`DATA_SUPPORT_LIMITS_EV`.

For every condensed medium in the catalog the plasma energy is the binding
term (the smallest catalog value is ~30.2 eV, against a 12 eV data-support
floor), so the floor is a derived, material-specific number rather than a round
one. The data-support term binds only in the dilute limit, which is exactly the
limiting case ``plasma_energy_eV`` is checked against.

Choosing where the nodes go *between* the floor and the ceiling -- refinement
near absorption edges and kinematic endpoints -- is a separate concern and is
deliberately not done here.
"""

from __future__ import annotations

import numpy as np

from ..materials import MediumSpec
from ..materials.attenuation import plasma_energy_eV

__all__ = [
    "DATA_SUPPORT_LIMITS_EV",
    "data_support_floor_eV",
    "geometric_continuum_grid",
    "photon_continuum_floor_eV",
]

#: Lowest energy [eV] at which each table the continuum pipeline evaluates has
#: real support. Every value is measured from the packaged data, not asserted:
#:
#: * ``eedl_photon_spectra`` -- the smallest photon energy in MF=26/MT=527 over
#:   every element in ``montecarlo.transport.TRANSPORT_ELEMENTS``; EEDL
#:   tabulates all of them from 0.1 eV, so bremsstrahlung emission itself is
#:   never the binding constraint.
#: * ``chantler_scattering_factors`` -- ``min(xraydb.chantler_energies(el))``,
#:   which is 1.01 eV for every element. ``materials/atomic.py::
#:   henke_dispersion`` admits the *strict* interior ``E > Emin``, so 1.01 eV
#:   itself reads as out of range and returns NaN.
#: * ``eaglexo_quantum_efficiency`` -- the first row of ``data/eaglexo_qe.csv``,
#:   whose header documents the digitized curve as valid over 12 eV - 29 keV;
#:   below it ``eaglexo_response.qe`` falls back to a thin-Si absorption
#:   extrapolation rather than measured ordinates.
DATA_SUPPORT_LIMITS_EV: dict[str, float] = {
    "eedl_photon_spectra": 0.1,
    "chantler_scattering_factors": 1.01,
    "eaglexo_quantum_efficiency": 12.0,
}


def data_support_floor_eV() -> float:
    """Lowest energy at which *every* continuum table has real support, in eV."""
    return max(DATA_SUPPORT_LIMITS_EV.values())


def photon_continuum_floor_eV(material: str | MediumSpec) -> float:
    """Lowest photon energy a continuum grid over ``material`` may carry, in eV.

    The larger of the medium's plasma energy and :func:`data_support_floor_eV`.
    Neither term is a tuning knob: the first is fixed by the medium's own
    catalog number densities, the second by where the packaged tables stop.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium.

    Returns
    -------
    float
        Strictly positive floor energy in eV.

    Validation: photon-continuum-floor
    """
    return max(plasma_energy_eV(material), data_support_floor_eV())


def geometric_continuum_grid(
    material: str | MediumSpec,
    stop_eV: float,
    num: int,
    *,
    floor_eV: float | None = None,
) -> np.ndarray:
    """Geometric continuum nodes from the derived floor to ``stop_eV``.

    The geometric baseline only -- no refinement near absorption edges or
    kinematic endpoints, which is a separate concern.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium.
    stop_eV
        Highest node energy, strictly above the floor.
    num
        Endpoint-inclusive node count, at least 2.
    floor_eV
        Override for the lowest node. Refused below
        :func:`photon_continuum_floor_eV`, because a lower bound would place
        nodes where the emission and escape models carry no validity; passing a
        *higher* value is an ordinary bandwidth choice and is allowed.

    Returns
    -------
    numpy.ndarray
        Strictly increasing nodes in eV, ``E[0]`` exactly the resolved floor.
    """
    derived = photon_continuum_floor_eV(material)
    if floor_eV is None:
        floor = derived
    else:
        floor = float(floor_eV)
        if not np.isfinite(floor):
            raise ValueError("floor_eV must be finite")
        if floor < derived:
            raise ValueError(
                f"floor_eV {floor:.6g} eV is below the derived continuum floor "
                f"{derived:.6g} eV for this medium; the continuum and photon-escape "
                "models carry no validity there, so widening the band downward needs a "
                "physics decision, not a smaller number"
            )
    stop = float(stop_eV)
    if not np.isfinite(stop) or stop <= floor:
        raise ValueError(f"stop_eV must be finite and above the floor {floor:.6g} eV")
    if int(num) != num or int(num) < 2:
        raise ValueError("num must be an integer of at least 2")
    return np.geomspace(floor, stop, int(num))
