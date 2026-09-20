"""The positive low-energy floor of a photon-continuum grid.

This package-root leaf is shared by ``campaign`` (which resolves a declared
band against the medium's floor when it builds cases) and by the
``energy_grid`` driver package, without making the former depend on the
latter -- the same reason ``_energy_grid_encoding`` and ``_line_grid_policy``
sit here. :mod:`pyrite.energy_grid.floor` re-exports everything below and
adds the grid builder that only that package needs.

A logarithmic or geometric grid cannot contain zero, so it needs a strictly
positive lower node. That bound is a physics choice: the bremsstrahlung
infrared rise makes the integrated background depend on it, so an arbitrarily
small epsilon chosen to keep ``log`` finite is an unexamined infrared cutoff
that silently sets a result. See
[Energy-grid semantics](../docs/physics/radiation-physics/energy-grid-semantics.md).

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

from .materials import MediumSpec, load_material_catalog
from .materials.attenuation import plasma_energy_eV

__all__ = [
    "DATA_SUPPORT_LIMITS_EV",
    "continuum_medium_key",
    "data_support_floor_eV",
    "floored_lattice_start_eV",
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


def continuum_medium_key(material: str | MediumSpec) -> str | MediumSpec:
    """The medium whose floor bounds ``material``'s photon continuum.

    A catalog *material* need not name a medium the attenuation tables know:
    ``mos2-on-sapphire`` and ``mos2-on-sio2-si`` are film-on-substrate entries
    whose emitting crystal is ``mos2``. The continuum is produced in, and its
    low-energy validity set by, that entry's own crystal
    (``MaterialSpec.crystal_key``), so catalog entries resolve through it.
    Anything the medium resolver already accepts -- a bare crystal or media
    key, or an explicit :class:`~pyrite.materials.MediumSpec` -- passes through
    unchanged.

    This is a naming lookup, not a physics choice: the substrate a photon
    escapes through has its own plasma energy, and a stack whose *substrate*
    floor exceeded its film's would need that decided rather than resolved
    here. No catalog stack is in that position today.
    """
    if not isinstance(material, str):
        return material
    try:
        spec = load_material_catalog().material(material)
    except KeyError:
        return material
    return spec.crystal_key


def floored_lattice_start_eV(material: str | MediumSpec, step_eV: float) -> float:
    """Lowest multiple of ``step_eV`` at or above ``material``'s derived floor.

    A uniform grid over the continuum keeps the node *coordinates* it always
    had -- absolute multiples of ``step_eV`` -- and simply starts above
    :func:`photon_continuum_floor_eV` rather than at ``0.0``, which sits below
    the band the emission and escape models are valid over. Snapping to the
    lattice rather than starting at the floor itself is what leaves every
    surviving node, and so any quantile measured on it, exactly where it was.

    Shared by the diagnostic band
    (:func:`~pyrite.energy_grid.derive.wide_brem_grid`) and the installed
    production grid (:mod:`pyrite.energy_grid.apply`), which must agree about
    where a medium's modelled band begins.
    """
    step = float(step_eV)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_eV must be a finite positive number")
    floor = photon_continuum_floor_eV(continuum_medium_key(material))
    return float(step * np.ceil(floor / step))
