"""Opt-in quantity adapters for deterministic, numeric material calculations (#816).

The computational functions keep their historic numeric contracts. These named
adapters require quantities for physical arguments, and return fresh quantities
while preserving complex values, broadcast shapes and unsupported-region NaNs.
Form factors are *relative to one electron*, so their Pint unit is dimensionless:
Pint's ``electron`` is an electrical charge, not an atomic-scattering unit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._units import _is_pint_quantity, normalize_unit


def _input(value: Any, unit: str, name: str, *, kind: str | None = None) -> Any:
    """Explicit quantity-only input; leave scientific domain handling to owner."""
    if not _is_pint_quantity(value):
        raise TypeError(f"{name} requires a Pint quantity in units convertible to {unit}")
    original = np.asarray(value.magnitude)
    if original.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have real-valued magnitudes")
    result = normalize_unit(value, unit, field=name, expected=unit, kind=kind)
    arr = np.asarray(result)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have real-valued magnitudes")
    # Converting long doubles and external registries must not silently narrow
    # a finite nonzero value to zero/inf in the numeric solver's float64 domain.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rounded = arr.astype(float)
    if np.any(np.isfinite(arr) & (~np.isfinite(rounded) | ((arr != 0) & (rounded == 0)))):
        raise ValueError(f"{name} cannot be represented in float64 {unit}")
    return rounded.item() if rounded.ndim == 0 else rounded.copy()


def _output(value: Any, unit: str):
    from ..units import Q_

    return Q_(np.array(value, copy=True), unit)


def _lattice(lattice: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a heterogeneous cell mapping without completing absent axes."""
    if not isinstance(lattice, Mapping):
        raise TypeError("lattice must map physical cell parameters to quantities")
    result = {}
    for key, value in lattice.items():
        if key in ("a", "b", "c"):
            result[key] = _input(value, "angstrom", f"lattice.{key}")
        elif key in ("alpha", "beta", "gamma"):
            result[key] = _input(value, "degree", f"lattice.{key}", kind="angle")
        else:
            result[key] = value
    return result


def crystal_constants() -> dict[str, Any]:
    """Named immutable numeric constants, read back through the canonical owner.

    ``E2_EV_ANG`` is the *Gaussian* ``e² = alpha * hbar*c`` energy-length,
    not SI Coulomb squared. ``M_E_EV`` is the electron *rest energy*.
    Each call returns fresh quantities from the shared registry.
    """
    from . import crystal as c

    return {
        "HC_EV_ANG": _output(c.HC_EV_ANG, "eV*angstrom"),
        "HBARC_EV_ANG": _output(c.HBARC_EV_ANG, "eV*angstrom"),
        "M_E_EV": _output(c.M_E_EV, "eV"),
        "R_E_ANG": _output(c.R_E_ANG, "angstrom"),
        "E2_EV_ANG": _output(c.E2_EV_ANG, "eV*angstrom"),
    }


def cromer_mann_f0_q(element: str, g):
    """Atomic f0 in dimensionless, electron-relative scattering-factor units."""
    from .atomic import cromer_mann_f0

    return _output(cromer_mann_f0(element, _input(g, "1/angstrom", "g")), "dimensionless")


def henke_dispersion_q(element: str, energy, on_out_of_range="nan"):
    """Electron-relative f' and f''; both preserve NaNs and grid shape."""
    from .atomic import henke_dispersion

    fp, fpp = henke_dispersion(element, _input(energy, "eV", "energy"), on_out_of_range)
    return _output(fp, "dimensionless"), _output(fpp, "dimensionless")


def atomic_form_factor_q(element: str, g, energy, on_out_of_range="nan"):
    """Complex electron-relative atomic amplitude on its broadcast grid."""
    from .atomic import atomic_form_factor

    result = atomic_form_factor(
        element,
        _input(g, "1/angstrom", "g"),
        _input(energy, "eV", "energy"),
        on_out_of_range,
    )
    return _output(result, "dimensionless")


def reciprocal_g_vector_q(hkl, lattice: Mapping[str, Any]):
    """Reciprocal three-vector and norm in inverse Å; Miller indices stay integers."""
    from .crystal import reciprocal_g_vector

    vec, magnitude = reciprocal_g_vector(hkl, _lattice(lattice))
    return _output(vec, "1/angstrom"), _output(magnitude, "1/angstrom")


def _crystal_args(energy, b_factor):
    return _input(energy, "eV", "energy"), _input(b_factor, "angstrom**2", "b_factor")


def structure_factor_q(crystal: str, hkl, energy, *, b_factor, use_henke=False):
    """Structure amplitude (electron-relative) and reciprocal magnitude."""
    from .crystal import structure_factor

    e, b = _crystal_args(energy, b_factor)
    sf, g = structure_factor(crystal, hkl, e, B_ang2=b, use_henke=use_henke)
    return _output(sf, "dimensionless"), _output(g, "1/angstrom")


def chi_g_q(crystal: str, hkl, energy, *, b_factor, use_henke=False):
    """Dimensionless, possibly complex dielectric susceptibility."""
    from .crystal import chi_g

    e, b = _crystal_args(energy, b_factor)
    return _output(chi_g(crystal, hkl, e, B_ang2=b, use_henke=use_henke), "dimensionless")


def U_g_q(crystal: str, hkl, energy, *, b_factor, use_henke=False):
    """Complex static crystal-potential Fourier amplitude in eV."""
    from .crystal import U_g

    e, b = _crystal_args(energy, b_factor)
    return _output(U_g(crystal, hkl, e, B_ang2=b, use_henke=use_henke), "eV")


def absorption_length_q(element: str, energy, number_density):
    """Elemental Beer-Lambert intensity length; NaNs/infinities stay intact."""
    from .crystal import absorption_length_ang

    result = absorption_length_ang(
        element,
        _input(energy, "eV", "energy"),
        _input(number_density, "1/angstrom**3", "number_density"),
    )
    return _output(result, "angstrom")


def dominant_reflections_q(
    crystal: str,
    *,
    n_families=4,
    reference_energy,
    b_factor,
    g_max,
    use_henke=False,
    representatives_only=False,
):
    """Ordered dimensionless Miller triples; physics limits accept quantities."""
    from .crystal import dominant_reflections

    return dominant_reflections(
        crystal,
        n_families=n_families,
        E_ref_eV=_input(reference_energy, "eV", "reference_energy"),
        B_ang2=_input(b_factor, "angstrom**2", "b_factor"),
        use_henke=use_henke,
        g_max_invang=_input(g_max, "1/angstrom", "g_max"),
        representatives_only=representatives_only,
    )


def linear_attenuation_q(material, energy):
    """EPDL total narrow-beam attenuation on a 1D photon grid, in 1/mm."""
    from .attenuation import linear_attenuation_inv_mm

    return _output(linear_attenuation_inv_mm(material, _input(energy, "eV", "energy")), "1/mm")


def photon_energies_q(
    materials,
    beam_energy,
    *,
    tilt,
    tilt_azimuth,
    observation_polar,
    observation_azimuth,
    hkl=None,
    mode="broadcast",
    catalog=None,
):
    """Vacuum resonant CXR photon energies, preserving broadcast/grid semantics."""
    from .emission import photon_energies

    result = photon_energies(
        materials,
        _input(beam_energy, "eV", "beam_energy"),
        tilt_deg=_input(tilt, "degree", "tilt", kind="angle"),
        tilt_azim_deg=_input(tilt_azimuth, "degree", "tilt_azimuth", kind="angle"),
        theta_obs_deg=_input(observation_polar, "degree", "observation_polar", kind="angle"),
        phi_obs_deg=_input(observation_azimuth, "degree", "observation_azimuth", kind="angle"),
        hkl=hkl,
        mode=mode,
        catalog=catalog,
    )
    return _output(result, "eV")


__all__ = [
    "crystal_constants",
    "cromer_mann_f0_q",
    "henke_dispersion_q",
    "atomic_form_factor_q",
    "reciprocal_g_vector_q",
    "structure_factor_q",
    "chi_g_q",
    "U_g_q",
    "absorption_length_q",
    "dominant_reflections_q",
    "linear_attenuation_q",
    "photon_energies_q",
]
