"""
materials.attenuation

Composition handling and X-ray self-absorption shared across transport,
spectrum and detector: normalize a single-element / compound material spec,
the total narrow-beam linear attenuation summed over elements
(photoabsorption + coherent + incoherent), and the layered
(film-on-substrate) Beer-Lambert optical depth.
"""

import numpy as np
from scipy.constants import electron_mass as _M_E
from scipy.constants import elementary_charge as _E_CHARGE
from scipy.constants import epsilon_0 as _EPS_0
from scipy.constants import hbar as _HBAR

from . import CATALOG, MediumSpec
from .atomic import Z_TABLE
from .photon_cross_sections import (
    EPDL_E_MAX_EV,
    EPDL_E_MIN_EV,
    total_photon_cross_section_ang2,
)


def linear_attenuation_inv_mm(material: str | MediumSpec, energy_eV: object) -> np.ndarray:
    """Return total linear X-ray attenuation on a positive energy grid [mm^-1].

    This is the *narrow-beam* coefficient: the probability per unit path that a
    photon is removed from the unscattered ray by any channel. Element
    contributions add as

        ``mu(E) = sum_i n_i sigma_tot,i(E)``,

    with ``sigma_tot`` the EPDL2025 per-atom total (photoelectric + coherent +
    incoherent + pair production) of
    :func:`~pyrite.materials.photon_cross_sections.total_photon_cross_section_ang2`.
    It is homogeneous and passive; this helper adds no fluorescence, diffraction,
    secondary production, or build-up factor, so it assumes good geometry --
    scattered photons leave the collection solid angle rather than reaching the
    pixel. As every elemental coefficient tends to zero, so does the returned
    coefficient.

    ``material`` is either a catalog crystal/medium key or an explicit
    :class:`~pyrite.materials.catalog.MediumSpec`. Runnable target-material
    keys are intentionally rejected because their film/stack composition is
    not a single homogeneous filter medium.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium.
    energy_eV
        Non-empty one-dimensional array of finite positive photon energies in eV.

    Returns
    -------
    numpy.ndarray
        Read-only attenuation coefficients in inverse mm, shaped like
        ``energy_eV``.

    Raises
    ------
    TypeError
        If the energy grid or material has an incompatible type.
    ValueError
        If energies, composition, or catalog key are invalid.

    Validation: positioned-filter-attenuation
    """
    try:
        energy = np.asarray(energy_eV, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("energy_eV must be a one-dimensional real array") from exc
    if energy.ndim != 1:
        raise ValueError("energy_eV must be one-dimensional")
    if energy.size == 0:
        raise ValueError("energy_eV must not be empty")
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("energy_eV must contain only finite positive values")

    composition = _resolve_composition(material)
    coefficient = np.asarray(_mu_total_inv_ang(composition, energy), dtype=float) * 1.0e7
    coefficient.setflags(write=False)
    return coefficient


def _resolve_composition(material: str | MediumSpec) -> tuple[tuple[str, float], ...]:
    """Catalog key or explicit medium to a validated ``(element, n)`` list.

    Runnable target-material keys are intentionally unreachable: their
    film/stack composition is not a single homogeneous medium.
    """
    if isinstance(material, str):
        if material in CATALOG.media:
            composition = CATALOG.media[material].composition
        elif material in CATALOG.crystals:
            composition = CATALOG.crystals[material].composition
        else:
            raise ValueError(
                f"unknown filter material {material!r}; expected a catalog crystal or medium key"
            )
    elif isinstance(material, MediumSpec):
        composition = material.composition
    else:
        raise TypeError("material must be a catalog key or MediumSpec")

    if not composition or any(
        not element or not np.isfinite(float(density)) or float(density) <= 0.0
        for element, density in composition
    ):
        raise ValueError("filter composition must contain positive element number densities")
    return composition


def plasma_energy_eV(material: str | MediumSpec) -> float:
    r"""Bulk free-electron plasma energy ``hbar*omega_p`` of a medium, in eV.

    This is the low-energy edge of the band in which this repository's X-ray
    optics is meaningful at all, so it is what sets the photon-continuum grid
    floor (``_photon_continuum_floor.py``); it is not itself part of any transport or
    emission kernel.

    Source equation. For an electron gas of number density ``n_e`` the Drude
    (collisionless, free-electron) dielectric function is
    ``eps(omega) = 1 - omega_p^2 / omega^2`` with

        omega_p = sqrt(n_e e^2 / (eps_0 m_e)),

    so the plasma energy is ``hbar*omega_p``. Jackson, *Classical
    Electrodynamics* 3rd ed., Sec. 7.5 (plasma frequency of a free-electron
    medium); the same quantity appears as ``hbar*omega_p = 28.816
    sqrt(rho <Z/A>) eV`` in the PDG's "Passage of particles through matter"
    presentation of the Sternheimer density effect. Here ``n_e = sum_i n_i Z_i``
    is summed over the medium's own catalog number densities ``n_i``
    [Angstrom^-3], converted to m^-3.

    Why this is the model edge. ``crystal.py::optical_constants`` writes the
    medium as ``n = 1 - delta - i beta`` with ``delta = (r_e lambda^2 / 2 pi)
    n_a f1``. In the high-frequency limit every electron responds freely,
    ``f1 -> Z``, and that expression is identically ``delta = omega_p^2 /
    (2 omega^2)``. At ``omega = omega_p`` it gives ``delta = 1/2``: the
    weakly-refracting, transparent-medium expansion that the photon-escape and
    self-absorption models rest on has collapsed, and below ``omega_p`` the
    medium reflects rather than transmits. So ``hbar*omega_p`` is where *this
    code's own* optics stops being valid, not an imported convention.

    Assumptions. All ``Z`` electrons respond as free (exact only for
    ``omega`` well above every binding energy; near and below ``omega_p`` the
    real response is collective and band-structure dependent, which is the
    point -- the model is not claimed to hold there). Homogeneous, isotropic
    bulk medium at the catalog's number densities, so no surface, porosity, or
    anisotropy term. No damping: the Drude collision frequency is dropped,
    which shifts a real plasmon resonance by order ``(1/tau)/omega_p``.

    Limiting case. ``n_e -> 0`` (vacuum, or an arbitrarily dilute medium) gives
    ``hbar*omega_p -> 0``: an empty medium imposes no low-energy bound, and the
    continuum floor then falls back to pure table support. Independent
    cross-check: inverting the PDG/Sternheimer ``C_bar = 2 ln(I / hbar omega_p)
    + 1`` with the packaged ``I`` and ``C_bar`` for silicon
    (``materials/_transport_data.py``) gives ``31.0482 eV`` against the
    ``31.0498 eV`` this function returns from the catalog number density --
    agreement to ``5e-5`` relative, from data that shares no code path with
    this one.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium.

    Returns
    -------
    float
        Plasma energy in eV; strictly positive for any real medium.

    Validation: photon-continuum-floor
    """
    composition = _resolve_composition(material)
    # Angstrom^-3 -> m^-3 is 1e30; Z from the shared atomic table.
    electron_density_per_m3 = (
        sum(float(n) * Z_TABLE[element] for element, n in composition) * 1.0e30
    )
    omega_p = np.sqrt(electron_density_per_m3 * _E_CHARGE**2 / (_EPS_0 * _M_E))
    return float(_HBAR * omega_p / _E_CHARGE)


def _normalize_composition(element, n_atoms_per_ang3, composition):
    """
    Accept either the single-element API (element=, n_atoms_per_ang3=) or a
    compound composition=[(element, number_density_1_per_Ang3), ...];
    return the latter form.
    """
    if composition is not None:
        return [(el, float(n)) for el, n in composition]
    if element is None or n_atoms_per_ang3 is None:
        raise ValueError(
            "specify the material: pass composition=[(element, n_per_Ang3), ...] "
            "(or both element= and n_atoms_per_ang3=). Refusing to fall back to a "
            "default element so a material can't be silently mis-loaded."
        )
    return [(element, float(n_atoms_per_ang3))]


def _mu_total_inv_ang(comp, E_eV):
    """Total narrow-beam linear attenuation [1/Angstrom] summed over elements.

    Per element this is the number density times the EPDL2025 per-atom total,

        mu_i = n_i [sigma_photo + sigma_coh + sigma_incoh
                    + sigma_pair,nuc + sigma_pair,el]_i(E),

    from :func:`~pyrite.materials.photon_cross_sections.total_photon_cross_section_ang2`.
    Every consumer of this helper -- filter plates, crystal-source
    self-absorption, the PXR/CBS line escape, the brem and characteristic
    escape factors, the hard-photon event scorer, and the detector window --
    models removal of a photon from an unscattered ray, so all of them want the
    total. Callers that want the photoabsorption coefficient of the refractive
    index (``beta`` from Chantler ``f2``) must use
    ``absorption_length_ang``/``optical_constants`` directly; that is a
    different compilation by design.

    The EPDL accessor is CPU-only, so the sum is always computed on the CPU.
    The result is returned on the SAME device as E_eV: a GPU array if the
    caller passed one (mc_spectrum, mixing it with on-device factors), a numpy
    array otherwise (detector_efficiency, whose output is multiplied into the
    host-side spectra in the notebook). Keying off the input device -- not the
    global _GPU flag -- keeps the CPU post-processing path numpy even when a
    GPU is present.

    Energies outside the EPDL band (1 eV to 100 GeV) are NaN. Callers must not
    read NaN as transparency: the line routes drop such samples, and the
    continuum routes reject them through :func:`_finite_mu_or_raise` (which
    keeps ``mu = 0`` only below 1 eV, under every modelled band).

    The backend import stays function-local for import cost, not for cycles:
    importing pyrite._backend runs the accelerator probe, and the catalog and
    CLI paths that pull in this module must not pay for it.

    Validation: narrow-beam-total-attenuation
    """
    from .._backend import REAL, _to_cpu, is_device_array, xp

    E_cpu = _to_cpu(E_eV)
    mu = 0.0
    for el, n_i in comp:
        mu = mu + n_i * total_photon_cross_section_ang2(el, E_cpu)
    if is_device_array(E_eV):
        return xp.asarray(mu, dtype=REAL)
    return mu


def _finite_mu_or_raise(mu, energy_eV, context):
    """Escape-factor ``mu``: refuse it where undefined, except below 1 eV.

    The continuum scorers must not read an undefined coefficient as
    transparency: above ``EPDL_E_MAX_EV`` a NaN ``mu`` would let a photon
    escape any thickness with unit transmission, so any non-finite value at
    ``E >= EPDL_E_MIN_EV`` raises. Nodes below ``EPDL_E_MIN_EV`` (``E = 0`` on
    a grid that was not floored) sit below every modelled band
    (``_photon_continuum_floor``) and keep the historical ``mu = 0``.

    Validation: narrow-beam-total-attenuation
    """
    from .._backend import array_namespace

    # The namespace that owns the operands, not the selected backend: the
    # hard-event scorer guards host arrays while an accelerator is selected.
    xp = array_namespace(mu, energy_eV)
    below_floor = xp.asarray(energy_eV) < EPDL_E_MIN_EV
    undefined = ~xp.isfinite(mu)
    if bool(xp.any(undefined & ~below_floor)):
        raise ValueError(
            f"{context}: photon attenuation is undefined at some energies; the EPDL2025 "
            f"cross sections cover {EPDL_E_MIN_EV:g} eV to {EPDL_E_MAX_EV:g} eV"
        )
    return xp.where(undefined, 0.0, mu)


# ---- layered (film-on-substrate) self-absorption ----------------------------
def _layer_dz(z_mid, n_z, z_top, z_bot):
    """z-extent of the layer [z_top, z_bot] that a photon leaving depth z_mid
    along n_hat (z-component n_z) crosses on its way out: toward z=0 when n_z<0
    (the entrance face) or the back face when n_z>0. numpy ufuncs are used so the
    same code serves numpy or cupy z_mid. Returns an array shaped like z_mid."""
    if n_z < 0:  # escape ray spans depths [0, z_mid]
        return np.maximum(np.minimum(z_mid, z_bot) - z_top, 0.0)
    return np.maximum(z_bot - np.maximum(z_mid, z_top), 0.0)  # spans [z_mid, z_total]


def _layer_path_length(z_mid, n_z, exit_distance_ang, z_top, z_bot):
    """Length within ``[z_top, z_bot]`` of a ray capped at ``exit_distance_ang``.

    For nonzero ``n_z``, intersect the forward ray interval
    ``[0, exit_distance_ang]`` with this layer's two z-face parameters.  A
    lateral ray remains entirely in its containing half-open z layer, avoiding
    double counting at shared layer boundaries.  NumPy ufunc dispatch preserves
    the NumPy/CuPy backend of array arguments.
    """
    if n_z == 0.0:
        in_layer = (z_mid >= z_top) & (z_mid < z_bot)
        return np.where(in_layer, exit_distance_ang, 0.0)

    t_top = (z_top - z_mid) / n_z
    t_bot = (z_bot - z_mid) / n_z
    path_start = np.maximum(np.minimum(t_top, t_bot), 0.0)
    path_end = np.minimum(np.maximum(t_top, t_bot), exit_distance_ang)
    return np.maximum(path_end - path_start, 0.0)


def _stack_tau(layers, z_mid, n_z, E, *, exit_distance_ang=None):
    """Beer-Lambert optical depth for a photon leaving each segment midpoint
    (depth z_mid) along n_hat through a LAYERED absorber stack:
        tau = (1/|n_z|) * sum_i mu_i(E) * dz_i
    layers = [(z_top, z_bot, composition), ...] top (entrance) first, contiguous,
    the deepest z_bot being the total stack thickness. z_mid and E are per-segment
    arrays (E the resonance energy); the result matches their device. A single
    layer over [0, total_thickness] reproduces the single-slab escape exactly,
    so passing layers=None elsewhere stays bit-for-bit identical.  Supplying an
    exit distance caps paths at the selected prism face.

    Validation: self-absorption
    """
    if exit_distance_ang is not None:
        tau = 0.0
        for z_top, z_bot, comp in layers:
            path = _layer_path_length(z_mid, n_z, exit_distance_ang, float(z_top), float(z_bot))
            tau = tau + _mu_total_inv_ang(comp, E) * path
        return tau

    inv = 1.0 / max(abs(float(n_z)), 1e-12)
    tau = 0.0
    for z_top, z_bot, comp in layers:
        dz = _layer_dz(z_mid, n_z, float(z_top), float(z_bot))
        tau = tau + _mu_total_inv_ang(comp, E) * dz * inv
    return tau


__all__ = ["linear_attenuation_inv_mm", "plasma_energy_eV"]
