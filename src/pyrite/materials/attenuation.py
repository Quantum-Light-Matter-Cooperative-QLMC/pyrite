"""
materials.attenuation

Composition handling and X-ray self-absorption shared across transport,
spectrum and detector: normalize a single-element / compound material spec,
the total linear attenuation summed over elements, and the layered
(film-on-substrate) Beer-Lambert optical depth.
"""

import numpy as np

from . import CATALOG, MediumSpec
from .crystal import absorption_length_ang


def linear_attenuation_inv_mm(material: str | MediumSpec, energy_eV: object) -> np.ndarray:
    """Return total linear X-ray attenuation on a positive energy grid [mm^-1].

    Element contributions add as
    ``mu(E) = sum_i 1 / L_abs,i(E)``. The underlying Henke/Chantler
    absorption lengths assume homogeneous, passive primary-beam attenuation;
    this helper adds no scattering, fluorescence, diffraction, or secondary
    production. As every elemental absorption coefficient tends to zero, so
    does the returned coefficient.

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
    coefficient = np.asarray(_mu_total_inv_ang(composition, energy), dtype=float) * 1.0e7
    coefficient.setflags(write=False)
    return coefficient


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
    """Total linear attenuation 1/L_abs [1/Angstrom] summed over elements.

    absorption_length_ang (from crystallography) is CPU-only, so the sum
    is always computed on the CPU. The result is returned on the SAME device as
    E_eV: a GPU array if the caller passed one (mc_spectrum, mixing it with
    on-device factors), a numpy array otherwise (detector_efficiency, whose
    output is multiplied into the host-side spectra in the notebook). Keying off
    the input device -- not the global _GPU flag -- keeps the CPU post-processing
    path numpy even when a GPU is present."""
    from ..montecarlo._backend import REAL, _to_cpu, is_device_array, xp

    E_cpu = _to_cpu(E_eV)
    mu = 0.0
    for el, n_i in comp:
        mu = mu + 1.0 / absorption_length_ang(el, E_cpu, n_i)
    if is_device_array(E_eV):
        return xp.asarray(mu, dtype=REAL)
    return mu


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


__all__ = ["linear_attenuation_inv_mm"]
