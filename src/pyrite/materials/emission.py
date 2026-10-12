"""Deterministic vacuum CXR line centres for catalogue-material comparisons."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .catalog import CrystalSpec, MaterialCatalog, load_material_catalog
from .crystal import HBARC_EV_ANG, M_E_EV, _orientation_R, beta_from_Ee, reciprocal_g_vector

__all__ = ["photon_energies"]


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex values are unsupported")
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numerical values") from exc


def _indices(value: Sequence[int], name: str) -> tuple[int, int, int]:
    indices = _real_array(value, name)
    if (
        indices.shape != (3,)
        or not np.all(np.isfinite(indices))
        or not np.all(indices == np.round(indices))
        or not np.any(indices)
    ):
        raise ValueError(f"{name} must contain three finite integer Miller indices, not (0, 0, 0)")
    return int(indices[0]), int(indices[1]), int(indices[2])


def _primary_hkl(crystal: CrystalSpec) -> tuple[int, int, int]:
    """Select the catalogue's primary reflection parallel to its cleavage cut.

    Pinned parallel families are authoritative. Otherwise test harmonics 1--32
    of the primitive reciprocal cut. Source: the standard crystallographic
    basis sum ``S_el(hkl) = sum_j exp(2*pi*i*hkl.r_j)`` over each element.
    A systematic extinction cancels every element's sum; unlike an accidental
    cancellation between elements this is independent of photon energy.
    Assumes fully occupied expanded fractional sites. A relative amplitude
    tolerance of 1e-5 accommodates CIF rounding. An unshifted single-site basis
    selects the primitive harmonic. This estimates allowed geometry, not yield.

    Validation: structure-factor
    """
    cut = crystal.surface_hkl
    if cut is None:
        raise ValueError(f"{crystal.key!r} has no reciprocal surface cut; provide hkl explicitly")
    primitive = np.asarray(cut, dtype=int)
    primitive = primitive // np.gcd.reduce(np.abs(primitive))
    parallel = [
        family for family in crystal.hkl_families if not np.any(np.cross(primitive, family))
    ]
    if parallel:
        return min(parallel, key=lambda family: sum(index**2 for index in family))

    sites: dict[str, list[NDArray[np.float64]]] = {}
    for element, site in crystal.basis:
        sites.setdefault(element, []).append(site)
    for harmonic in range(1, 33):
        candidate = primitive * harmonic
        if any(
            abs(np.exp(2j * np.pi * (np.asarray(positions) @ candidate)).sum())
            > 1e-5 * len(positions)
            for positions in sites.values()
        ):
            return int(candidate[0]), int(candidate[1]), int(candidate[2])
    raise ValueError(f"{crystal.key!r} has no resolved primary reflection; provide hkl explicitly")


def photon_energies(
    materials: str | Sequence[str],
    beam_energy_eV: ArrayLike,
    *,
    tilt_deg: ArrayLike = 0.0,
    tilt_azim_deg: ArrayLike = 0.0,
    theta_obs_deg: ArrayLike = 90.0,
    phi_obs_deg: ArrayLike = 0.0,
    hkl: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
    mode: Literal["broadcast", "grid"] = "broadcast",
    catalog: MaterialCatalog | str | Path | None = None,
) -> NDArray[np.float64]:
    """Return vacuum CXR photon energies across materials and beam geometry.

    Parameters
    ----------
    materials
        Catalogue material key or ordered sequence of keys. Duplicate keys and
        empty sequences are preserved. The entrance crystal is used for stacks.
    beam_energy_eV
        Finite, strictly positive electron kinetic energies in eV.
    tilt_deg, tilt_azim_deg
        Sample polar tilt and tilt azimuth about the lab beam +z, in degrees.
        Positive polar tilt points the slab normal toward +x at zero azimuth,
        matching ``Slab``. The crystal retains its catalogue cut and zero
        in-plane roll. These angles tilt the whole sample, not just the crystal.
    theta_obs_deg, phi_obs_deg
        Lab detector polar angle from beam +z and azimuth from +x, in degrees.
        Detector azimuth does not change this straight-beam vacuum resonance,
        but participates in broadcasting and grid shape. Default polar: 90 deg.
    hkl
        Optional Miller triple shared by all materials, or a mapping of material
        keys to triples. Missing mapping entries use the default. Defaults choose
        the lowest pinned family parallel to the catalogue surface normal, or
        the first non-extinct harmonic (1--32) of its primitive cut. This gives
        basal/c-axis reflections for the usual layered materials, and the
        catalogue cleavage analogue for other cells. Ambiguous cuts require an
        override. Explicit triples compute kinematics even for extinct planes;
        they do not guarantee a nonzero intensity.
    mode
        ``"broadcast"`` (default) pairs numerical inputs using NumPy
        broadcasting, returning ``(n_materials, *broadcast_shape)``. Scalar
        inputs return shape ``(n_materials,)``. ``"grid"`` takes independent
        scalar/1-D axes and returns shape ``(n_materials, n_beam_energy,
        n_tilt, n_tilt_azim, n_theta_obs, n_phi_obs)``. Scalars keep size-one
        axes; empty axes remain empty. Neither mode squeezes the material axis.
    catalog
        Loaded catalogue, path to a complete catalogue, or ``None`` for current
        configured/environment-selected catalogue resolution.

    Returns
    -------
    numpy.ndarray
        Photon energies in eV. Each Miller triple denotes its Friedel pair:
        choose the sign with positive beam projection. A zero projection or a
        resonance above the electron kinetic energy gives shape-preserving NaN.
        Values within 1e-12 of a grazing projection (relative to ``|g|``) are zero.

    Raises
    ------
    ValueError
        For invalid/nonfinite numerical inputs, incompatible shapes, invalid
        Miller triples, unknown/noncrystalline materials, or unresolved defaults.

    Notes
    -----
    Source: Zhai 2025 main-text Eq. (1), Feranchuk--Spence 2000: in vacuum,
    ``E_gamma = hbar*c*beta*abs(v_hat.g)/(1 - beta*v_hat.n_hat)``;
    ``|g| = 2*pi/d_hkl``. The numerator selects the radiating Friedel mate
    under the production positive-g convention. Units are eV and angstroms.
    Assumes a straight electron at the supplied entrance energy, vacuum photon
    propagation, and a single ideal orientation. In-medium dispersion, transport
    broadening, mosaic averaging, intensity, and detector response are excluded.
    A basal reflection aligned with the slab normal reduces to
    ``hbar*c*beta*|g|*abs(cos(tilt))/(1-beta*cos(theta_obs))`` and is independent
    of tilt azimuth. Production in-medium line centres can differ.

    Validation: line-energy-dispersion, surface-hkl-orientation
    """
    if mode not in ("broadcast", "grid"):
        raise ValueError("mode must be 'broadcast' or 'grid'")
    keys = [materials] if isinstance(materials, str) else list(materials)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError("materials must be a key or a one-dimensional sequence of keys")
    names = ("beam_energy_eV", "tilt_deg", "tilt_azim_deg", "theta_obs_deg", "phi_obs_deg")
    values = [
        _real_array(value, name)
        for name, value in zip(
            names,
            (beam_energy_eV, tilt_deg, tilt_azim_deg, theta_obs_deg, phi_obs_deg),
            strict=True,
        )
    ]
    for name, value in zip(names, values, strict=True):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        if mode == "grid" and value.ndim > 1:
            raise ValueError(f"{name} must be scalar or one-dimensional in grid mode")
    if np.any(values[0] <= 0.0):
        raise ValueError("beam_energy_eV must be strictly positive")
    if mode == "grid":
        values = [
            value.reshape((1,) * axis + (value.size,) + (1,) * (4 - axis))
            for axis, value in enumerate(values)
        ]
    try:
        energy, tilt, azimuth, observation, _ = np.broadcast_arrays(*values)
    except ValueError as exc:
        raise ValueError("numerical inputs have incompatible broadcast shapes") from exc

    overrides: dict[str, tuple[int, int, int]] = {}
    shared_hkl = None
    if isinstance(hkl, Mapping):
        if set(hkl) - set(keys):
            raise ValueError("hkl mapping contains keys not requested in materials")
        overrides = {key: _indices(value, f"hkl[{key!r}]") for key, value in hkl.items()}
    elif hkl is not None:
        shared_hkl = _indices(hkl, "hkl")
    resolved = (
        catalog
        if isinstance(catalog, MaterialCatalog)
        else load_material_catalog(None if catalog is None else Path(catalog))
    )

    # Sparse grid inputs expand only as views; numerical work is vectorized
    # over geometry, with one small orientation calculation per material.
    tilt, azimuth, observation = (np.deg2rad(angle) for angle in (tilt, azimuth, observation))
    with np.errstate(over="ignore", under="ignore"):
        beta = beta_from_Ee(energy)
        inverse_gamma = M_E_EV / (M_E_EV + energy)
        # Stable forward-angle form of 1 - beta*cos(theta), including beta~1.
        denominator = inverse_gamma**2 / (1.0 + beta) + 2.0 * beta * np.sin(observation / 2) ** 2
    st, ct = np.sin(tilt), np.cos(tilt)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    result = np.full((len(keys), *energy.shape), np.nan, dtype=float)
    for index, key in enumerate(keys):
        if key not in resolved.materials:
            raise ValueError(f"unknown catalogue material {key!r}")
        crystal_key = resolved.materials[key].crystal_key
        if crystal_key not in resolved.crystals:
            raise ValueError(f"material {key!r} is noncrystalline")
        crystal = resolved.crystals[crystal_key]
        indices = overrides.get(key, shared_hkl)
        if indices is None:
            indices = _primary_hkl(crystal)
        g, magnitude = reciprocal_g_vector(indices, crystal.lattice)
        orientation = _orientation_R(
            crystal.lattice, crystal.beam_uvw, 0.0, surface_hkl=crystal.surface_hkl
        )
        if orientation is not None:
            g = orientation @ g
        # R_sample_to_lab's final row is (-sin(t)*cos(a), -sin(t)*sin(a), cos(t)).
        projection = np.abs(-st * (ca * g[0] + sa * g[1]) + ct * g[2])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            photon = HBARC_EV_ANG * beta * projection / denominator
        keep = (projection > 1e-12 * magnitude) & (photon > 0) & (photon <= energy)
        result[index] = np.where(keep, photon, np.nan)
    return result
