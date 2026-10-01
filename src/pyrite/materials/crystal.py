"""
materials.crystal

General-purpose X-ray crystallography / diffraction primitives, shared by the
Monte-Carlo pipeline (montecarlo, sweep, the detector forward models) and the
Feranchuk-Spence analytic checks (src/pyrite/validation/feranchuk_spence.py). Nothing here is
specific to the Feranchuk amplitude framework -- it is the reusable layer above
materials.atomic:

  * physical constants (hc, hbar c, alpha, m_e, r_e),
  * lattice geometry: direct/reciprocal vectors, |g| for any crystal system,
  * the catalog-backed crystal database (CIF -> CATALOG -> CRYSTALS),
  * Debye-Waller, structure factor S(g), and the polarizability / crystal-
    potential Fourier components chi_g (PXR) and U_g (CBS),
  * the g=0 susceptibility chi_0 and the complex refractive index
    n = sqrt(1 + chi_0) it defines,
  * photoabsorption length from Henke f2, and the coherent+incoherent
    scattering term that completes the narrow-beam attenuation coefficient,
  * dominant_reflections (rank reflection families by |S| e^{-W} / g^2),
  * _rotation_between (minimal rotation matrix, used to orient crystals).

Units: energies eV, lengths Angstrom, angles radians.

Crystal structures are projected from the immutable material catalog into the
mapping-style ``CRYSTALS`` compatibility registry. Depends on materials.atomic
(cromer_mann_f0, atomic_form_factor, henke_dispersion, Z_TABLE).
"""

import numpy as np

from ._cif import (
    crystals_crystal_to_crystal_info as crystals_crystal_to_crystal_info,
)
from ._cif import load_crystal_from_cif as load_crystal_from_cif
from .atomic import (
    Z_TABLE,
    atomic_form_factor,
    cromer_mann_f0,
    henke_dispersion,
)

# ---- constants --------------------------------------------------------------
HC_EV_ANG = 12398.4198  # h c [eV*Angstrom]
HBARC_EV_ANG = 1973.269804  # hbar c [eV*Angstrom]
ALPHA_FS = 1.0 / 137.035999
M_E_EV = 510998.95  # electron rest energy [eV]
R_E_ANG = 2.8179403e-5  # classical electron radius [Angstrom]
E2_EV_ANG = ALPHA_FS * HBARC_EV_ANG  # e^2 (Gaussian) = alpha hbar c = 14.3996 [eV*Angstrom]

# Elements whose edges fall in the soft-x-ray band -> force Chantler correction.
# xraydb edge energies confirm that the 350--3500 eV catalog line grids cross
# Fe L (707--845 eV), Bi M4/M5/M3 (2580--3177 eV), Re M (1883--2932 eV),
# and Ta M (1735--2708 eV), in addition to the previously reviewed elements.
_EDGE_PRONE = {"P", "Si", "Fe", "Ge", "Mo", "Nb", "Se", "Te", "Ta", "Re", "Bi"}


# ---- lattice geometry --------------------------------------------------------
def _direct_lattice_vectors(lattice):
    """
    Direct lattice vectors (a1, a2, a3) [Angstrom] from a lattice dict:
        {"system": "cubic",       "a": a}
        {"system": "tetragonal",  "a": a, "c": c}
        {"system": "orthorhombic","a": a, "b": b, "c": c}
        {"system": "hexagonal",   "a": a, "c": c}
        {"system": "general",     "a": a, "b": b, "c": c,
         "alpha": alpha, "beta": beta, "gamma": gamma}   # angles in DEGREES
    """
    sysname = lattice["system"]
    if sysname == "cubic":
        a = lattice["a"]
        return (
            np.array([a, 0.0, 0.0]),
            np.array([0.0, a, 0.0]),
            np.array([0.0, 0.0, a]),
        )
    if sysname == "tetragonal":
        a, c = lattice["a"], lattice["c"]
        return (
            np.array([a, 0.0, 0.0]),
            np.array([0.0, a, 0.0]),
            np.array([0.0, 0.0, c]),
        )
    if sysname == "orthorhombic":
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        return (
            np.array([a, 0.0, 0.0]),
            np.array([0.0, b, 0.0]),
            np.array([0.0, 0.0, c]),
        )
    if sysname == "hexagonal":
        a, c = lattice["a"], lattice["c"]
        # standard hexagonal: gamma = 120 deg between a1 and a2
        return (
            np.array([a, 0.0, 0.0]),
            np.array([-a / 2.0, a * np.sqrt(3) / 2.0, 0.0]),
            np.array([0.0, 0.0, c]),
        )
    if sysname == "general":
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        al = np.radians(lattice["alpha"])
        be = np.radians(lattice["beta"])
        ga = np.radians(lattice["gamma"])
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
        return a1, a2, np.array([cx, cy, cz])
    raise ValueError(f"unknown crystal system '{sysname}'")


def _cross3(u, v):
    """Cross product for plain 3-vectors (much faster than np.cross)."""
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


_RECIP_BASIS_CACHE = {}


def _reciprocal_basis(lattice):
    """Rows are b1, b2, b3 [1/Angstrom]; cached per lattice (hot path)."""
    key = tuple(sorted(lattice.items()))
    B = _RECIP_BASIS_CACHE.get(key)
    if B is None:
        a1, a2, a3 = _direct_lattice_vectors(lattice)
        V = np.dot(a1, _cross3(a2, a3))
        B = 2.0 * np.pi * np.array([_cross3(a2, a3), _cross3(a3, a1), _cross3(a1, a2)]) / V
        _RECIP_BASIS_CACHE[key] = B
    return B


def reciprocal_g_vector(hkl, lattice):
    """Return a reciprocal-lattice vector for any crystal system.

    Parameters
    ----------
    hkl
        Three Miller indices.
    lattice
        Lattice mapping containing cell lengths in angstroms and angles in
        degrees.

    Returns
    -------
    g_vec, g_mag
        Three-vector and scalar magnitude in inverse angstroms. The convention
        is ``|g| = 2*pi/d_hkl``.
    """
    g_vec = np.asarray(hkl, dtype=float) @ _reciprocal_basis(lattice)
    return g_vec, np.linalg.norm(g_vec)


# ---- catalog-backed crystal database ---------------------------------------
def load_crystals(catalog=None):
    """Project CIF-backed catalog crystals into mapping-style physics entries."""
    if catalog is None:
        from .catalog import _get_default_catalog

        catalog = _get_default_catalog()
    return {
        key: {
            "lattice": spec.lattice,
            "basis": spec.basis,
            "V_cell": spec.V_cell,
            "mosaic_fwhm_deg": spec.mosaic_fwhm_deg,
        }
        for key, spec in catalog.crystals.items()
    }


CRYSTALS = load_crystals()


# ---- kinematics -------------------------------------------------------------
def beta_from_Ee(Ee_eV):
    g = 1.0 + Ee_eV / M_E_EV
    return np.sqrt(1.0 - 1.0 / g**2)


def g_mag(d_ang):
    """|g| = 2 pi / d  [1/Angstrom]."""
    return 2.0 * np.pi / d_ang


# ---- structure & Debye-Waller ----------------------------------------------
def debye_waller(g_invang, B_ang2):
    """Amplitude Debye-Waller factor exp(-W), W = B (sin(theta)/lambda)^2
    = B (g/4pi)^2, for a tabulated B-factor [Angstrom^2] (B = 8 pi^2 <u_x^2>).
    Intensities carry exp(-2W) = the square of this.

    Validation: structure-factor
    """
    s = g_invang / (4.0 * np.pi)
    return np.exp(-B_ang2 * s**2)


def _atom_F(element, g, photon_E_eV, use_henke):
    """
    Per-atom form factor policy: complex Henke-corrected f0+f'+if'' for
    edge-prone elements (or when use_henke is set), else non-resonant
    Cromer-Mann f0 per Eq. (3). Always returns complex.
    """
    if use_henke or element in _EDGE_PRONE:
        return atomic_form_factor(element, g, photon_E_eV)
    return cromer_mann_f0(element, g) + 0.0j


def _basis_F(basis, g, photon_E_eV, use_henke):
    """Form factor per basis atom, computed once per unique element."""
    cache = {}
    for el, _ in basis:
        if el not in cache:
            cache[el] = _atom_F(el, g, photon_E_eV, use_henke)
    return [cache[el] for el, _ in basis]


def structure_factor(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """Return the crystal structure factor and reciprocal magnitude.

    Implements ``S(g) = sum_i F_i(g) exp(i*g.R_i) exp(-W_i)`` from Eq. (3).
    ``F_i`` is non-resonant ``f0`` unless ``use_henke`` enables anomalous
    corrections.

    Parameters
    ----------
    crystal
        Catalog crystal key.
    hkl
        Three Miller indices.
    photon_E_eV
        Photon energy in eV.
    B_ang2
        Isotropic Debye--Waller ``B`` factor in square angstroms.
    use_henke
        Include anomalous energy-dependent corrections.

    Returns
    -------
    structure_factor, g_mag
        Complex structure factor in electron units and reciprocal magnitude in
        inverse angstroms.

    Validation: structure-factor
    """
    info = CRYSTALS[crystal]
    hkl = np.asarray(hkl, dtype=float)
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    dwf = debye_waller(g, B_ang2)
    S = 0.0 + 0.0j
    for (_el, R), F in zip(
        info["basis"], _basis_F(info["basis"], g, photon_E_eV, use_henke), strict=False
    ):
        phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))
        S += F * phase * dwf
    return S, g


# ---- couplings: chi_g (PXR) and U_g (CBS) ----------------------------------
def chi_g(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """Return the polarizability Fourier component from Feranchuk Eq. (3).

    With the Debye--Waller factor already included in ``S(g)`` by
    :func:`structure_factor`,

        chi_g = -(4 pi e^2 / m omega^2) * S(g) / V.

    In the direct electron-density form with classical electron radius r_e:

        chi_g = - r_e lambda^2 / (pi V_cell) * S(g)     [dimensionless]

    Assumes the kinematic, independent-atom susceptibility convention used by
    Feranchuk--Spence (2000). An extinct reflection has ``S(g) -> 0`` and hence
    ``chi_g -> 0``; at high photon energy, ``chi_g`` falls as ``lambda**2``.

    Parameters
    ----------
    crystal, hkl
        Catalog crystal key and three Miller indices.
    photon_E_eV
        Scalar or array photon energy in eV.
    B_ang2
        Isotropic Debye--Waller ``B`` factor in square angstroms.
    use_henke
        Include anomalous energy-dependent form-factor corrections.

    Returns
    -------
    complex or numpy.ndarray
        Dimensionless complex polarizability on the photon-energy shape.

    Validation: pxr-amplitude
    """
    S, _ = structure_factor(crystal, hkl, photon_E_eV, B_ang2, use_henke)
    lam = HC_EV_ANG / photon_E_eV  # wavelength [Angstrom]
    return -R_E_ANG * lam**2 / (np.pi * CRYSTALS[crystal]["V_cell"]) * S


def chi_0(crystal, photon_E_eV, use_henke=True):
    """Return the g=0 (bulk mean) polarizability of the unit cell.

    This is :func:`chi_g` evaluated at ``g = 0``, where the Debye-Waller factor
    ``exp(-B s^2)`` and every basis phase ``exp(i 2 pi hkl . R)`` reduce to 1,
    so the structure factor collapses to the forward-scattering sum over the
    cell:

        S(0)  = sum_i (f1_i + i f2_i),   f1 = Z + f',  f2 = f''
        chi_0 = - r_e lambda^2 / (pi V_cell) * S(0)     [dimensionless]

    The forward form factors use the Henke convention ``f1 = Z + f'`` taken
    from ``Z_TABLE`` and ``henke_dispersion``, not ``cromer_mann_f0(el, 0)``,
    so that this shares one normalization with :func:`optical_constants` and
    :func:`absorption_length_ang` exactly rather than to within the
    Cromer-Mann ``f0(0) ~ Z`` fit residual.

    ``use_henke`` defaults to True here, unlike :func:`chi_g`: the dispersive
    ``f'`` is the entire content of the real refractive correction, and
    dropping it would leave only the non-resonant Thomson term.

    Assumes the same kinematic, independent-atom susceptibility convention as
    :func:`chi_g`, and a homogeneous medium on the scale of the photon
    wavelength (bulk response; interface/Fresnel effects are not included).

    Limiting case: with ``f' = f'' = 0``, ``chi_0 -> -r_e lambda^2 Z_cell /
    (pi V_cell)``, purely real and negative, giving the textbook Thomson
    ``n = 1 - delta`` with ``delta = r_e lambda^2 n_e / (2 pi)``.

    Out-of-range energies propagate NaN from ``henke_dispersion``, consistent
    with the rest of the module.

    Validation: xray-chi-zero
    """
    info = CRYSTALS[crystal]
    E = np.asarray(photon_E_eV, dtype=float)
    forward_F = {}
    for el, _R in info["basis"]:
        if el in forward_F:
            continue
        if use_henke:
            fp, fpp = henke_dispersion(el, E)
            forward_F[el] = (Z_TABLE[el] + fp) + 1j * fpp
        else:
            forward_F[el] = np.full(E.shape, float(Z_TABLE[el]), dtype=complex)
    S = np.zeros(E.shape, dtype=complex)
    for el, _R in info["basis"]:
        S = S + forward_F[el]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = HC_EV_ANG / E  # wavelength [Angstrom]
        return -R_E_ANG * lam**2 / (np.pi * info["V_cell"]) * S


def refractive_index(crystal, photon_E_eV, use_henke=True):
    """Complex X-ray refractive index of a catalog crystal.

    From the Maxwell dispersion relation in a homogeneous dielectric,
    ``k^2 = (1 + chi_0) omega^2``, so

        n(E) = sqrt(1 + chi_0(E)) ~ 1 - delta - i beta

    with the module-wide ``n = 1 - delta - i beta`` convention of
    :func:`optical_constants` (time factor ``exp(+i omega t)``, matching the
    coherent propagation phase ``exp{i[omega t - k.r]}`` used by the spectrum
    kernels). The square root is taken exactly rather than linearized; the two
    agree to O(chi_0^2) ~ 1e-10 in the X-ray regime, but the exact form is what
    the in-medium wavevector is defined from.

    Because ``|chi_0| << 1``, ``delta ~ -Re(chi_0)/2`` is typically 1e-5..1e-3;
    it is small per Angstrom but accumulates over micron-scale trajectories,
    which is precisely the effect this is here to track.

    Limiting case: ``chi_0 -> 0`` (vacuum, or photon energy far above all
    edges) gives ``n -> 1`` and every in-medium expression collapses to its
    vacuum form.

    Validation: xray-refractive-index
    """
    return np.sqrt(1.0 + chi_0(crystal, photon_E_eV, use_henke))


def U_g(crystal, hkl, photon_E_eV, B_ang2=0.0, use_henke=False):
    """Return the crystal-potential Fourier component for CBS coupling.

    Implements Eq. (4), folded with the inverse-volume factor of Eq. (14) and
    one electron charge: ``e*U_g/V = 4*pi*e**2*sum_i(exp(i*g*R_i) *``
    ``(Z_i-F_i)/g**2 * exp(-W))/V``. The result is in eV; divide by
    ``m_e*c**2`` for the dimensionless Eq. (14) coupling.

    Parameters
    ----------
    crystal, hkl
        Catalog crystal key and three Miller indices.
    photon_E_eV
        Scalar or array photon energy in eV.
    B_ang2
        Isotropic Debye--Waller ``B`` factor in square angstroms.
    use_henke
        Include anomalous form-factor corrections.

    Returns
    -------
    complex or numpy.ndarray
        Crystal-potential Fourier component in eV.

    Validation: cbs-amplitude
    """
    info = CRYSTALS[crystal]
    hkl = np.asarray(hkl, dtype=float)
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    dwf = debye_waller(g, B_ang2)
    acc = 0.0 + 0.0j
    for (el, R), F in zip(
        info["basis"], _basis_F(info["basis"], g, photon_E_eV, use_henke), strict=False
    ):
        phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))
        acc += phase * (Z_TABLE[el] - F.real) / g**2 * dwf
    result = 4.0 * np.pi * E2_EV_ANG * acc / info["V_cell"]
    # Without anomalous corrections the CBS coupling is energy-independent,
    # but callers still receive one value per requested energy so it can share
    # the chi_g tabulation/interpolation path.
    if np.ndim(photon_E_eV):
        return np.broadcast_to(result, np.shape(photon_E_eV))
    return result


def reflection_coupling_tables(crystal, hkl_list, photon_E_eV, B_ang2=0.0, use_henke=False, xp=np):
    """Tabulate :func:`chi_g` and ``U_g/m_e`` for several reflections at once.

    Both couplings are linear in the per-atom form factor
    ``F_i(g, E) = f0_i(g) + f'_i(E) + i f''_i(E)`` (or ``f0_i(g)`` alone when
    no anomalous correction applies), and ``f'``/``f''`` depend only on the
    element and energy. Grouping the basis by element,

        S(g, E)   = DWF(g) [c_g + sum_el P_el(g) (f'_el(E) + i f''_el(E))]
        acc(g, E) = DWF(g)/g**2 [a_g - sum_el P_el(g) f'_el(E)]

    with ``P_el(g) = sum_{i in el} exp(i 2 pi hkl.R_i)``,
    ``c_g = sum_el P_el f0_el(g)`` and ``a_g = sum_el P_el (Z_el - f0_el(g))``.
    Then ``chi_g = -r_e lambda**2 S / (pi V_cell)`` and
    ``U_g/m_e = 4 pi e**2 acc / (V_cell m_e c**2)``, exactly as
    :func:`chi_g`/:func:`U_g`. The anomalous tables are read once per element
    for every reflection, and the per-energy arithmetic runs on ``xp`` (CuPy on
    a GPU), in float64. Only summation order differs from the per-atom
    functions, so results agree to float64 rounding; NaN outside the
    Chantler range propagates the same way.

    Parameters
    ----------
    crystal
        Catalog crystal key.
    hkl_list
        Sequence of three Miller indices per reflection.
    photon_E_eV
        One-dimensional photon energies in eV.
    B_ang2
        Isotropic Debye--Waller ``B`` factor in square angstroms.
    use_henke
        Include anomalous corrections for every element (edge-prone elements
        always carry them).
    xp
        Array module for the energy arithmetic.

    Returns
    -------
    chi_re, chi_im, u_re, u_im
        ``(N_hkl, N_E)`` float64 ``xp`` arrays: dimensionless ``chi_g`` and
        dimensionless ``U_g/m_e``.

    Validation: line-reflection-coupling-tables
    """
    info = CRYSTALS[crystal]
    basis = info["basis"]
    E_host = np.asarray(photon_E_eV, dtype=np.float64)
    E = xp.asarray(E_host)
    dispersive = {}
    for el, _R in basis:
        if el not in dispersive and (use_henke or el in _EDGE_PRONE):
            fp, fpp = henke_dispersion(el, E_host)
            dispersive[el] = (xp.asarray(fp), xp.asarray(fpp))
    chi_scale = -R_E_ANG * (HC_EV_ANG / E) ** 2 / (np.pi * info["V_cell"])
    u_scale = 4.0 * np.pi * E2_EV_ANG / info["V_cell"] / M_E_EV
    rows = ([], [], [], [])
    for hkl in hkl_list:
        hkl = np.asarray(hkl, dtype=float)
        _, g = reciprocal_g_vector(hkl, info["lattice"])
        dwf = debye_waller(g, B_ang2)
        weights = {}
        for el, R in basis:
            weights[el] = weights.get(el, 0.0) + np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))
        c = sum(w * cromer_mann_f0(el, g) for el, w in weights.items())
        a = sum(w * (Z_TABLE[el] - cromer_mann_f0(el, g)) for el, w in weights.items())
        s_re = xp.full(E.shape, float(np.real(c)))
        s_im = xp.full(E.shape, float(np.imag(c)))
        acc_re = xp.full(E.shape, float(np.real(a)))
        acc_im = xp.full(E.shape, float(np.imag(a)))
        for el, w in weights.items():
            if el not in dispersive:
                continue
            fp, fpp = dispersive[el]
            s_re += w.real * fp - w.imag * fpp
            s_im += w.imag * fp + w.real * fpp
            acc_re -= w.real * fp
            acc_im -= w.imag * fp
        rows[0].append(chi_scale * (dwf * s_re))
        rows[1].append(chi_scale * (dwf * s_im))
        rows[2].append(u_scale * (dwf / g**2) * acc_re)
        rows[3].append(u_scale * (dwf / g**2) * acc_im)
    return tuple(xp.stack(row) for row in rows)


# ---- absorption length ------------------------------------------------------
def absorption_length_ang(element, photon_E_eV, number_density_per_ang3):
    """
    Return the Beer-Lambert intensity attenuation length in Angstrom.

    Starting from ``I(z) = I(0) exp(-mu z)`` and the Henke imaginary
    refractive-index coefficient
    ``beta = r_e lambda**2 n f2 / (2 pi)``, this uses
    ``mu = 2 k beta = 2 r_e lambda n f2`` with ``k = 2 pi / lambda``.
    The assumptions are a homogeneous elemental medium, passive attenuation,
    and positive photon energy in the tabulated Henke range. For compounds,
    inverse lengths add through ``sum_i n_i f2_i``. As ``n`` or ``f2`` tends
    to zero, ``mu`` tends to zero and the returned length diverges.

    Outside the Chantler table range, including at the wide bremsstrahlung
    grid's 0 eV bin, ``henke_dispersion`` returns NaN and this function
    therefore returns NaN. Downstream ``nan_to_num`` policy handles those
    out-of-domain bins. The ``errstate`` guard only suppresses the associated
    divide-by-zero and invalid-value warnings; the derivation above applies to
    positive energies inside the tabulated range.

    Parameters
    ----------
    element
        Element symbol.
    photon_E_eV
        Scalar or array photon energy in eV.
    number_density_per_ang3
        Element number density in atoms per cubic angstrom.

    Returns
    -------
    numpy.ndarray
        Beer--Lambert intensity attenuation length in angstroms, shaped like
        ``photon_E_eV``.

    Validation: absorption-length
    """
    _, f2 = henke_dispersion(element, photon_E_eV)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = HC_EV_ANG / photon_E_eV
        beta_idx = R_E_ANG * lam**2 / (2.0 * np.pi) * number_density_per_ang3 * f2
        k = 2.0 * np.pi / lam
        mu = 2.0 * k * beta_idx  # 1/Angstrom
        return 1.0 / mu


def crystal_absorption_length_ang(crystal, photon_E_eV):
    """
    Beer--Lambert attenuation length [Angstrom] of a catalog crystal, summing
    the inverse lengths of its basis elements.

    `absorption_length_ang` is elemental; its own docstring states the compound
    rule, that inverse lengths add through ``sum_i n_i f2_i``. This applies that
    rule to a `CRYSTALS` entry, taking each element's number density from the
    basis occupancy of the unit cell, ``n_i = N_i / V_cell``. For a
    single-element crystal it reduces exactly to the elemental call at
    ``n = len(basis) / V_cell``.

    Out-of-range energies propagate NaN from `henke_dispersion`, as in the
    elemental function.

    Parameters
    ----------
    crystal
        Key into `CRYSTALS`.
    photon_E_eV
        Scalar or array photon energy in eV.

    Returns
    -------
    numpy.ndarray
        Attenuation length in angstroms, shaped like ``photon_E_eV``.

    Validation: absorption-length
    """
    info = CRYSTALS[crystal]
    counts: dict[str, int] = {}
    for element, _ in info["basis"]:
        counts[element] = counts.get(element, 0) + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_mu = sum(
            1.0 / absorption_length_ang(element, photon_E_eV, n_sites / info["V_cell"])
            for element, n_sites in counts.items()
        )
        return 1.0 / inv_mu


# ---- complex refractive index (grazing-incidence optics) --------------------
def optical_constants(element, photon_E_eV, number_density_per_ang3):
    """
    Complex refractive index n = 1 - delta - i*beta of a material, from the
    element's Henke/Chantler anomalous scattering factors f'(E), f''(E)
    (`atomic_form_factors.henke_dispersion`), the Henke convention
    f1 = Z + f' (forward-scattering factor):

        delta(E) = (r_e lambda^2 / 2 pi) * n_atomic * f1(E),   f1 = Z + f'(E)
        beta(E)  = (r_e lambda^2 / 2 pi) * n_atomic * f2(E)

    Standard result, e.g. Als-Nielsen & McMorrow, "Elements of Modern X-ray
    Physics" 2nd ed., Ch. 3 (index of refraction from the atomic scattering
    factor); equivalently Attwood & Sakdinawat, "X-Rays and Extreme Ultraviolet
    Radiation" 2nd ed., Ch. 3. Same r_e, lambda, 2*pi normalization as this
    module's `absorption_length_ang`, whose `beta_idx` IS this beta -- that
    function's mu = 2 k beta is the textbook absorption coefficient, so
    beta computed here reproduces it exactly (limiting-case cross-check).

    element : str element symbol.
    photon_E_eV : float or array, photon energy [eV].
    number_density_per_ang3 : float, atomic number density [1/Angstrom^3].

    Returns (delta, beta), each shaped like photon_E_eV. Out-of-range energies
    (see henke_dispersion) return NaN, consistent with the rest of the module.

    Limiting case: f2 -> 0 (far from any edge, fully non-absorbing idealization)
    gives beta -> 0, the lossless dielectric limit used by `Grating.reflectivity`'s
    total-external-reflection / power-law-falloff checks.

    Validation: grazing-optical-constants
    """
    fp, f2 = henke_dispersion(element, photon_E_eV)
    f1 = Z_TABLE[element] + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = HC_EV_ANG / np.asarray(photon_E_eV, float)  # wavelength [Angstrom]
        pref = R_E_ANG * lam**2 / (2.0 * np.pi) * number_density_per_ang3
        delta = pref * f1
        beta = pref * f2
    return delta, beta


# ---- crystal orientation ----------------------------------------------------
def _rotation_between(u_hat, t_hat):
    """Minimal rotation matrix R such that R @ u_hat = t_hat (unit vectors)."""
    c = float(np.dot(u_hat, t_hat))
    axis = _cross3(u_hat, t_hat)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: 180 deg about any axis perpendicular to u_hat
        tmp = np.array([1.0, 0.0, 0.0]) if abs(u_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _cross3(u_hat, tmp)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * K @ K
    axis = axis / s
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# ---- reflection ranking -----------------------------------------------------
def dominant_reflections(
    crystal,
    n_families=4,
    E_ref_eV=1000.0,
    B_ang2=0.0,
    use_henke=False,
    g_max_invang=8.0,
    representatives_only=False,
):
    """
    Automatically select the strongest reflection FAMILIES of a crystal,
    Zhai-style (their Table 5 keeps the four planes of largest ``abs(chi_g)`` per
    crystal; everything weaker contributes < ~30%).

    Enumerates reciprocal vectors with ``abs(g) <= g_max_invang`` and ranks by
    ``abs(S(g))*exp(-W)/g**2``, which is proportional to ``abs(chi_g)`` at each
    reflection's own line
    energy (omega_res scales with g, and chi ~ S/omega^2). Symmetry-
    equivalent members are grouped by identical reciprocal magnitudes and metrics -- no explicit
    space-group code needed -- and ALL members of the top n_families are
    returned as a sorted list of (h, k, l) tuples (including Friedel mates).
    Set ``representatives_only`` to return one deterministic member per ranked
    family, useful when each family should appear once in a visualization.

    NOTE: this ranks by the crystal STRUCTURE only. Texture constraints are
    yours to impose -- e.g. HOPG must be restricted to (00l) by hand, since
    its in-plane reflections are incoherent across fiber-textured grains.

    Parameters
    ----------
    crystal
        Catalog crystal key.
    n_families
        Maximum number of ranked symmetry families.
    E_ref_eV
        Reference photon energy in eV for anomalous form factors.
    B_ang2
        Isotropic Debye--Waller ``B`` factor in square angstroms.
    use_henke
        Include anomalous form-factor corrections.
    g_max_invang
        Maximum reciprocal-vector magnitude in inverse angstroms.
    representatives_only
        Return one deterministic member per family instead of all mates.

    Returns
    -------
    list of tuple
        Miller-index triples ordered by decreasing family strength.
    """
    info = CRYSTALS[crystal]
    B = _reciprocal_basis(info["lattice"])
    a_vecs = _direct_lattice_vectors(info["lattice"])

    # exact per-axis index bounds: |h_i| <= g_max |a_i| / 2 pi
    nmax = [int(np.floor(g_max_invang * np.linalg.norm(a) / (2.0 * np.pi))) for a in a_vecs]
    grids = np.meshgrid(*(np.arange(-n, n + 1) for n in nmax), indexing="ij")
    hkl = np.column_stack([G.ravel() for G in grids]).astype(float)
    g_vec = hkl @ B
    g_mag = np.linalg.norm(g_vec, axis=1)
    keep = (g_mag > 1e-9) & (g_mag <= g_max_invang)
    hkl, g_mag = hkl[keep], g_mag[keep]

    # |S(g)| with the per-element form-factor policy, vectorized over hkl
    dwf = debye_waller(g_mag, B_ang2)
    F_el = {}
    for el in {el for el, _ in info["basis"]}:
        F_el[el] = _atom_F(el, g_mag, E_ref_eV, use_henke)
    S = np.zeros(g_mag.shape, dtype=complex)
    for el, R_frac in info["basis"]:
        S += F_el[el] * np.exp(2j * np.pi * (hkl @ R_frac)) * dwf
    metric = np.abs(S) / g_mag**2

    # group symmetry mates: identical (|g|, metric) to rounding
    fams = {}
    for i in range(hkl.shape[0]):
        key = (round(float(g_mag[i]), 6), round(float(metric[i]), 9))
        fams.setdefault(key, []).append(tuple(int(x) for x in hkl[i]))
    ranked = sorted(fams.items(), key=lambda kv: -kv[0][1])

    out = []
    for (_, m), members in ranked[:n_families]:
        if m < 1e-9 * ranked[0][0][1]:
            break  # forbidden/negligible families
        ordered = sorted(members)
        out.extend([ordered[-1]] if representatives_only else ordered)
    return out
