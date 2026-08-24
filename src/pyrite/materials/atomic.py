"""
materials.atomic

Atomic form factor F(g, E) = f0(g) + f'(E) + i f''(E), sourced from xraydb
(https://github.com/xraypy/XrayDB, MIT code / CC0 data) -- no hard-coded tables.

  f0(g)        Waasmaier & Kirfel (1995) parameterization  -> xraydb.f0
  f'(E),f''(E) Chantler / NIST FFAST anomalous corrections  -> xraydb.f1_chantler,
               xraydb.f2_chantler  (these return f' and f'' DIRECTLY, i.e. the
               anomalous parts only -- NOT the Henke-style f1 = Z + f')
  Z            atomic numbers                                -> xraydb.atomic_number

The public surface is unchanged from the previous Henke/CXRO + Cromer-Mann
implementation -- cromer_mann_f0, henke_dispersion, atomic_form_factor, Z_TABLE,
load_henke -- so callers (materials.crystal, montecarlo, the checks/) are
untouched. The names cromer_mann_f0 / henke_dispersion are kept for compatibility
even though the underlying data is now Waasmaier-Kirfel / Chantler.

Conventions:
  g = |reciprocal lattice vector| = 2*pi / d_hkl   [1/Angstrom]  (NOT 1/d)
  E = photon energy                                 [eV]
  f0(s), s = sin(theta)/lambda = g/(4*pi).
"""

import hashlib
from collections import OrderedDict
from functools import cache

import numpy as np
import xraydb


# ---- atomic numbers ---------------------------------------------------------
@cache
def _z_of(element):
    """Atomic number for an element symbol, or None if xraydb doesn't know it."""
    try:
        return int(xraydb.atomic_number(element))
    except Exception:
        return None


class _ElementZTable:
    """Mapping element-symbol -> atomic number, backed by xraydb (replaces the
    old hard-coded Z_TABLE dict). Supports the two access patterns callers use:
        Z_TABLE[el]      -> int Z, or KeyError if the symbol is unknown
        el in Z_TABLE    -> bool (is it a valid element symbol?)
    """

    def __getitem__(self, element):
        z = _z_of(element)
        if z is None:
            raise KeyError(f"Unknown element symbol '{element}'.")
        return z

    def __contains__(self, element):
        return _z_of(element) is not None


Z_TABLE = _ElementZTable()


# ---- f0(g): non-resonant, energy-independent --------------------------------
def cromer_mann_f0(element, g):
    """Return the energy-independent atomic form factor ``f0(g)``.

    Uses Waasmaier--Kirfel through xraydb; the historical function name is
    retained for API compatibility with the old Cromer--Mann implementation.

    Parameters
    ----------
    element
        Element symbol, for example ``"C"``.
    g
        Scalar or array reciprocal-vector magnitude in inverse angstroms,
        using ``g = 2*pi/d`` rather than ``1/d``.

    Returns
    -------
    numpy.ndarray
        Form factor in electron units, shaped like ``g``. ``f0(0) = Z``.

    """
    g_arr = np.asarray(g, dtype=float)
    s = (g_arr / (4.0 * np.pi)).ravel()
    f0 = np.asarray(xraydb.f0(element, s), dtype=float)
    return f0.reshape(g_arr.shape)


# ---- f'(E), f''(E): resonant dispersion + absorption ------------------------
@cache
def _chantler_bounds(element):
    """(Emin, Emax) [eV] of the element's tabulated Chantler/FFAST grid."""
    E = np.asarray(xraydb.chantler_energies(element), dtype=float)
    return float(E.min()), float(E.max())


# f'(E)/f''(E) depend only on (element, energy grid), never on g/hkl, but the
# per-reflection line-spectrum loop re-requests the same (element, E_tab) pair
# ~N_hkl times per case -- each hit is an xraydb FITPACK spline eval. Memoize on
# the exact energy bytes so the repeats collapse to one spline pass per element.
# Returns bit-identical arrays (frozen read-only so a hit can't be mutated).
_HENKE_MEMO: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_HENKE_MEMO_MAX = 256


def henke_dispersion(element, E_eV, on_out_of_range="nan"):
    """Return energy-dependent anomalous dispersion corrections.

    Parameters
    ----------
    element
        Element symbol.
    E_eV
        Scalar or array photon energies in eV.
    on_out_of_range
        ``"nan"`` preserves shape and fills unsupported energies with NaN;
        ``"raise"`` raises :class:`ValueError`.

    Returns
    -------
    f_prime, f_double_prime
        Read-only arrays shaped like ``E_eV`` containing the real and positive
        imaginary anomalous corrections in electron units.

    Raises
    ------
    KeyError
        If ``element`` is unknown.
    ValueError
        If strict mode encounters energy outside the Chantler table.

    Notes
    -----
    Energy-dependent dispersion corrections (Chantler/FFAST via xraydb; name kept
    for API compatibility with the old Henke/CXRO implementation).

    """
    if element not in Z_TABLE:
        raise KeyError(f"Unknown element symbol '{element}'.")
    Emin, Emax = _chantler_bounds(element)
    E = np.asarray(E_eV, dtype=float)
    shape = E.shape

    # "raise" mode can throw, so it is not cached; every other request keys on
    # the exact energy bytes and short-circuits to the frozen spline result.
    # (ascontiguousarray only for hashing -- it would promote a 0-d scalar to
    # shape (1,), so it must not touch the returned array's shape.)
    if on_out_of_range != "raise":
        digest = hashlib.blake2b(np.ascontiguousarray(E).tobytes()).digest()
        key = (element, on_out_of_range, shape, digest)
        cached = _HENKE_MEMO.get(key)
        if cached is not None:
            _HENKE_MEMO.move_to_end(key)
            return cached

    Eflat = np.atleast_1d(E).ravel()

    # strict interior: f1_chantler can fail right at the table endpoints, and the
    # brem grid passes E=0 (<= Emin) which must read as out-of-range anyway.
    in_range = (Eflat > Emin) & (Eflat < Emax)

    if on_out_of_range == "raise" and not np.all(in_range):
        bad = Eflat[~in_range]
        raise ValueError(
            f"E={bad} eV outside Chantler range [{Emin:.1f}, {Emax:.1f}] eV for {element}."
        )

    fp = np.full(Eflat.shape, np.nan)
    fpp = np.full(Eflat.shape, np.nan)
    if in_range.any():
        Ein = Eflat[in_range]
        fp[in_range] = np.asarray(xraydb.f1_chantler(element, Ein), dtype=float)
        fpp[in_range] = np.asarray(xraydb.f2_chantler(element, Ein), dtype=float)
    out_fp, out_fpp = fp.reshape(shape), fpp.reshape(shape)

    if on_out_of_range != "raise":
        out_fp.flags.writeable = False
        out_fpp.flags.writeable = False
        _HENKE_MEMO[key] = (out_fp, out_fpp)
        if len(_HENKE_MEMO) > _HENKE_MEMO_MAX:
            _HENKE_MEMO.popitem(last=False)
    return out_fp, out_fpp


def atomic_form_factor(element, g, E_eV, on_out_of_range="nan"):
    """Return ``F(g, E) = f0(g) + f'(E) + i f''(E)``.

    ``f0`` uses Waasmaier--Kirfel through ``xraydb.f0`` with the crystallographic
    argument ``q = g / (4*pi)``. ``f'`` and positive ``f''`` use the
    Chantler/FFAST tables through xraydb. This assumes the repository's
    structure-factor phase and passive-medium sign convention.

    Parameters
    ----------
    element
        Element symbol.
    g
        Reciprocal-vector magnitude in inverse angstroms, ``2*pi/d``.
    E_eV
        Photon energy in eV. ``g`` and ``E_eV`` must broadcast together.
    on_out_of_range
        ``"nan"`` for aligned NaN results or ``"raise"`` for strict bounds.

    Returns
    -------
    numpy.ndarray
        Complex form factor in electron units on the broadcast input shape.

    Out-of-range energies return NaN (shape preserved) by default, so the result
    stays index-aligned with E_eV. Pass on_out_of_range="raise" for strict mode.

    In the forward-scattering limit ``g -> 0``, ``f0 -> Z``; when anomalous
    terms vanish, the result reduces to the ordinary elastic form factor.

    Validation: atomic-form-factor
    """
    f0 = cromer_mann_f0(element, g)  # real, energy-independent
    fp, fpp = henke_dispersion(element, E_eV, on_out_of_range)
    return (f0 + fp) + 1j * fpp  # NaN where E out of range


# ---- Henke-style (E, f1, f2) table -----------------------------------------
@cache
def load_henke(element):
    """Element's anomalous-scattering table as (E [eV], f1, f2) arrays on the
    native Chantler/FFAST energy grid (which densely samples absorption edges --
    montecarlo relies on this to resolve the edge jumps). f1 = Z + f' is the
    Henke-convention forward-scattering factor; f2 = f''. Cached per element.
    """
    E = np.asarray(xraydb.chantler_energies(element), dtype=float)
    E = E[E.min() < E]  # drop the single endpoint where f1_chantler is unstable
    Z = Z_TABLE[element]
    fp = np.asarray(xraydb.f1_chantler(element, E), dtype=float)
    f2 = np.asarray(xraydb.f2_chantler(element, E), dtype=float)
    return E, Z + fp, f2
