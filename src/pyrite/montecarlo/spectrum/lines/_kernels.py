"""Leaf numerics for the CXR line spectrum: lineshape, interpolation, and
per-segment geometry.

Every function here is a pure array kernel over the ``xp`` backend -- no setup
state, no dispatch policy, no accumulation bookkeeping -- so both the per-hkl
and the batched route can share them and the focused kernel tests can call them
directly.
"""

import numpy as np

from ...._backend import REAL, array_namespace, xp
from ....materials.attenuation import _mu_total_inv_ang
from ....materials.crystal import HBARC_EV_ANG, M_E_EV, U_g, chi_g
from ...geometry import first_prism_exit
from ...groove import _THETA_TOL
from ...transport import (
    _BS_PREFACTOR,
    _LN2,
    _MC2_KEV,
    TRANSPORT_ELEMENTS,
    _element_crossover_keV,
    beta_from_keV,
)

# ---- segment-sum CXR spectrum ------------------------------------------------
# Owning registry for every per-row transform: layer filtering, population
# cutoff clipping, and device staging. A per-row array missing from it would
# survive a row mask at full length and silently desynchronize from the rows.
# Optional fields are guarded by presence, so listing them here is safe.
_SEG_ARRAYS = (
    "r_mid",
    "v_hat",
    "L_ang",
    "E_keV",
    "E_start_keV",
    "E_end_keV",
    "E_repr_keV",
    "t_ang",
    "t_start_ang",
    "t_end_ang",
    "t0_ang",
    "elec_id",
    "electron_id",
    "flight_id",
    "substep_id",
    "layer",
)


def _sincsq_lineshape(a_width_j, E_grid, E_r_j):
    """``sinc(a_width*(E - E_res)/pi)**2`` -- the finite-time line profile.

    Previously spelled ``xp.sinc(x)**2`` over a freshly built ``x``; on the GPU
    that was 5+ separate CuPy elementwise kernels (subtract, multiply, divide,
    sinc, square) each allocating a full ``[seg, E]`` temporary -- the launch
    storm that dominated the 300 keV line profile (``cxr.lines`` ~75 s, ~187k
    ``cuLaunchKernel``). Written as one function it JIT-fuses to a single kernel
    under CuPy (see ``xp.fuse`` wrap below); on NumPy it runs eager but is
    bit-for-bit identical to the old ``np.sinc(x)**2`` -- ``np.sinc`` is exactly
    ``y = pi*where(x==0, 1e-20, x); sin(y)/y`` -- so CPU goldens are unchanged.
    """
    x = a_width_j * (E_grid - E_r_j) / xp.pi
    y = xp.pi * xp.where(x == 0.0, 1.0e-20, x)
    s = xp.sin(y) / y
    return s * s


if hasattr(xp, "fuse"):  # CuPy exposes fuse(); NumPy/dpnp do not -> eager fallback
    _sincsq_lineshape = xp.fuse()(_sincsq_lineshape)


def _line_amp_sq_core(
    chi_re,
    chi_im,
    u_re,
    u_im,
    v_dot_kg,
    g_dot_e,
    om,
    v_dot_e,
    vdg,
    k_dot_g,
    k_dot_v,
    gamma,
    detuning,
):
    """One polarization's |A|^2, |A_PXR|^2, |A_CBS|^2 (Zhai Eq. 13/14) as a single
    fused GPU kernel. Replaces the per-reflection storm of complex CuPy elementwise
    kernels (chi/detuning, the CBS braced product, three ``abs()**2``) that made up
    the bulk of the residual ``cxr.lines`` GPU-idle after the sincsq fuse.

    Split the complex amplitudes into real/imag: A_PXR = chi * f_pxr and
    A_CBS = eUg/m * f_cbs with REAL scalars

        f_pxr = (v.kg * g.e - omega^2 v.e) / detuning
        f_cbs = -({g.e - (v.g)(v.e)} + v.e * {k.g - (k.v)(v.g)}/(v.g)) / (gamma * v.g)

    so every op is real and the kernel fuses. |A_PXR + A_CBS|^2 is expanded on the
    real/imag components, |A_PXR|^2 = |chi|^2 f_pxr^2, |A_CBS|^2 = |eUg/m|^2 f_cbs^2.

    NOT bit-for-bit vs the old complex expression -- the multiply/divide are
    reassociated (chi*(N/detuning) vs (chi/detuning)*N), so the line/PXR/CBS
    goldens move at float-rounding level.
    Ledgered `filtered` (measured reassociation envelope in the row); REGEN still
    required -- regenerate the affected spectrum goldens (regen-golden) before this
    is signed off. Validation: line-amplitude-fusion
    """
    f_pxr = (v_dot_kg * g_dot_e - om * om * v_dot_e) / detuning
    braced_ge = g_dot_e - vdg * v_dot_e
    braced_kg = k_dot_g - k_dot_v * vdg
    f_cbs = -(braced_ge + v_dot_e * braced_kg / vdg) / (gamma * vdg)
    re = chi_re * f_pxr + u_re * f_cbs
    im = chi_im * f_pxr + u_im * f_cbs
    a2 = re * re + im * im
    a2_pxr = (chi_re * chi_re + chi_im * chi_im) * f_pxr * f_pxr
    a2_cbs = (u_re * u_re + u_im * u_im) * f_cbs * f_cbs
    return a2, a2_pxr, a2_cbs


if hasattr(xp, "fuse"):
    _line_amp_sq_core = xp.fuse()(_line_amp_sq_core)


def _dot3_core(a0, a1, a2, b0, b1, b2):
    """Length-3 contraction ``a.b`` written as three fused multiply-adds.

    Serves both ``M @ v`` (``v.shape == (3,)`` -> ``b*`` are scalars) and the
    row-wise dot ``einsum('ij,ij->i', A, B)`` (``b*`` are columns of a second
    ``(N,3)`` array). cuBLAS GEMV/GEMM on an inner dimension of 3 is
    pathologically inefficient -- it is tuned for large K, so the skinny
    length-3 contractions in the line path (``v@g``, ``v@n``, ``k@g``, ``v@e``,
    the ``kg.kg`` / ``v.kg`` einsums, ``r@g`` phase) were dispatched as ~6.3k
    ``internal::gemvx`` launches carrying ~24% of the 300 keV line-path GPU
    time. Spelled elementwise this fuses to a single CuPy kernel (no cuBLAS
    handle, no launch) and runs bit-for-bit identical on NumPy at ``float32``
    (measured; at NumPy ``float64`` the reference dispatches through BLAS and the
    two differ by ~1e-10 relative).

    NOT bit-for-bit vs BLAS: the sum is reassociated to ``(a0*b0 + a1*b1) +
    a2*b2``, which differs from GEMV accumulation at float-rounding level.
    Ledgered `filtered`; REGEN still required -- regenerate the affected spectrum
    goldens (regen-golden) before sign-off.
    Validation: line-gemv-elementwise
    """
    return a0 * b0 + a1 * b1 + a2 * b2


if hasattr(xp, "fuse"):
    _dot3_core = xp.fuse()(_dot3_core)


def _matvec3(m, v):
    """``m @ v`` for ``m.shape == (N, 3)`` and ``v.shape == (3,)`` without a
    cuBLAS GEMV -- see _dot3_core."""
    return _dot3_core(m[:, 0], m[:, 1], m[:, 2], v[0], v[1], v[2])


def _rowdot3(a, b):
    """``einsum('ij,ij->i', a, b)`` for ``(N, 3)`` operands without cuBLAS -- see
    _dot3_core."""
    return _dot3_core(a[:, 0], a[:, 1], a[:, 2], b[:, 0], b[:, 1], b[:, 2])


def _interp1(x, grid, f):
    """Linear interpolation of ``x`` (any shape) against an ascending ``grid``
    (1-D, size n) with a SINGLE shared table ``f`` (1-D, size n), clamped to the
    endpoints like ``xp.interp``. The on-device sibling of ``_batch_interp`` for
    a g-independent table (the absorption coefficient mu(E), shared across
    reflections). Reassociated vs ``xp.interp`` at float-rounding level."""
    n = grid.size
    idx = xp.clip(xp.searchsorted(grid, x), 1, n - 1)
    x0 = grid[idx - 1]
    x1 = grid[idx]
    f0 = f[idx - 1]
    f1 = f[idx]
    y = f0 + (x - x0) / (x1 - x0) * (f1 - f0)
    y = xp.where(x <= grid[0], f[0], y)
    y = xp.where(x >= grid[-1], f[n - 1], y)
    return y


# Pre-tabulated PXR prefactor denominator 4 pi^2 hbar c: one Python float, so
# ``ALPHA_FS * om / _PREF_C1 * ...`` below is the SAME division as the inline
# ``/ (4.0 * xp.pi**2 * HBARC_EV_ANG)`` it replaces (bit-for-bit).
_PREF_C1 = 4.0 * xp.pi**2 * HBARC_EV_ANG


def _interp_index(x, grid):
    """Shared linear-interp bookkeeping for ``x`` against an ascending ``grid``:
    the clipped bracket index, the blend fraction, and the below/above endpoint
    masks. The batched line path interpolates the SAME ``E_res`` on the SAME
    ``E_tab_g`` grid four+one times (chi/U real/imag + mu); computing the
    ``searchsorted``/clip/fraction ONCE here and gathering per table (see
    ``_interp_gather{2d,1d}``) removes that fivefold-redundant index launch
    storm. The per-table gather+blend (``_interp_gather2d`` / ``_interp_gather1d``)
    is bit-for-bit the old per-reflection blend, so no new reassociation beyond
    the ``line-hkl-batch`` debt the batched path already carries.

    Namespace comes from the operands (``array_namespace``), not from the
    selected backend: the bracket is pure index algebra with no device state,
    and the tabulation tests call it on host fp64 grids that must interpolate
    identically whatever backend the session selected."""
    _xp = array_namespace(x, grid)
    n = grid.size
    idx = _xp.clip(_xp.searchsorted(grid, x), 1, n - 1)
    frac = (x - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
    below = x <= grid[0]
    above = x >= grid[-1]
    return idx, frac, below, above


def _log_interp_fraction(x, grid, idx):
    """Stable log-energy interpolation fraction for a precomputed bracket.

    ``log(x/x0) / log(x1/x0)`` is evaluated with ``log1p`` so adjacent
    float32 edge nodes do not lose their separation to cancellation.
    Endpoint masks remain the responsibility of the gather, matching the
    existing linear-table interpolation policy.

    Validation: line-absorption-tabulation
    """
    _xp = array_namespace(x, grid, idx)
    x0 = grid[idx - 1]
    x1 = grid[idx]
    bounded_x = _xp.minimum(_xp.maximum(x, x0), x1)
    return _xp.log1p((bounded_x - x0) / x0) / _xp.log1p((x1 - x0) / x0)


def _interp_elemental_mu(idx, log_frac, below, above, log_mu_table):
    """Interpolate elemental ``log(mu_i)`` rows, exponentiate, then sum.

    xraydb's non-``f1`` Chantler rule is log-linear in energy. Since
    ``mu_i = 2 r_e lambda n_i f2_i`` is proportional to ``f2_i / E``, each
    ``log(mu_i)`` is linear on the same native interval. Compound attenuation
    is the sum of the interpolated elemental coefficients, not a log-linear
    interpolation of their total. Validation: line-absorption-tabulation
    """
    _xp = array_namespace(idx, log_frac, log_mu_table)
    f0 = log_mu_table[:, idx - 1]
    values = _xp.exp(f0 + log_frac[None, ...] * (log_mu_table[:, idx] - f0))
    endpoint_shape = (log_mu_table.shape[0],) + (1,) * idx.ndim
    low = _xp.exp(log_mu_table[:, 0]).reshape(endpoint_shape)
    high = _xp.exp(log_mu_table[:, -1]).reshape(endpoint_shape)
    values = _xp.where(below[None, ...], low, values)
    values = _xp.where(above[None, ...], high, values)
    return _xp.sum(values, axis=0)


def _elemental_log_mu_table(composition, energy_grid):
    """Return CPU ``log(mu_i)`` rows [log(1/Angstrom)] for ``composition``.

    Validation: line-absorption-tabulation
    """
    rows = [
        np.asarray(_mu_total_inv_ang([(element, density)], energy_grid))
        for element, density in composition
    ]
    return np.log(np.stack(rows, axis=0))


def _line_tabulation_grid(crystal_info, composition, lo, hi):
    """Shared 1 eV/native-Chantler line grid, including absorber elements.

    Validation: line-absorption-tabulation
    """
    from ....materials.atomic import load_henke

    grids = [np.arange(lo, hi + 1.0, 1.0)]
    elements = {element for element, _ in crystal_info["basis"]}
    elements.update(element for element, _density in composition)
    for element in elements:
        try:
            native_energy = load_henke(element)[0]
            grids.append(native_energy[(native_energy >= lo) & (native_energy <= hi)])
        except Exception:
            pass
    return np.unique(np.concatenate(grids))


def _interp_gather2d(idx, frac, below, above, tables, gcol):
    """Per-column gather+blend for the shared ``_interp_index`` bracket against
    per-``g`` ``tables`` of shape ``(N_g, n)`` (``gcol == arange(N_g)``, hoisted
    once). Bit-for-bit the batched per-reflection ``xp.interp`` blend it replaces
    (Validation: line-hkl-batch). This remains unchanged debt."""
    _xp = array_namespace(idx, frac, tables)
    f0 = tables[gcol, idx - 1]
    y = f0 + frac * (tables[gcol, idx] - f0)
    y = _xp.where(below, tables[gcol, 0], y)
    y = _xp.where(above, tables[gcol, tables.shape[1] - 1], y)
    return y


def _interp_gather1d(idx, frac, below, above, f):
    """Single-table gather+blend for the shared ``_interp_index`` bracket
    against a g-independent 1-D table ``f`` (the mu(E) column). Bit-for-bit
    identical to ``_interp1``."""
    _xp = array_namespace(idx, frac, f)
    f0 = f[idx - 1]
    y = f0 + frac * (f[idx] - f0)
    y = _xp.where(below, f[0], y)
    y = _xp.where(above, f[f.size - 1], y)
    return y


_INTERP_GATHER_LINE_TABLES_F32 = None
if hasattr(xp, "ElementwiseKernel"):
    # A single launch gathers chi/U real+imag and mu for every (segment, g)
    # pair.  Explicit round-to-nearest intrinsics prevent NVCC from contracting
    # the old f0 + frac*(f1-f0) sequence into an FMA, preserving the accepted
    # line-hkl-batch rounding for each interpolation blend.
    _INTERP_GATHER_LINE_TABLES_F32 = xp.ElementwiseKernel(
        "raw I idx, raw float32 frac, raw float32 log_frac, raw bool below, raw bool above, "
        "raw float32 chi_re_tab, raw float32 chi_im_tab, "
        "raw float32 u_re_tab, raw float32 u_im_tab, raw float32 log_mu_tab, "
        "int32 n_g, int32 n_mu, int32 n_tab",
        "float32 chi_re, float32 chi_im, float32 u_re, float32 u_im, float32 mu",
        r"""
        const long long bracket = (long long)idx[i];
        const int g = (int)(i % (size_t)n_g);
        const long long base = (long long)g * (long long)n_tab;
        const long long lo = base + bracket - 1;
        const long long hi = base + bracket;
        const bool use_lo = below[i];
        const bool use_hi = above[i];
        const float f = frac[i];

        float a0 = chi_re_tab[lo];
        chi_re = use_lo ? chi_re_tab[base]
                 : (use_hi ? chi_re_tab[base + n_tab - 1]
                           : __fadd_rn(a0, __fmul_rn(f, __fsub_rn(chi_re_tab[hi], a0))));
        a0 = chi_im_tab[lo];
        chi_im = use_lo ? chi_im_tab[base]
                 : (use_hi ? chi_im_tab[base + n_tab - 1]
                           : __fadd_rn(a0, __fmul_rn(f, __fsub_rn(chi_im_tab[hi], a0))));
        a0 = u_re_tab[lo];
        u_re = use_lo ? u_re_tab[base]
               : (use_hi ? u_re_tab[base + n_tab - 1]
                         : __fadd_rn(a0, __fmul_rn(f, __fsub_rn(u_re_tab[hi], a0))));
        a0 = u_im_tab[lo];
        u_im = use_lo ? u_im_tab[base]
               : (use_hi ? u_im_tab[base + n_tab - 1]
                         : __fadd_rn(a0, __fmul_rn(f, __fsub_rn(u_im_tab[hi], a0))));

        const float lf = log_frac[i];
        mu = 0.0f;
        for (int element = 0; element < n_mu; ++element) {
            const long long mu_base = (long long)element * (long long)n_tab;
            const long long mu_lo = mu_base + bracket - 1;
            const float log0 = log_mu_tab[mu_lo];
            const float log_mu = use_lo ? log_mu_tab[mu_base]
                               : (use_hi ? log_mu_tab[mu_base + n_tab - 1]
                                         : __fadd_rn(log0, __fmul_rn(lf, __fsub_rn(log_mu_tab[mu_base + bracket], log0))));
            mu += expf(log_mu);
        }
        """,
        "cxr_interp_gather_line_tables_f32",
    )


def _interp_gather_line_tables(
    idx,
    frac,
    log_frac,
    below,
    above,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    log_mu_tab,
):
    """Gather every line-coupling table from one shared interpolation bracket."""
    if (
        _INTERP_GATHER_LINE_TABLES_F32 is not None
        and idx.dtype.kind in "iu"
        and frac.dtype == xp.float32
        and chi_re_tab.dtype == xp.float32
        and log_mu_tab.dtype == xp.float32
    ):
        n_g, n_tab = chi_re_tab.shape
        n_mu = log_mu_tab.shape[0]
        flat = _INTERP_GATHER_LINE_TABLES_F32(
            idx,
            frac,
            log_frac,
            below,
            above,
            chi_re_tab,
            chi_im_tab,
            u_re_tab,
            u_im_tab,
            log_mu_tab,
            np.int32(n_g),
            np.int32(n_mu),
            np.int32(n_tab),
            size=idx.size,
        )
        return tuple(value.reshape(idx.shape) for value in flat)

    n_g = chi_re_tab.shape[0]
    gcol = xp.arange(n_g)
    return (
        _interp_gather2d(idx, frac, below, above, chi_re_tab, gcol),
        _interp_gather2d(idx, frac, below, above, chi_im_tab, gcol),
        _interp_gather2d(idx, frac, below, above, u_re_tab, gcol),
        _interp_gather2d(idx, frac, below, above, u_im_tab, gcol),
        _interp_elemental_mu(idx, log_frac, below, above, log_mu_tab),
    )


def _line_kin_core(vx, vy, vz, gx, gy, gz, denom, g2, n_dot_g):
    """Resonance frequency + photon kinematics with g-only algebra hoisted.

    For unit detector direction ``n``, ``|(omega*n)+g|^2-omega^2`` is exactly
    ``g^2 + 2*omega*(n.g)``. Likewise ``k.g = omega*(n.g)`` and
    ``v.(k+g) = v.g + k.v``. ``g2`` and ``n_dot_g`` are therefore computed once
    per reflection/orientation rather than rebuilding ``k`` and ``k+g`` for every
    segment. The identities are exact; floating-point association differs from
    the expanded vector form at rounding level.

    The caller passes the in-medium ``denom`` and ``n_dot_g`` (the latter
    carrying its factor of ``Re n``), both per (segment, g) rather than hoisted.
    Every identity above survives that substitution unchanged -- see
    ``_in_medium_kinematics``.
    """
    v_dot_g = vx * gx + vy * gy + vz * gz
    omega_res = v_dot_g / denom
    k_dot_v = omega_res * (1.0 - denom)
    k_dot_g = omega_res * n_dot_g
    v_dot_kg = v_dot_g + k_dot_v
    detuning = g2 + 2.0 * k_dot_g
    return omega_res, v_dot_g, detuning, k_dot_g, v_dot_kg, k_dot_v


if hasattr(xp, "fuse"):
    _line_kin_core = xp.fuse()(_line_kin_core)


def _line_weight_core(omega_res, t_L, L_esc, mu, alpha_fs, pref_c1):
    """PXR prefactor incl. the Beer-Lambert escape factor as one fused kernel:
    ``T_abs = exp(-L_esc mu)`` folded into ``alpha_fs om / (4 pi^2 hbar c)
    t_L^2 T_abs``. Pure kernel-merge of the identical inline step 6/7 expression
    (``t_L**2`` written ``t_L*t_L``) -> bit-for-bit, no new validation debt."""
    T_abs = xp.exp(-(L_esc * mu))
    return alpha_fs * omega_res / pref_c1 * (t_L * t_L) * T_abs


if hasattr(xp, "fuse"):
    _line_weight_core = xp.fuse()(_line_weight_core)


# Relative movement of ``denom`` on the last fixed-point pass that still counts
# as converged. A genuine contraction moves it by ~delta**3 ~ 1e-15 in float64
# and is floored by float32 rounding (~1e-7) on the device twin; the 2-cycle the
# guard exists to catch moves it by O(1). Five orders of margin either side.
_RESONANCE_ROOT_RTOL = 1e-3


def _in_medium_kinematics(v_dot_n, v_dot_g, n_re_tab, E_tab):
    """In-medium resonance denominator and refractive factor per segment.

    Energy-momentum conservation on a segment is ``omega = v.(k + g)``, and the
    Maxwell dispersion relation in the bulk dielectric is ``k = n(omega) omega``
    along the observation direction, so the vacuum resonance
    ``omega_res = v.g / (1 - v.n_hat)`` becomes implicit:

        omega_res = v.g / (1 - Re n(omega_res) (v.n_hat))

    Only ``Re n`` enters. ``Im n`` is the same absorption already carried as the
    Beer-Lambert ``mu(E)`` escape factor, so folding it in here as well would
    double-count it.

    Solved by fixed-point iteration from the vacuum root. The map's derivative is
    ``(v.n_hat) (dn/dE) (dE/ddenom) ~ delta ~ 1e-5``, so each pass gains ~5
    digits and two are already at float64 rounding; three are taken for margin.

    That contraction rate assumes ``Re n = 1 - delta`` with ``delta ~ 1e-5-1e-3``,
    which is only true in the X-ray regime. A segment scattered nearly
    perpendicular to ``g`` puts the vacuum root down in the optical/UV, where the
    tabulations are honest about ``Re n > 1`` (carbon: 6.24-285 eV, peaking at
    4.766 at 6.40 eV). There ``Re n (v.n_hat)`` can approach unity, ``denom``
    collapses toward a spurious Cherenkov-like zero, the map stops contracting,
    and the iteration settles into a 2-cycle instead: three passes then return
    whichever half of the cycle pass three lands on, and a keV-scale ``E_res``
    comes back attached to a ``v.g`` four orders below the median. Downstream the
    CBS amplitude's ``1/(gamma (v.g)^2)`` turns that into a line total ten orders
    too large, and it is finite, so nothing flags it.

    So convergence is checked rather than assumed: the last pass must have moved
    ``denom`` by less than ``_RESONANCE_ROOT_RTOL``, which a genuine contraction
    clears by five orders. Pairs that fail carry NaN out of ``denom`` and drop on
    the caller's finite mask, the same route out-of-range tabulation energies
    take. Rejection is the correct handling, not a workaround: those samples
    violate the CBS amplitude's own perturbative validity condition
    ``|U_g| g^2 / (gamma m c^2 (v.g)^2) << 1`` -- by a factor 15.6 on the sample
    this was traced from.

    Returns ``(denom, n_re)`` with the shape of ``v_dot_g``: the caller forms the
    remaining in-medium scalars from ``n_re`` rather than re-deriving them, since
    ``k.v = omega (1 - denom)`` still holds exactly while ``k.g`` and ``k^2``
    pick up one and two powers of ``n_re`` respectively.

    Out-of-range tabulation energies carry NaN out of ``n_re``, which propagates
    to ``denom`` and drops the segment on the caller's finite/window mask -- the
    same convention as the chi/U/mu tabulations.

    Validation: xray-in-medium-resonance
    """
    _xp = array_namespace(v_dot_n, v_dot_g, n_re_tab, E_tab)
    denom = 1.0 - v_dot_n
    n_re = None
    previous = denom
    for _ in range(3):
        E_res = HBARC_EV_ANG * (v_dot_g / denom)
        _ix, _fr, _blw, _abv = _interp_index(E_res, E_tab)
        n_re = _interp_gather1d(_ix, _fr, _blw, _abv, n_re_tab)
        previous = denom
        denom = 1.0 - n_re * v_dot_n
    # NaN denominators compare False here and stay NaN, which is the wanted
    # outcome: an out-of-range root is already a rejected pair.
    settled = _xp.abs(denom - previous) <= _RESONANCE_ROOT_RTOL * _xp.abs(denom)
    denom = _xp.where(settled, denom, REAL(_xp.nan))
    return denom, n_re


def _segments_in_layer(segments, L):
    """A view of `segments` restricted to those emitted in layer index L, keeping
    the scalar fields (Ne, thickness_ang, ...) so the per-electron normalization
    and geometry are unchanged. For a single-layer stack, layer 0 returns all
    segments (same values), so the single-material path is unaffected."""
    mask = segments["layer"] == L
    out = dict(segments)
    for k in _SEG_ARRAYS:
        if k in out:
            out[k] = out[k][mask]
    return out


def _spliced_stopping_magnitude_xp(E_eval, layer_index, layer_compositions):
    """Spliced |dE/ds| [keV/Ang] per row at the given evaluation energy.

    This is the device-array twin of
    ``transport.spliced_stopping_keV_per_ang``: the rows here may live on
    the GPU, so it cannot delegate to the host helper. It must stay in step
    with it -- ``test_stopping_mirrors_agree`` pins the two together.
    """
    tau = E_eval / REAL(_MC2_KEV)
    gamma = REAL(1.0) + tau
    beta_sq = REAL(1.0) - REAL(1.0) / (gamma * gamma)
    f_minus = (
        REAL(1.0)
        - beta_sq
        + (tau * tau / REAL(8.0) - (REAL(2.0) * tau + REAL(1.0)) * REAL(_LN2)) / (gamma * gamma)
    )

    stopping = xp.zeros_like(E_eval)
    for index, comp in enumerate(layer_compositions):
        joy_luo_total = xp.zeros_like(E_eval)
        bs_total = xp.zeros_like(E_eval)
        for element, n_i in comp:
            params = TRANSPORT_ELEMENTS[element]
            Z = REAL(params["Z"])
            J = REAL(params["J_keV"])
            k = REAL(0.731 + 0.0688 * np.log10(float(Z)))
            coeff = REAL((n_i / 0.602214076) * float(Z))
            E_cross = REAL(
                _element_crossover_keV(
                    element, float(Z), float(params["A"]), float(params["J_keV"])
                )
            )
            I_rel = J / REAL(_MC2_KEV)
            use_bs = E_eval >= E_cross
            joy_luo_total += xp.where(
                use_bs, REAL(0.0), coeff * xp.log(REAL(1.166) * (E_eval + k * J) / J)
            )
            bs_total += xp.where(
                use_bs,
                coeff
                * (xp.log(tau * tau * (tau + REAL(2.0)) / (REAL(2.0) * I_rel * I_rel)) + f_minus),
                REAL(0.0),
            )
        # magnitude: callers divide an energy drop by it
        layer_stopping = (
            REAL(7.85e-4) / E_eval * joy_luo_total + REAL(_BS_PREFACTOR) / beta_sq * bs_total
        )
        stopping = xp.where(layer_index == index, layer_stopping, stopping)
    return stopping


def _clip_segments_to_cutoff(segments, E_cut_keV, composition, layers=None):
    """Clip terminal material flights to a population-specific energy floor.

    Transport may be shared by radiation populations with different cutoffs.
    Reapply the transport core's stopping rule here so a higher-cutoff consumer
    cannot recover radiation from the lower-cutoff tail. Segment start time and
    energy remain unchanged; length and midpoint are shortened from the original
    start point.

    Rows carrying ``E_end_keV`` came from ``energy_model="midpoint"``, so the
    truncation distance solves ``E_end == E_cut`` under that rule (as the
    transport core does) and the shortened flight's end/representative state is
    reconstructed exactly rather than dropped. Frozen rows keep the historical
    left-endpoint solve bit-for-bit.
    """
    if E_cut_keV is None:
        return segments

    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)
    keep = seg_E >= REAL(E_cut_keV)
    # Gather with an explicit index, NOT with the boolean mask. CuPy has to know
    # the survivor COUNT to size a boolean-mask result, and reads it back to the
    # host to find out -- a blocking device->host sync for EVERY key, i.e. one
    # stalled queue per segment array per call. Resolving the mask to an index
    # once pays that readback a single time and then gathers with a known output
    # size. Same survivors in the same order, so the selection is bit-for-bit
    # what the mask produced. Measured on qlmc (RTX 5080, hopg Ne=450, 3 cases
    # after warmup): this line alone was 6.89 ms/case -- the single largest cost
    # in the spectrum phase -- and the phase went 23.9 -> 17.8 ms/case, with
    # blocking device->host syncs 46 -> 19 per case and kernel launches 471 ->
    # 366. Keep it an index; reverting to `out[key][keep]` restores the stall.
    keep_idx = xp.nonzero(keep)[0]
    # ``keep_idx`` lives on the selected backend, so the rows it gathers have to
    # as well. This runs BEFORE ``_segments_on_device``, i.e. on whatever the
    # transport handed us -- host arrays on every accelerator run -- and
    # ``host_rows[device_idx]`` is not an upload: NumPy asks the index for
    # ``__array__`` and CuPy refuses to sync implicitly. Stage the row here (a
    # no-op once the caller already staged, and no copy at all on NumPy) rather
    # than fetch the index back to the host, which would undo the readback this
    # very index exists to pay only once. Staging with ``_stage_row`` rather
    # than a bare ``asarray`` keeps the clip's own outputs at the same dtype
    # ``_segments_on_device`` would give them a moment later, so the fields it
    # derives below (E_end/E_repr/t_end) are consistent with the rows it emits
    # instead of mixing transport fp64 with a REAL solve.
    out = dict(segments)
    for key in _SEG_ARRAYS:
        if key in out:
            out[key] = _stage_row(out[key])[keep_idx]

    E = xp.asarray(out["E_keV"], dtype=REAL)
    old_L = xp.asarray(out["L_ang"], dtype=REAL)
    layer_index = xp.asarray(out["layer"])
    layer_compositions = [composition] if layers is None else [item[2] for item in layers]

    def _stopping_at(E_eval):
        return _spliced_stopping_magnitude_xp(E_eval, layer_index, layer_compositions)

    midpoint_rows = "E_end_keV" in out and "E_repr_keV" in out
    E_cut = REAL(E_cut_keV)
    if midpoint_rows:
        E_repr_cut = REAL(0.5) * (E + E_cut)
        cutoff_L = (E - E_cut) / _stopping_at(E_repr_cut)
    else:
        cutoff_L = (E - E_cut) / _stopping_at(E)
    new_L = xp.minimum(old_L, xp.maximum(REAL(0.0), cutoff_L))
    direction = xp.asarray(out["v_hat"], dtype=REAL)
    old_mid = xp.asarray(out["r_mid"], dtype=REAL)
    start = old_mid - REAL(0.5) * old_L[:, None] * direction
    out["L_ang"] = new_L
    out["r_mid"] = start + REAL(0.5) * new_L[:, None] * direction
    if midpoint_rows:
        # A clipped flight is a shorter flight, so the transported end state no
        # longer describes it -- but the clip rule above is the transport core's
        # own cutoff solve, whose end state is E_cut by construction.
        shortened = new_L < old_L
        out["E_end_keV"] = xp.where(shortened, E_cut, xp.asarray(out["E_end_keV"], dtype=REAL))
        out["E_repr_keV"] = xp.where(
            shortened, E_repr_cut, xp.asarray(out["E_repr_keV"], dtype=REAL)
        )
        if "t_end_ang" in out:
            t_start = xp.asarray(out["t_start_ang"], dtype=REAL)
            t_end_clipped = t_start + new_L / beta_from_keV(E_repr_cut)
            out["t_end_ang"] = xp.where(
                shortened, t_end_clipped, xp.asarray(out["t_end_ang"], dtype=REAL)
            )
    return out


def _segments_on_device(segments):
    """One backend copy of the segment arrays, to be shared by a case's kernels.

    Each kernel reaches for what it needs with ``xp.asarray(segments[k],
    dtype=REAL)``, so on an accelerator every call re-uploads its own slice of
    the transport's output -- and a case runs the line sum (twice when it also
    wants the coherent one) and the brem sum over the SAME segments, once per
    layer. Staging the upload here turns those calls into no-ops: `xp.asarray`
    hands back a device array that already carries the requested dtype without
    copying it. The kernels therefore need not know whether their caller staged
    or not, and the ones still called on host segments (the repair paths) behave
    exactly as before.

    Values are unchanged either way: this is the same cast to `REAL` the kernels
    would each have done, hoisted to happen once.

    On the CPU backend `xp` is NumPy and the arrays already carry the kernels'
    dtype, so nothing is copied at all."""
    out = dict(segments)
    for k in _SEG_ARRAYS:
        a = out.get(k)
        if a is None:
            continue
        out[k] = _stage_row(a)
    return out


def _stage_row(a):
    """One segment row array on the selected backend at the kernels' dtype.

    Float rows carry ``REAL``; index rows (``elec_id``, ``flight_id``,
    ``layer``, ...) keep their integer dtype. Shared by ``_segments_on_device``
    and by the cutoff clip, which runs before staging and so must apply the
    same rule rather than leave the rows it emits at the transport's dtype.
    """

    return xp.asarray(a, dtype=REAL) if a.dtype.kind == "f" else xp.asarray(a)


def _polarization_pair(k_hat, g_vec):
    n_plane = np.cross(k_hat, g_vec)
    npl = np.linalg.norm(n_plane)
    if npl < 1e-12:
        tmp = np.array([0.0, 1.0, 0.0])
        n_plane = np.cross(k_hat, tmp)
        npl = np.linalg.norm(n_plane)
    e_s = n_plane / npl
    e_p = np.cross(e_s, k_hat)
    return e_s, e_p / np.linalg.norm(e_p)


def _observation_direction(theta_obs_rad, n_hat):
    if n_hat is None:
        return np.array([np.sin(theta_obs_rad), 0.0, np.cos(theta_obs_rad)])
    n_hat = np.asarray(n_hat, dtype=float)
    return n_hat / np.linalg.norm(n_hat)


def _validate_groove_escape_direction(n_hat, groove):
    """Require the exact working-facet normal encoded by ``groove``."""
    tp = groove.tilt_polar_rad
    expected = np.array([np.cos(tp), 0.0, -np.sin(tp)])
    if not np.allclose(n_hat, expected, rtol=0.0, atol=_THETA_TOL):
        raise ValueError(
            "groove escape requires n_hat = (cos(tilt_polar), 0, "
            "-sin(tilt_polar)) along the working-facet normal"
        )


def _escape_length(z_mid, thickness, n_z):
    return z_mid / -n_z if n_z < 0 else (thickness - z_mid) / n_z


def _segment_escape_distance(segments, n_hat, *, xp):
    """Photon distance from segment midpoints to their first crystal face.

    Omitting both transverse dimensions recovers the original z-only slab
    escape path.  A finite rectangular footprint instead chooses the nearest
    of all six prism faces along the fixed far-field observation direction.
    """
    r = xp.asarray(segments["r_mid"], dtype=REAL)
    width = segments.get("crystal_width_ang")
    height = segments.get("crystal_height_ang")
    if width is None or height is None:
        return _escape_length(r[:, 2], segments["thickness_ang"], n_hat[2])
    distance, _ = first_prism_exit(
        r,
        xp.asarray(n_hat, dtype=REAL),
        z_min_ang=0.0,
        z_max_ang=segments["thickness_ang"],
        width_ang=width,
        height_ang=height,
        xp=xp,
    )
    return distance


def _flight_blocks(bounds, chunk):
    """Split flight groups into row blocks of at most ``chunk`` rows.

    ``bounds`` holds the group start offsets plus the total row count. Yields
    ``(first_group, last_group_exclusive)`` pairs whose row spans never split a
    group, which is what keeps a flight's substeps inside one coherent sum. A
    single group larger than ``chunk`` is emitted whole rather than split.
    """
    n_groups = bounds.size - 1
    ka = 0
    for kb in range(1, n_groups + 1):
        if bounds[kb] - bounds[ka] >= chunk:
            yield ka, kb
            ka = kb
    if ka < n_groups:
        yield ka, n_groups


def _reflection_tabulation(crystal, hkl, E_tab, B_ang2, use_henke):
    """Tabulate one reflection's susceptibility and structure factor.

    ``U_g/m_e`` is stored rather than ``U_g`` because the mass scaling is
    energy-independent: the division happens once per reflection instead of
    once per interpolated pair.

    Both accumulation routes call this rather than ``chi_g``/``U_g`` directly,
    so the per-reflection evaluation count has exactly one seam -- the coherent
    emission tests patch ``chi_g`` here and see both routes.
    """
    chi_tab = np.asarray(chi_g(crystal, hkl, E_tab, B_ang2, use_henke))
    u_tab = np.asarray(U_g(crystal, hkl, E_tab, B_ang2, use_henke)) / M_E_EV
    return chi_tab, u_tab
