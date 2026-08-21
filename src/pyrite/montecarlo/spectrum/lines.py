"""
montecarlo.spectrum

Radiation from the transported segments (Zhai SI S1): the finite-interaction-
time CXR (PXR + CBS) line spectrum mc_spectrum, its solid-angle-integrated
wrapper, the Bethe-Heitler bremsstrahlung background, and the external-brem
loader. The array-heavy inner loops run on the GPU backend (``xp``) when
available.
"""

import numpy as np

from ...materials.attenuation import (
    _mu_total_inv_ang,
    _normalize_composition,
    _stack_tau,
)
from ...materials.crystal import (
    ALPHA_FS,
    CRYSTALS,
    HBARC_EV_ANG,
    M_E_EV,
    U_g,
    chi_g,
    reciprocal_g_vector,
    refractive_index,
)
from .._backend import REAL, _to_cpu, xp
from ..geometry import _mosaic_quadrature, _orientation_R, first_prism_exit
from ..groove import _THETA_TOL, escape_distance_ang
from ..transport import (
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
_USE_JIT_LINE_REDUCTION = True
_USE_JIT_COHERENT_REDUCTION = True
_USE_JIT_COHERENT_STREAM = True
_JIT_LINE_BATCH_TARGET = 400_000
_JIT_COHERENT_PAIR_TARGET = 1_000_000


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
    the ``line-hkl-batch`` debt the batched path already carries."""
    n = grid.size
    idx = xp.clip(xp.searchsorted(grid, x), 1, n - 1)
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
    x0 = grid[idx - 1]
    x1 = grid[idx]
    bounded_x = xp.minimum(xp.maximum(x, x0), x1)
    return xp.log1p((bounded_x - x0) / x0) / xp.log1p((x1 - x0) / x0)


def _interp_elemental_mu(idx, log_frac, below, above, log_mu_table):
    """Interpolate elemental ``log(mu_i)`` rows, exponentiate, then sum.

    xraydb's non-``f1`` Chantler rule is log-linear in energy. Since
    ``mu_i = 2 r_e lambda n_i f2_i`` is proportional to ``f2_i / E``, each
    ``log(mu_i)`` is linear on the same native interval. Compound attenuation
    is the sum of the interpolated elemental coefficients, not a log-linear
    interpolation of their total. Validation: line-absorption-tabulation
    """
    f0 = log_mu_table[:, idx - 1]
    values = xp.exp(f0 + log_frac[None, ...] * (log_mu_table[:, idx] - f0))
    endpoint_shape = (log_mu_table.shape[0],) + (1,) * idx.ndim
    low = xp.exp(log_mu_table[:, 0]).reshape(endpoint_shape)
    high = xp.exp(log_mu_table[:, -1]).reshape(endpoint_shape)
    values = xp.where(below[None, ...], low, values)
    values = xp.where(above[None, ...], high, values)
    return xp.sum(values, axis=0)


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
    from ...materials.atomic import load_henke

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
    (Validation: line-hkl-batch, unchanged debt)."""
    f0 = tables[gcol, idx - 1]
    y = f0 + frac * (tables[gcol, idx] - f0)
    y = xp.where(below, tables[gcol, 0], y)
    y = xp.where(above, tables[gcol, tables.shape[1] - 1], y)
    return y


def _interp_gather1d(idx, frac, below, above, f):
    """Single-table gather+blend for the shared ``_interp_index`` bracket
    against a g-independent 1-D table ``f`` (the mu(E) column). Bit-for-bit
    identical to ``_interp1``."""
    f0 = f[idx - 1]
    y = f0 + frac * (f[idx] - f0)
    y = xp.where(below, f[0], y)
    y = xp.where(above, f[f.size - 1], y)
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
    settled = xp.abs(denom - previous) <= _RESONANCE_ROOT_RTOL * xp.abs(denom)
    denom = xp.where(settled, denom, REAL(xp.nan))
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
    out = dict(segments)
    for key in _SEG_ARRAYS:
        if key in out:
            out[key] = out[key][keep_idx]

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
        out[k] = xp.asarray(a, dtype=REAL) if a.dtype.kind == "f" else xp.asarray(a)
    return out


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


def mc_spectrum(
    segments,
    E_grid_eV,
    crystal,
    hkl_list,
    theta_obs_rad=np.deg2rad(119.0),
    B_ang2=None,
    use_henke=True,
    absorber_element="C",
    chunk=40000,
    n_hat=None,
    composition=None,
    beam_uvw=None,
    azimuth_rad=0.0,
    recip_miscut_rad: tuple[float, float] | None = None,
    sinc_cutoff=None,
    components=False,
    layers=None,
    mosaic_fwhm_rad=None,
    mosaic_nodes=1,
    surface_hkl: tuple[int, int, int] | None = None,
    groove=None,
    coherent=False,
    electron_limit=None,
    E_cut_keV=None,
    _table_cache=None,
):
    """
    Per-electron CXR spectrum d2N/dE dOmega [photons / eV / sr / electron]
    on E_grid_eV, summed incoherently over the trajectory segments and the
    listed reflections (their resonances are spectrally separated and their
    relative phase decorrelates over the segment midpoints, so cross-g
    coherence is negligible -- bounded under `cross-reflection-coherence`).

    Per segment and reflection (Zhai SI Eqs. 5-7, nonrelativistic):
      omega_res = beta v_hat.g / (1 - beta v_hat.n)         [Eq. 10 resonance]
      d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |A|^2 t_L^2
                      sinc^2[(1 - beta v.n)(omega-omega_res) t_L / 2] T_abs
    with A = A_PXR + A_CBS per polarization (Feranchuk Eqs. 13/14 evaluated
    at omega_res with the segment's velocity vector), t_L = L_seg/beta, and
    T_abs the Beer-Lambert escape factor from the segment midpoint.

    The finite-time factor follows from integrating ``exp(i 2 P t)`` over a
    centered segment duration ``t_L``, giving
    ``t_L**2 sinc(P t_L / pi)**2`` under NumPy's normalized-sinc convention,
    with ``P = (1 - beta v_hat.n)(omega - omega_res) / 2``. It assumes a
    constant segment velocity and amplitude. Writing this factor as
    ``|Q(P, t_L)|**2``, its exact normalized long-duration limit is
    ``|Q|**2 / (pi t_L) -> delta(P)`` distributionally. At zero detuning the
    unnormalized factor has value ``t_L**2``.

    Validation: finite-time-lineshape

    n_hat: detector direction in the SAMPLE frame; overrides theta_obs_rad
    when given (use tilted_geometry() for a tilted sample).
    composition: [(element, n_per_Ang3), ...] for compound self-absorption;
    defaults to the single absorber_element at the crystal's total atom
    density (exact for elemental crystals).

    Validation: line-absorption-tabulation

    layers: optional film-on-substrate absorber stack
    [(z_top, z_bot, composition), ...] (top/entrance first). When given, the
    escape attenuation is the piecewise mu_i*dz_i sum across the whole stack
    rather than the single slab; the RADIATION still comes from crystal/hkl_list
    (the film). None -> single slab (bit-for-bit unchanged).

    Validation: self-absorption

    Finite transverse dimensions stored on ``segments`` attenuate each photon to
    the first of the rectangular prism's six faces along the fixed far-field
    ``n_hat``. When both dimensions are omitted, the original z-only slab
    attenuation branch is retained unchanged.

    Validation: finite-transverse-crystal

    beam_uvw: CRYSTAL AXIS along the slab normal (+z). Default None keeps the
    construction-frame convention, i.e. [001] (the c-axis for hexagonal
    crystals, a cube edge for cubic ones). Passing e.g. (1, 1, 1) cuts the
    slab perpendicular to the [111] direct-lattice direction: every g in
    hkl_list is rotated by the MINIMAL rotation taking [uvw] -> +z, then by
    azimuth_rad about +z (the in-plane setting of the crystal relative to
    the detector azimuth -- it matters for individual family members).

    surface_hkl: reciprocal-lattice PLANE NORMAL along the slab normal (+z).
    This is mutually exclusive with beam_uvw and is the exact cleavage-plane
    contract for nonorthogonal crystals. The reciprocal vector is rotated by
    the minimal proper rotation, followed by azimuth_rad about +z.

    Validation: surface-hkl-orientation

    recip_miscut_rad: optional (polar_rad, azim_rad) crystal miscut -- an
    EXTRA tilt applied only to the reciprocal vectors g, leaving the
    transported slab normal/beam_dir (set by tilted_geometry() upstream,
    outside this function) untouched. None (default) is a strict no-op: g
    stays aligned with the slab normal, today's behavior bit-for-bit. See
    :func:`geometry._orientation_R`. Not wired into any grid/study yet; the
    escape hatch for a future asymmetric reflection (g not parallel to n).

    sinc_cutoff: None (default) evaluates every segment's lineshape over the
    FULL grid (exact). A number C truncates each lineshape at |P t_L| > C,
    i.e. |E - E_res| > C/a_width -- segments are processed in resonance-
    sorted blocks against only the relevant grid window, which is several
    times faster on wide grids. Tail loss is ~1/(pi C) of each line's
    integral (0.3% at C = 100); peak heights are unaffected. Requires a
    UNIFORM E_grid.

    mosaic_fwhm_rad / mosaic_nodes: the EXACT crystal-mosaicity route
    (docs/physics/materials/crystal-mosaicity.md (2)). None / nodes<=1 (the default) is a perfect
    crystal -- today's single-orientation result bit-for-bit. Otherwise the
    spectrum is incoherently averaged over crystallite orientations drawn from a
    Gaussian mosaic of rocking-curve FWHM ``mosaic_fwhm_rad`` [rad], via a 2-D
    Gauss-Hermite product quadrature of ``mosaic_nodes`` nodes per tilt axis (so
    K = mosaic_nodes**2 evaluations of the per-reflection block). Unlike the
    analytic mosaic_fwhm_eV (energy-shift only, applied at detector convolution),
    this broadens BOTH PXR and CBS, captures the amplitude/polarization variation
    across the cone, and yields the correct (generally asymmetric) lineshape and
    integrated yield. Do NOT also apply the analytic term to the result (double
    count); build_cases handles that mutual exclusion.

    Validation: mosaic-mc

    groove: optional GrooveSpec (montecarlo.groove) cutting the beam-entrance
    face into a blazed sawtooth relief profile instead of a flat face. The
    working facet is perpendicular to n_hat and the relief facet is
    perpendicular to the beam (zero shadowing by construction), so a photon's
    straight-line escape path along n_hat is shortened to the nearest working
    facet rather than running the full flat-face distance z_mid/(-n_hat[2])
    (source: elementary ray-plane intersection on a periodic sawtooth --
    montecarlo.groove.escape_distance_ang). This ONLY replaces the escape
    DISTANCE fed into the existing straight-ray incoherent Beer-Lambert
    attenuation (T_abs = exp(-mu*L_esc)); the radiation amplitudes (Eqs.
    13/14) and resonance kinematics (Eq. 10) are unchanged -- grooving is
    purely an absorption-path effect in this v1 model.

    Assumptions: profile invariant along y; photons travel straight along
    n_hat with no wave-optics diffraction off the groove edges (consistent
    with the rest of mc_spectrum's incoherent transport). A finite crystal
    footprint (crystal_width_ang/crystal_height_ang) is permitted and only
    classifies launch hit/miss in transport (hit_frac); the groove escape here
    ignores it and treats the sawtooth as laterally periodic. At emission depth
    z, the side-edge-affected strip has width
    ``L_esc*cos(tp) = z*cot(tp) + O(groove spacing)``; its fractional width is
    ``min(z*cot(tp)/crystal_width + O(groove spacing/crystal_width), 1)``.

    v1 exclusions (raise ValueError rather than silently mismodeling):
    groove with layers (single-slab absorber only); groove with n_hat[2] >= 0
    (escape must be back out the entrance face -- the relief geometry is defined
    for that exit only). None (default) is a strict no-op: today's flat-face
    result bit-for-bit.

    The first working-facet crossing is also the complete material path:
    ``n_hat`` is parallel to every relief facet and points outward through the
    working facets, so an escaped coherent photon cannot intersect a later
    material interval. This differs from a scattered electron, whose arbitrary
    direction can leave one facet and re-enter through another. Source: exact
    periodic ray-plane intersections; see
    ``docs/validation/geometry/blazed-groove-geometry.md``.

    Validation: blazed-groove-geometry

    The line kinematics always run on the IN-MEDIUM photon dispersion: the bulk
    crystal dielectric response gives ``k = n(omega) omega`` with
    ``n = sqrt(1 + chi_0)`` (materials.crystal.refractive_index), which shifts
    the resonance denominator to ``1 - Re n (v.n_hat)`` and carries the
    corresponding ``n`` powers into ``k.g`` and the PXR numerator's ``k^2``.
    Only the real part is applied: ``Im n`` is the same absorption already
    carried by the Beer-Lambert ``mu(E)`` escape factor. Bulk response only --
    interface/Fresnel refraction is not modelled, so grazing observation
    geometry is out of scope. There is no vacuum-dispersion switch; ``k = omega``
    is recovered only in the physical ``chi_0 -> 0`` (high-energy) limit.

    With ``coherent=True`` the segment-to-segment propagation phase rides the
    same dispersion relation: each segment's field picks up
    ``-delta(E) omega(E) L_esc,j`` over its in-crystal escape path, the real
    partner of the Beer-Lambert amplitude factor already applied over that same
    path. Refused for LAYERED absorbers, whose per-layer delta is not modelled.

    Validation: xray-in-medium-resonance, xray-in-medium-propagation-phase

    coherent: opt-in coherent (phased) segment sum. None/False (default) is the
    incoherent path above, bit-for-bit. When True the spectrum is
    ``d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |sum_j A_j Q_j e^{i phi_j}|^2``
    -- the SAME per-segment quantities (resonance omega, complex A = A_PXR+A_CBS
    per polarization, the finite-time factor and the escape factor) accumulated
    as a COMPLEX field per polarization and squared at the end, rather than
    accumulating ``|A_j|^2 * |Q_j|^2`` incoherently. Concretely, per segment the
    complex field is

        E_j(omega) = sqrt(alpha*omega/(4 pi^2 hbar c) * T_abs_j)
                     * A_j * Q_j(omega)
                     * exp{i[omega t_abs,j - (omega n_hat + g).r_j]},

    with the UN-squared finite-time factor ``Q_j = t_L sinc(P t_L / pi)`` (whose
    modulus-square is the incoherent ``t_L^2 sinc^2``), the emission-time phase
    ``omega t_abs,j`` (``t_abs = t_ang + L_ang/(2 beta) + t0_ang``: midpoint
    transport age plus the per-electron bunch offset, in Ang with c=1) and the
    far-field retardation ``omega n_hat.r_j``. Transport keeps ``t_ang`` as
    segment-start age for compatibility; the midpoint correction pairs time
    with the stored midpoint position under the same constant-velocity segment
    assumption as the finite-time factor. The reciprocal-harmonic spatial phase
    ``g.r_j`` follows the repository's structure-factor convention
    ``S(g)=sum F exp(+i g.R)``, whose susceptibility harmonic is
    ``chi_g exp(-i g.r)``. The coherent sum runs WITHIN each reflection and
    orientation; reflections/orientations still add incoherently, i.e. the
    cross-reflection terms of ``|sum_g F_g|^2`` are DROPPED. Two independent
    mechanisms suppress them -- spectral separation (each reflection's sinc
    line is narrow against the harmonic spacing, so the Cauchy-Schwarz bound
    ``2 sqrt(S_g S_g')`` collapses wherever either line is strong) and
    reciprocal-lattice decorrelation (the residual ``exp[-i(g-g').r_j]`` phase
    random-walks to ~``1/sqrt(n_seg)`` over midpoints spread across thousands
    of lattice spacings, which the diagonal ``|F_g|^2`` does not carry). Over
    the catalog's basal-plane families the dropped term bounds below ~1.6% of
    the integrated yield, and below ~1.4e-2 at the line centres of the forward
    harmonics that carry it; a reflection set with near-degenerate resonances
    at the observation angle is NOT covered.

    Validation: cross-reflection-coherence

    Two coherence scales fall
    out of the one sum: intra-electron
    (segments of a trajectory) and inter-electron / superradiant (the spread of
    ``t0_ang`` across the bunch, whose ``|<e^{i omega t0}>|^2`` is the Gaussian
    bunch form factor ``exp[-(omega sigma_z)^2]``).

    Limiting cases: coherent=False recovers the incoherent path bit-for-bit; a
    single segment / single electron has only the self-term and is identical to
    incoherent. A bunch much longer than the wavelength (or independently
    scrambled per-electron ``t0``) removes inter-electron cross terms but
    preserves each electron's intra-trajectory field,
    ``sum_e |sum_{j in e} E_j|^2``; it equals the per-segment incoherent path
    only when each electron contributes one segment or its internal segment
    phases also decohere. A bunch much shorter than the wavelength phases every
    emitter together into the ``|sum A_j|^2`` N^2-scaling limit.
    ``bunch_length_fs=None`` (all ``t0_ang=0``) is the documented degenerate
    pure-geometry (position-phase) limit, still physics.

    coherent is mutually exclusive with components (the PXR/CBS split is
    ambiguous once the cross term ``A_PXR A_CBS*`` survives) -- v1 raises.

    Validation: coherent-emission, coherent-segment-midpoint-time
    """
    if coherent and components:
        raise ValueError(
            "coherent=True is incompatible with components=True: the PXR/CBS "
            "split is ambiguous under coherence (the A_PXR*A_CBS cross term "
            "survives). Request the coherent total, or components incoherently."
        )
    if B_ang2 is None:
        raise ValueError(
            "mc_spectrum: B_ang2 (Debye-Waller B-factor [Ang^2]) is required; "
            "pass the material's value (no silent default)."
        )
    if coherent and layers is not None:
        raise NotImplementedError(
            "the in-medium dispersion does not cover the coherent path through "
            "a LAYERED absorber: the dispersive propagation phase needs a "
            "per-layer delta accumulated along the escape path -- the real "
            "partner of _stack_tau's per-layer mu -- which is not modelled. "
            "Single-slab absorbers (with or without a groove or a finite "
            "footprint) are supported."
        )
    info = CRYSTALS[crystal]
    n_atoms = len(info["basis"]) / info["V_cell"]
    abs_comp = _normalize_composition(absorber_element, n_atoms, composition)
    segments = _clip_segments_to_cutoff(segments, E_cut_keV, abs_comp, layers)

    # crystal orientation: rotation applied to all reciprocal vectors
    R_orient = _orientation_R(
        info["lattice"],
        beam_uvw,
        azimuth_rad,
        recip_miscut_rad,
        surface_hkl=surface_hkl,
    )
    thickness = segments["thickness_ang"]
    if electron_limit is None:
        Ne = segments["Ne"]
    else:
        Ne = electron_limit

    n_hat = _observation_direction(theta_obs_rad, n_hat)
    if groove is not None:
        if layers is not None:
            raise ValueError("groove escape is v1 single-slab only (no layers)")
        _validate_groove_escape_direction(n_hat, groove)
    E_grid = xp.asarray(E_grid_eV, dtype=REAL)
    spec = xp.zeros(E_grid.size, dtype=REAL)
    spec_pxr = xp.zeros(E_grid.size, dtype=REAL)
    spec_cbs = xp.zeros(E_grid.size, dtype=REAL)

    # Every per-row emission coefficient is evaluated at ONE energy along the
    # row. Under ``energy_model="midpoint"`` transport supplies the propagator's
    # own representative energy, so the row becomes a midpoint evaluation of its
    # emission integral rather than a left-endpoint one; the resulting t_L =
    # L/beta(E_repr) is then exactly the transported flight duration
    # ``t_end - t_start``, and ``t_ang + t_L/2`` exactly its midpoint age.
    # Frozen rows carry no representative energy and stay bit-for-bit.
    # Validation: substep-radiation-invariance
    E_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    seg_E = xp.asarray(segments[E_field], dtype=REAL)
    seg_v = xp.asarray(segments["v_hat"], dtype=REAL)
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)
    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)
    seg_elec_id = xp.asarray(segments["elec_id"])
    line_electron = seg_elec_id < Ne
    beta_all = beta_from_keV(seg_E)  # speed/c per segment
    v_all = beta_all[:, None] * seg_v  # velocity vectors (c=1)

    # Physical-flight grouping for the DEFAULT (incoherent) reduction. Numerical
    # substeps of one flight are integration detail: summing |A_j Q_j|^2 over
    # them treats each as an independent emitter and divides the line peak by
    # the substep count (each row carries (t_L/N)^2 where the flight carries
    # t_L^2), so a tighter energy tolerance would silently destroy the line.
    # Rows of one flight therefore add COHERENTLY and only whole flights add
    # incoherently. Groups are keyed by ``(electron_id, flight_id)`` VALUE, not
    # by adjacency: the lockstep core emits step-major, so one flight's substeps
    # are separated by every other electron's rows for that step.
    # Validation: substep-radiation-invariance
    gid_all = None
    grouped = False
    if not coherent and segments.get("flight_id") is not None and seg_E.size:
        flight_key = _to_cpu(xp.asarray(segments["flight_id"]))
        electron_key = _to_cpu(xp.asarray(segments["elec_id"]))
        order = np.lexsort((flight_key, electron_key))
        new_group = np.empty(flight_key.size, dtype=bool)
        new_group[0] = True
        new_group[1:] = (flight_key[order][1:] != flight_key[order][:-1]) | (
            electron_key[order][1:] != electron_key[order][:-1]
        )
        # All-singleton groups are the one-row-per-flight case, whose grouped
        # reduction is algebraically the incoherent one; keep the proven path.
        grouped = not bool(new_group.all())
        if grouped:
            gid_all = np.empty(flight_key.size, dtype=np.int64)
            gid_all[order] = np.cumsum(new_group) - 1
    if grouped and getattr(xp, "__name__", "") != "numpy":
        raise ValueError(
            "flight-grouped incoherent CXR is host-only: the segmented complex "
            "reduction over numerical substeps has no device port yet. Run the "
            "spectrum on the NumPy backend, or transport without max_dE_frac."
        )
    if grouped and layers is not None:
        # Same unmodelled per-layer delta as the coherent path: the grouped
        # reduction carries the dispersive propagation phase too.
        raise NotImplementedError(
            "the in-medium dispersion does not cover numerical substeps through "
            "a LAYERED absorber: the intra-flight coherent sum needs the "
            "per-layer dispersive propagation phase, which is not modelled."
        )
    if grouped and components:
        raise ValueError(
            "components=True is incompatible with numerical substeps: the "
            "PXR/CBS split is ambiguous once a flight's substep fields add "
            "coherently (the A_PXR*A_CBS cross term survives). Request the "
            "total, or transport without max_dE_frac."
        )

    # chi_g / U_g are smooth in energy AWAY from absorption edges, so evaluate
    # them on a tabulation grid and interpolate at the per-segment resonance
    # energies ON THE GPU (step 3) -- a few ms over ~10^3 grid points instead of
    # ~0.25 s/case over ~10^5 segments, and E_res stays on the device (no
    # per-reflection GPU->CPU->GPU round-trip; that structure-factor CPU cost was
    # what capped GPU utilisation on a fast card). The grid is a 1 eV mesh UNION
    # the basis and absorber elements' native Henke energies, which densely
    # sample the edges -- a plain uniform mesh mis-resolves the edge jumps (tens
    # of % at e.g. the C K-edge). Window matches the keep mask below.
    _pad = 0.2 * (float(E_grid_eV[-1]) - float(E_grid_eV[0]))
    _lo, _hi = float(E_grid_eV[0]) - _pad, float(E_grid_eV[-1]) + _pad
    _lo = max(_lo, 1.0)  # keep tabulation energies positive: chi_g/U_g need lambda = HC_EV_ANG / E
    E_tab = _line_tabulation_grid(info, abs_comp, _lo, _hi)
    E_tab_g = xp.asarray(E_tab, dtype=REAL)
    # Each elemental coefficient is stored as log(mu_i) [log(1/Ang)] and
    # interpolated linearly against log(E) before summing. This reproduces the
    # pinned xraydb non-f1 Chantler rule; interpolating the compound total in
    # either linear or log space does not. Explicit absorber elements contribute
    # their native nodes to E_tab even when absent from the crystal basis.
    # Applied only to single-slab/finite-footprint routes. Layered and grooved
    # escape retain exact per-point mu. No per-segment host/device transfer is
    # restored. Independently rederived with no physics divergence; human
    # sign-off remains required.
    # Validation: line-absorption-tabulation
    log_mu_tab_g = xp.asarray(_elemental_log_mu_table(abs_comp, E_tab), dtype=REAL)

    # Real part of the crystal's bulk refractive index n(E) = sqrt(1 + chi_0(E)),
    # tabulated on the SAME edge-resolved grid as chi/U/mu (delta = 1 - Re n has
    # its own edge structure, from the f1 cusp). The in-medium dispersion is
    # unconditional, so this table is always built.
    n_re_tab_g = xp.asarray(
        np.asarray(refractive_index(crystal, E_tab, use_henke).real), dtype=REAL
    )

    n_hat_d = xp.asarray(n_hat, dtype=REAL)  # detector dir is g-independent: hoist

    # Segment-only kinematics shared by every reflection/orientation and by both
    # the batched and compatibility paths. These used to be recomputed inside
    # ``_accumulate`` for every g row.
    v_dot_n_all = _matvec3(v_all, n_hat_d)
    denom_all = 1.0 - v_dot_n_all
    gamma_all = 1.0 / xp.sqrt(1.0 - beta_all * beta_all)
    t_L_all = seg_L / beta_all

    # coherent (phased) sum precompute: the per-segment retardation scalar
    # d_j = t_abs,j - n_hat.r_j [Ang, c=1] (emission-time phase minus far-field
    # retardation) and photon wavenumber k_gamma(E) = E / hbar c [1/Ang].
    # ``t_ang`` remains segment-start age; add half the constant-velocity flight
    # time so it describes the same midpoint as ``r_mid``. This one d_all feeds
    # both coherent reduction routes. Validation: coherent-segment-midpoint-time.
    # Each reflection adds its spatial susceptibility phase -g.r_j inside
    # _accumulate. All-zero t0_ang leaves the physical trajectory phase.
    # Inert unless a complex per-row field is actually reduced, i.e. under
    # coherent=True or the flight-grouped incoherent path.
    if coherent or grouped:
        cdtype = xp.result_type(REAL, 1j)
        seg_t0 = xp.asarray(segments.get("t0_ang", np.zeros(seg_E.size)), dtype=REAL)
        seg_t = xp.asarray(segments.get("t_ang", np.zeros(seg_E.size)), dtype=REAL)
        seg_t_mid = seg_t + 0.5 * t_L_all
        d_all = (seg_t_mid + seg_t0) - _matvec3(seg_r, n_hat_d)
        omega_grid = E_grid / HBARC_EV_ANG

        # In-medium propagation phase. The observation-time phase is
        # omega (t_j + n_med L_esc,j + L_vac,j), and the geometric total path
        # L_esc + L_vac = R - n_hat.r_j to first order, so the vacuum
        # ``omega d_j`` picks up exactly
        #
        #     omega (Re n(E) - 1) L_esc,j = - delta(E) omega(E) L_esc,j
        #
        # where L_esc,j is the SAME in-crystal escape distance the Beer-Lambert
        # factor already runs over. That is not a coincidence: the escape leg
        # contributes exp(i n omega L) = exp(i omega L) exp(-i delta omega L)
        # exp(-beta omega L), and the last factor is sqrt(exp(-mu L)) = ``amp``.
        # The dispersive phase is the real partner of an absorption the coherent
        # path already carries; only the two together are one complex n.
        #
        # Note this is NOT ``k(E) n_hat.r_j``: that form would charge the medium
        # for the whole flight to the detector. The two agree only when the
        # photon exits along the face normal (where L_esc and n_hat.r differ by a
        # segment-independent constant, i.e. a global phase).
        #
        # Tabulated on the OUTPUT grid: it is a propagation phase read across the
        # whole spectrum, not a coupling frozen at the line energy.
        # Validation: xray-in-medium-propagation-phase
        delta_omega_grid = (
            xp.asarray(
                1.0 - np.asarray(refractive_index(crystal, E_grid_eV, use_henke).real),
                dtype=REAL,
            )
            * omega_grid
        )

    # Empirical inter-electron decoherence for the coherent path. Today's
    # coherent sum bakes each electron's SAMPLED longitudinal offset
    # (``t0_ang``) and transverse entry offset (via ``seg_r``'s trajectory
    # position) into the segment phase and squares once -- one Monte Carlo
    # realization. For a squared coherent sum that is speckle, not shot
    # noise: the spurious enhancement does not shrink with electron count
    # (Rayleigh statistics), unlike ordinary incoherent MC noise.
    #
    # Fresh-context result: per (row, energy), Total = (1-F)*Grouped +
    # F*Flat, where Flat = |sum_e S_e|^2 is TODAY'S coherent reduction but
    # fed the electron's INTRINSIC (offset-free) position/time -- so it
    # needs no new reduction code, only feeding ``d_all_geom``/``seg_r_geom``
    # in place of ``d_all``/``seg_r`` at the handful of points that build a
    # row's phase -- and Grouped = sum_e|S_e|^2 groups the SAME segments by
    # electron, squares each electron's own sum, then adds (new reduction,
    # ``_coherent_electron_grouped_row`` below). F is the empirical
    # characteristic function of the ACTUAL per-electron offsets transport
    # already draws (``initial_t0_ang``/``initial_r_ang``, Ne-long
    # population arrays, NOT the per-segment gathered/duplicated ones):
    # F(row) = |mean_e exp(i*(omega*t0_e - q_perp(row).dr_perp,e))|^2. This
    # needs no per-policy sigma-resolution logic (legacy gaussian/uniform
    # bunch, long_offsets_fs, compressed, microtrain, and elliptical/
    # Courant-Snyder transverse spots all fall out for free -- the only
    # requirement is t0 sampled independently of the transverse offset,
    # true here since they use independent RNG child streams), and it
    # converges to the closed-form exp[-(omega sigma_z)^2-(q_perp
    # sigma_perp)^2] via ordinary 1/sqrt(Ne) statistics rather than the
    # non-converging speckle the naive sum shows.
    #
    # F must be applied PER ROW (reflection x mosaic orientation), before
    # summing across rows: q_perp depends on g, which differs row to row,
    # so a single scalar F(E) on the row-summed spectrum would be wrong
    # whenever more than one row contributes to the same energy bin.
    #
    # Validation: coherent-inter-electron-decoherence
    decoherence_active = False
    if coherent:
        seg_r_geom = seg_r
        d_all_geom = seg_t_mid - _matvec3(seg_r, n_hat_d)
        t0_pop = xp.asarray(segments.get("initial_t0_ang", np.zeros(0)), dtype=REAL)[:Ne]
        xy0_pop = xp.asarray(
            np.asarray(segments.get("initial_r_ang", np.zeros((0, 3))))[:, :2], dtype=REAL
        )[:Ne]
        decoherence_active = bool(
            (t0_pop.size and xp.any(t0_pop != 0.0)) or (xy0_pop.size and xp.any(xy0_pop != 0.0))
        )
        if decoherence_active:
            finite_footprint_now = (
                segments.get("crystal_width_ang") is not None
                and segments.get("crystal_height_ang") is not None
            )
            if finite_footprint_now:
                raise ValueError(
                    "coherent emission with a finite crystal footprint "
                    "(crystal_width_mm/crystal_height_mm) and a nonzero "
                    "bunch_length_fs/beam_fwhm_mm is not yet supported: the "
                    "transverse offset also perturbs escape attenuation "
                    "there, which the inter-electron decoherence form "
                    "factor does not model (see "
                    "docs/validation/radiation-physics/"
                    "coherent-inter-electron-decoherence.md)"
                )
            seg_r_geom = seg_r.copy()
            seg_elec_id_clamped = xp.clip(seg_elec_id, 0, max(Ne - 1, 0))
            seg_r_geom[:, :2] = seg_r_geom[:, :2] - xy0_pop[seg_elec_id_clamped]
            d_all_geom = seg_t_mid - _matvec3(seg_r_geom, n_hat_d)
            # A_e = t0_e - n_hat_perp . dr_perp,e does not depend on g (the
            # reciprocal vector varies per row; n_hat is fixed for the whole
            # call), so hoist it once here; B_e(row) = g_perp . dr_perp,e is
            # cheap and stays inside the per-row helper below.
            decoherence_A_pop = t0_pop - xy0_pop @ n_hat_d[:2]

        def _row_decoherence_factor(g_vec_d):
            """Empirical |<e^{i*phase_e}>|^2 for one row's g, from the
            actual per-electron offset population (chunked to bound peak
            memory the same way the field reduction below already is)."""
            if not decoherence_active:
                return None
            B_pop = xy0_pop @ g_vec_d[:2]
            chi_sum = xp.zeros(E_grid.size, dtype=cdtype)
            for j0 in range(0, decoherence_A_pop.size, chunk):
                sl = slice(j0, min(j0 + chunk, decoherence_A_pop.size))
                phase = omega_grid[None, :] * decoherence_A_pop[sl][:, None] - B_pop[sl][:, None]
                chi_sum += xp.exp(1j * phase).sum(axis=0)
            chi = chi_sum / decoherence_A_pop.size
            return (chi.real**2 + chi.imag**2).astype(REAL)

        def _coherent_electron_grouped_row(elec_id_sel, a_width_sel, E_r_sel, d_geom_sel,
                                            g_phase_sel, L_esc_sel, coefs_sel):
            """sum_e |sum_{j in e} E_j|^2 for one row's kept, finite segments,
            using the SAME group-then-reduce-then-square pattern as the
            flight-grouped incoherent path (7b) above, keyed by electron
            instead of flight. Every ``*_sel`` array is already restricted
            to this row's kept, finite segments and shares one length;
            ``coefs_sel``: per-polarization complex per-segment
            coefficients (same restriction)."""
            gid = _to_cpu(elec_id_sel)
            perm = np.argsort(gid, kind="stable")
            gid = gid[perm]
            starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
            bounds = np.append(starts, gid.size)
            row_total = xp.zeros(E_grid.size, dtype=REAL)
            perm_xp = xp.asarray(perm)
            aw_p = a_width_sel[perm_xp]
            Er_p = E_r_sel[perm_xp]
            d_p = d_geom_sel[perm_xp]
            gp_p = g_phase_sel[perm_xp]
            Lesc_p = L_esc_sel[perm_xp]
            coefs_p = [c[perm_xp] for c in coefs_sel]
            for ka, kb in _flight_blocks(bounds, chunk):
                rows = slice(bounds[ka], bounds[kb])
                x = aw_p[rows][:, None] * (E_grid[None, :] - Er_p[rows][:, None]) / xp.pi
                arg = d_p[rows][:, None] * omega_grid[None, :] - gp_p[rows][:, None]
                arg = arg - Lesc_p[rows][:, None] * delta_omega_grid[None, :]
                SP = xp.sinc(x).astype(cdtype) * xp.exp(1j * arg)
                offsets = bounds[ka:kb] - bounds[ka]
                for c in coefs_p:
                    field = xp.add.reduceat(c[rows][:, None] * SP, offsets, axis=0)
                    row_total += (xp.abs(field) ** 2).sum(axis=0)
            return row_total

        def _coherent_jit_grouped_row(elec_id_sel, per_line_sel, L_esc_sel, out):
            """sum_e |sum_{j in e} E_j|^2 for one row on the float32 CUDA-JIT
            reduction kernel -- the device counterpart of
            ``_coherent_electron_grouped_row`` above.

            No new device code is needed: ``run_coherent_reduction_kernel``
            already computes |sum of the lines it is handed|^2 and ACCUMULATES
            (``spec[k] += wm * ...``), so calling it once per electron over that
            electron's own lines, with ``mosaic_weight=1``, into one zeroed
            buffer sums the per-electron squares exactly. The cost is Ne extra
            launches per row (the segments themselves are still touched once in
            total); the mosaic weight is applied outside, by the blend.

            ``per_line_sel`` is the 8-tuple the kernel takes (E_r, a_width,
            phase_slope, g_phase, and the two complex polarization coefficients
            split into real/imag), already restricted to this row's kept lines
            and sharing one length with ``elec_id_sel``/``L_esc_sel``."""
            from .coherent_jit_kernel import (
                DEFAULT_COHERENT_KERNEL_CONFIG,
                run_coherent_reduction_kernel,
            )

            gid = _to_cpu(elec_id_sel)
            if gid.size == 0:
                return out
            perm = np.argsort(gid, kind="stable")
            gid = gid[perm]
            starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
            bounds = np.append(starts, gid.size)
            perm_xp = xp.asarray(perm)
            cols = [xp.ascontiguousarray(a[perm_xp], dtype=REAL) for a in per_line_sel]
            L_p = xp.ascontiguousarray(L_esc_sel[perm_xp], dtype=REAL)
            E_grid_c = xp.ascontiguousarray(E_grid, dtype=REAL)
            dom_c = xp.ascontiguousarray(delta_omega_grid, dtype=REAL)
            for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
                sl = slice(int(b0), int(b1))
                run_coherent_reduction_kernel(
                    *(c[sl] for c in cols),
                    E_grid_c,  # ty: ignore[too-many-positional-arguments]
                    out=out,
                    mosaic_weight=1.0,
                    L_esc=L_p[sl],
                    delta_omega=dom_c,
                    config=DEFAULT_COHERENT_KERNEL_CONFIG,
                )
            return out

    # mosaic crystallite-orientation quadrature: None -> perfect crystal (default;
    # today's single-orientation result bit-for-bit). Otherwise a list of
    # (rotation, weight) tilting g across the Gaussian mosaic cone, summed
    # incoherently below (docs/physics/materials/crystal-mosaicity.md route 2).
    mosaic_quad = _mosaic_quadrature(mosaic_fwhm_rad, mosaic_nodes)

    # NVTX sub-ranges to split the coarse ``cxr.lines`` range into structure-
    # factor tabulation vs per-reflection accumulation (no-op off the profiled
    # GPU path). Lazy import: runner imports this module, so a top-level import
    # would be circular.
    from ..runner import _nsys_pop, _nsys_push

    def _accumulate(
        g_vec_d, e_s, e_p, g2, n_dot_g, g_dot_es, g_dot_ep, chi_re, chi_im, u_re, u_im, wm
    ):
        """Add one reflection's contribution for crystallite reciprocal vector
        ``g_vec_d``, scaled by the mosaic-quadrature weight ``wm``, into spec /
        spec_pxr / spec_cbs in place. Every argument is a DEVICE array uploaded
        once by the caller's stacking prologue (row views): the structure-factor
        tabulations (chi/u on E_tab_g) depend on hkl and energy only -- NOT on
        the mosaic orientation -- while g and its sigma/pi polarization pair
        (``e_s``, ``e_p``) vary per orientation. wm = 1.0 for the perfect-crystal
        path."""

        # -- 1. per-segment resonance energy (Eq. 10) ---------------------------
        #   omega_res = v.g / (1 - v.n)   [1/Ang]   (>0 required to radiate)
        v_dot_g = _matvec3(v_all, g_vec_d)
        denom, n_re_seg = _in_medium_kinematics(v_dot_n_all, v_dot_g, n_re_tab_g, E_tab_g)
        omega_res = v_dot_g / denom
        E_res = HBARC_EV_ANG * omega_res  # -> eV

        # -- 2. drop segments whose line misses the spectral window -------------
        # (pad by 20% so sinc tails that reach into the window still count)
        pad = 0.2 * (E_grid[-1] - E_grid[0])
        keep = (
            line_electron
            & (E_res > float(E_grid[0] - pad))
            & (E_res > 10.0)
            & (E_res < E_grid[-1] + pad)
        )
        if not keep.any():
            return
        idx = xp.flatnonzero(keep)

        E_r = E_res[idx]  # line energy per kept segment [eV]
        om = omega_res[idx]  # same in 1/Ang
        v = v_all[idx]  # velocity vectors
        t_L = t_L_all[idx]  # interaction time [Ang] (c=1)
        dnm = denom[idx]
        vdg = v_dot_g[idx]

        # -- 3. couplings AT each segment's resonance energy --------------------
        # All five tabulations share E_r and E_tab_g, so bracket once. U_g tables
        # are pre-scaled by 1/m_e during construction.
        _ix, _fr, _blw, _abv = _interp_index(E_r, E_tab_g)
        _log_fr = _log_interp_fraction(E_r, E_tab_g, _ix)
        chi_re_i = _interp_gather1d(_ix, _fr, _blw, _abv, chi_re)
        chi_im_i = _interp_gather1d(_ix, _fr, _blw, _abv, chi_im)
        u_re_i = _interp_gather1d(_ix, _fr, _blw, _abv, u_re)
        u_im_i = _interp_gather1d(_ix, _fr, _blw, _abv, u_im)
        mu_i = _interp_elemental_mu(_ix, _log_fr, _blw, _abv, log_mu_tab_g)
        chi = chi_re_i + 1j * chi_im_i
        eUg_over_m = u_re_i + 1j * u_im_i

        # -- 4. photon kinematics per segment ------------------------------------
        # k = omega*n, so detuning = g^2 + 2*omega*(n.g), k.g = omega*(n.g),
        # and v.(k+g) = v.g + k.v. The g-only scalars are precomputed once.
        # In medium k = n omega n_hat, so k.v = omega(1 - denom) still holds
        # exactly (denom absorbed the n), while k.g takes one power of n and
        # |k+g|^2 - k^2 = g^2 + 2 k.g keeps its form. k_mag = |k| is what the
        # PXR numerator's k^2 needs; it is omega in vacuum.
        k_mag = om if n_re_seg is None else om * n_re_seg[idx]
        k_dot_v = om * (1.0 - dnm)
        k_dot_g = k_mag * n_dot_g
        v_dot_kg = vdg + k_dot_v
        detuning = g2 + 2.0 * k_dot_g

        # -- 5. Eq. (13) + relativistic Eq. (14) amplitudes, per segment ----------
        # CBS braced product {a;b} = a.b - (a.v)(b.v) and 1/gamma prefactor
        # (Zhai SI Eq. 6).
        gamma = gamma_all[idx]
        A2 = xp.zeros(idx.size, dtype=REAL)
        A2_pxr = xp.zeros(idx.size, dtype=REAL)
        A2_cbs = xp.zeros(idx.size, dtype=REAL)
        pol_A = []  # complex A = A_PXR + A_CBS per polarization (coherent path)
        for e_d, g_dot_e in ((e_s, g_dot_es), (e_p, g_dot_ep)):
            v_dot_e = _matvec3(v, e_d)
            if coherent or grouped:
                # Complex amplitudes retained verbatim -- the phased-field paths
                # (global-coherent and flight-grouped) keep the un-reassociated
                # expression and their goldens are unaffected.
                A_PXR = chi / detuning * (v_dot_kg * g_dot_e - k_mag**2 * v_dot_e)
                braced_ge = g_dot_e - vdg * v_dot_e
                braced_kg = k_dot_g - k_dot_v * vdg
                A_CBS = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
                pol_A.append(A_PXR + A_CBS)  # keep phase: orthogonal pols still add incoherently
                continue
            # Incoherent |A|^2 in one fused real kernel (reassociated, NOT
            # bit-for-bit; ledger + regen required -- see _line_amp_sq_core,
            # Validation: line-amplitude-fusion).
            a2, a2_pxr, a2_cbs = _line_amp_sq_core(
                chi.real,
                chi.imag,
                eUg_over_m.real,
                eUg_over_m.imag,
                v_dot_kg,
                g_dot_e,
                k_mag,
                v_dot_e,
                vdg,
                k_dot_g,
                k_dot_v,
                gamma,
                detuning,
            )
            A2 += a2
            A2_pxr += a2_pxr
            A2_cbs += a2_cbs

        # -- 6. Beer-Lambert escape factor from the segment midpoint -------------
        # straight path along n_hat to whichever face the photon exits. With a
        # LAYERED absorber (layers) the optical depth sums mu_i*dz_i across the
        # film-on-substrate stack; otherwise it's the single-slab path. The
        # geometric path is mosaic-independent; the optical depth uses E_r (the
        # orientation-shifted line energy), so it is recomputed per orientation.
        z_mid = seg_r[idx, 2]
        if groove is not None:
            # Blazed sawtooth entrance face: closed-form path to the working
            # facet (grooves shorten, never lengthen, the flat-face path). Takes
            # precedence over any finite footprint -- the mm-scale crystal extent
            # only classifies launch hit/miss (hit_frac, set in transport); the
            # groove escape treats the slab as laterally periodic. At depth z,
            # the side-edge-affected strip is
            # L_esc*cos(tp) = z*cot(tp) + O(groove spacing), capped by the
            # crystal width. The guard above guarantees layers is None and
            # n_hat[2] < 0 here.
            # Validation: blazed-groove-geometry
            L_esc = escape_distance_ang(seg_r[idx, 0], z_mid, groove)
            tau = L_esc * _mu_total_inv_ang(abs_comp, E_r)
        elif finite_footprint:
            # the escape DISTANCE is g-independent, so it is computed once per
            # case (L_esc_all, below the loop's stacking prologue) instead of per
            # reflection/orientation; only the idx selection is per-g.
            assert L_esc_all is not None  # set whenever finite_footprint and no groove
            L_esc = L_esc_all[idx]
            if layers is None:
                tau = L_esc * mu_i
            else:
                tau = _stack_tau(layers, z_mid, n_hat[2], E_r, exit_distance_ang=L_esc)
        else:
            if layers is None:
                if n_hat[2] < 0:
                    L_esc = z_mid / (-n_hat[2])  # out the entrance face
                else:
                    L_esc = (thickness - z_mid) / n_hat[2]  # out the back face
                tau = L_esc * mu_i
            else:
                tau = _stack_tau(layers, z_mid, n_hat[2], E_r)
        T_abs = xp.exp(-tau)

        # -- 7b. flight-grouped incoherent accumulation ---------------------------
        # The same complex per-row field the coherent path builds, but reduced
        # per PHYSICAL FLIGHT: substeps of one flight add coherently, whole
        # flights add incoherently. At frozen energy and clock this is an exact
        # algebraic identity with the unsplit row (the substep sinc times the
        # Dirichlet sum over substep offsets rebuilds the parent's
        # ``t_L sinc(P t_L / pi)``), so refining the energy tolerance changes
        # only the quadrature of the energy sweep along the flight -- which is
        # the point -- and not the number of independent emitters.
        if gid_all is not None:  # i.e. ``grouped``, narrowed for the gather below
            amp = xp.sqrt(ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * T_abs)
            a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
            d = d_all[idx]
            g_phase = _matvec3(seg_r[idx], g_vec_d)
            coefs = [(amp * t_L) * A_e for A_e in pol_A]
            good = xp.isfinite(amp) & (amp > 0) & (t_L > 0)
            sel = np.flatnonzero(good)
            if sel.size == 0:
                return
            gid = gid_all[idx[sel]]
            # Gather this flight's rows together; a stable sort leaves already
            # grouped input (and each group's internal row order) untouched.
            perm = np.argsort(gid, kind="stable")
            sel = sel[perm]
            gid = gid[perm]
            starts = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
            bounds = np.append(starts, sel.size)
            for ka, kb in _flight_blocks(bounds, chunk):
                rows = sel[bounds[ka] : bounds[kb]]
                x = a_width[rows][:, None] * (E_grid[None, :] - E_r[rows][:, None]) / xp.pi
                arg = d[rows][:, None] * omega_grid[None, :] - g_phase[rows][:, None]
                arg = arg - L_esc[rows][:, None] * delta_omega_grid[None, :]
                SP = xp.sinc(x).astype(cdtype) * xp.exp(1j * arg)
                # Blocks break only on flight boundaries, so no flight is split
                # across two reductions and squared twice.
                offsets = bounds[ka:kb] - bounds[ka]
                for c in coefs:
                    field = np.add.reduceat(c[rows][:, None] * SP, offsets, axis=0)
                    spec[:] += (xp.abs(field) ** 2).sum(axis=0) * wm
            return

        # -- 7c. coherent (phased) accumulation -----------------------------------
        # Build the complex field per polarization within THIS reflection and
        # orientation, square it, and add |field|^2 * wm to spec (reflections and
        # mosaic orientations remain incoherent). The un-squared finite-time
        # factor Q = t_L sinc(a_width(E-E_res)/pi) carries the amplitude scale
        # (|Q|^2 = t_L^2 sinc^2); the emission-time/retardation phase is
        # exp[i omega(E) d_j] with d_j = t_abs,j - n_hat.r_j.
        if coherent:
            amp = xp.sqrt(ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * T_abs)
            a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
            # Geometric-only (offset-free) phase: identical to d_all/seg_r
            # when no decoherence-relevant offset is configured, so this is
            # a no-op swap in that (default) case. See the
            # coherent-inter-electron-decoherence block above.
            d = d_all_geom[idx]
            g_phase = _matvec3(seg_r_geom[idx], g_vec_d)
            coefs = [(amp * t_L) * A_e for A_e in pol_A]  # complex per polarization
            good = xp.isfinite(amp) & (amp > 0) & (t_L > 0)

            if decoherence_active and sinc_cutoff is not None:
                raise ValueError(
                    "coherent emission with a nonzero bunch_length_fs/"
                    "beam_fwhm_mm and sinc_cutoff together is not yet "
                    "supported (the electron-grouped decoherence floor "
                    "does not implement sinc_cutoff windowing)"
                )

            # GPU float32 fast path: reduce the two complex polarization fields
            # directly in a raw kernel. This avoids materializing the dense
            # complex SP[segment, energy] matrix and avoids both complex GEMVs.
            # The exact CuPy path below remains the fallback for CPU/other
            # backends, float64, and sinc_cutoff windowing. A nonzero
            # bunch_length_fs/beam_fwhm_mm stays ON the kernel: the fused
            # kernel squares whatever line set it is handed and ACCUMULATES,
            # so the electron-grouped floor sum_e|S_e|^2 is just one call per
            # electron into a zeroed buffer (``_coherent_jit_grouped_row``),
            # and |sum_e S_e|^2 is the same single call this path already
            # makes -- both on the geometric (offset-free) phase, blended by
            # F(row, E) outside the kernel.
            # The reduction kernel folds the vacuum phase as ``slope_j * E``, a
            # single per-line scalar against the energy axis. The in-medium term
            # is a SECOND (per-segment scalar) x (per-energy table) product, so
            # it rides along as its own ``L_esc``/``delta_omega`` pair rather
            # than being absorbed into that slope.
            _use_jit_coherent_reduction = (
                _USE_JIT_COHERENT_REDUCTION
                and getattr(xp, "__name__", "") == "cupy"
                and np.dtype(REAL) == np.dtype(np.float32)
                and sinc_cutoff is None
            )
            if _use_jit_coherent_reduction:
                from .coherent_jit_kernel import (
                    DEFAULT_COHERENT_KERNEL_CONFIG,
                    run_coherent_reduction_kernel,
                )

                sel = xp.flatnonzero(good)
                if sel.size:
                    c_s, c_p = coefs
                    per_line_sel = (
                        xp.ascontiguousarray(E_r[sel], dtype=REAL),
                        xp.ascontiguousarray(a_width[sel], dtype=REAL),
                        xp.ascontiguousarray(d[sel] / HBARC_EV_ANG, dtype=REAL),
                        xp.ascontiguousarray(g_phase[sel], dtype=REAL),
                        xp.ascontiguousarray(c_s[sel].real, dtype=REAL),
                        xp.ascontiguousarray(c_s[sel].imag, dtype=REAL),
                        xp.ascontiguousarray(c_p[sel].real, dtype=REAL),
                        xp.ascontiguousarray(c_p[sel].imag, dtype=REAL),
                    )
                    L_esc_sel = xp.ascontiguousarray(L_esc[sel], dtype=REAL)
                    E_grid_c = xp.ascontiguousarray(E_grid, dtype=REAL)
                    dom_c = xp.ascontiguousarray(delta_omega_grid, dtype=REAL)
                    # Decoherence-inactive: unchanged single fused call
                    # straight into spec with the row's mosaic weight.
                    out_flat = spec if not decoherence_active else xp.zeros(E_grid.size, dtype=REAL)
                    run_coherent_reduction_kernel(
                        *per_line_sel,
                        E_grid_c,
                        out=out_flat,
                        mosaic_weight=1.0 if decoherence_active else wm,
                        L_esc=L_esc_sel,
                        delta_omega=dom_c,
                        config=DEFAULT_COHERENT_KERNEL_CONFIG,
                    )
                    if decoherence_active:
                        grouped_total = _coherent_jit_grouped_row(
                            seg_elec_id[idx][sel],
                            per_line_sel,
                            L_esc_sel,
                            xp.zeros(E_grid.size, dtype=REAL),
                        )
                        F_row = _row_decoherence_factor(g_vec_d)
                        spec[:] += ((1.0 - F_row) * grouped_total + F_row * out_flat) * wm
                return

            fields = [xp.zeros(E_grid.size, dtype=cdtype) for _ in coefs]
            if sinc_cutoff is None:
                for j0 in range(0, idx.size, chunk):
                    sl = slice(j0, min(j0 + chunk, idx.size))
                    m = good[sl]
                    if not m.any():
                        continue
                    x = a_width[sl][m, None] * (E_grid[None, :] - E_r[sl][m, None]) / xp.pi
                    arg = d[sl][m, None] * omega_grid[None, :] - g_phase[sl][m, None]
                    arg = arg - L_esc[sl][m, None] * delta_omega_grid[None, :]
                    ph = xp.exp(1j * arg)
                    SP = xp.sinc(x).astype(cdtype) * ph
                    for c, f in zip(coefs, fields, strict=True):
                        f += c[sl][m] @ SP
            else:
                dE = E_grid[1] - E_grid[0]
                order = xp.argsort(E_r)
                blk = 8192
                for j0 in range(0, order.size, blk):
                    sel = order[j0 : j0 + blk]
                    sel = sel[good[sel]]
                    if sel.size == 0:
                        continue
                    half = sinc_cutoff / a_width[sel]
                    lo = float(_to_cpu((E_r[sel] - half).min()))
                    hi = float(_to_cpu((E_r[sel] + half).max()))
                    i0 = max(int((lo - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))), 0)
                    i1 = min(
                        int((hi - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))) + 2,
                        E_grid.size,
                    )
                    if i1 <= i0:
                        continue
                    x = a_width[sel][:, None] * (E_grid[None, i0:i1] - E_r[sel][:, None]) / xp.pi
                    arg = d[sel][:, None] * omega_grid[None, i0:i1] - g_phase[sel][:, None]
                    arg = arg - L_esc[sel][:, None] * delta_omega_grid[None, i0:i1]
                    ph = xp.exp(1j * arg)
                    SP = xp.sinc(x).astype(cdtype) * ph
                    for c, f in zip(coefs, fields, strict=True):
                        f[i0:i1] += c[sel] @ SP
            flat_total = sum(xp.abs(f) ** 2 for f in fields)
            if decoherence_active:
                sel_full = xp.flatnonzero(good)
                grouped_total = _coherent_electron_grouped_row(
                    seg_elec_id[idx][sel_full],
                    a_width[sel_full],
                    E_r[sel_full],
                    d[sel_full],
                    g_phase[sel_full],
                    L_esc[sel_full],
                    [c[sel_full] for c in coefs],
                )
                F_row = _row_decoherence_factor(g_vec_d)
                spec[:] += ((1.0 - F_row) * grouped_total + F_row * flat_total) * wm
            else:
                spec[:] += flat_total * wm
            return

        # -- 7. accumulate the finite-segment lineshape ---------------------------
        # d2N/dE dOmega = alpha*omega/(4 pi^2 hbar c) |A|^2 t_L^2
        #                  * sinc^2[(1 - v.n)(omega - omega_res) t_L / 2] * T_abs
        # weight = everything except the sinc^2 (times the mosaic weight wm);
        # a_width converts (E - E_res) to the sinc argument: P t_L = a_width(E - E_res).
        _nsys_push("cxr.lines.accum")
        pref = ALPHA_FS * om / (4.0 * xp.pi**2 * HBARC_EV_ANG) * t_L**2 * T_abs
        weight = pref * A2 * wm
        targets = [(weight, spec)]
        if components:
            targets += [(pref * A2_pxr * wm, spec_pxr), (pref * A2_cbs * wm, spec_cbs)]
        a_width = dnm * t_L / (2.0 * HBARC_EV_ANG)
        good = xp.isfinite(weight) & (weight > 0)

        if sinc_cutoff is None:
            for j0 in range(0, idx.size, chunk):
                sl = slice(j0, min(j0 + chunk, idx.size))
                m = good[sl]
                if not m.any():
                    continue
                S = _sincsq_lineshape(a_width[sl][m, None], E_grid[None, :], E_r[sl][m, None])
                for w, tgt in targets:
                    tgt += w[sl][m] @ S
        else:
            dE = E_grid[1] - E_grid[0]
            order = xp.argsort(E_r)
            blk = 8192
            for j0 in range(0, order.size, blk):
                sel = order[j0 : j0 + blk]
                sel = sel[good[sel]]
                if sel.size == 0:
                    continue
                half = sinc_cutoff / a_width[sel]
                lo = float(_to_cpu((E_r[sel] - half).min()))
                hi = float(_to_cpu((E_r[sel] + half).max()))
                i0 = max(int((lo - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))), 0)
                i1 = min(
                    int((hi - float(_to_cpu(E_grid[0]))) // float(_to_cpu(dE))) + 2,
                    E_grid.size,
                )
                if i1 <= i0:
                    continue
                S = _sincsq_lineshape(a_width[sel][:, None], E_grid[None, i0:i1], E_r[sel][:, None])
                for w, tgt in targets:
                    tgt[i0:i1] += w[sel] @ S
        _nsys_pop()

    # The batched path below runs the whole hkl set through steps 1-6 in one
    # vectorized (n_seg, N_g) pass. It covers the single-slab absorber (with or
    # without a finite crystal footprint -- the escape DISTANCE is g-independent
    # either way, so it hoists) for BOTH coherent and incoherent emission. The
    # layered / grooved absorbers, and coherent runs that ask for sinc_cutoff
    # windowing, stay on the proven per-hkl _accumulate loop, bit-for-bit.
    finite_footprint = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )
    # The flight-grouped reduction lives on the per-hkl loop only: its segmented
    # complex sum has no batched or device counterpart yet (slice H owns those
    # ports), and correctness of the default incoherent yield outranks the
    # batched path's launch-count win on the substepped configuration.
    if (
        (coherent and sinc_cutoff is not None)
        or groove is not None
        or layers is not None
        or grouped
    ):
        # Stacking prologue: every host->device transfer this path needs is done
        # ONCE per case here, not once per (reflection, orientation) inside the
        # loop. Previously each pass re-uploaded the four chi/U tabulations plus
        # g and its two polarization vectors -- ~90 xp.asarray calls per case,
        # 15% of GPU-phase tottime on the 3060 Ti profile (hopg_coherent, 4
        # reflections). The chi/U rows are keyed by REFLECTION (they do not
        # depend on the mosaic orientation), the geometry rows by
        # (reflection, orientation). U_g/m_e is now pre-scaled at table build and
        # the fallback shares one interpolation bracket; these are algebraically
        # identical with only float-rounding-level movement.
        _nsys_push("cxr.lines.tab")
        g_rows, es_rows, ep_rows, wm_rows, hkl_of_row = [], [], [], [], []
        cr_rows, ci_rows, ur_rows, ui_rows = [], [], [], []
        orients = ((None, 1.0),) if mosaic_quad is None else mosaic_quad
        for i_hkl, hkl in enumerate(hkl_list):
            # reciprocal vector in the sample frame: construction frame by default
            # ([001] along the slab normal), rotated if beam_uvw given
            g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
            if R_orient is not None:
                g_vec = R_orient @ g_vec
            chi_tab = np.asarray(chi_g(crystal, hkl, E_tab, B_ang2, use_henke))
            u_tab = np.asarray(U_g(crystal, hkl, E_tab, B_ang2, use_henke)) / M_E_EV
            # Store U_g/m_e in the table: the mass scaling is energy-independent.
            cr_rows.append(chi_tab.real)
            ci_rows.append(chi_tab.imag)
            ur_rows.append(u_tab.real)
            ui_rows.append(u_tab.imag)
            for R_m, wm in orients:  # None -> perfect crystal, one orientation, weight 1
                gd = g_vec if R_m is None else R_m @ g_vec
                e_s, e_p = _polarization_pair(n_hat, gd)
                g_rows.append(gd)
                es_rows.append(e_s)
                ep_rows.append(e_p)
                wm_rows.append(wm)
                hkl_of_row.append(i_hkl)
        G = xp.asarray(np.array(g_rows), dtype=REAL)  # (N_g, 3)
        ES = xp.asarray(np.array(es_rows), dtype=REAL)
        EP = xp.asarray(np.array(ep_rows), dtype=REAL)
        CHI_RE = xp.asarray(np.array(cr_rows), dtype=REAL)  # (N_hkl, N_tab)
        CHI_IM = xp.asarray(np.array(ci_rows), dtype=REAL)
        U_RE = xp.asarray(np.array(ur_rows), dtype=REAL)
        U_IM = xp.asarray(np.array(ui_rows), dtype=REAL)
        G2 = _rowdot3(G, G)
        N_DOT_G = _matvec3(G, n_hat_d)
        G_DOT_ES = _rowdot3(G, ES)
        G_DOT_EP = _rowdot3(G, EP)
        # g-independent escape distance: one pass per case, sliced per g inside
        # _accumulate (only the finite-footprint, non-grooved branch reads it).
        L_esc_all = (
            _segment_escape_distance(segments, n_hat, xp=xp)
            if finite_footprint and groove is None
            else None
        )
        _nsys_pop()

        for i_row, (wm, i_hkl) in enumerate(zip(wm_rows, hkl_of_row, strict=True)):
            _accumulate(
                G[i_row],
                ES[i_row],
                EP[i_row],
                G2[i_row],
                N_DOT_G[i_row],
                G_DOT_ES[i_row],
                G_DOT_EP[i_row],
                CHI_RE[i_hkl],
                CHI_IM[i_hkl],
                U_RE[i_hkl],
                U_IM[i_hkl],
                wm,
            )
    else:
        # ---- batched line accumulation (options A + B) -----------------------
        # Every reflection/orientation shares the segment geometry, so run the
        # per-orientation block (steps 1-6) ONCE over an (n_seg, N_g) grid rather
        # than ~N_g separate _accumulate passes -- this collapses the tiny-kernel
        # launch storm that starved the GPU (cxr.lines setup ~59 s at 6% util).
        # Length-3 contractions expand to component-wise fused multiply-adds (no
        # cuBLAS, no (n_seg, N_g, 3) temporaries), and the g-independent segment
        # quantities are hoisted out of the pass.
        #
        # NOT bit-for-bit vs the per-hkl loop: reassociates the float reductions
        # (component dots, batched linear interp, union-order sinc matmul), same
        # rounding-level move the chunk-invariance rtol gate already covers.
        # Ledgered `filtered`; REGEN still required before sign-off, and the
        # measured A/B against the per-hkl loop is float64 only (no CUDA device
        # in the verifying environment). Validation: line-hkl-batch
        #
        # coherent=True shares steps 1-6 verbatim and diverges only at step 5/7:
        # it keeps the COMPLEX amplitude per polarization and defers the square
        # to a per-(reflection, orientation) reduction, so reflections and mosaic
        # orientations stay incoherent. Validation: coherent-line-hkl-batch
        e_lo = float(_to_cpu(E_grid[0]))
        e_hi = float(_to_cpu(E_grid[-1]))
        pad = 0.2 * (e_hi - e_lo)  # keep sinc tails that reach into the window
        lo_keep, hi_keep = e_lo - pad, e_hi + pad

        # The live runner evaluates incoherent and coherent spectra back-to-back
        # over the same case.  Their reflection/orientation tables are identical,
        # so retain this preparation in the pair-local cache instead of rebuilding
        # and re-uploading it for the second kernel.
        table_key = (
            "batched",
            id(E_grid_eV),
            crystal,
            tuple(tuple(hkl) for hkl in hkl_list),
            float(B_ang2),
            bool(use_henke),
            None if beam_uvw is None else tuple(beam_uvw),
            None if surface_hkl is None else tuple(surface_hkl),
            float(azimuth_rad),
            None if recip_miscut_rad is None else tuple(recip_miscut_rad),
            None if mosaic_fwhm_rad is None else float(mosaic_fwhm_rad),
            int(mosaic_nodes),
            tuple(float(value) for value in n_hat),
        )
        tables = None if _table_cache is None else _table_cache.get(table_key)
        if tables is None:
            _nsys_push("cxr.lines.tab")
            g_rows, es_rows, ep_rows, wm_rows = [], [], [], []
            cr_rows, ci_rows, ur_rows, ui_rows = [], [], [], []
            orients = ((None, 1.0),) if mosaic_quad is None else mosaic_quad
            for hkl in hkl_list:
                g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
                if R_orient is not None:
                    g_vec = R_orient @ g_vec
                chi_tab = np.asarray(chi_g(crystal, hkl, E_tab, B_ang2, use_henke))
                u_tab = np.asarray(U_g(crystal, hkl, E_tab, B_ang2, use_henke)) / M_E_EV
                # Store U_g/m_e once instead of dividing every interpolated pair.
                for R_m, wm in orients:
                    gd = g_vec if R_m is None else R_m @ g_vec
                    e_s, e_p = _polarization_pair(n_hat, gd)
                    g_rows.append(gd)
                    es_rows.append(e_s)
                    ep_rows.append(e_p)
                    wm_rows.append(wm)
                    cr_rows.append(chi_tab.real)
                    ci_rows.append(chi_tab.imag)
                    ur_rows.append(u_tab.real)
                    ui_rows.append(u_tab.imag)
            G = xp.asarray(np.array(g_rows), dtype=REAL)  # (N_g, 3)
            ES = xp.asarray(np.array(es_rows), dtype=REAL)
            EP = xp.asarray(np.array(ep_rows), dtype=REAL)
            WM = xp.asarray(np.array(wm_rows), dtype=REAL)[None, :]  # (1, N_g)
            CHI_RE = xp.asarray(np.array(cr_rows), dtype=REAL)  # (N_g, N_tab)
            CHI_IM = xp.asarray(np.array(ci_rows), dtype=REAL)
            U_RE = xp.asarray(np.array(ur_rows), dtype=REAL)
            U_IM = xp.asarray(np.array(ui_rows), dtype=REAL)
            G2 = _rowdot3(G, G)
            N_DOT_G = _matvec3(G, n_hat_d)
            G_DOT_ES = _rowdot3(G, ES)
            G_DOT_EP = _rowdot3(G, EP)
            tables = (
                G,
                ES,
                EP,
                WM,
                CHI_RE,
                CHI_IM,
                U_RE,
                U_IM,
                G2,
                N_DOT_G,
                G_DOT_ES,
                G_DOT_EP,
                wm_rows,
            )
            if _table_cache is not None:
                _table_cache[table_key] = tables
            _nsys_pop()
        else:
            (
                G,
                ES,
                EP,
                WM,
                CHI_RE,
                CHI_IM,
                U_RE,
                U_IM,
                G2,
                N_DOT_G,
                G_DOT_ES,
                G_DOT_EP,
                wm_rows,
            ) = tables

        N_g = G.shape[0]
        gx, gy, gz = G[:, 0][None, :], G[:, 1][None, :], G[:, 2][None, :]  # (1, N_g)
        nz = float(n_hat[2])
        g2 = G2[None, :]
        n_dot_g = N_DOT_G[None, :]

        # option B: g-independent per-segment quantities, computed once
        denom_full = denom_all[:, None]
        gamma_full = gamma_all[:, None]
        t_L_full = t_L_all[:, None]
        # escape distance is g-independent (straight ray along n_hat): a finite
        # footprint picks the nearest prism face, else the plain slab path.
        if finite_footprint:
            L_esc_full = _segment_escape_distance(segments, n_hat, xp=xp)[:, None]
        else:
            L_esc_full = _escape_length(seg_r[:, 2], thickness, nz)[:, None]

        # Experimental fused CUDA line reduction.
        # Keep the existing CuPy/NumPy accumulation as the fallback for:
        #   - CPU execution
        #   - components=True
        #   - sinc_cutoff support
        #
        # Lazy import is intentional: spectrum.py also supports NumPy CPU workers.
        _use_jit_line_reduction = (
            _USE_JIT_LINE_REDUCTION
            and getattr(xp, "__name__", "") == "cupy"
            and np.dtype(REAL) == np.dtype(np.float32)
            and not components
            and sinc_cutoff is None
        )

        if _use_jit_line_reduction:
            from .line_jit_kernel import (
                DEFAULT_SPECTRUM_KERNEL_CONFIG,
                run_reduction_kernel,
            )

        # Coherent CUDA fast path: stream one bounded segment block at a time
        # through a fused (segment, g) prologue and into persistent per-g complex
        # field planes.  The field, not the intensity, is additive across segment
        # blocks, so this is algebraically the coherent sum while avoiding the old
        # all-lines ``coh_blocks`` retention, boolean compaction, row-count D2H
        # sync, and per-g concatenate/slice pass.  The final kernel squares each g
        # row independently and only then mosaic-weights/sums rows, so reflections
        # and orientations remain incoherent.  Validation: coherent-line-hkl-batch
        # ``run_coherent_prologue_kernel`` is an independent CUDA port of steps
        # 1-6, so the refractive model reaches it as its own arguments: the
        # per-segment ``v.n_hat`` plus the ``Re n(E)`` table let it solve the same
        # implicit in-medium resonance on the device, and it then returns the
        # half-width in the pair layout because that denominator is per
        # (segment, g). ``_field_kernel_*`` carries the propagation-phase half.
        _use_jit_coherent_stream = (
            coherent
            and _USE_JIT_COHERENT_STREAM
            and getattr(xp, "__name__", "") == "cupy"
            and np.dtype(REAL) == np.dtype(np.float32)
            and sinc_cutoff is None
        )
        if _use_jit_coherent_stream:
            from .coherent_stream_jit_kernel import (
                DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
                allocate_coherent_fields,
                finalize_coherent_fields,
                run_coherent_field_accumulation_kernel,
                run_coherent_prologue_kernel,
            )

            _coh_g = G.reshape(-1)
            _coh_es = ES.reshape(-1)
            _coh_ep = EP.reshape(-1)
            _coh_g2 = xp.ascontiguousarray(G2, dtype=REAL)
            _coh_n_dot_g = xp.ascontiguousarray(N_DOT_G, dtype=REAL)
            _coh_g_dot_es = xp.ascontiguousarray(G_DOT_ES, dtype=REAL)
            _coh_g_dot_ep = xp.ascontiguousarray(G_DOT_EP, dtype=REAL)
            _coh_chi_re = CHI_RE.reshape(-1)
            _coh_chi_im = CHI_IM.reshape(-1)
            _coh_u_re = U_RE.reshape(-1)
            _coh_u_im = U_IM.reshape(-1)
            # The in-medium prologue builds the sinc half-width per (segment, g)
            # from the in-medium denominator, so it cannot be hoisted per segment
            # the way a vacuum k = omega kernel would (``aw_seg`` stays unset).
            # Geometric-only (offset-free) phase, matching _accumulate's 7c and
            # the batched fallback below: identical to d_all/seg_r when no
            # decoherence-relevant offset is configured, so this is a no-op swap
            # in that (default) case. See the
            # coherent-inter-electron-decoherence block above.
            _coh_phase_slope = xp.ascontiguousarray(d_all_geom / HBARC_EV_ANG, dtype=REAL)
            coherent_fields = allocate_coherent_fields(N_g, E_grid.size)

            def _stream_segment_block(sel, n_sel):
                """Prologue + field accumulation for ONE segment selection into
                the persistent ``coherent_fields`` planes.

                ``sel`` is a contiguous slice on the ordinary (flat) pass and an
                electron's segment index array on the decoherence grouped pass;
                the kernels only ever see a compact block either way."""
                _nsys_push("cxr.lines.coherent_prologue")
                coh_line_data = run_coherent_prologue_kernel(
                    v_all[sel].reshape(-1),
                    denom_full[sel].reshape(-1),
                    gamma_full[sel].reshape(-1),
                    t_L_full[sel].reshape(-1),
                    L_esc_full[sel].reshape(-1),
                    line_electron[sel],
                    seg_r_geom[sel].reshape(-1),
                    _coh_phase_slope[sel],
                    _coh_g,
                    _coh_es,
                    _coh_ep,
                    E_tab_g,
                    _coh_chi_re,
                    _coh_chi_im,
                    _coh_u_re,
                    _coh_u_im,
                    log_mu_tab_g,
                    lo_keep=lo_keep,
                    hi_keep=hi_keep,
                    hbarc=HBARC_EV_ANG,
                    root_rtol=_RESONANCE_ROOT_RTOL,
                    electron_mass_eV=1.0,  # U tables are already U_g/m_e
                    alpha_fs=ALPHA_FS,
                    pref_c1=_PREF_C1,
                    n_hat=n_hat,
                    n_g=N_g,
                    g2=_coh_g2,
                    n_dot_g=_coh_n_dot_g,
                    g_dot_es=_coh_g_dot_es,
                    g_dot_ep=_coh_g_dot_ep,
                    v_dot_n=v_dot_n_all[sel],
                    n_re_tab=n_re_tab_g,
                    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
                )
                _nsys_pop()
                _nsys_push("cxr.lines.coherent_field")
                run_coherent_field_accumulation_kernel(
                    *coh_line_data,
                    E_grid,  # ty: ignore[too-many-positional-arguments]
                    fields=coherent_fields,
                    n_g=N_g,
                    n_seg=n_sel,
                    # g-independent, so it stays segment-sized here even though
                    # the prologue's other outputs are pair-sized.
                    L_esc=L_esc_full[sel].reshape(-1),
                    delta_omega=delta_omega_grid,
                    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
                )
                _nsys_pop()
                del coh_line_data  # release scratch before the next block allocates

            def _stream_field_mag2():
                """|f_s|^2 + |f_p|^2 of the current field planes, as (N_g, n_E).

                Deliberately NOT ``finalize_coherent_fields``: that kernel folds
                the mosaic-weighted sum over rows into ``spec`` in the same
                pass, but F depends on the row (through q_perp = (omega n + g)_perp)
                and must multiply BEFORE the rows are summed."""
                mag2 = sum(plane * plane for plane in coherent_fields)
                return mag2.reshape(N_g, E_grid.size)
        else:
            coherent_fields = ()

        # Existing per-row coherent reducer remains the compatibility fallback for
        # non-streaming coherent execution (e.g. disabling the experimental stage).
        _use_jit_coherent_reduction = (
            coherent
            and not _use_jit_coherent_stream
            and _USE_JIT_COHERENT_REDUCTION
            and getattr(xp, "__name__", "") == "cupy"
            and np.dtype(REAL) == np.dtype(np.float32)
        )
        if _use_jit_coherent_reduction:
            from .coherent_jit_kernel import (
                DEFAULT_COHERENT_KERNEL_CONFIG,
                run_coherent_reduction_kernel,
            )

        n_seg = v_all.shape[0]
        pair_target = _JIT_COHERENT_PAIR_TARGET if coherent else 1_000_000
        seg_block = max(1, pair_target // max(1, N_g))  # bound (n_block, N_g) temporaries

        # Compatibility-fallback buffers. The streaming RawKernel path above
        # does not retain line records across segment blocks; these stay empty when
        # it is active.
        coh_blocks = []
        coh_counts = []

        pending_E_r = []
        pending_aw = []
        pending_w = []
        pending_n_lines = 0

        def _flush_line_batch():
            nonlocal pending_n_lines

            if pending_n_lines == 0:
                return

            if len(pending_E_r) == 1:
                E_r_batch = pending_E_r[0]
                aw_batch = pending_aw[0]
                w_batch = pending_w[0]
            else:
                E_r_batch = xp.concatenate(pending_E_r)
                aw_batch = xp.concatenate(pending_aw)
                w_batch = xp.concatenate(pending_w)

            _nsys_push("cxr.lines.reduce")

            run_reduction_kernel(
                E_r_batch,
                aw_batch,
                w_batch,
                E_grid,
                out=spec,
                config=DEFAULT_SPECTRUM_KERNEL_CONFIG,
            )

            _nsys_pop()

            pending_E_r.clear()
            pending_aw.clear()
            pending_w.clear()
            pending_n_lines = 0

        for s0 in range(0, n_seg, seg_block):
            sb = slice(s0, min(s0 + seg_block, n_seg))

            if _use_jit_coherent_stream:
                _stream_segment_block(sb, sb.stop - sb.start)
                continue

            vx = v_all[sb, 0][:, None]  # (nb, 1)F
            vy = v_all[sb, 1][:, None]
            vz = v_all[sb, 2][:, None]
            # The in-medium root depends on g through E_res, so denom stops being
            # a hoisted column and n.g stops being a hoisted row: both become
            # (nb, N_g). _line_kin_core is elementwise, so it takes them
            # unchanged.
            denom, n_re_blk = _in_medium_kinematics(
                v_dot_n_all[sb][:, None],
                vx * gx + vy * gy + vz * gz,
                n_re_tab_g,
                E_tab_g,
            )
            n_dot_g_blk = n_re_blk * n_dot_g
            gamma = gamma_full[sb]
            t_L = t_L_full[sb]
            L_esc = L_esc_full[sb]

            # -- 1+4. resonance energy + photon kinematics (fused kernel) --------
            # steps 1 and 4 are reduced algebraically inside one fused launch;
            # g-only invariants are hoisted outside the segment-block loop.
            omega_res, v_dot_g, detuning, k_dot_g, v_dot_kg, k_dot_v = _line_kin_core(
                vx, vy, vz, gx, gy, gz, denom, g2, n_dot_g_blk
            )
            vdg = v_dot_g
            k_mag = omega_res * n_re_blk
            E_res = HBARC_EV_ANG * omega_res

            line_electron_block = line_electron[sb][:, None]

            keep = line_electron_block & (E_res > lo_keep) & (E_res > 10.0) & (E_res < hi_keep)

            # -- 3. couplings at each resonance energy (shared interp index) -----
            # chi/U real+imag all sample the SAME E_res on the SAME E_tab_g grid,
            # so bracket ONCE (_interp_index) and gather per table -- kills the
            # fourfold-redundant searchsorted/clip. Bit-for-bit vs _batch_interp.
            _ix, _fr, _blw, _abv = _interp_index(E_res, E_tab_g)
            _log_fr = _log_interp_fraction(E_res, E_tab_g, _ix)
            chi_re, chi_im, u_re, u_im, mu = _interp_gather_line_tables(
                _ix,
                _fr,
                _log_fr,
                _blw,
                _abv,
                CHI_RE,
                CHI_IM,
                U_RE,
                U_IM,
                log_mu_tab_g,
            )
            if coherent:
                # -- 5c. COMPLEX amplitude per polarization ---------------------
                # Eq. (13) PXR + relativistic Eq. (14) CBS, the SAME expression
                # tree the per-hkl coherent path evaluates, now on the
                # (n_block, N_g) grid. Assumptions are inherited unchanged:
                # amplitudes frozen at E_res across the narrow line, orthogonal
                # polarizations add incoherently, kinematics from steps 1/4.
                # Every operation here is elementwise, so this step introduces NO
                # reduction of its own -- the batch debt is the shared steps
                # 1/3/4 (component dots, gathered interp).
                chi = chi_re + 1j * chi_im
                eUg_over_m = u_re + 1j * u_im  # u_re/u_im already carry 1/M_E_EV
                pol_A = []
                for E_pol, g_dot_e in ((ES, G_DOT_ES[None, :]), (EP, G_DOT_EP[None, :])):
                    ex, ey, ez = E_pol[:, 0][None, :], E_pol[:, 1][None, :], E_pol[:, 2][None, :]
                    v_dot_e = vx * ex + vy * ey + vz * ez  # (nb, N_g)
                    A_PXR = chi / detuning * (v_dot_kg * g_dot_e - k_mag**2 * v_dot_e)
                    braced_ge = g_dot_e - vdg * v_dot_e
                    braced_kg = k_dot_g - k_dot_v * vdg
                    A_CBS = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
                    pol_A.append(A_PXR + A_CBS)

                # -- 6c. Beer-Lambert escape + field coefficients ---------------
                # amp = sqrt(alpha omega / (4 pi^2 hbar c) * T_abs) is the
                # UN-squared prefactor (its square is the incoherent ``pref``
                # without t_L^2); the finite-time factor t_L sinc(.) and the
                # phase exp[i(omega d_j - g.r_j)] are applied by the reduction.
                amp = xp.sqrt(ALPHA_FS * omega_res / _PREF_C1 * xp.exp(-(L_esc * mu)))
                a_width = denom * t_L / (2.0 * HBARC_EV_ANG) * xp.ones_like(omega_res)
                good = keep & xp.isfinite(amp) & (amp > 0) & (t_L > 0)
                shape = omega_res.shape
                # Flatten g-MAJOR so each row's kept lines land contiguously:
                # one masked copy serves all N_g rows, and the per-row split is
                # pure slicing. The row lengths stay ON DEVICE here and are
                # fetched in a single sync after the block loop -- syncing per
                # block drained the queue between blocks and cost more than the
                # transfer itself.
                gm = xp.ascontiguousarray(good.T).reshape(-1)
                c_s, c_p = ((amp * t_L) * A_e for A_e in pol_A)
                # Geometric-only (offset-free) phase: identical to
                # d_all/seg_r when no decoherence-relevant offset is
                # configured, so this is a no-op swap in that (default)
                # case. See the coherent-inter-electron-decoherence block
                # above (_accumulate's 7c does the same swap).
                gxr, gyr, gzr = (
                    seg_r_geom[sb, 0][:, None],
                    seg_r_geom[sb, 1][:, None],
                    seg_r_geom[sb, 2][:, None],
                )
                per_line = (
                    E_res,
                    a_width,
                    (d_all_geom[sb] / HBARC_EV_ANG)[:, None],  # phase slope vs E
                    gxr * gx + gyr * gy + gzr * gz,  # g.r_j
                    c_s.real,
                    c_s.imag,
                    c_p.real,
                    c_p.imag,
                )
                # Escape distance rides along as the second phase slope, the one
                # that multiplies delta(E) omega(E) instead of E. The emitting
                # electron id rides along too, needed only when
                # decoherence_active blends in the electron-grouped floor.
                per_line = (*per_line, L_esc, seg_elec_id[sb][:, None])
                coh_blocks.append(
                    [
                        xp.ascontiguousarray(xp.broadcast_to(f, shape).T).reshape(-1)[gm]
                        for f in per_line
                    ]
                )
                coh_counts.append(good.sum(axis=0))  # (N_g,), still on device
                continue

            # -- 5. |A|^2 summed over both polarizations ------------------------
            A2 = xp.zeros_like(omega_res)
            A2_pxr = xp.zeros_like(omega_res)
            A2_cbs = xp.zeros_like(omega_res)
            for E_pol, g_dot_e in ((ES, G_DOT_ES[None, :]), (EP, G_DOT_EP[None, :])):
                ex, ey, ez = E_pol[:, 0][None, :], E_pol[:, 1][None, :], E_pol[:, 2][None, :]
                v_dot_e = vx * ex + vy * ey + vz * ez  # (nb, N_g)
                a2, a2p, a2c = _line_amp_sq_core(
                    chi_re,
                    chi_im,
                    u_re,
                    u_im,
                    v_dot_kg,
                    g_dot_e,
                    k_mag,
                    v_dot_e,
                    vdg,
                    k_dot_g,
                    k_dot_v,
                    gamma,
                    detuning,
                )
                A2 = A2 + a2
                A2_pxr = A2_pxr + a2p
                A2_cbs = A2_cbs + a2c

            # -- 6+7. Beer-Lambert escape factor + weights (fused prefactor) -----
            # mu(E_res) reuses the step-3 interp bracket (no fresh searchsorted);
            # _line_weight_core folds T_abs = exp(-L_esc mu) into the PXR
            # prefactor as one launch. a_width stays inline (needs ones_like).
            pref = _line_weight_core(omega_res, t_L, L_esc, mu, ALPHA_FS, _PREF_C1)
            a_width = denom * t_L / (2.0 * HBARC_EV_ANG) * xp.ones_like(omega_res)
            weight = pref * A2 * WM
            good = keep & xp.isfinite(weight) & (weight > 0)
            # Resolve the mask once. CuPy otherwise performs one blocking
            # survivor-count readback for ``any`` and one for every boolean
            # gather below. Integer gathers preserve the same flattened order.
            gm_idx = xp.flatnonzero(good.reshape(-1))
            if gm_idx.size == 0:
                continue
            E_r_f = E_res.reshape(-1)[gm_idx]
            aw_f = a_width.reshape(-1)[gm_idx]
            w_f = weight.reshape(-1)[gm_idx]

            _nsys_push("cxr.lines.accum")

            if _use_jit_line_reduction:
                pending_E_r.append(E_r_f)
                pending_aw.append(aw_f)
                pending_w.append(w_f)

                pending_n_lines += int(E_r_f.size)

                if pending_n_lines >= _JIT_LINE_BATCH_TARGET:
                    _flush_line_batch()

            else:
                # Existing accumulation retained as the CPU / compatibility fallback.
                targets = [(weight, spec)]
                if components:
                    targets.append((pref * A2_pxr * WM, spec_pxr))
                    targets.append((pref * A2_cbs * WM, spec_cbs))
                for w, tgt in targets:
                    w_f = w.reshape(-1)[gm_idx]
                    for j0 in range(0, E_r_f.size, chunk):
                        sl2 = slice(j0, min(j0 + chunk, E_r_f.size))
                        S = _sincsq_lineshape(
                            aw_f[sl2][:, None], E_grid[None, :], E_r_f[sl2][:, None]
                        )
                        tgt += w_f[sl2] @ S
            _nsys_pop()

        # Flush the residual line batch even when it never reached the target.
        if _use_jit_line_reduction:
            _flush_line_batch()

        if _use_jit_coherent_stream:
            _nsys_push("cxr.lines.coherent_finalize")
            if decoherence_active:
                # The planes now hold |sum_e S_e| per row (the "flat" term);
                # snapshot its magnitude before the grouped passes reuse them.
                flat_mag2 = _stream_field_mag2()
                # sum_e |S_e|^2 with no new device code: re-stream one
                # electron's segments at a time into the SAME planes (zeroed
                # first) and square. Every segment is still touched once in
                # total across the electron loop; the extra cost is Ne
                # zero+square passes over the (N_g, n_E) planes plus Ne kernel
                # launch pairs, traded for keeping this path on the fused
                # float32 kernels instead of the dense-matrix CuPy fallback.
                grouped_mag2 = xp.zeros((N_g, E_grid.size), dtype=REAL)
                elec_cpu = _to_cpu(seg_elec_id)
                order = np.argsort(elec_cpu, kind="stable")
                gid_e = elec_cpu[order]
                order = order[gid_e < Ne]  # secondaries never radiate a line
                gid_e = gid_e[gid_e < Ne]
                if gid_e.size:
                    e_starts = np.flatnonzero(np.concatenate(([True], gid_e[1:] != gid_e[:-1])))
                    e_bounds = np.append(e_starts, gid_e.size)
                    for b0, b1 in zip(e_bounds[:-1], e_bounds[1:], strict=True):
                        idx_e = order[int(b0) : int(b1)]
                        for plane in coherent_fields:
                            plane.fill(0)
                        for j0 in range(0, idx_e.size, seg_block):
                            sub = xp.asarray(idx_e[j0 : j0 + seg_block])
                            _stream_segment_block(sub, int(sub.size))
                        grouped_mag2 += _stream_field_mag2()
                # F PER ROW, before the incoherent row sum.
                F_rows = xp.stack([_row_decoherence_factor(G[i_row]) for i_row in range(N_g)])
                blended = (1.0 - F_rows) * grouped_mag2 + F_rows * flat_mag2
                spec[:] += (WM.reshape(-1)[:, None] * blended).sum(axis=0)
            else:
                finalize_coherent_fields(
                    coherent_fields,
                    WM.reshape(-1),
                    out=spec,
                    n_g=N_g,
                    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
                )
            _nsys_pop()
        elif coherent:
            # -- 7c. per-(reflection, orientation) coherent reduction ----------
            # Sum the phased complex field over ALL of ONE row's segments, square
            # it, and add |F_s|^2 + |F_p|^2 weighted by that row's mosaic weight.
            # Rows are reduced independently, so reflections and mosaic
            # orientations stay INCOHERENT (docs/physics/materials/crystal-mosaicity.md route 2)
            # while the segment sum inside a row keeps its phase -- the defining
            # property the per-hkl coherent path had, preserved verbatim.
            # Limiting case: one row, one segment collapses to the incoherent
            # self-term |A|^2 t_L^2 sinc^2 (test_coherent_emission.py).
            _nsys_push("cxr.lines.accum")
            # ONE sync for every block's row lengths, then pure slicing.
            counts = _to_cpu(xp.stack(coh_counts)) if coh_blocks else np.zeros((0, N_g), int)
            starts = np.concatenate(
                [np.zeros((counts.shape[0], 1), counts.dtype), np.cumsum(counts, axis=1)], axis=1
            )
            for i_row in range(N_g):
                blocks = [
                    [f[int(starts[b, i_row]) : int(starts[b, i_row + 1])] for f in flat]
                    for b, flat in enumerate(coh_blocks)
                    if counts[b, i_row]
                ]
                if not blocks:
                    continue
                if len(blocks) == 1:
                    row = blocks[0]
                else:
                    row = [xp.concatenate(parts) for parts in zip(*blocks, strict=True)]
                E_r_i, aw_i, ps_i, gp_i, csr, csi, cpr, cpi = row[:8]
                L_i = row[8]  # escape distance for the in-medium phase
                elec_id_i = row[9]  # emitting electron id (decoherence_active only)
                wm_i = float(wm_rows[i_row])
                if _use_jit_coherent_reduction:
                    per_line_i = (E_r_i, aw_i, ps_i, gp_i, csr, csi, cpr, cpi)
                    out_flat = spec if not decoherence_active else xp.zeros(E_grid.size, dtype=REAL)
                    run_coherent_reduction_kernel(
                        *per_line_i,
                        E_grid,
                        out=out_flat,
                        mosaic_weight=1.0 if decoherence_active else wm_i,
                        L_esc=L_i,
                        delta_omega=delta_omega_grid,
                        config=DEFAULT_COHERENT_KERNEL_CONFIG,
                    )
                    if decoherence_active:
                        grouped_total = _coherent_jit_grouped_row(
                            elec_id_i,
                            per_line_i,
                            L_i,
                            xp.zeros(E_grid.size, dtype=REAL),
                        )
                        F_row = _row_decoherence_factor(G[i_row])
                        spec[:] += ((1.0 - F_row) * grouped_total + F_row * out_flat) * wm_i
                    continue
                f_s = xp.zeros(E_grid.size, dtype=cdtype)
                f_p = xp.zeros(E_grid.size, dtype=cdtype)
                for j0 in range(0, E_r_i.size, chunk):
                    sl = slice(j0, min(j0 + chunk, E_r_i.size))
                    x = aw_i[sl][:, None] * (E_grid[None, :] - E_r_i[sl][:, None]) / xp.pi
                    arg = ps_i[sl][:, None] * E_grid[None, :] - gp_i[sl][:, None]
                    arg = arg - L_i[sl][:, None] * delta_omega_grid[None, :]
                    ph = xp.exp(1j * arg)
                    SP = xp.sinc(x).astype(cdtype) * ph
                    f_s += (csr[sl] + 1j * csi[sl]) @ SP
                    f_p += (cpr[sl] + 1j * cpi[sl]) @ SP
                flat_total = xp.abs(f_s) ** 2 + xp.abs(f_p) ** 2
                if decoherence_active:
                    grouped_total = _coherent_electron_grouped_row(
                        elec_id_i,
                        aw_i,
                        E_r_i,
                        ps_i * HBARC_EV_ANG,  # ps_i is d_geom/HBARC_EV_ANG; undo the fold
                        gp_i,
                        L_i,
                        [csr + 1j * csi, cpr + 1j * cpi],
                    )
                    F_row = _row_decoherence_factor(G[i_row])
                    spec[:] += ((1.0 - F_row) * grouped_total + F_row * flat_total) * wm_i
                else:
                    spec[:] += flat_total * wm_i
            _nsys_pop()

    if components:
        return _to_cpu(spec / Ne), _to_cpu(spec_pxr / Ne), _to_cpu(spec_cbs / Ne)
    return _to_cpu(spec / Ne)


def mc_spectrum_solid_angle(
    segments,
    E_grid_eV,
    crystal,
    hkl_list,
    *,
    n_hats,
    weights,
    groove=None,
    **kwargs,
):
    """
    Solid-angle-INTEGRATED per-electron line spectrum dN/dE [photons / eV /
    electron], already x Omega: ``sum_i weights_i * mc_spectrum(n_hat=n_hats_i)``.

    The finite detector face is tiled by detector_directions() into directions
    ``n_hats`` (sample frame) carrying solid-angle ``weights``. Because the
    resonance energy AND the amplitudes depend on n_hat, summing per-direction
    spectra yields the true, generally ASYMMETRIC integrated lineshape and the
    across-face intensity gradient -- the first-principles replacement for the
    flat-Omega + analytic aperture_fwhm_eV pair (docs/physics/detectors/detector-solid-angle.md). It
    reuses the validated single-angle mc_spectrum, so a 1-direction grid
    reproduces ``spec * Omega`` exactly (the regression anchor).

    Units: the result ALREADY includes the solid angle (the weights carry
    dOmega). When consuming it do NOT multiply by domega_sr again, and drop the
    aperture_fwhm_eV term from the detector convolution (keep the EDS-resolution
    term). This is an opt-in tool; it does not change the checkpoint pipeline's
    single-n_hat unit convention. ``**kwargs`` forward to mc_spectrum (B_ang2,
    use_henke, layers, composition, beam_uvw, azimuth_rad, mosaic_*, ...).

    groove: forwarded to mc_spectrum's blazed-groove escape model, but ONLY
    when ``n_hats`` carries a single direction (the ``n_side=1`` case of
    detector_directions() -- N == n_side**2, so N > 1 means n_side > 1).
    Tiling the detector face into multiple directions breaks the relief-facet
    parallelism the groove geometry assumes (each tile would need its own
    working-facet family), so a multi-direction grid with groove set raises
    ValueError rather than silently mixing per-tile escape paths.

    Validation: blazed-groove-geometry
    """
    n_hats = np.asarray(n_hats, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if n_hats.ndim != 2 or n_hats.shape[1] != 3:
        raise ValueError("n_hats must be (N, 3)")
    if weights.shape != (n_hats.shape[0],):
        raise ValueError("weights must be (N,) matching n_hats")
    if groove is not None and n_hats.shape[0] > 1:
        raise ValueError(
            "groove escape supports n_side=1 only -- detector tiles break "
            "the relief-facet parallelism"
        )
    total = None
    for n_i, w_i in zip(n_hats, weights, strict=True):
        spec_i = np.asarray(
            mc_spectrum(segments, E_grid_eV, crystal, hkl_list, n_hat=n_i, groove=groove, **kwargs)
        )
        contrib = float(w_i) * spec_i
        total = contrib if total is None else total + contrib
    return total


# ---- bremsstrahlung background -------------------------------------------------
