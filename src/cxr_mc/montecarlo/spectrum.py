"""
montecarlo.spectrum

Radiation from the transported segments (Zhai SI S1): the finite-interaction-
time CXR (PXR + CBS) line spectrum mc_spectrum, its solid-angle-integrated
wrapper, the Bethe-Heitler bremsstrahlung background, and the external-brem
loader. The array-heavy inner loops run on the GPU backend (``xp``) when
available.
"""

import numpy as np

from ..materials.attenuation import (
    _layer_dz,
    _layer_path_length,
    _mu_total_inv_ang,
    _normalize_composition,
    _stack_tau,
)
from ..materials.crystal import (
    ALPHA_FS,
    CRYSTALS,
    HBARC_EV_ANG,
    M_E_EV,
    U_g,
    chi_g,
    reciprocal_g_vector,
)
from ._backend import REAL, _to_cpu, xp
from .geometry import _mosaic_quadrature, _orientation_R, first_prism_exit
from .groove import _THETA_TOL, escape_distance_ang
from .transport import TRANSPORT_ELEMENTS, beta_from_keV

# ---- segment-sum CXR spectrum ------------------------------------------------
_SEG_ARRAYS = ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "t0_ang", "elec_id", "layer")
_USE_JIT_LINE_REDUCTION = True
# Staged behind an explicit opt-in until the new kernel passes CUDA compilation,
# numerical goldens, repeatability, and A/B timing on the supported GPUs.
_USE_JIT_LINE_PROLOGUE = False
_USE_JIT_COHERENT_REDUCTION = True
_USE_JIT_BREM_REDUCTION = True
_JIT_LINE_BATCH_TARGET = 400_000


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
    LEDGER + REGEN REQUIRED: add a physics-ledger row and regenerate the affected
    spectrum goldens (regen-golden) before this is signed off. Validation: line-amplitude-fusion
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
    handle, no launch) and runs bit-for-bit identical on NumPy.

    NOT bit-for-bit vs BLAS: the sum is reassociated to ``(a0*b0 + a1*b1) +
    a2*b2``, which differs from GEMV accumulation at float-rounding level.
    LEDGER + REGEN REQUIRED: fold into the physics-ledger row and regenerate the
    affected spectrum goldens (regen-golden) before sign-off.
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
        "raw I idx, raw float32 frac, raw bool below, raw bool above, "
        "raw float32 chi_re_tab, raw float32 chi_im_tab, "
        "raw float32 u_re_tab, raw float32 u_im_tab, raw float32 mu_tab, "
        "int32 n_g, int32 n_tab",
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

        const long long mu_lo = bracket - 1;
        a0 = mu_tab[mu_lo];
        mu = use_lo ? mu_tab[0]
             : (use_hi ? mu_tab[n_tab - 1]
                       : __fadd_rn(a0, __fmul_rn(f, __fsub_rn(mu_tab[bracket], a0))));
        """,
        "cxr_interp_gather_line_tables_f32",
    )


def _interp_gather_line_tables(
    idx,
    frac,
    below,
    above,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    mu_tab,
):
    """Gather every line-coupling table from one shared interpolation bracket."""
    if (
        _INTERP_GATHER_LINE_TABLES_F32 is not None
        and idx.dtype.kind in "iu"
        and frac.dtype == xp.float32
        and chi_re_tab.dtype == xp.float32
        and mu_tab.dtype == xp.float32
    ):
        n_g, n_tab = chi_re_tab.shape
        flat = _INTERP_GATHER_LINE_TABLES_F32(
            idx,
            frac,
            below,
            above,
            chi_re_tab,
            chi_im_tab,
            u_re_tab,
            u_im_tab,
            mu_tab,
            np.int32(n_g),
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
        _interp_gather1d(idx, frac, below, above, mu_tab),
    )


def _line_kin_core(vx, vy, vz, gx, gy, gz, denom, nx, ny, nz):
    """Resonance frequency + photon kinematics for the batched line path as a
    single fused GPU kernel: collapses the ~15 tiny elementwise launches of
    inline steps 1+4 (``v.g``, ``omega_res``, ``k``, ``kg``, ``detuning``,
    ``k.g``, ``v.kg``, ``k.v``) into one. Pure kernel-merge of the identical
    expression tree (no reassociation, ``x**2`` written ``x*x``) -> bit-for-bit
    vs the inline code; carries NO new validation debt of its own."""
    v_dot_g = vx * gx + vy * gy + vz * gz
    omega_res = v_dot_g / denom
    kx, ky, kz = omega_res * nx, omega_res * ny, omega_res * nz
    kgx, kgy, kgz = kx + gx, ky + gy, kz + gz
    kg2 = kgx * kgx + kgy * kgy + kgz * kgz
    detuning = kg2 - omega_res * omega_res
    k_dot_g = kx * gx + ky * gy + kz * gz
    v_dot_kg = vx * kgx + vy * kgy + vz * kgz
    k_dot_v = omega_res * (1.0 - denom)
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
):
    """
    Per-electron CXR spectrum d2N/dE dOmega [photons / eV / sr / electron]
    on E_grid_eV, summed incoherently over the trajectory segments and the
    listed reflections (their resonances are spectrally separated, so
    cross-g coherence is negligible).

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
    (docs/crystal-mosaicity.md (2)). None / nodes<=1 (the default) is a perfect
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
    ``docs/superpowers/specs/2026-07-24-groove-aware-transport-design.md``.

    Validation: blazed-groove-geometry

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
    ``omega t_abs,j`` (``t_abs = t_ang + t0_ang``, the segment age plus the
    per-electron bunch offset, in Ang with c=1) and the far-field retardation
    ``omega n_hat.r_j``. The reciprocal-harmonic spatial phase
    ``g.r_j`` follows the repository's structure-factor convention
    ``S(g)=sum F exp(+i g.R)``, whose susceptibility harmonic is
    ``chi_g exp(-i g.r)``. Distinct reflections are spectrally separated, so
    the coherent sum runs WITHIN each reflection and orientation and
    reflections/orientations still add incoherently.

    # TODO: To be confirmed numerically that lines are sufficiently separated to ignore cross-g coherence.
    # TODO: This needs to be added to the validation ledger/documentation.

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

    Validation: coherent-emission
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
    info = CRYSTALS[crystal]
    n_atoms = len(info["basis"]) / info["V_cell"]
    abs_comp = _normalize_composition(absorber_element, n_atoms, composition)

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

    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)
    seg_v = xp.asarray(segments["v_hat"], dtype=REAL)
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)
    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)
    seg_elec_id = xp.asarray(segments["elec_id"])
    line_electron = seg_elec_id < Ne
    if E_cut_keV is not None:
        # See mc_brem_spectrum: shared dual-use electrons are transported to the
        # LOWER of the two cutoffs, so restore this population's own floor.
        line_electron = line_electron & (seg_E >= REAL(E_cut_keV))
    beta_all = beta_from_keV(seg_E)  # speed/c per segment
    v_all = beta_all[:, None] * seg_v  # velocity vectors (c=1)

    # chi_g / U_g are smooth in energy AWAY from absorption edges, so evaluate
    # them on a tabulation grid and interpolate at the per-segment resonance
    # energies ON THE GPU (step 3) -- a few ms over ~10^3 grid points instead of
    # ~0.25 s/case over ~10^5 segments, and E_res stays on the device (no
    # per-reflection GPU->CPU->GPU round-trip; that structure-factor CPU cost was
    # what capped GPU utilisation on a fast card). The grid is a 1 eV mesh UNION
    # the basis elements' native Henke energies, which densely sample the edges --
    # a plain uniform mesh mis-resolves the edge jumps (tens of % at e.g. the
    # C K-edge). Window matches the keep mask below.
    from ..materials.atomic import load_henke

    _pad = 0.2 * (float(E_grid_eV[-1]) - float(E_grid_eV[0]))
    _lo, _hi = float(E_grid_eV[0]) - _pad, float(E_grid_eV[-1]) + _pad
    _lo = max(_lo, 1.0)  # keep tabulation energies positive: chi_g/U_g need lambda = HC_EV_ANG / E
    _grids = [np.arange(_lo, _hi + 1.0, 1.0)]
    for _el in {el for el, _ in info["basis"]}:
        try:
            _Eh = load_henke(_el)[0]
            _grids.append(_Eh[(_Eh >= _lo) & (_Eh <= _hi)])
        except Exception:
            pass
    E_tab = np.unique(np.concatenate(_grids))
    E_tab_g = xp.asarray(E_tab, dtype=REAL)
    # Absorption coefficient mu(E) [1/Ang] tabulated on the SAME edge-resolved
    # E_tab grid as chi/U, for on-device interpolation at each resonance energy
    # (_interp1) in both the batched and per-hkl line paths. Previously mu was
    # evaluated exactly per (segment, reflection) via a CPU xraydb spline behind
    # a per-block _to_cpu sync -- the real GPU-starving cost (~59 s at ~9% GPU
    # utilisation). mu uses only f2 and is smoother between nodes than chi/U (no
    # f1 edge cusp), so this is at least as accurate as the chi/U interpolation
    # already accepted here. Applied to the single-slab and finite-footprint
    # branches (coherent AND incoherent alike, so their single-segment
    # limiting-case identity still holds bit-for-bit); the layered _stack_tau
    # and grooved escape paths keep exact per-point mu.
    #
    # Physics-METHOD change (tabulated vs exact absorption), not a reassociation.
    # LEDGER + REGEN + human sign-off REQUIRED. Validation: line-absorption-tabulation
    mu_tab_g = xp.asarray(np.asarray(_mu_total_inv_ang(abs_comp, E_tab)), dtype=REAL)

    n_hat_d = xp.asarray(n_hat, dtype=REAL)  # detector dir is g-independent: hoist

    # coherent (phased) sum precompute: the per-segment retardation scalar
    # d_j = t_abs,j - n_hat.r_j [Ang, c=1] (emission-time phase minus far-field
    # retardation) and photon wavenumber k_gamma(E) = E / hbar c [1/Ang].
    # Each reflection adds its spatial susceptibility phase -g.r_j inside
    # _accumulate. All-zero t0_ang leaves the physical trajectory phase.
    # Inert unless coherent=True.
    if coherent:
        cdtype = xp.result_type(REAL, 1j)
        seg_t0 = xp.asarray(segments.get("t0_ang", np.zeros(seg_E.size)), dtype=REAL)
        seg_t = xp.asarray(segments.get("t_ang", np.zeros(seg_E.size)), dtype=REAL)
        d_all = (seg_t + seg_t0) - _matvec3(seg_r, n_hat_d)
        omega_grid = E_grid / HBARC_EV_ANG
    # mosaic crystallite-orientation quadrature: None -> perfect crystal (default;
    # today's single-orientation result bit-for-bit). Otherwise a list of
    # (rotation, weight) tilting g across the Gaussian mosaic cone, summed
    # incoherently below (docs/crystal-mosaicity.md route 2).
    mosaic_quad = _mosaic_quadrature(mosaic_fwhm_rad, mosaic_nodes)

    # NVTX sub-ranges to split the coarse ``cxr.lines`` range into structure-
    # factor tabulation vs per-reflection accumulation (no-op off the profiled
    # GPU path). Lazy import: runner imports this module, so a top-level import
    # would be circular.
    from .runner import _nsys_pop, _nsys_push

    def _accumulate(g_vec_d, e_s, e_p, chi_re, chi_im, u_re, u_im, wm):
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
        denom = 1.0 - _matvec3(v_all, n_hat_d)  # the Doppler-like denominator
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
        beta = beta_all[idx]
        t_L = seg_L[idx] / beta  # interaction time [Ang] (c=1)
        dnm = denom[idx]
        vdg = v_dot_g[idx]

        # -- 3. couplings AT each segment's resonance energy --------------------
        # (amplitudes vary slowly across the narrow line; freezing them at E_res
        # is accurate to the linewidth/E_res level). Interpolated at E_res ON THE
        # GPU from the per-reflection tabulation; chi_g/U_g are complex, so the
        # real and imaginary parts are interpolated separately.
        chi = xp.interp(E_r, E_tab_g, chi_re) + 1j * xp.interp(E_r, E_tab_g, chi_im)
        eUg_over_m = (xp.interp(E_r, E_tab_g, u_re) + 1j * xp.interp(E_r, E_tab_g, u_im)) / M_E_EV

        # -- 4. photon kinematics per segment ------------------------------------
        k_vec = om[:, None] * n_hat_d  # photon wavevector omega * n_hat
        kg_vec = k_vec + g_vec_d  # diffracted wavevector k + g
        kg2 = _rowdot3(kg_vec, kg_vec)
        detuning = kg2 - om**2  # PXR denominator (~g^2, never small)
        k_dot_g = _matvec3(k_vec, g_vec_d)
        # v.kg is polarization-independent (kg fixed per segment): hoist out of
        # the e_s/e_p loop so it is contracted once, not twice.
        v_dot_kg = _rowdot3(v, kg_vec)

        # -- 5. Eq. (13) + relativistic Eq. (14) amplitudes, per segment ----------
        # CBS braced product {a;b} = a.b - (a.v)(b.v) and 1/gamma prefactor
        # (Zhai SI Eq. 6); v here is the SEGMENT velocity, so k.v = omega(1-dnm)
        gamma = 1.0 / xp.sqrt(1.0 - beta**2)
        k_dot_v = om * (1.0 - dnm)
        A2 = xp.zeros(idx.size, dtype=REAL)
        A2_pxr = xp.zeros(idx.size, dtype=REAL)
        A2_cbs = xp.zeros(idx.size, dtype=REAL)
        pol_A = []  # complex A = A_PXR + A_CBS per polarization (coherent path)
        for e_d in (e_s, e_p):  # sum |A|^2 over both polarizations
            g_dot_e = g_vec_d @ e_d  # scalar (e fixed per reflection)
            v_dot_e = _matvec3(v, e_d)
            if coherent:
                # Complex amplitudes retained verbatim -- the coherent path sums
                # phased fields, so it keeps the un-reassociated expression and
                # its goldens are unaffected.
                A_PXR = chi / detuning * (v_dot_kg * g_dot_e - om**2 * v_dot_e)
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
                om,
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
                tau = L_esc * _interp1(E_r, E_tab_g, mu_tab_g)
            else:
                tau = _stack_tau(layers, z_mid, n_hat[2], E_r, exit_distance_ang=L_esc)
        else:
            if layers is None:
                if n_hat[2] < 0:
                    L_esc = z_mid / (-n_hat[2])  # out the entrance face
                else:
                    L_esc = (thickness - z_mid) / n_hat[2]  # out the back face
                tau = L_esc * _interp1(E_r, E_tab_g, mu_tab_g)
            else:
                tau = _stack_tau(layers, z_mid, n_hat[2], E_r)
        T_abs = xp.exp(-tau)

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
            d = d_all[idx]
            g_phase = _matvec3(seg_r[idx], g_vec_d)
            coefs = [(amp * t_L) * A_e for A_e in pol_A]  # complex per polarization
            good = xp.isfinite(amp) & (amp > 0) & (t_L > 0)

            # GPU float32 fast path: reduce the two complex polarization fields
            # directly in a raw kernel. This avoids materializing the dense
            # complex SP[segment, energy] matrix and avoids both complex GEMVs.
            # The exact CuPy path below remains the fallback for CPU/other
            # backends, float64, and sinc_cutoff windowing.
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
                    run_coherent_reduction_kernel(
                        xp.ascontiguousarray(E_r[sel], dtype=REAL),
                        xp.ascontiguousarray(a_width[sel], dtype=REAL),
                        xp.ascontiguousarray(d[sel] / HBARC_EV_ANG, dtype=REAL),
                        xp.ascontiguousarray(g_phase[sel], dtype=REAL),
                        xp.ascontiguousarray(c_s[sel].real, dtype=REAL),
                        xp.ascontiguousarray(c_s[sel].imag, dtype=REAL),
                        xp.ascontiguousarray(c_p[sel].real, dtype=REAL),
                        xp.ascontiguousarray(c_p[sel].imag, dtype=REAL),
                        xp.ascontiguousarray(E_grid, dtype=REAL),
                        out=spec,
                        mosaic_weight=wm,
                        config=DEFAULT_COHERENT_KERNEL_CONFIG,
                    )
                return

            fields = [xp.zeros(E_grid.size, dtype=cdtype) for _ in coefs]
            if sinc_cutoff is None:
                for j0 in range(0, idx.size, chunk):
                    sl = slice(j0, min(j0 + chunk, idx.size))
                    m = good[sl]
                    if not m.any():
                        continue
                    x = a_width[sl][m, None] * (E_grid[None, :] - E_r[sl][m, None]) / xp.pi
                    ph = xp.exp(1j * (d[sl][m, None] * omega_grid[None, :] - g_phase[sl][m, None]))
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
                    ph = xp.exp(
                        1j * (d[sel][:, None] * omega_grid[None, i0:i1] - g_phase[sel][:, None])
                    )
                    SP = xp.sinc(x).astype(cdtype) * ph
                    for c, f in zip(coefs, fields, strict=True):
                        f[i0:i1] += c[sel] @ SP
            for f in fields:
                spec[:] += xp.abs(f) ** 2 * wm
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
    if (coherent and sinc_cutoff is not None) or groove is not None or layers is not None:
        # Stacking prologue: every host->device transfer this path needs is done
        # ONCE per case here, not once per (reflection, orientation) inside the
        # loop. Previously each pass re-uploaded the four chi/U tabulations plus
        # g and its two polarization vectors -- ~90 xp.asarray calls per case,
        # 15% of GPU-phase tottime on the 3060 Ti profile (hopg_coherent, 4
        # reflections). The chi/U rows are keyed by REFLECTION (they do not
        # depend on the mosaic orientation), the geometry rows by
        # (reflection, orientation). Values are unchanged -- same CPU inputs,
        # same float64 -> REAL cast, row views handed to _accumulate -- so the
        # per-hkl path stays bit-for-bit.
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
            u_tab = np.asarray(U_g(crystal, hkl, E_tab, B_ang2, use_henke))
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
        # LEDGER + REGEN REQUIRED before sign-off. Validation: line-hkl-batch
        #
        # coherent=True shares steps 1-6 verbatim and diverges only at step 5/7:
        # it keeps the COMPLEX amplitude per polarization and defers the square
        # to a per-(reflection, orientation) reduction, so reflections and mosaic
        # orientations stay incoherent. Validation: coherent-line-hkl-batch
        e_lo = float(_to_cpu(E_grid[0]))
        e_hi = float(_to_cpu(E_grid[-1]))
        pad = 0.2 * (e_hi - e_lo)  # keep sinc tails that reach into the window
        lo_keep, hi_keep = e_lo - pad, e_hi + pad

        # -- stack all (hkl, orientation) g-vectors + per-reflection tabulations --
        _nsys_push("cxr.lines.tab")
        g_rows, es_rows, ep_rows, wm_rows = [], [], [], []
        cr_rows, ci_rows, ur_rows, ui_rows = [], [], [], []
        orients = ((None, 1.0),) if mosaic_quad is None else mosaic_quad
        for hkl in hkl_list:
            g_vec, _g = reciprocal_g_vector(hkl, info["lattice"])
            if R_orient is not None:
                g_vec = R_orient @ g_vec
            chi_tab = np.asarray(chi_g(crystal, hkl, E_tab, B_ang2, use_henke))
            u_tab = np.asarray(U_g(crystal, hkl, E_tab, B_ang2, use_henke))
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
        _nsys_pop()

        N_g = G.shape[0]
        gx, gy, gz = G[:, 0][None, :], G[:, 1][None, :], G[:, 2][None, :]  # (1, N_g)
        nx, ny, nz = float(n_hat[0]), float(n_hat[1]), float(n_hat[2])

        # option B: g-independent per-segment quantities, computed once
        v_dot_n = _matvec3(v_all, n_hat_d)
        denom_full = (1.0 - v_dot_n)[:, None]  # (n_seg, 1)
        gamma_full = (1.0 / xp.sqrt(1.0 - beta_all**2))[:, None]
        t_L_full = (seg_L / beta_all)[:, None]
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
            from .spectrum_jit_kernel import (
                DEFAULT_SPECTRUM_KERNEL_CONFIG,
                run_reduction_kernel,
            )

        _use_jit_line_prologue = _use_jit_line_reduction and _USE_JIT_LINE_PROLOGUE and not coherent
        if _use_jit_line_prologue:
            from .line_prologue_jit_kernel import (
                DEFAULT_LINE_PROLOGUE_KERNEL_CONFIG,
                run_line_prologue_kernel,
            )

            # C-order flattened views are constructed once; slicing v_all by
            # complete rows below remains contiguous and needs no copy.
            _prologue_g = G.reshape(-1)
            _prologue_es = ES.reshape(-1)
            _prologue_ep = EP.reshape(-1)
            _prologue_wm = WM.reshape(-1)
            _prologue_chi_re = CHI_RE.reshape(-1)
            _prologue_chi_im = CHI_IM.reshape(-1)
            _prologue_u_re = U_RE.reshape(-1)
            _prologue_u_im = U_IM.reshape(-1)

        # The coherent reduction kernel stays PER-(reflection, orientation) --
        # its signature is unchanged. Squaring is intrinsically per-row (fields
        # from different g must never mix before |.|^2), so a g-batched kernel
        # would need a 2-D grid plus per-row line offsets to save N_g launches
        # per case (4 on the profiled hopg_coherent shape). The launch storm this
        # branch removes is in steps 1-6, not in the reduction; the row loop
        # below therefore feeds the existing kernel from batched inputs.
        _use_jit_coherent_reduction = (
            coherent
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
        seg_block = max(1, 1_000_000 // max(1, N_g))  # bound (n_block, N_g) temporaries

        # Line buffers for the coherent path. |sum_j|^2 is NOT additive over
        # segment blocks, so unlike the incoherent flush these cannot be reduced
        # incrementally: each row's kept lines are collected across all blocks
        # and reduced once, after the loop. Peak footprint is 8 REAL values per
        # kept (segment, g) pair.
        coh_blocks = []  # per block: 8 g-major, mask-compacted line arrays
        coh_counts = []  # per block: (N_g,) kept-line counts, device-side

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

            if _use_jit_line_prologue:
                _nsys_push("cxr.lines.prologue")
                E_r_f, aw_f, w_f = run_line_prologue_kernel(
                    v_all[sb].reshape(-1),
                    denom_full[sb].reshape(-1),
                    gamma_full[sb].reshape(-1),
                    t_L_full[sb].reshape(-1),
                    L_esc_full[sb].reshape(-1),
                    line_electron[sb],
                    _prologue_g,
                    _prologue_es,
                    _prologue_ep,
                    _prologue_wm,
                    E_tab_g,
                    _prologue_chi_re,
                    _prologue_chi_im,
                    _prologue_u_re,
                    _prologue_u_im,
                    mu_tab_g,
                    lo_keep=lo_keep,
                    hi_keep=hi_keep,
                    hbarc=HBARC_EV_ANG,
                    electron_mass_eV=M_E_EV,
                    alpha_fs=ALPHA_FS,
                    pref_c1=_PREF_C1,
                    n_hat=n_hat,
                    config=DEFAULT_LINE_PROLOGUE_KERNEL_CONFIG,
                )
                _nsys_pop()
                _nsys_push("cxr.lines.reduce")
                run_reduction_kernel(
                    E_r_f,
                    aw_f,
                    w_f,
                    E_grid,
                    out=spec,
                    config=DEFAULT_SPECTRUM_KERNEL_CONFIG,
                )
                _nsys_pop()
                continue

            vx = v_all[sb, 0][:, None]  # (nb, 1)F
            vy = v_all[sb, 1][:, None]
            vz = v_all[sb, 2][:, None]
            denom = denom_full[sb]
            gamma = gamma_full[sb]
            t_L = t_L_full[sb]
            L_esc = L_esc_full[sb]

            # -- 1+4. resonance energy + photon kinematics (fused kernel) --------
            # steps 1 and 4 are the same ~15 tiny elementwise ops for every seg
            # block; _line_kin_core JIT-merges them into ONE launch, bit-for-bit.
            omega_res, v_dot_g, detuning, k_dot_g, v_dot_kg, k_dot_v = _line_kin_core(
                vx, vy, vz, gx, gy, gz, denom, nx, ny, nz
            )
            vdg = v_dot_g
            E_res = HBARC_EV_ANG * omega_res

            line_electron_block = line_electron[sb][:, None]

            keep = line_electron_block & (E_res > lo_keep) & (E_res > 10.0) & (E_res < hi_keep)

            # -- 3. couplings at each resonance energy (shared interp index) -----
            # chi/U real+imag all sample the SAME E_res on the SAME E_tab_g grid,
            # so bracket ONCE (_interp_index) and gather per table -- kills the
            # fourfold-redundant searchsorted/clip. Bit-for-bit vs _batch_interp.
            _ix, _fr, _blw, _abv = _interp_index(E_res, E_tab_g)
            chi_re, chi_im, u_re, u_im, mu = _interp_gather_line_tables(
                _ix,
                _fr,
                _blw,
                _abv,
                CHI_RE,
                CHI_IM,
                U_RE,
                U_IM,
                mu_tab_g,
            )
            u_re = u_re / M_E_EV
            u_im = u_im / M_E_EV

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
                for E_pol in (ES, EP):
                    ex, ey, ez = E_pol[:, 0][None, :], E_pol[:, 1][None, :], E_pol[:, 2][None, :]
                    g_dot_e = gx * ex + gy * ey + gz * ez  # (1, N_g)
                    v_dot_e = vx * ex + vy * ey + vz * ez  # (nb, N_g)
                    A_PXR = chi / detuning * (v_dot_kg * g_dot_e - omega_res**2 * v_dot_e)
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
                gxr, gyr, gzr = seg_r[sb, 0][:, None], seg_r[sb, 1][:, None], seg_r[sb, 2][:, None]
                per_line = (
                    E_res,
                    a_width,
                    (d_all[sb] / HBARC_EV_ANG)[:, None],  # phase slope vs E
                    gxr * gx + gyr * gy + gzr * gz,  # g.r_j
                    c_s.real,
                    c_s.imag,
                    c_p.real,
                    c_p.imag,
                )
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
            for E_pol in (ES, EP):
                ex, ey, ez = E_pol[:, 0][None, :], E_pol[:, 1][None, :], E_pol[:, 2][None, :]
                g_dot_e = gx * ex + gy * ey + gz * ez  # (1, N_g)
                v_dot_e = vx * ex + vy * ey + vz * ez  # (nb, N_g)
                a2, a2p, a2c = _line_amp_sq_core(
                    chi_re,
                    chi_im,
                    u_re,
                    u_im,
                    v_dot_kg,
                    g_dot_e,
                    omega_res,
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
            gm = good.reshape(-1)
            if not bool(gm.any()):
                continue
            E_r_f = E_res.reshape(-1)[gm]
            aw_f = a_width.reshape(-1)[gm]
            w_f = weight.reshape(-1)[gm]

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
                    w_f = w.reshape(-1)[gm]
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

        if coherent:
            # -- 7c. per-(reflection, orientation) coherent reduction ----------
            # Sum the phased complex field over ALL of ONE row's segments, square
            # it, and add |F_s|^2 + |F_p|^2 weighted by that row's mosaic weight.
            # Rows are reduced independently, so reflections and mosaic
            # orientations stay INCOHERENT (docs/crystal-mosaicity.md route 2)
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
                    E_r_i, aw_i, ps_i, gp_i, csr, csi, cpr, cpi = blocks[0]
                else:
                    E_r_i, aw_i, ps_i, gp_i, csr, csi, cpr, cpi = (
                        xp.concatenate(parts) for parts in zip(*blocks, strict=True)
                    )
                wm_i = float(wm_rows[i_row])
                if _use_jit_coherent_reduction:
                    run_coherent_reduction_kernel(
                        E_r_i,
                        aw_i,
                        ps_i,
                        gp_i,
                        csr,
                        csi,
                        cpr,
                        cpi,
                        E_grid,
                        out=spec,
                        mosaic_weight=wm_i,
                        config=DEFAULT_COHERENT_KERNEL_CONFIG,
                    )
                    continue
                f_s = xp.zeros(E_grid.size, dtype=cdtype)
                f_p = xp.zeros(E_grid.size, dtype=cdtype)
                for j0 in range(0, E_r_i.size, chunk):
                    sl = slice(j0, min(j0 + chunk, E_r_i.size))
                    x = aw_i[sl][:, None] * (E_grid[None, :] - E_r_i[sl][:, None]) / xp.pi
                    ph = xp.exp(1j * (ps_i[sl][:, None] * E_grid[None, :] - gp_i[sl][:, None]))
                    SP = xp.sinc(x).astype(cdtype) * ph
                    f_s += (csr[sl] + 1j * csi[sl]) @ SP
                    f_p += (cpr[sl] + 1j * cpi[sl]) @ SP
                spec[:] += (xp.abs(f_s) ** 2 + xp.abs(f_p) ** 2) * wm_i
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
    flat-Omega + analytic aperture_fwhm_eV pair (docs/detector-solid-angle.md). It
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
R_E_CM2 = 7.9407877e-26  # classical electron radius squared [cm^2]
_BREM_MC2_KEV = 510.99895  # electron rest energy [keV]


def _brem_dsigma_dk_core(T_i, k, Z):
    """Pure-elementwise Bethe-Heitler + Elwert core, one fused GPU kernel.

    ``T_i`` (kinetic energy [keV]) and ``k`` (photon energy [keV]) are the
    already-reshaped, mutually broadcasting operands (segments x grid); ``Z`` is
    the scalar atomic number. Split out of ``_brem_dsigma_dk`` so the whole chain
    (sqrt/divide/log/exp/where/maximum -- ~15 CuPy elementwise kernels, each
    allocating a full [seg, E] temporary and dominating ``cxr.brem`` at 300 keV)
    JIT-fuses to a SINGLE kernel under CuPy (see ``xp.fuse`` wrap below). On NumPy
    it runs eager with the identical ops, so it is bit-for-bit the old inline
    expression and the brem-spectrum golden is unchanged."""
    mc2 = _BREM_MC2_KEV
    T_f = T_i - k
    ok = (T_f > 1e-6) & (k > 0.0)  # k>0: no photon (and no 1/k blowup) at k=0
    T_f = xp.where(ok, T_f, 1e-6)

    p_i = xp.sqrt(T_i * (T_i + 2.0 * mc2)) / mc2
    p_f = xp.sqrt(T_f * (T_f + 2.0 * mc2)) / mc2
    beta_i = p_i / (1.0 + T_i / mc2)
    beta_f = p_f / (1.0 + T_f / mc2)

    born_log = xp.log((p_i + p_f) / xp.maximum(p_i - p_f, 1e-30))
    elwert = (
        beta_i
        / beta_f
        * (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_i))
        / (1.0 - xp.exp(-2.0 * xp.pi * Z * ALPHA_FS / beta_f))
    )

    dsig = (
        16.0
        / 3.0
        * ALPHA_FS
        * R_E_CM2
        * Z**2
        / xp.maximum(k * 1e3, 1e-30)
        / p_i**2
        * born_log
        * elwert
    )  # per eV
    return xp.where(ok, dsig, 0.0)


if hasattr(xp, "fuse"):  # CuPy exposes fuse(); NumPy/dpnp do not -> eager fallback
    _brem_dsigma_dk_core = xp.fuse()(_brem_dsigma_dk_core)


def _brem_dsigma_dk(Z, T_keV, k_eV):
    """
    Bremsstrahlung cross section differential in photon energy,
    dsigma/dk [cm^2/eV]: nonrelativistic Bethe-Heitler in Born approximation
    with the Elwert Coulomb correction (cf. Koch & Motz, Rev. Mod. Phys. 31,
    920 (1959)), evaluated with relativistic electron momenta:

        dsigma/dk = (16/3) alpha r_e^2 Z^2 (1/k) (1/p_i^2)
                    ln[(p_i+p_f)/(p_i-p_f)] * f_Elwert,
        f_Elwert  = (beta_i/beta_f) (1-exp(-2 pi Z alpha/beta_i))
                                  / (1-exp(-2 pi Z alpha/beta_f)),

    with p in units of m_e c. Broadcasts T_keV (segments) against k_eV
    (spectral grid); zero where k >= T. Adequate for Z <~ 30 and
    T <~ 100 keV; swap in Seltzer-Berger tables for better accuracy. The
    elementwise math lives in the fused ``_brem_dsigma_dk_core``.

    Validation: brem-spectrum
    """
    T_i = xp.asarray(T_keV, dtype=REAL)[:, None]
    k = xp.asarray(k_eV, dtype=REAL)[None, :] / 1e3  # keV
    # Z must arrive dtype-tagged, not as a bare Python int: cupy.fuse types an
    # untyped scalar operand by value (min_scalar_type), so the
    # 16/3*alpha*r_e^2 prefactor -- a scalar*scalar product with Z**2 -- gets
    # inferred as float16 and flushes ~3e-27 to zero, silently zeroing the
    # whole cross section on every fused (non-raw-kernel) GPU path.
    Z = REAL(Z)
    return _brem_dsigma_dk_core(T_i, k, Z)


def mc_brem_spectrum(
    segments,
    E_grid_eV,
    element=None,
    n_atoms_per_ang3=None,
    theta_obs_rad=np.deg2rad(119.0),
    n_hat=None,
    chunk=20000,
    composition=None,
    layers=None,
    groove=None,
    electron_limit=None,
    E_cut_keV=None,
):
    """
    Incoherent bremsstrahlung background d2N/dE dOmega
    [photons / eV / sr / electron] from the same Monte Carlo segments as
    mc_spectrum: each segment radiates n * dsigma/dk * L_seg photons/eV at
    its (start) kinetic energy, attenuated by the Beer-Lambert escape factor
    from the segment midpoint along the observation direction.

    Approximations: emission taken isotropic (1/4pi) -- the standard
    assumption at weakly relativistic energies once electron directions are
    scattering-randomized (and the one Zhai et al. adopt for their
    estimates); the tiny coherent fraction of the continuum (which is what
    forms the CBS lines) is not subtracted.

    NOTE: run the transport with a LOW E_cut_keV (~1 keV) for backgrounds --
    electrons below the CXR cutoff still radiate in the soft X-ray window.

    composition: [(element, n_per_Ang3), ...] for compounds; the emission is
    additive over elements (each weighted by its own Z^2 cross section), and
    the self-absorption uses the summed attenuation.

    Finite transverse dimensions stored on ``segments`` attenuate each photon
    to the first of the rectangular prism's six faces along the fixed far-field
    observation direction. With both dimensions omitted, the original z-only
    slab escape branches are retained unchanged.

    groove: optional GrooveSpec replacing the flat/prism escape length with the
    exact periodic working-facet distance from ``escape_distance_ang``. In the
    supported blazed geometry, ``n_hat = (cos(tp), 0, -sin(tp))`` crosses
    working facets outward and is parallel to relief facets, so the first
    crossing is the complete material path and cannot be followed by re-entry.
    This changes only the existing Beer--Lambert factor; emission cross sections
    and kinematics remain unchanged. Transport supplies material segments only,
    so vacuum legs do not radiate.

    Assumptions: straight photon rays, y-invariant and laterally periodic
    grooves, single-slab absorption, and the exact working-facet-normal
    observation direction. Layers and other directions raise rather than
    silently using flat attenuation. Source: exact periodic ray-plane
    intersections; see
    ``docs/superpowers/specs/2026-07-24-groove-aware-transport-design.md``.
    Limiting case: ``groove=None`` retains the original flat/prism path
    bit-for-bit; vanishing groove depth tends to the flat entrance-face path.

    Validation: brem-spectrum, finite-transverse-crystal, blazed-groove-geometry
    """
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
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
    mu = _mu_total_inv_ang(comp, E_grid)  # (NE,) [1/Ang], single-slab fallback
    # The Henke absorption tables span ~20 eV - 30 keV; outside that the wide
    # brem grid gets NaN (above 30 keV) or inf (at E=0), and a single bad bin
    # makes brem_wide -- and its integrated count rate -- NaN. Hard X-rays escape
    # essentially unattenuated, so treat an unavailable mu as zero (transparent).
    mu = xp.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    # layered (film-on-substrate) absorber: precompute each layer's mu(E_grid);
    # the per-segment z-path dz folds in inside the chunk loop. None -> single slab.
    if layers is not None:
        inv_nz = 1.0 / max(abs(float(n_hat[2])), 1e-12)
        layer_mu = [
            xp.nan_to_num(_mu_total_inv_ang(c, E_grid), nan=0.0, posinf=0.0, neginf=0.0)
            for (_, _, c) in layers
        ]

    seg_elec_id = xp.asarray(segments["elec_id"])
    brem_electron = seg_elec_id < Ne
    if E_cut_keV is not None:
        # Dual-use transport runs the shared electrons down to
        # min(E_cut_lines, E_cut_brem), so segments below THIS population's own
        # cutoff exist for the shared electrons but not for the brem-only ones.
        # Drop them, or the ensemble mixes two cutoffs. Approximate at the
        # boundary (the straddling segment is dropped whole rather than
        # truncated), exact whenever the two cutoffs coincide.
        brem_electron = brem_electron & (
            xp.asarray(segments["E_keV"], dtype=REAL) >= REAL(E_cut_keV)
        )

    seg_r = xp.asarray(segments["r_mid"], dtype=REAL)[brem_electron]
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)[brem_electron]
    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)[brem_electron]

    z_mid = seg_r[:, 2]
    finite_footprint = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )
    if groove is not None:
        L_esc = xp.asarray(
            escape_distance_ang(seg_r[:, 0], z_mid, groove),
            dtype=REAL,
        )
    elif finite_footprint:
        L_esc = _segment_escape_distance(segments, n_hat, xp=xp)[brem_electron]
    else:
        L_esc = _escape_length(z_mid, thickness, n_hat[2])

    spec = xp.zeros(E_grid.size, dtype=REAL)

    # GPU float32 fast path: evaluate Bethe-Heitler/Elwert, absorption, and the
    # segment reduction in one raw kernel. Instead of a dense T_abs[M, NE]
    # matrix, pass only each segment's path length through each absorber layer.
    _use_jit_brem_reduction = (
        _USE_JIT_BREM_REDUCTION
        and getattr(xp, "__name__", "") == "cupy"
        and np.dtype(REAL) == np.dtype(np.float32)
    )
    if _use_jit_brem_reduction:
        from .brem_jit_kernel import (
            DEFAULT_BREM_KERNEL_CONFIG,
            run_brem_reduction_kernel,
        )

        if layers is None:
            path_by_layer = L_esc[:, None]
            mu_by_layer = mu[None, :]
        else:
            path_cols = []
            if finite_footprint:
                for z_top, z_bot, _ in layers:
                    path_cols.append(
                        _layer_path_length(z_mid, n_hat[2], L_esc, float(z_top), float(z_bot))
                    )
            else:
                for z_top, z_bot, _ in layers:
                    dz = _layer_dz(z_mid, n_hat[2], float(z_top), float(z_bot))
                    path_cols.append(dz * inv_nz)
            path_by_layer = xp.stack(path_cols, axis=1)
            mu_by_layer = xp.stack(layer_mu, axis=0)

        path_by_layer = xp.ascontiguousarray(path_by_layer, dtype=REAL)
        mu_by_layer = xp.ascontiguousarray(mu_by_layer, dtype=REAL)
        path_flat = path_by_layer.reshape(-1)
        mu_flat = mu_by_layer.reshape(-1)
        T_jit = xp.ascontiguousarray(seg_E, dtype=REAL)
        L_jit = xp.ascontiguousarray(seg_L, dtype=REAL)
        E_jit = xp.ascontiguousarray(E_grid, dtype=REAL)
        n_abs_layers = int(path_by_layer.shape[1])

        for el_i, n_i in comp:
            Z_i = TRANSPORT_ELEMENTS[el_i]["Z"]
            run_brem_reduction_kernel(
                T_jit,
                L_jit,
                path_flat,
                mu_flat,
                E_jit,
                Z=Z_i,
                density_cm3=n_i * 1e24,
                n_layers=n_abs_layers,
                out=spec,
                config=DEFAULT_BREM_KERNEL_CONFIG,
            )
        return _to_cpu(spec / (4.0 * xp.pi) / Ne)

    M = seg_E.size
    for j0 in range(0, M, chunk):
        sl = slice(j0, min(j0 + chunk, M))
        if layers is None:
            T_abs = xp.exp(-L_esc[sl][:, None] * mu[None, :])
        elif finite_footprint:
            tau = 0.0
            for (z_top, z_bot, _), mu_i in zip(layers, layer_mu, strict=False):
                path = _layer_path_length(
                    z_mid[sl], n_hat[2], L_esc[sl], float(z_top), float(z_bot)
                )
                tau = tau + path[:, None] * mu_i[None, :]
            T_abs = xp.exp(-tau)
        else:
            tau = 0.0
            for (z_top, z_bot, _), mu_i in zip(layers, layer_mu, strict=False):
                dz = _layer_dz(z_mid[sl], n_hat[2], float(z_top), float(z_bot))
                tau = tau + (dz * inv_nz)[:, None] * mu_i[None, :]
            T_abs = xp.exp(-tau)
        path_cm = seg_L[sl] * 1e-8
        for el_i, n_i in comp:
            Z_i = TRANSPORT_ELEMENTS[el_i]["Z"]
            dsig = _brem_dsigma_dk(Z_i, seg_E[sl], E_grid)
            spec += (n_i * 1e24 * path_cm) @ (dsig * T_abs)
    return _to_cpu(spec / (4.0 * xp.pi) / Ne)


def load_external_brem(path, E_grid_eV):
    """
    Interpolate an EXTERNAL bremsstrahlung background onto the spectral grid
    -- e.g. a NIST DTSA-II simulation, which is what Zhai et al. use both
    for their simulated backgrounds (refs 96-100) and, with a PIXE-style
    numerical fit, for their experimental subtraction (SI S3).

    File format: two columns (energy [eV], intensity), whitespace- or
    comma-separated; '#' comment lines and non-numeric headers are skipped.
    The intensity must already be in DETECTED units matching your plots
    (e.g. Phs/eV/s/nA: from a DTSA-II counts export, divide counts/channel
    by channel width [eV] x live time [s] x beam current [nA]). It is
    treated as an as-detected spectrum: window efficiency and detector
    resolution are NOT re-applied. Energies outside the file's range
    interpolate to zero.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2 or parts[0].startswith(("#", "//")):
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # header / text line
    if not rows:
        raise ValueError(f"no numeric (E, intensity) rows found in {path}")
    arr = np.array(sorted(rows))
    return np.interp(np.asarray(E_grid_eV, dtype=float), arr[:, 0], arr[:, 1], left=0.0, right=0.0)
