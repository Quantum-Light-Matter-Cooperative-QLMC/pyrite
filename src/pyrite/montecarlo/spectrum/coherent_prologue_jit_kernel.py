"""Prologue stage of the streaming CUDA coherent CXR pipeline.

``run_coherent_prologue_kernel`` evaluates the energy-independent line data
once per ``(g, segment)`` pair (stage 1 of
:mod:`pyrite.montecarlo.spectrum.coherent_stream_jit_kernel`, split out of
that module for its line budget): the first-order Snell root with exact
recoil, the escape/absorption constants, polarization amplitudes and the
recoil age phase. Outputs use g-major fixed order; rejected pairs keep zero
field coefficients. Exact recoil makes ``aw`` and ``phase_slope`` per
(segment, g) (``line-recoil-in-medium-root``).
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from .coherent_stream_jit_kernel import (
    DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
    F32_FORMATION_SMALL_Q,
    F32_HALF,
    F32_MAX,
    F32_ONE,
    F32_QUARTER,
    F32_TEN,
    F32_TWO,
    F32_ZERO,
    U32_ONE,
    U32_THREE,
    U32_TWO,
    U32_ZERO,
    _dummy,
    _validate_threads,
)


@jit.rawkernel(device=True)
def _interp_row(table, row, upper_idx, fraction, below, above, n_samples):
    row_offset = row * n_samples
    if below:
        return table[row_offset]
    if above:
        return table[row_offset + n_samples - U32_ONE]
    lower_value = table[row_offset + upper_idx - U32_ONE]
    return lower_value + fraction * (table[row_offset + upper_idx] - lower_value)


@jit.rawkernel(device=True)
def _interp_shared(table, upper_idx, fraction, below, above, n_samples):
    if below:
        return table[U32_ZERO]
    if above:
        return table[n_samples - U32_ONE]
    lower_value = table[upper_idx - U32_ONE]
    return lower_value + fraction * (table[upper_idx] - lower_value)


@jit.rawkernel(device=True)
def _interp_elemental_mu(
    log_mu_table, upper_idx, log_fraction, below, above, n_elements, n_samples
):
    """Sum per-element log-log attenuation interpolation on device.

    Validation: line-absorption-tabulation
    """
    mu = F32_ZERO
    element = U32_ZERO
    while element < n_elements:
        row_offset = element * n_samples
        if below:
            log_mu = log_mu_table[row_offset]
        elif above:
            log_mu = log_mu_table[row_offset + n_samples - U32_ONE]
        else:
            lower_log_mu = log_mu_table[row_offset + upper_idx - U32_ONE]
            log_mu = lower_log_mu + log_fraction * (
                log_mu_table[row_offset + upper_idx] - lower_log_mu
            )
        mu += xp.exp(log_mu)
        element += U32_ONE
    return mu


@jit.rawkernel(device=True)
def _bracket_index(grid, energy_eV, n_samples):
    """Equivalent to ``clip(searchsorted(grid, energy_eV), 1, n_samples - 1)``."""
    lower_idx = U32_ZERO
    upper_idx = n_samples
    while lower_idx < upper_idx:
        mid_idx = (lower_idx + upper_idx) // U32_TWO
        if grid[mid_idx] < energy_eV:
            lower_idx = mid_idx + U32_ONE
        else:
            upper_idx = mid_idx
    if lower_idx < U32_ONE:
        return U32_ONE
    if lower_idx >= n_samples:
        return n_samples - U32_ONE
    return lower_idx


@jit.rawkernel()
def _coherent_prologue_kernel(
    v_flat,
    denom,
    grad_flat,
    gamma,
    t_L,
    L_esc,
    L_start,
    L_end,
    cxr_electron,
    r_flat,
    g_flat,
    es_flat,
    ep_flat,
    g2,
    n_dot_g,
    g_dot_es,
    g_dot_ep,
    E_tab,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    log_mu_tab,
    table_row,
    delta_tab,
    inverse_energy,
    age,
    centre_ang,
    E_r_out,
    aw_out,
    phase_slope_out,
    g_phase_out,
    cs_re_out,
    cs_im_out,
    cp_re_out,
    cp_im_out,
    apb_out,
    bma_out,
    q_out,
    lo_keep,
    hi_keep,
    hbarc,
    root_rtol,
    alpha_fs,
    pref_c1,
    use_medium,
    use_formation,
    pair_count,
    n_segments,
    n_reflections,
    n_elements,
    n_samples,
):
    # g-major fixed order: pair_index = reflection*n_segments + segment.  The reduction kernel then
    # reads each g row contiguously while its threads stride over segments.
    """Validation: xray-in-medium-resonance, line-kinematics-precision,
    line-recoil-in-medium-root"""
    pair_index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if pair_index >= pair_count:
        return

    reflection = pair_index // n_segments
    segment = pair_index - reflection * n_segments

    # A zero coefficient is a complete rejected-pair marker. Geometry buffers
    # may remain untouched on rejection because the reducer reads coefficients
    # first and only loads E/phase data for a live field contribution.
    cs_re_out[pair_index] = F32_ZERO
    cs_im_out[pair_index] = F32_ZERO
    cp_re_out[pair_index] = F32_ZERO
    cp_im_out[pair_index] = F32_ZERO

    if not cxr_electron[segment]:
        return

    velocity_offset = segment * U32_THREE
    reflection_offset = reflection * U32_THREE
    vx = v_flat[velocity_offset]
    vy = v_flat[velocity_offset + U32_ONE]
    vz = v_flat[velocity_offset + U32_TWO]
    gx = g_flat[reflection_offset]
    gy = g_flat[reflection_offset + U32_ONE]
    gz = g_flat[reflection_offset + U32_TWO]

    # Exact recoil (CPU twin ``_line_recoil_terms``): v.g -> v.g - r_num and
    # D -> D + r_den, with inverse_energy = hbar c / E [Angstrom]; zero is the
    # classical hbar -> 0 limit. Validation: line-recoil-in-medium-root
    inv_energy = inverse_energy[segment]
    recoil_num = F32_HALF * g2[reflection] * inv_energy
    recoil_den = n_dot_g[reflection] * inv_energy
    resonance_denom = denom[segment] + recoil_den
    vacuum_denom = resonance_denom
    v_dot_g = vx * gx + vy * gy + vz * gz
    line_numerator = v_dot_g - recoil_num

    # First-order Snell fixed point: D = (1-v.n) - delta v.grad L on the
    # cancellation-free vacuum factor and decrement table (see the CPU twin).
    # Retain the CPU twin's three passes and float32 convergence guard;
    # reject critical-angle gradients before producing any amplitude.
    # Validation: xray-in-medium-resonance, line-kinematics-precision
    delta = F32_ZERO
    grad_x = grad_flat[velocity_offset]
    grad_y = grad_flat[velocity_offset + U32_ONE]
    grad_z = grad_flat[velocity_offset + U32_TWO]
    gradient_norm_sq = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z
    velocity_dot_grad = vx * grad_x + vy * grad_y + vz * grad_z
    if use_medium:
        previous_denominator = resonance_denom
        for _it in range(3):
            trial_energy_eV = hbarc * (line_numerator / resonance_denom)
            trial_below = trial_energy_eV <= E_tab[U32_ZERO]
            trial_above = trial_energy_eV >= E_tab[n_samples - U32_ONE]
            trial_upper = _bracket_index(E_tab, trial_energy_eV, n_samples)
            trial_lower_eV = E_tab[trial_upper - U32_ONE]
            trial_fraction = (trial_energy_eV - trial_lower_eV) / (
                E_tab[trial_upper] - trial_lower_eV
            )
            delta = _interp_shared(
                delta_tab, trial_upper, trial_fraction, trial_below, trial_above, n_samples
            )
            previous_denominator = resonance_denom
            resonance_denom = vacuum_denom - delta * velocity_dot_grad
        # `abs` is not a cupyx.jit builtin; `xp.abs` is. Written as `>` rather
        # than `not (<=)` so a NaN root falls through to the existing NaN
        # handling below instead of taking a new exit.
        if (
            xp.abs(resonance_denom - previous_denominator) > root_rtol * xp.abs(resonance_denom)
            or F32_TWO * xp.abs(delta) * gradient_norm_sq >= F32_ONE
        ):
            return

    omega = line_numerator / resonance_denom
    resonance_eV = hbarc * omega
    if resonance_eV <= lo_keep or resonance_eV <= F32_TEN or resonance_eV >= hi_keep:
        return
    # Positive-mass-shell guard (``_recoil_line_allowed``): D > 0 and hbar
    # omega <= T, i.e. omega hbar c/E <= 1 - 1/gamma.
    recoil_fraction = omega * inv_energy
    if not (resonance_denom > F32_ZERO and recoil_fraction <= F32_ONE - F32_ONE / gamma[segment]):
        return

    below = resonance_eV <= E_tab[U32_ZERO]
    above = resonance_eV >= E_tab[n_samples - U32_ONE]
    upper_idx = _bracket_index(E_tab, resonance_eV, n_samples)
    lower_eV = E_tab[upper_idx - U32_ONE]
    fraction = (resonance_eV - lower_eV) / (E_tab[upper_idx] - lower_eV)
    bounded_energy = min(max(resonance_eV, lower_eV), E_tab[upper_idx])
    log_fraction = xp.log1p((bounded_energy - lower_eV) / lower_eV) / xp.log1p(
        (E_tab[upper_idx] - lower_eV) / lower_eV
    )

    row = table_row[reflection]
    chi_re = _interp_row(chi_re_tab, row, upper_idx, fraction, below, above, n_samples)
    chi_im = _interp_row(chi_im_tab, row, upper_idx, fraction, below, above, n_samples)
    u_re = _interp_row(u_re_tab, row, upper_idx, fraction, below, above, n_samples)
    u_im = _interp_row(u_im_tab, row, upper_idx, fraction, below, above, n_samples)
    mu = _interp_elemental_mu(
        log_mu_tab, upper_idx, log_fraction, below, above, n_elements, n_samples
    )

    # Scalar products of k_eff = omega(n_hat + delta grad L).
    k_mag = omega
    if use_medium:
        if gradient_norm_sq != F32_ZERO:
            k_mag = omega * xp.sqrt(F32_ONE - F32_TWO * delta + delta * delta * gradient_norm_sq)
    k_dot_v = omega * ((F32_ONE - resonance_denom) + recoil_den)
    k_dot_g = omega * (n_dot_g[reflection] + delta * (grad_x * gx + grad_y * gy + grad_z * gz))
    v_dot_kg = v_dot_g + k_dot_v
    detuning = g2[reflection] + F32_TWO * k_dot_g

    duration = t_L[segment]
    lorentz_factor = gamma[segment]
    # Formation mode (issue #181): the attenuation moves out of the amplitude
    # into the per-piece formation constants of ``_formation.
    # formation_coefficients`` -- tau at the piece ends, ordered along travel --
    # which the field kernels turn into the complex factor F.
    # Validation: coherent-formation-absorption
    transmission = F32_ONE
    end_sum = F32_ZERO
    end_diff = F32_ZERO
    absorb_q = F32_ZERO
    if use_formation:
        start_optical_depth = L_start[segment] * mu
        end_optical_depth = L_end[segment] * mu
        absorb_q = F32_QUARTER * (end_optical_depth - start_optical_depth)
        if xp.abs(absorb_q) < F32_FORMATION_SMALL_Q:
            mid_attenuation = xp.exp(-(F32_QUARTER * (start_optical_depth + end_optical_depth)))
            end_sum = F32_TWO * mid_attenuation * xp.cosh(absorb_q)
            end_diff = -(F32_TWO * mid_attenuation * xp.sinh(absorb_q))
        else:
            start_amplitude = xp.exp(-(F32_HALF * start_optical_depth))
            end_amplitude = xp.exp(-(F32_HALF * end_optical_depth))
            end_sum = start_amplitude + end_amplitude
            end_diff = end_amplitude - start_amplitude
        # Written so a NaN falls through to rejection, like the root guard.
        if not (end_sum > F32_ZERO and end_sum < F32_MAX and end_diff == end_diff):
            return
    else:
        transmission = xp.exp(-(L_esc[segment] * mu))
    amplitude = xp.sqrt(alpha_fs * omega / pref_c1 * transmission)
    if (
        amplitude != amplitude
        or amplitude <= F32_ZERO
        or amplitude >= F32_MAX
        or duration <= F32_ZERO
    ):
        return
    field_scale = amplitude * duration

    esx = es_flat[reflection_offset]
    esy = es_flat[reflection_offset + U32_ONE]
    esz = es_flat[reflection_offset + U32_TWO]
    epx = ep_flat[reflection_offset]
    epy = ep_flat[reflection_offset + U32_ONE]
    epz = ep_flat[reflection_offset + U32_TWO]

    # sigma polarization: A = chi*f_pxr + (U/m)*f_cbs, with both scalar
    # kinematic factors REAL. This is the real/imag expansion of the existing
    # coherent complex expression, preserving its phase information.
    g_dot_e = g_dot_es[reflection]
    v_dot_e = vx * esx + vy * esy + vz * esz
    k_dot_e = omega * delta * (grad_x * esx + grad_y * esy + grad_z * esz)
    numerator = v_dot_kg * (g_dot_e + k_dot_e) - k_mag * k_mag * v_dot_e
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    bracket = braced_ge + v_dot_e * braced_kg / v_dot_g
    cbs_denominator = lorentz_factor * v_dot_g
    # Preserve the coherent eager expression order as closely as scalar real/imag
    # arithmetic allows: (chi/detuning)*numerator and (-u/cbs_denominator)*bracket.
    amplitude_real = (chi_re / detuning) * numerator + (-u_re / cbs_denominator) * bracket
    amplitude_imag = (chi_im / detuning) * numerator + (-u_im / cbs_denominator) * bracket
    sigma_re = field_scale * amplitude_real
    sigma_im = field_scale * amplitude_imag

    # pi polarization.
    g_dot_e = g_dot_ep[reflection]
    v_dot_e = vx * epx + vy * epy + vz * epz
    k_dot_e = omega * delta * (grad_x * epx + grad_y * epy + grad_z * epz)
    numerator = v_dot_kg * (g_dot_e + k_dot_e) - k_mag * k_mag * v_dot_e
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    bracket = braced_ge + v_dot_e * braced_kg / v_dot_g
    amplitude_real = (chi_re / detuning) * numerator + (-u_re / cbs_denominator) * bracket
    amplitude_imag = (chi_im / detuning) * numerator + (-u_im / cbs_denominator) * bracket
    pi_re = field_scale * amplitude_real
    pi_im = field_scale * amplitude_imag

    E_r_out[pair_index] = resonance_eV
    # The sinc half-width rides on the resonance denominator, so under the
    # refractive model it stops being a per-segment hoist and is emitted here in
    # the pair layout the field reducer already understands.
    # The exact slope kappa D (``_recoil_slope_factor``) carries the width,
    # so it is per (segment, g) even in vacuum.
    aw_out[pair_index] = (resonance_denom * (F32_ONE / (F32_ONE - recoil_fraction)) * duration) / (
        F32_TWO * hbarc
    )
    # Formation mode centres the line on the VACUUM sinc centre and width; the
    # field kernels add the escape-path refractive slope (``_formation``).
    # CPU twin of the exact centre, width and age phase: ``_recoil_formation_line``.
    vacuum_omega = line_numerator / vacuum_denom
    vacuum_fraction = vacuum_omega * inv_energy
    vacuum_kappa = F32_ONE / (F32_ONE - vacuum_fraction)
    if use_formation:
        # Same rounding order as the CPU twin: (hbar c N) / D.
        E_r_out[pair_index] = (hbarc * line_numerator) / vacuum_denom
        aw_out[pair_index] = (vacuum_kappa * vacuum_denom) * duration / (F32_TWO * hbarc)
        apb_out[pair_index] = end_sum
        bma_out[pair_index] = end_diff
        q_out[pair_index] = absorb_q
    segment_age = age[segment]
    d_shift = vacuum_kappa * (vacuum_fraction * denom[segment] + recoil_den) * segment_age
    g_shift = vacuum_kappa * (vacuum_fraction * v_dot_g - recoil_num) * segment_age
    # (d + d_shift) / hbar c, the CPU twin's rounding order.
    phase_slope_out[pair_index] = (centre_ang[segment] + d_shift) / hbarc
    position_offset = segment * U32_THREE
    g_phase_out[pair_index] = (
        r_flat[position_offset] * gx
        + r_flat[position_offset + U32_ONE] * gy
        + r_flat[position_offset + U32_TWO] * gz
    ) + g_shift
    cs_re_out[pair_index] = sigma_re
    cs_im_out[pair_index] = sigma_im
    cp_re_out[pair_index] = pi_re
    cp_im_out[pair_index] = pi_im


def _recoil_args(inverse_energy, age, centre_ang, default_centre, n_segments):
    """Per-segment float32 ``hbar c/E``, age and centre ``d``.

    An omitted ``hbar c/E`` or age is zero; an omitted centre is
    ``default_centre``.
    """
    if centre_ang is None:
        centre_ang = default_centre
    out = []
    named = (("inverse_energy", inverse_energy), ("age", age), ("centre_ang", centre_ang))
    for name, values in named:
        if values is None:
            values = xp.zeros(n_segments, dtype=xp.float32)
        values = xp.ascontiguousarray(values, dtype=xp.float32)
        if int(values.size) != n_segments:
            raise ValueError(f"{name} must have one entry per segment")
        out.append(values)
    return tuple(out)


def run_coherent_prologue_kernel(
    v_flat,
    denom,
    gamma,
    t_L,
    L_esc,
    cxr_electron,
    r_flat,
    phase_slope_seg,
    g_flat,
    es_flat,
    ep_flat,
    E_tab,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    log_mu_tab,
    *,
    lo_keep,
    hi_keep,
    hbarc,
    root_rtol,
    electron_mass_eV,
    alpha_fs,
    pref_c1,
    n_hat,
    n_g,
    g2=None,
    n_dot_g=None,
    g_dot_es=None,
    g_dot_ep=None,
    aw_seg=None,
    delta_tab=None,
    escape_gradient=None,
    L_start=None,
    L_end=None,
    table_row=None,
    inverse_energy=None,
    age=None,
    centre_ang=None,
    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
):
    """Build g-major coherent line data for one contiguous segment block.

    The public/internal call shape is retained for compatibility. The returned
    tuple also retains its historical eight entries; ``aw`` and ``phase_slope``
    are in the pair layout. The field reducer accepts both layouts, so callers
    that construct segment-sized geometry directly remain valid.

    ``g2``/``n_dot_g``/``g_dot_e*`` are optional precomputed hoists. When
    omitted they are constructed once here. ``aw_seg`` is accepted for call
    compatibility but ignored: the exact-recoil half-width is per pair.

    Under the in-medium dispersion the caller supplies ``delta_tab``, the
    decrement ``1 - Re n`` on the ``E_tab`` grid. The kernel then solves the
    implicit in-medium resonance per (segment, g) from the hoisted vacuum
    ``denom = 1 - v.n_hat`` instead of using that denominator directly, and
    returns ``aw`` in the pair layout because the sinc half-width follows the
    in-medium root. Omitted, the kernel evaluates the vacuum kinematics
    unchanged.

    ``root_rtol`` is the CPU twin's ``_in_medium_kinematics._RESONANCE_ROOT_RTOL``,
    passed in for the same reason ``hbarc`` is: this module cannot import the
    line core. It bounds how far the last fixed-point pass may move ``denom``
    before the pair is rejected as non-converged.

    Formation mode (issue #181) takes the per-segment escape distances
    ``L_start``/``L_end`` at the piece ends, ordered along travel, together and
    only under the refractive model. The coefficients then carry no
    transmission, ``resonances_eV``/``aw`` are the vacuum sinc centre and width, and the
    tuple gains three pair-layout entries ``(apb, bma, q)`` of
    ``_formation.formation_coefficients`` for the field kernels.
    Validation: coherent-formation-absorption

    ``table_row[g]`` names the coupling-table row g row reads, so mosaic
    orientations share their reflection's tables; omitted, row ``g`` reads
    table row ``g``.
    Validation: line-reflection-coupling-tables

    Exact recoil takes the per-segment ``inverse_energy = hbar c / E``
    [Angstrom] and the electron's own midpoint ``age`` [Angstrom]; omitted,
    both are zero (the classical root). The root, width, and the recoil
    phase over the age then follow ``cxr._recoil._recoil_formation_line``,
    so ``phase_slope`` is returned in the pair layout as ``(d + d_shift) /
    hbar c``. ``centre_ang`` is the per-segment ``d`` [Angstrom]; omitted, it
    is ``phase_slope_seg * hbarc``.
    Validation: line-recoil-in-medium-root
    """
    n_threads = int(config.prologue_nthreads)
    _validate_threads(n_threads, "prologue_nthreads")
    n_g = int(n_g)
    n_tab = int(E_tab.size)
    if log_mu_tab.ndim != 2 or int(log_mu_tab.shape[1]) != n_tab:
        raise ValueError("log_mu_tab must have shape (elements, E_tab.size)")
    n_mu = int(log_mu_tab.shape[0])
    log_mu_kernel = log_mu_tab.reshape(-1)
    n_segments = int(denom.size)
    if n_g <= 0 or n_tab < 2:
        raise ValueError("coherent prologue requires at least one g row and two tabulation points")
    n_pairs = n_g * n_segments
    if table_row is None:
        table_row = xp.arange(n_g, dtype=xp.uint32)
    else:
        table_row = xp.ascontiguousarray(table_row, dtype=xp.uint32)
        if int(table_row.size) != n_g:
            raise ValueError("table_row must have one entry per g row")

    use_medium = delta_tab is not None
    if use_medium:
        if int(delta_tab.size) != n_tab:
            raise ValueError("delta_tab must use the E_tab grid")
        if aw_seg is not None:
            raise ValueError("aw_seg cannot be hoisted per segment under the refractive model")
    if (L_start is None) != (L_end is None):
        raise ValueError("L_start and L_end must be given together")
    use_formation = L_start is not None
    if use_formation:
        if not use_medium:
            raise ValueError("the formation factor needs the refractive model (delta_tab)")
        if int(L_start.size) != n_segments or int(L_end.size) != n_segments:
            raise ValueError("L_start and L_end must have one entry per segment")

    G = g_flat.reshape(n_g, 3)
    ES = es_flat.reshape(n_g, 3)
    EP = ep_flat.reshape(n_g, 3)
    if g2 is None:
        g2 = G[:, 0] * G[:, 0] + G[:, 1] * G[:, 1] + G[:, 2] * G[:, 2]
    if n_dot_g is None:
        nx, ny, nz = np.float32(n_hat[0]), np.float32(n_hat[1]), np.float32(n_hat[2])
        n_dot_g = G[:, 0] * nx + G[:, 1] * ny + G[:, 2] * nz
    if g_dot_es is None:
        g_dot_es = G[:, 0] * ES[:, 0] + G[:, 1] * ES[:, 1] + G[:, 2] * ES[:, 2]
    if g_dot_ep is None:
        g_dot_ep = G[:, 0] * EP[:, 0] + G[:, 1] * EP[:, 1] + G[:, 2] * EP[:, 2]

    # Backward-compatible direct callers may still supply raw U_g tables. The
    # optimized main path supplies pre-scaled U_g/m_e tables and sets mass=1.
    if float(electron_mass_eV) != 1.0:
        inv_m = np.float32(1.0 / float(electron_mass_eV))
        u_re_kernel = u_re_tab * inv_m
        u_im_kernel = u_im_tab * inv_m
    else:
        u_re_kernel = u_re_tab
        u_im_kernel = u_im_tab

    # Only g-dependent quantities live in pair scratch; under the refractive
    # model the half-width joins them.
    n_slots = 8 + (3 if use_formation else 0)
    storage = xp.empty(n_slots * n_pairs, dtype=xp.float32)
    phase_slope_pair = storage[(n_slots - 1) * n_pairs : n_slots * n_pairs]
    resonances_eV = storage[0 * n_pairs : 1 * n_pairs]
    g_phase = storage[1 * n_pairs : 2 * n_pairs]
    sigma_coeffs_re = storage[2 * n_pairs : 3 * n_pairs]
    sigma_coeffs_im = storage[3 * n_pairs : 4 * n_pairs]
    pi_coeffs_re = storage[4 * n_pairs : 5 * n_pairs]
    pi_coeffs_im = storage[5 * n_pairs : 6 * n_pairs]
    # Exact recoil puts kappa on the half-width, so it is never hoisted.
    aw_seg = storage[6 * n_pairs : 7 * n_pairs]
    formation = ()
    if use_formation:
        formation = tuple(storage[i * n_pairs : (i + 1) * n_pairs] for i in (7, 8, 9))
    line_data = (
        resonances_eV,
        aw_seg,
        phase_slope_pair,
        g_phase,
        sigma_coeffs_re,
        sigma_coeffs_im,
        pi_coeffs_re,
        pi_coeffs_im,
        *formation,
    )
    if n_pairs == 0:
        return line_data

    if not use_medium:
        delta_tab = _dummy()
    inverse_energy, age, centre_ang = _recoil_args(
        inverse_energy, age, centre_ang, phase_slope_seg * np.float32(hbarc), n_segments
    )
    if use_formation:
        apb_out, bma_out, q_out = formation
    else:
        L_start = L_end = apb_out = bma_out = q_out = _dummy()

    if escape_gradient is None:
        # Compatibility for direct callers without escape geometry.
        escape_gradient = xp.broadcast_to(-xp.asarray(n_hat, dtype=xp.float32), (n_segments, 3))
    grad_flat = xp.ascontiguousarray(escape_gradient, dtype=xp.float32).reshape(-1)
    if int(grad_flat.size) != 3 * n_segments:
        raise ValueError("escape_gradient must have shape (n_seg, 3)")
    n_blocks = (n_pairs + n_threads - 1) // n_threads
    _coherent_prologue_kernel(
        (n_blocks,),
        (n_threads,),
        (
            v_flat,
            denom,
            grad_flat,
            gamma,
            t_L,
            L_esc,
            L_start,
            L_end,
            cxr_electron,
            r_flat,
            g_flat,
            es_flat,
            ep_flat,
            g2,
            n_dot_g,
            g_dot_es,
            g_dot_ep,
            E_tab,
            chi_re_tab,
            chi_im_tab,
            u_re_kernel,
            u_im_kernel,
            log_mu_kernel,
            table_row,
            delta_tab,
            inverse_energy,
            age,
            centre_ang,
            resonances_eV,
            aw_seg,
            phase_slope_pair,
            g_phase,
            sigma_coeffs_re,
            sigma_coeffs_im,
            pi_coeffs_re,
            pi_coeffs_im,
            apb_out,
            bma_out,
            q_out,
            np.float32(lo_keep),
            np.float32(hi_keep),
            np.float32(hbarc),
            np.float32(root_rtol),
            np.float32(alpha_fs),
            np.float32(pref_c1),
            np.uint32(1 if use_medium else 0),
            np.uint32(1 if use_formation else 0),
            np.uint32(n_pairs),
            np.uint32(n_segments),
            np.uint32(n_g),
            np.uint32(n_mu),
            np.uint32(n_tab),
        ),
    )
    return line_data
