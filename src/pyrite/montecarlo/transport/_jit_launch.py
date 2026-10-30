"""Host-side launchers for the CUDA transport kernels.

Flattening, dtype narrowing, and scalar typing happen here rather than inside
the kernels in :mod:`pyrite.montecarlo.transport._jit_kernel`, so the launcher
signatures can track their CPU reference cores positionally.

Like its siblings, this module imports ``cupy`` at module scope, so it must
stay out of the package ``__init__``; ``api.py`` and ``batching.py`` reach it
only through deferred, function-local imports.

Validation: sbethe-corrected-stopping
Validation: shell-secondary-transport
Validation: shell-soft-hard-transport
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np

from ._jit_kernel_variants import (
    TransportKernelSwitches,
    monolithic_transport_kernel,
    transport_kernel_variant,
)
from ._jit_lut_kernel import _transport_lut_kernel


@dataclass(frozen=True)
class TransportKernelConfig:
    """Launch geometry. Does not affect results -- output slots are addressed by
    electron index, not by thread or block index."""

    nthreads: int = 128


DEFAULT_TRANSPORT_KERNEL_CONFIG = TransportKernelConfig()

# Launch the exact kernel compiled for the case's feature switches (#827). Off,
# every launch uses the runtime-switched monolithic kernel; results are bitwise
# identical either way, so this is a bisection and benchmarking switch only.
_SPECIALIZE_TRANSPORT_KERNEL = True


def run_transport_lut_kernel(
    run,
    control,
    geometry,
    lut,
    materials,
    state,
    segments,
    pe_out,
    straggling,
    config=DEFAULT_TRANSPORT_KERNEL_CONFIG,
):
    """Launch the energy-LUT transport kernel.

    Signature matches the CPU ``_transport_core_ungrooved_perelectron_lut``
    positionally. ``materials`` is the padded per-element
    ``(L_Js, L_Zs, L_ks, L_coeffs, L_E_cross)`` tables plus the SBETHE group the
    Urban sampler reads; without straggling they are one-row dummies the kernel
    never touches. ``straggling`` is ``(straggle_on, stragg_dE)``.
    """
    (e_start, e_count, cap, stream_key) = run
    (max_steps, _max_segments, elastic_model_code, energy_model_code, max_dE_frac) = control
    (
        n_layers,
        internal_bounds,
        z_total,
        finite_footprint,
        width_ang,
        height_ang,
        L_nel,
        L_top,
        L_bot,
    ) = geometry
    (
        lut_log_E_min,
        lut_inv_dlogE,
        lut_n_energy,
        lut_total_rate,
        lut_dEds,
        lut_inv_beta,
        lut_cdf,
        lut_alpha,
    ) = lut
    (L_Js, L_Zs, L_ks, L_coeffs, L_E_cross) = materials[:5]
    sbethe_on = bool(materials[5])
    (alive, clock, pos, dirs, E_keV, E_cut_by_electrons) = state
    (
        seg_dir,
        seg_mid,
        seg_len,
        seg_E,
        seg_t0,
        seg_id,
        seg_lay,
        seg_E_end,
        seg_t_end,
        seg_flight,
        seg_substep,
        seg_event,
    ) = segments
    (seg_count, exit_code) = pe_out
    (straggle_on, stragg_dE) = straggling
    n_threads = int(config.nthreads)
    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    e_count = int(e_count)
    if e_count == 0:
        return

    max_el = int(lut_cdf.shape[1])
    n_blocks = (e_count + n_threads - 1) // n_threads
    _transport_lut_kernel(
        (n_blocks,),
        (n_threads,),
        (
            np.int32(e_start),
            np.int32(e_count),
            np.int32(cap),
            stream_key,
            alive.astype(xp.uint8, copy=False),
            np.int32(max_steps),
            np.int32(n_layers),
            internal_bounds,
            np.int32(elastic_model_code),
            np.int32(energy_model_code),
            np.float64(max_dE_frac),
            np.float64(z_total),
            np.int32(1 if finite_footprint else 0),
            np.float64(width_ang),
            np.float64(height_ang),
            clock,
            pos.reshape(-1),
            dirs.reshape(-1),
            E_cut_by_electrons,
            L_nel.astype(xp.int32, copy=False),
            np.int32(max_el),
            L_top,
            L_bot,
            np.float64(lut_log_E_min),
            np.float64(lut_inv_dlogE),
            np.int32(lut_n_energy),
            lut_total_rate.reshape(-1),
            lut_dEds.reshape(-1),
            lut_inv_beta,
            lut_cdf.reshape(-1),
            lut_alpha.reshape(-1),
            E_keV,
            seg_dir.reshape(-1),
            seg_mid.reshape(-1),
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            seg_event,
            seg_count,
            exit_code,
            np.int32(1 if straggle_on else 0),
            stragg_dE,
            L_Js.reshape(-1),
            L_Zs.reshape(-1),
            L_ks.reshape(-1),
            L_coeffs.reshape(-1),
            L_E_cross.reshape(-1),
            np.int32(L_Zs.shape[1]),
            np.int32(1 if sbethe_on else 0),
        ),
    )


def make_cuda_transport_lut_core(config=DEFAULT_TRANSPORT_KERNEL_CONFIG):
    """Return the LUT CUDA core and CuPy array module for the shared driver."""

    def core(*args):
        run_transport_lut_kernel(*args, config=config)

    return core, xp


def run_transport_kernel(
    run,
    control,
    geometry,
    materials,
    mott,
    state,
    segments,
    pe_out,
    straggling,
    inelastic_args=None,
    radiative_args=None,
    config=DEFAULT_TRANSPORT_KERNEL_CONFIG,
    *,
    continuation=None,
):
    """Launch one thread per electron over ``[e_start, e_start + e_count)``.

    Signature matches :func:`_transport_core_ungrooved_perelectron` positionally
    so ``_run_per_electron_transport`` can drive either. Flattening, dtype
    narrowing, and scalar typing happen here rather than in the kernel.

    ``straggle_on``/``stragg_dE`` are the straggling gate and Ne-sized
    per-electron accumulator (see the reference core's docstring). As of slice F
    the sampled loss is *applied* here, not merely accumulated, under slice E's
    crossing rule. CUDA parity is not bit-for-bit: ``_urban_poisson_scalar``'s
    host counterpart branches on a ``log``/``exp`` comparison that a last-bit
    libm difference can move across a Poisson CDF boundary, so the claim is
    few-ulp per flight, per slice C's own recommendation. Applying the loss
    makes that a *divergence* claim rather than a per-row one: a flipped
    Poisson count changes the electron's energy and the trajectory diverges
    from there, so only the first row of each electron is comparable to the
    host, exactly as ``test_cuda_first_step_agrees_with_the_cpu_reference``
    already asserts for the deterministic path.

    Hardware re-validation passed 5/5 straggling tests on an NVIDIA GeForce RTX
    5080 (driver 610.47, CuPy 14.1.1). Those tests cover disabled-path identity,
    replay, energy bookkeeping, first-row entering state, and ensemble
    agreement. Direct parity of the first applied loss remains an anchor gap:
    the current first-row test compares row-start ``E_keV``, not ``E_end_keV``.

    ``inelastic_args`` is the shell soft/hard ``ShellInelasticTables.core_args``
    tuple the CPU inelastic cores unpack; ``segments`` then carries the three
    hard row columns (the last, secondary directions, empty when off) after
    its twelve midpoint fields. An empty tuple, which
    the batch driver passes ahead of ``radiative_args``, also means off.

    ``radiative_args`` is the coupled radiative ``(keys, cutoff_eV, *packed)``
    tuple the CPU radiative cores unpack, ``packed`` being
    :func:`.hard_radiative.pack_radiative_layer_tables`; its two photon row
    columns follow any hard-inelastic ones in ``segments``.

    ``continuation`` (#789) is ``(slots, state_i32, state_u64, state_f64)``
    from :mod:`.continuation`; ``None`` keeps run-to-completion. Thread ``i``
    then owns history ``slots[i]``, not ``e_start + i``, and ``cap`` is this
    launch's row quantum, not a track capacity. After ``cap`` rows a running
    history yields at the top of its step loop, where its only loop-carried
    values are ``(step, flight_id, substep_id)`` (int32), the elastic, hard
    and radiative stream counters (uint64) and open optical depths (float64),
    three per history each; position, direction, energy, clock and the
    straggling accumulator are already global. It stores them and reports
    ``I8_SUSPENDED``. A history with nonzero ``step`` resumes from them, so its
    arithmetic, draws and rows are those of one uninterrupted run. Rows land
    at ``i * cap`` plus their index within the launch, and ``seg_count``
    counts this launch's rows only.

    Validation: gpu-transport-core
    """
    (e_start, e_count, cap, stream_key) = run
    (max_steps, _max_segments, elastic_model_code, energy_model_code, max_dE_frac) = control
    (
        n_layers,
        internal_bounds,
        z_total,
        finite_footprint,
        width_ang,
        height_ang,
        L_nel,
        L_top,
        L_bot,
    ) = geometry
    (
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_E_cross,
        L_ncm3,
        L_sr_rate_numer,
        L_mott_numer,
        L_mott_denom1,
        L_mott_denom2,
        L_sr_joy_numer,
        sbethe_on,
        L_sbethe_n,
        L_sbethe_logE,
        L_sbethe_logS,
    ) = materials
    (mott_has_table, mott_start, mott_len, mott_logE_flat, mott_logA_flat) = mott[:5]
    (_, el_start, el_len, el_logE, el_log_rate, el_cdf, el_pdf, el_mu, el_guide) = mott[5:]
    (alive, clock, pos, dirs, E_keV, E_cut_by_electrons) = state
    (
        seg_dir,
        seg_mid,
        seg_len,
        seg_E,
        seg_t0,
        seg_id,
        seg_lay,
        seg_E_end,
        seg_t_end,
        seg_flight,
        seg_substep,
        seg_event,
    ) = segments[:12]
    (seg_count, exit_code) = pe_out
    (straggle_on, stragg_dE) = straggling
    inelastic_on = bool(inelastic_args)
    shell_arguments = _shell_kernel_args(inelastic_args if inelastic_on else None, segments[12:15])
    radiative_segments = segments[15:17] if inelastic_on else segments[12:14]
    radiative_arguments = _radiative_kernel_args(radiative_args, radiative_segments)
    n_threads = int(config.nthreads)
    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    e_count = int(e_count)
    if e_count == 0:
        return

    max_el = int(L_Zs.shape[1])
    sbethe_width = int(L_sbethe_logE.shape[1])
    n_blocks = (e_count + n_threads - 1) // n_threads
    kernel = monolithic_transport_kernel
    if _SPECIALIZE_TRANSPORT_KERNEL:
        kernel = transport_kernel_variant(
            TransportKernelSwitches.normalized(
                elastic_model_code=elastic_model_code,
                energy_model_code=energy_model_code,
                finite_footprint=bool(finite_footprint),
                sbethe_on=bool(sbethe_on),
                straggle_on=bool(straggle_on),
                inelastic_on=inelastic_on,
                rad_on=radiative_arguments[0],
                cont_on=continuation is not None,
            )
        )
    kernel(
        (n_blocks,),
        (n_threads,),
        (
            np.int32(e_start),
            np.int32(e_count),
            np.int32(cap),
            stream_key,
            alive.astype(xp.uint8, copy=False),
            np.int32(max_steps),
            np.int32(n_layers),
            internal_bounds,
            np.int32(elastic_model_code),
            np.int32(energy_model_code),
            np.float64(max_dE_frac),
            np.float64(z_total),
            np.int32(1 if finite_footprint else 0),
            np.float64(width_ang),
            np.float64(height_ang),
            clock,
            pos.reshape(-1),
            dirs.reshape(-1),
            E_cut_by_electrons,
            L_Js.reshape(-1),
            L_Zs.reshape(-1),
            L_ks.reshape(-1),
            L_coeffs.reshape(-1),
            L_E_cross.reshape(-1),
            L_ncm3.reshape(-1),
            L_sr_rate_numer.reshape(-1),
            L_mott_numer.reshape(-1),
            L_mott_denom1.reshape(-1),
            L_mott_denom2.reshape(-1),
            L_sr_joy_numer.reshape(-1),
            np.int32(1 if sbethe_on else 0),
            L_sbethe_n.astype(xp.int32, copy=False),
            L_sbethe_logE.reshape(-1),
            L_sbethe_logS.reshape(-1),
            np.int32(sbethe_width),
            L_nel.astype(xp.int32, copy=False),
            np.int32(max_el),
            L_top,
            L_bot,
            mott_has_table.reshape(-1).astype(xp.uint8, copy=False),
            mott_start.reshape(-1).astype(xp.int32, copy=False),
            mott_len.reshape(-1).astype(xp.int32, copy=False),
            mott_logE_flat,
            mott_logA_flat,
            el_start.reshape(-1).astype(xp.int32, copy=False),
            el_len.reshape(-1).astype(xp.int32, copy=False),
            el_logE,
            el_log_rate,
            el_cdf.reshape(-1),
            el_pdf.reshape(-1),
            el_mu,
            np.int32(el_mu.size),
            el_guide.reshape(-1).astype(xp.int32, copy=False),
            np.int32(el_guide.shape[1] - 1),
            E_keV,
            seg_dir.reshape(-1),
            seg_mid.reshape(-1),
            seg_len,
            seg_E,
            seg_t0,
            seg_id,
            seg_lay,
            seg_E_end,
            seg_t_end,
            seg_flight,
            seg_substep,
            seg_event,
            seg_count,
            exit_code,
            np.int32(1 if straggle_on else 0),
            stragg_dE,
            *shell_arguments,
            *radiative_arguments,
            *_continuation_kernel_args(continuation),
        ),
    )


def _continuation_kernel_args(continuation):
    """Continuation slot/state arrays, or inert placeholders."""
    if continuation is None:
        return (
            np.int32(0),
            xp.zeros(1, dtype=xp.int32),
            xp.zeros(1, dtype=xp.int32),
            xp.zeros(1, dtype=xp.uint64),
            xp.zeros(1, dtype=xp.float64),
        )
    slots, state_i32, state_u64, state_f64 = continuation
    return (
        np.int32(1),
        slots.astype(xp.int32, copy=False),
        state_i32.reshape(-1),
        state_u64.reshape(-1),
        state_f64.reshape(-1),
    )


def _shell_kernel_args(inelastic_args, hard_segments):
    """Flattened shell soft/hard kernel arguments, or inert placeholders."""
    if inelastic_args is None:
        dummy_float64 = xp.zeros(1, dtype=xp.float64)
        dummy_int32 = xp.zeros(1, dtype=xp.int32)
        return (
            np.int32(0),
            xp.zeros(1, dtype=xp.uint64),
            np.float64(0.0),
            dummy_int32,
            dummy_float64,
            dummy_float64,
            dummy_float64,
            np.int32(1),
            dummy_int32,
            np.int32(1),
            dummy_float64,
            dummy_float64,
            dummy_float64,
            dummy_int32,
            xp.zeros(1, dtype=xp.int16),
            dummy_float64,
            xp.zeros(1, dtype=xp.int16),
            np.int32(0),
            dummy_float64,
        )
    (
        keys,
        cutoff_eV,
        energy_counts,
        log_e,
        rate,
        omega2,
        channel_counts,
        ch_rate,
        ch_u,
        ch_w,
        ch_branch,
        ch_code,
    ) = inelastic_args
    seg_hard_W, seg_hard_ch, seg_hard_dir = hard_segments
    # Secondary launch directions (#94): an empty column means off.
    sec_on = seg_hard_dir.shape[0] > 0
    return (
        np.int32(1),
        keys,
        np.float64(cutoff_eV),
        energy_counts.astype(xp.int32, copy=False),
        log_e.reshape(-1),
        rate.reshape(-1),
        omega2.reshape(-1),
        np.int32(log_e.shape[1]),
        channel_counts.astype(xp.int32, copy=False),
        np.int32(ch_rate.shape[1]),
        ch_rate.reshape(-1),
        ch_u.reshape(-1),
        ch_w.reshape(-1),
        ch_branch.reshape(-1).astype(xp.int32, copy=False),
        ch_code.reshape(-1).astype(xp.int16, copy=False),
        seg_hard_W,
        seg_hard_ch,
        np.int32(1 if sec_on else 0),
        seg_hard_dir.reshape(-1) if sec_on else xp.zeros(1, dtype=xp.float64),
    )


def _radiative_kernel_args(radiative_args, photon_segments):
    """Flattened coupled radiative kernel arguments, or inert placeholders."""
    if radiative_args is None:
        dummy_float64 = xp.zeros(1, dtype=xp.float64)
        dummy_int32 = xp.zeros(1, dtype=xp.int32)
        return (
            np.int32(0),
            xp.zeros(1, dtype=xp.uint64),
            np.float64(0.0),
            dummy_int32,
            dummy_int32,
            dummy_int32,
            dummy_float64,
            dummy_float64,
            dummy_float64,
            dummy_float64,
            dummy_float64,
            dummy_float64,  # inert plasma-energy array for uncoupled kernels
            np.int32(1),
            np.int32(1),
            np.int32(1),
            dummy_float64,
            xp.zeros(1, dtype=xp.int16),
        )
    (
        keys,
        cutoff_eV,
        element_counts,
        incident_energy_counts,
        z,
        ncm3,
        incident,
        nominal,
        top,
        chi,
        plasma_by_layer,
    ) = radiative_args
    seg_rad_k, seg_rad_Z = photon_segments
    return (
        np.int32(1),
        keys,
        np.float64(cutoff_eV),
        element_counts.astype(xp.int32, copy=False),
        incident_energy_counts.reshape(-1).astype(xp.int32, copy=False),
        z.reshape(-1).astype(xp.int32, copy=False),
        ncm3.reshape(-1),
        incident.reshape(-1),
        nominal.reshape(-1),
        top.reshape(-1),
        chi.reshape(-1),
        plasma_by_layer.reshape(-1),
        np.int32(z.shape[1]),
        np.int32(incident.shape[2]),
        np.int32(nominal.shape[2]),
        seg_rad_k,
        seg_rad_Z,
    )


def make_cuda_transport_core(config=DEFAULT_TRANSPORT_KERNEL_CONFIG):
    """Return ``(core, array_module)`` for ``_run_per_electron_transport``."""

    def core(*args, **kwargs):
        run_transport_kernel(*args, config=config, **kwargs)

    return core, xp
