"""Host-side launchers for the CUDA transport kernels.

Flattening, dtype narrowing, and scalar typing happen here rather than inside
the kernels in :mod:`pyrite.montecarlo.transport._jit_kernel`, so the launcher
signatures can track their CPU reference cores positionally.

Like its siblings, this module imports ``cupy`` at module scope, so it must
stay out of the package ``__init__``; ``api.py`` and ``batching.py`` reach it
only through deferred, function-local imports.
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np

from ._jit_kernel import _transport_kernel, _transport_lut_kernel


@dataclass(frozen=True)
class TransportKernelConfig:
    """Launch geometry. Does not affect results -- output slots are addressed by
    electron index, not by thread or block index."""

    nthreads: int = 128


DEFAULT_TRANSPORT_KERNEL_CONFIG = TransportKernelConfig()


def run_transport_lut_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
    L_nel,
    L_top,
    L_bot,
    lut_E_min_keV,
    lut_inv_dE_keV,
    lut_n_energy,
    lut_total_rate,
    lut_dEds,
    lut_inv_beta,
    lut_cdf,
    lut_alpha,
    E_keV,
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
    seg_count,
    exit_code,
    config=DEFAULT_TRANSPORT_KERNEL_CONFIG,
):
    """Launch the energy-LUT transport kernel."""
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    e_count = int(e_count)
    if e_count == 0:
        return

    max_el = int(lut_cdf.shape[1])
    nblocks = (e_count + nthreads - 1) // nthreads
    _transport_lut_kernel(
        (nblocks,),
        (nthreads,),
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
            np.float64(lut_E_min_keV),
            np.float64(lut_inv_dE_keV),
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
            seg_count,
            exit_code,
        ),
    )


def make_cuda_transport_lut_core(config=DEFAULT_TRANSPORT_KERNEL_CONFIG):
    """Return the LUT CUDA core and CuPy array module for the shared driver."""

    def core(*args):
        # Shared driver (_run_per_electron_transport_lut) appends the
        # straggling params (L_Js, L_Zs, L_ks, L_coeffs, L_E_cross,
        # straggle_on, stragg_dE) for the CPU LUT core
        # (_transport_core_ungrooved_perelectron_lut); the CUDA LUT kernel
        # has no straggling support (see api.py's NotImplementedError for
        # straggle_on=True on this path), so drop them before forwarding.
        run_transport_lut_kernel(*args[:-7], config=config)

    return core, xp


def run_transport_kernel(
    e_start,
    e_count,
    cap,
    stream_key,
    alive,
    max_steps,
    n_layers,
    internal_bounds,
    elastic_model_code,
    energy_model_code,
    max_dE_frac,
    z_total,
    finite_footprint,
    width_ang,
    height_ang,
    clock,
    pos,
    dirs,
    E_cut_by_electrons,
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
    L_nel,
    L_top,
    L_bot,
    mott_has_table,
    mott_start,
    mott_len,
    mott_logE_flat,
    mott_logA_flat,
    E_keV,
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
    seg_count,
    exit_code,
    straggle_on,
    stragg_dE,
    config=DEFAULT_TRANSPORT_KERNEL_CONFIG,
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
    """
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    e_count = int(e_count)
    if e_count == 0:
        return

    max_el = int(L_Zs.shape[1])
    nblocks = (e_count + nthreads - 1) // nthreads
    _transport_kernel(
        (nblocks,),
        (nthreads,),
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
            L_nel.astype(xp.int32, copy=False),
            np.int32(max_el),
            L_top,
            L_bot,
            mott_has_table.reshape(-1).astype(xp.uint8, copy=False),
            mott_start.reshape(-1).astype(xp.int32, copy=False),
            mott_len.reshape(-1).astype(xp.int32, copy=False),
            mott_logE_flat,
            mott_logA_flat,
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
            seg_count,
            exit_code,
            np.int32(1 if straggle_on else 0),
            stragg_dE,
        ),
    )


def make_cuda_transport_core(config=DEFAULT_TRANSPORT_KERNEL_CONFIG):
    """Return ``(core, array_module)`` for ``_run_per_electron_transport``."""

    def core(*args):
        run_transport_kernel(*args, config=config)

    return core, xp
