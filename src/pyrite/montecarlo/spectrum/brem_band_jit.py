"""CUDA per-primary band scoring, without segment × photon matrices.

Cell physics remains the existing FP32 scalar. Quadrature and row/primary
accumulation use FP64; reduction order differs from the portable FP32 GEMV.
Memory is linear in escaped rows plus staged EEDL panels and layer paths.
"""

from functools import cache
from types import FunctionType

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from .brem_jit_kernel import (
    U32_TWO,
    U32_ZERO,
    _bremslib_or_fallback_weighted_scalar,
    _transmission_scalar,
)

ROW_ACCUMULATION_DTYPE = np.float64
_ACCUM_REAL = np.float64


def _bremslib_band_template(
    kinetic_energies_keV,
    incident_momenta,
    bethe_heitler_prefactor,
    eedl_weight,
    eedl_lower_panel,
    eedl_panel_fraction,
    eedl_available,
    path_flat,
    mu_flat,
    energies_eV,
    panel_probability,
    bremslib_weight,
    bremslib_lower_top,
    bremslib_upper_top,
    bremslib_energy_fraction,
    bremslib_available,
    bremslib_lower_values,
    bremslib_upper_values,
    nominal_ratio,
    row_sum,
    band_weights,
    atomic_number,
    plasma_eV,
    n_segments,
    n_energies,
    n_layers,
):
    """One block per escaped row; FP64 photon quadrature and shared reduction.

    Validation: bremslib-angular-model
    """
    line_id = jit.blockIdx.x
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    spectrum_sum = _ACCUM_REAL(0.0)
    energy_index = thread_idx
    while energy_index < n_energies:
        photon_eV = energies_eV[energy_index]
        weighted_cross_section = _bremslib_or_fallback_weighted_scalar(
            kinetic_energies_keV[line_id],
            photon_eV,
            energy_index,
            line_id,
            incident_momenta[line_id],
            bethe_heitler_prefactor[line_id],
            eedl_weight[line_id],
            eedl_lower_panel[line_id],
            eedl_panel_fraction[line_id],
            eedl_available[line_id],
            panel_probability,
            bremslib_weight[line_id],
            bremslib_lower_top[line_id],
            bremslib_upper_top[line_id],
            bremslib_energy_fraction[line_id],
            bremslib_available[line_id],
            bremslib_lower_values,
            bremslib_upper_values,
            nominal_ratio,
            n_energies,
            atomic_number,
            plasma_eV,
        )
        transmission = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index, n_layers, n_energies
        )
        spectrum_sum += _ACCUM_REAL(weighted_cross_section * transmission) * _ACCUM_REAL(
            band_weights[energy_index]
        )
        energy_index += n_threads

    shared = jit.shared_memory(_ACCUM_REAL, None)
    shared[thread_idx] = spectrum_sum
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[thread_idx] += shared[thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        row_sum[line_id] = shared[U32_ZERO]


@cache
def _band_kernel(dtype):
    namespace = dict(_bremslib_band_template.__globals__, _ACCUM_REAL=dtype)
    template = FunctionType(
        _bremslib_band_template.__code__,
        namespace,
        "_bremslib_band_template",
        _bremslib_band_template.__defaults__,
        _bremslib_band_template.__closure__,
    )
    return jit.rawkernel()(template)


@jit.rawkernel()
def _group_rows(row_sum, row_electron, out, n_rows):
    row = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if row < n_rows:
        jit.atomic_add(out, row_electron[row], row_sum[row])


def run_bremslib_band_kernel(arguments, band_weights, row_electron, out, n_threads):
    """Reduce rows to original primary IDs, including histories with no rows.

    ``arguments`` has the spectrum kernel's array/scalar layout. Production photon-quadrature partials
    and original-ID primary atomics are FP64. Private benchmark overrides
    select FP32 scoring buffers; trajectory fields and CXR never change.
    """
    n_rows = int(row_electron.size)
    row_sum = xp.empty(n_rows, dtype=ROW_ACCUMULATION_DTYPE)
    args = (*arguments[:19], row_sum, band_weights, *arguments[20:])
    _band_kernel(ROW_ACCUMULATION_DTYPE)(
        (n_rows,),
        (n_threads,),
        args,
        shared_mem=n_threads * np.dtype(ROW_ACCUMULATION_DTYPE).itemsize,
    )
    # Match the atomic value type to the output (CUDA atomicAdd overload).
    values = row_sum.astype(out.dtype, copy=False)
    _group_rows(((n_rows + 255) // 256,), (256,), (values, row_electron, out, np.uint32(n_rows)))
    return out
