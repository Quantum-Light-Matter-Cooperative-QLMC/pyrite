import cupy as xp
import numpy as np

from .._cupy_jit import jit

F32_ZERO = np.float32(0.0)
F32_TINY = np.float32(1.0e-20)

U32_ZERO = np.uint32(0)
U32_ONE = np.uint32(1)
U32_TWO = np.uint32(2)
U32_THREE = np.uint32(3)
U32_FOUR = np.uint32(4)

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumKernelConfig:
    nthreads: int
    energies_per_block: int


DEFAULT_SPECTRUM_KERNEL_CONFIG = SpectrumKernelConfig(
    nthreads=512,
    energies_per_block=3,
)


@jit.rawkernel()
def _kernel_1e(
    resonances_eV,
    sinc_slopes,
    line_weights,
    energies_eV,
    spectrum,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x
    energy_index0 = energy_offset
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x

    # energy_index0 is always valid for every launched block.
    photon0_eV = energies_eV[energy_index0]

    # One private accumulator per energy.
    spectrum_sum_0 = F32_ZERO

    # Each thread gets:
    # thread_idx, thread_idx+n_threads, thread_idx+2*n_threads, ...
    line_id = thread_idx

    while line_id < n_lines:
        line_weight = line_weights[line_id]
        if line_weight != F32_ZERO:
            resonance_eV = resonances_eV[line_id]
            sinc_slope = sinc_slopes[line_id]

            # Energy 0
            sinc_argument_0 = sinc_slope * (photon0_eV - resonance_eV)
            if sinc_argument_0 == F32_ZERO:
                sinc_argument_0 = F32_TINY

            sinc_value_0 = xp.sin(sinc_argument_0) / sinc_argument_0
            spectrum_sum_0 += line_weight * sinc_value_0 * sinc_value_0

        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)

    offset0 = U32_ZERO
    shared[offset0 + thread_idx] = spectrum_sum_0

    jit.syncthreads()

    # Tree-reduce
    stride = n_threads // U32_TWO

    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # Thread zero owns the final reduced values.
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[offset0]


@jit.rawkernel()
def _kernel_2e(
    resonances_eV,
    sinc_slopes,
    line_weights,
    energies_eV,
    spectrum,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x * U32_TWO

    energy_index0 = energy_offset
    energy_index1 = energy_offset + U32_ONE

    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x

    # energy_index0 is always valid for every launched block.
    photon0_eV = energies_eV[energy_index0]
    has_energy_1 = energy_index1 < n_energies

    photon1_eV = F32_ZERO

    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]

    # One private accumulator per energy.
    spectrum_sum_0 = F32_ZERO
    spectrum_sum_1 = F32_ZERO

    # Each thread gets:
    # thread_idx, thread_idx+n_threads, thread_idx+2*n_threads, ...
    line_id = thread_idx

    while line_id < n_lines:
        line_weight = line_weights[line_id]
        if line_weight != F32_ZERO:
            # Load line geometry only for a live pair.  The fused prologue
            # leaves rejected pairs in deterministic order with zero weight.
            resonance_eV = resonances_eV[line_id]
            sinc_slope = sinc_slopes[line_id]

            # Energy 0
            sinc_argument_0 = sinc_slope * (photon0_eV - resonance_eV)
            if sinc_argument_0 == F32_ZERO:
                sinc_argument_0 = F32_TINY

            sinc_value_0 = xp.sin(sinc_argument_0) / sinc_argument_0
            spectrum_sum_0 += line_weight * sinc_value_0 * sinc_value_0

            # Energy 1
            if has_energy_1:
                sinc_argument_1 = sinc_slope * (photon1_eV - resonance_eV)
                if sinc_argument_1 == F32_ZERO:
                    sinc_argument_1 = F32_TINY

                sinc_value_1 = xp.sin(sinc_argument_1) / sinc_argument_1
                spectrum_sum_1 += line_weight * sinc_value_1 * sinc_value_1

        line_id += n_threads

    # Two shared-memory reduction regions:
    #
    # [ spectrum_sum_0 partials ]
    # [ spectrum_sum_1 partials ]
    shared = jit.shared_memory(xp.float32, None)

    offset0 = U32_ZERO
    offset1 = n_threads

    shared[offset0 + thread_idx] = spectrum_sum_0
    shared[offset1 + thread_idx] = spectrum_sum_1

    jit.syncthreads()

    # Tree-reduce both energies simultaneously.
    stride = n_threads // U32_TWO

    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # Thread zero owns the final reduced values.
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[offset0]

        if has_energy_1:
            spectrum[energy_index1] += shared[offset1]


@jit.rawkernel()
def _kernel_3e(
    resonances_eV,
    sinc_slopes,
    line_weights,
    energies_eV,
    spectrum,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x * U32_THREE

    energy_index0 = energy_offset
    energy_index1 = energy_offset + U32_ONE
    energy_index2 = energy_offset + U32_TWO

    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x

    # energy_index0 is always valid for every launched block.
    photon0_eV = energies_eV[energy_index0]

    # The final block may contain fewer than three valid energies.
    has_energy_1 = energy_index1 < n_energies
    has_energy_2 = energy_index2 < n_energies

    photon1_eV = F32_ZERO
    photon2_eV = F32_ZERO

    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]

    if has_energy_2:
        photon2_eV = energies_eV[energy_index2]

    # One private accumulator per energy.
    spectrum_sum_0 = F32_ZERO
    spectrum_sum_1 = F32_ZERO
    spectrum_sum_2 = F32_ZERO

    # Each thread gets:
    # thread_idx, thread_idx+n_threads, thread_idx+2*n_threads, ...
    line_id = thread_idx

    while line_id < n_lines:
        line_weight = line_weights[line_id]
        if line_weight != F32_ZERO:
            # Load once and reuse for all three energies.
            resonance_eV = resonances_eV[line_id]
            sinc_slope = sinc_slopes[line_id]

            # Energy 0
            sinc_argument_0 = sinc_slope * (photon0_eV - resonance_eV)
            if sinc_argument_0 == F32_ZERO:
                sinc_argument_0 = F32_TINY

            sinc_value_0 = xp.sin(sinc_argument_0) / sinc_argument_0
            spectrum_sum_0 += line_weight * sinc_value_0 * sinc_value_0

            # Energy 1
            if has_energy_1:
                sinc_argument_1 = sinc_slope * (photon1_eV - resonance_eV)
                if sinc_argument_1 == F32_ZERO:
                    sinc_argument_1 = F32_TINY

                sinc_value_1 = xp.sin(sinc_argument_1) / sinc_argument_1
                spectrum_sum_1 += line_weight * sinc_value_1 * sinc_value_1

            # Energy 2
            if has_energy_2:
                sinc_argument_2 = sinc_slope * (photon2_eV - resonance_eV)
                if sinc_argument_2 == F32_ZERO:
                    sinc_argument_2 = F32_TINY

                sinc_value_2 = xp.sin(sinc_argument_2) / sinc_argument_2
                spectrum_sum_2 += line_weight * sinc_value_2 * sinc_value_2

        line_id += n_threads

    # Three shared-memory reduction regions:
    #
    # [ spectrum_sum_0 partials ]
    # [ spectrum_sum_1 partials ]
    # [ spectrum_sum_2 partials ]
    shared = jit.shared_memory(xp.float32, None)

    offset0 = U32_ZERO
    offset1 = n_threads
    offset2 = U32_TWO * n_threads

    shared[offset0 + thread_idx] = spectrum_sum_0
    shared[offset1 + thread_idx] = spectrum_sum_1
    shared[offset2 + thread_idx] = spectrum_sum_2

    jit.syncthreads()

    # Tree-reduce all three energies simultaneously.
    stride = n_threads // U32_TWO

    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
            shared[offset2 + thread_idx] += shared[offset2 + thread_idx + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # Thread zero owns the final reduced values.
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[offset0]

        if has_energy_1:
            spectrum[energy_index1] += shared[offset1]

        if has_energy_2:
            spectrum[energy_index2] += shared[offset2]


@jit.rawkernel()
def _kernel_4e(
    resonances_eV,
    sinc_slopes,
    line_weights,
    energies_eV,
    spectrum,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x * U32_FOUR

    energy_index0 = energy_offset
    energy_index1 = energy_offset + U32_ONE
    energy_index2 = energy_offset + U32_TWO
    energy_index3 = energy_offset + U32_THREE

    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x

    # energy_index0 is always valid for every launched block.
    photon0_eV = energies_eV[energy_index0]

    # The final block may contain fewer than four valid energies.
    has_energy_1 = energy_index1 < n_energies
    has_energy_2 = energy_index2 < n_energies
    has_energy_3 = energy_index3 < n_energies

    photon1_eV = F32_ZERO
    photon2_eV = F32_ZERO
    photon3_eV = F32_ZERO

    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]

    if has_energy_2:
        photon2_eV = energies_eV[energy_index2]

    if has_energy_3:
        photon3_eV = energies_eV[energy_index3]

    # One private accumulator per energy.
    spectrum_sum_0 = F32_ZERO
    spectrum_sum_1 = F32_ZERO
    spectrum_sum_2 = F32_ZERO
    spectrum_sum_3 = F32_ZERO

    # Each thread gets:
    # thread_idx, thread_idx+n_threads, thread_idx+2*n_threads, ...
    line_id = thread_idx

    while line_id < n_lines:
        line_weight = line_weights[line_id]
        if line_weight != F32_ZERO:
            # Load once and reuse for all four energies.
            resonance_eV = resonances_eV[line_id]
            sinc_slope = sinc_slopes[line_id]

            # Energy 0
            sinc_argument_0 = sinc_slope * (photon0_eV - resonance_eV)
            if sinc_argument_0 == F32_ZERO:
                sinc_argument_0 = F32_TINY

            sinc_value_0 = xp.sin(sinc_argument_0) / sinc_argument_0
            spectrum_sum_0 += line_weight * sinc_value_0 * sinc_value_0

            # Energy 1
            if has_energy_1:
                sinc_argument_1 = sinc_slope * (photon1_eV - resonance_eV)
                if sinc_argument_1 == F32_ZERO:
                    sinc_argument_1 = F32_TINY

                sinc_value_1 = xp.sin(sinc_argument_1) / sinc_argument_1
                spectrum_sum_1 += line_weight * sinc_value_1 * sinc_value_1

            # Energy 2
            if has_energy_2:
                sinc_argument_2 = sinc_slope * (photon2_eV - resonance_eV)
                if sinc_argument_2 == F32_ZERO:
                    sinc_argument_2 = F32_TINY

                sinc_value_2 = xp.sin(sinc_argument_2) / sinc_argument_2
                spectrum_sum_2 += line_weight * sinc_value_2 * sinc_value_2

            # Energy 3
            if has_energy_3:
                sinc_argument_3 = sinc_slope * (photon3_eV - resonance_eV)
                if sinc_argument_3 == F32_ZERO:
                    sinc_argument_3 = F32_TINY

                sinc_value_3 = xp.sin(sinc_argument_3) / sinc_argument_3
                spectrum_sum_3 += line_weight * sinc_value_3 * sinc_value_3

        line_id += n_threads

    # ---------------------------------------------------------
    # BLOCK REDUCTION
    # ---------------------------------------------------------
    shared = jit.shared_memory(xp.float32, None)

    offset0 = U32_ZERO
    offset1 = n_threads
    offset2 = U32_TWO * n_threads
    offset3 = U32_THREE * n_threads

    shared[offset0 + thread_idx] = spectrum_sum_0
    shared[offset1 + thread_idx] = spectrum_sum_1
    shared[offset2 + thread_idx] = spectrum_sum_2
    shared[offset3 + thread_idx] = spectrum_sum_3

    jit.syncthreads()

    # Tree-reduce all four energies simultaneously.
    stride = n_threads // U32_TWO

    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
            shared[offset2 + thread_idx] += shared[offset2 + thread_idx + stride]
            shared[offset3 + thread_idx] += shared[offset3 + thread_idx + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    # Thread zero owns the final reduced values.
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[offset0]

        if has_energy_1:
            spectrum[energy_index1] += shared[offset1]

        if has_energy_2:
            spectrum[energy_index2] += shared[offset2]

        if has_energy_3:
            spectrum[energy_index3] += shared[offset3]


# * ---------------------------------------------------------
# * Define kernel runner
# * ---------------------------------------------------------

_REDUCTION_KERNELS = {
    1: _kernel_1e,
    2: _kernel_2e,
    3: _kernel_3e,
    4: _kernel_4e,
}


def run_reduction_kernel(
    E_r,
    aw,
    w,
    E_grid,
    *,
    out=None,
    config,
):
    n_threads = config.nthreads
    energies_per_block = config.energies_per_block

    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")

    try:
        kernel = _REDUCTION_KERNELS[energies_per_block]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_REDUCTION_KERNELS)}") from None

    n_energies = int(E_grid.size)

    # Kernels accumulate so callers can stream deterministic fixed-order
    # prologue blocks directly into the final spectrum without a CuPy add
    # launch.  A fresh call retains the old return-new-array contract.
    if out is None:
        out = xp.zeros(n_energies, dtype=xp.float32)
    if n_energies == 0:
        return out

    # ceil(n_energies / energies_per_block)
    n_blocks = (n_energies + energies_per_block - 1) // energies_per_block

    grid = (n_blocks,)
    block = (int(n_threads),)

    shared_memory_bytes = energies_per_block * n_threads * np.dtype(np.float32).itemsize

    kernel(
        grid,
        block,
        (
            E_r,
            aw,
            w,
            E_grid,
            out,
            np.uint32(E_r.size),
            np.uint32(n_energies),
        ),
        shared_mem=shared_memory_bytes,
    )

    return out
