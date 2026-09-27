"""CUDA coherent-line kernel owners.

Imported only by GPU paths; keeping these re-exports out of ``spectrum``'s
package initializer preserves the CPU import path without a CuPy dependency.
"""

from .coherent_grouped_jit_kernel import run_coherent_grouped_intensity_kernel
from .coherent_jit_kernel import (
    DEFAULT_COHERENT_KERNEL_CONFIG,
    CoherentKernelConfig,
    run_coherent_reduction_kernel,
)
from .coherent_stream_jit_kernel import (
    DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
    CoherentStreamKernelConfig,
    finalize_coherent_fields,
    run_coherent_field_accumulation_kernel,
    run_coherent_prologue_kernel,
)

__all__ = [
    "CoherentKernelConfig",
    "CoherentStreamKernelConfig",
    "DEFAULT_COHERENT_KERNEL_CONFIG",
    "DEFAULT_COHERENT_STREAM_KERNEL_CONFIG",
    "finalize_coherent_fields",
    "run_coherent_field_accumulation_kernel",
    "run_coherent_grouped_intensity_kernel",
    "run_coherent_prologue_kernel",
    "run_coherent_reduction_kernel",
]
