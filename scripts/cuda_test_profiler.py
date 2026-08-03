import cupy as xp
import numpy as np

from cxr_mc.montecarlo import SpectrumKernelConfig, run_reduction_kernel

data = np.load("/tmp/line_accum_debug.npz")
E_r = xp.asarray(data["E_r_f_np"])
aw = xp.asarray(data["aw_f_np"])
w = xp.asarray(data["w_f_np"])
E_grid = xp.asarray(data["E_grid_np"])

DEFAULT_SPECTRUM_KERNEL_CONFIG = SpectrumKernelConfig(
    nthreads=512,
    energies_per_block=3,
)


# warmup / force JIT compilation
run_reduction_kernel(
    E_r,
    aw,
    w,
    E_grid,
    config=DEFAULT_SPECTRUM_KERNEL_CONFIG,
)

# Test robustness
xp.cuda.Stream.null.synchronize()
for epb in (1, 2, 3, 4):
    for n_E_test in (1, 2, 3, 4, 5, 7, 8, 9):
        config = SpectrumKernelConfig(nthreads=512, energies_per_block=epb)
        spec = run_reduction_kernel(
            E_r,
            aw,
            w,
            E_grid=xp.asarray(np.arange(n_E_test, dtype=np.float32)),
            config=config,
        )

# xp.cuda.Stream.null.synchronize()
# # One profiled launch
# for epb in (2, 3, 4):
#     for nthreads in (128, 256, 512):
#         spec = run_spectrum_reduction(
#             E_r,
#             aw,
#             w,
#             E_grid,
#             nthreads=nthreads,
#             energies_per_block=epb,
#         )
# xp.cuda.Stream.null.synchronize()
