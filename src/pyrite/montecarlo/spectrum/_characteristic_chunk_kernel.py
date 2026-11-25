"""One fused launch per segment chunk for characteristic line yields (#1455, option B).

The grouped scorer (:func:`.characteristic._accumulate_line_yields`) runs one
shell interpolation per chunk, then per line group three GEMMs (both escape
depths and the shell-to-line contraction), the fused weighted transmission and
a cuBLAS segment sum. On fine grids and high-tilt cases the device then waits
on launch overhead (#1432). This kernel does all of it in one launch per chunk:

    thread  = one segment row r of the chunk
    sigma_s = masked linear interpolation of shell s at the row's energy
              (the ``_characteristic_fused`` interpolation body), times the
              row's channeling flux weight when present
    per line j:
        sigma_j = sum_s sigma_s * response[s, j]            (shells in order)
        tau0_j  = sum_k path_start[r, k] * mu[k, j]          (layers in order)
        tau1_j  = sum_k path_end[r, k] * mu[k, j]
        y_rj    = path_r * sigma_j * mean_transmission(tau0_j, tau1_j)
        block tree reduction of y_rj over the block's rows -> partial[block, j]

The host then sums ``partial`` over blocks in FP64 and adds it to the FP64
line yields, as the grouped scorer adds its per-group sums. Per-element
arithmetic is that of the grouped chain; the BLAS contractions and the
segment sum regroup, so yields agree with it at FP64 summation order only
(#1313 ``fp64-line-sum-v1``). No atomics: one chunk's result is a fixed
function of its rows and the block size, so reruns are bitwise identical.

The kernel is a CuPy ``RawKernel`` compiled without FMA contraction for
CUDA or ROCm/HIP (:func:`pyrite._backend.exact_rawkernel_options`); its
block reduction uses shared memory only, no warp shuffles, so the 64-wide
HIP wavefront needs no special case. :func:`host_chunk_line_yields` runs the
same algorithm and reduction tree in NumPy: the CPU twin used by tests and
the GPU check script.

The route is opt-in while it awaits device validation:
``PYRITE_MC_CHARACTERISTIC_KERNEL=fused`` selects it, anything else keeps the
grouped chain. It also declines (returns ``None``) off the CuPy float32 path,
without packed shell tables, or above :data:`MAX_SHELLS` subshells.
"""

import numpy as np

from ..._backend import xp
from ..._util.env import env_value

#: Rows per block; a power of two for the shared-memory tree reduction.
BLOCK_ROWS = 256
#: Largest subshell count one thread holds (EEDL MF=23 has at most 39).
MAX_SHELLS = 40
#: Environment switch between the fused kernel and the grouped chain.
KERNEL_ENV = "PYRITE_MC_CHARACTERISTIC_KERNEL"

SOURCE = r"""
__device__ __forceinline__ double _nan_f64() { return __longlong_as_double(0x7ff8000000000000LL); }

__device__ __forceinline__ double _min_nan(double a, double b) {
    return (isnan(a) | isnan(b)) ? _nan_f64() : fmin(a, b);
}

// CuPy's cupy_interp body after searchsorted(side="right"), then the
// scorer's in-range mask; see _characteristic_fused._INTERP_BODY.
__device__ double _shell_interp(const float xv, const float* fx, const float* fy, long long len) {
    if (!((xv >= fx[0]) & (xv <= fx[len - 1]))) {
        return 0.0;
    }
    long long lo = 0;
    long long hi = len;
    while (lo < hi) {
        const long long m = lo + (hi - lo) / 2;
        if (fx[m] <= xv) { lo = m + 1; } else { hi = m; }
    }
    const long long x_idx = lo - 1;
    if (isnan(xv)) { return xv; }
    if (x_idx < 0) { return fy[0]; }
    if (xv == fx[len - 1]) { return fy[len - 1]; }
    if (x_idx >= len - 1) { return fy[len - 1]; }
    if (xv == fx[x_idx]) { return fy[x_idx]; }
    const double slope = (double)(fy[x_idx + 1] - fy[x_idx])
                         / ((double)fx[x_idx + 1] - (double)fx[x_idx]);
    double out = slope * ((double)xv - (double)fx[x_idx]) + (double)fy[x_idx];
    if (isnan(out)) {
        out = slope * ((double)xv - (double)fx[x_idx + 1]) + (double)fy[x_idx + 1];
        if (isnan(out) && (fy[x_idx] == fy[x_idx + 1])) {
            out = fy[x_idx];
        }
    }
    return out;
}

// mean_transmission(t0, t1) * path * sigma, in _characteristic_fused order.
__device__ __forceinline__ double _line_yield(double t0, double t1, double path, double sigma) {
    const double lo = _min_nan(t0, t1);
    const double d = fabs(t1 - t0);
    const bool big = d > 1.0e-4;
    const double safe = big ? d : 1.0;
    const double direct = -expm1(-safe) / safe;
    const double series = (1.0 - 0.5 * d) + (d * d) / 6.0;
    const double ratio = big ? direct : series;
    return (path * sigma) * (exp(-lo) * ratio);
}

extern "C" __global__ void characteristic_chunk_yield(
    const float* seg_E,          // keV, (n_segment,)
    const double* path_cm,       // n * L, (n_segment,)
    const double* path_start,    // (n_segment, n_layer)
    const double* path_end,      // (n_segment, n_layer)
    const double* weight,        // (n_segment, n_shell) flux weight if has_weight
    const float* table_E,        // packed shell energies, eV
    const float* table_sigma,    // packed shell cross sections, cm^2
    const long long* offset,     // (n_shell + 1,)
    const double* response,      // (n_shell, n_line)
    const double* mu,            // (n_layer, n_line), 1/angstrom
    const int n_shell,
    const int n_layer,
    const int n_line,
    const int has_weight,
    const long long start,
    const long long rows,
    double* partial              // (gridDim.x, n_line)
) {
    __shared__ double buffer[BLOCK_ROWS];
    const int tid = threadIdx.x;
    const long long local = (long long)blockIdx.x * BLOCK_ROWS + tid;
    const bool valid = local < rows;
    const long long r = start + local;
    double shell[MAX_SHELLS];
    double path = 0.0;
    if (valid) {
        const float xv = seg_E[r] * 1.0e3f;
        path = path_cm[r];
        for (int s = 0; s < n_shell; ++s) {
            const long long o = offset[s];
            double value = _shell_interp(xv, &table_E[o], &table_sigma[o], offset[s + 1] - o);
            if (has_weight) {
                value = value * weight[r * n_shell + s];
            }
            shell[s] = value;
        }
    }
    for (int j = 0; j < n_line; ++j) {
        double y = 0.0;
        if (valid) {
            double sigma = 0.0;
            for (int s = 0; s < n_shell; ++s) {
                sigma = sigma + shell[s] * response[(long long)s * n_line + j];
            }
            double t0 = 0.0;
            double t1 = 0.0;
            for (int k = 0; k < n_layer; ++k) {
                const double m = mu[(long long)k * n_line + j];
                t0 = t0 + path_start[r * n_layer + k] * m;
                t1 = t1 + path_end[r * n_layer + k] * m;
            }
            y = _line_yield(t0, t1, path, sigma);
        }
        buffer[tid] = y;
        __syncthreads();
        for (int stride = BLOCK_ROWS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                buffer[tid] = buffer[tid] + buffer[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial[(long long)blockIdx.x * n_line + j] = buffer[0];
        }
        __syncthreads();
    }
}
"""

_KERNELS: dict[int, object] = {}


def fused_requested() -> bool:
    """Whether ``PYRITE_MC_CHARACTERISTIC_KERNEL`` selects the fused route."""
    return env_value(KERNEL_ENV, "").strip().lower() == "fused"


def _kernel(cp):
    cached = _KERNELS.get(id(cp))
    if cached is None:
        from ..._backend import exact_rawkernel_options

        defines = f"#define BLOCK_ROWS {BLOCK_ROWS}\n#define MAX_SHELLS {MAX_SHELLS}\n"
        cached = _KERNELS[id(cp)] = cp.RawKernel(
            defines + SOURCE, "characteristic_chunk_yield", options=exact_rawkernel_options(cp)
        )
    return cached


def _is_cupy(*arrays) -> bool:
    return getattr(xp, "__name__", "") == "cupy" and all(isinstance(a, xp.ndarray) for a in arrays)


def _f64(array):
    return xp.ascontiguousarray(xp.asarray(array, dtype=np.float64))


def path_lengths(seg_L, number_density_ang3):
    """``n * L`` per segment, in the grouped scorer's expression and dtype."""
    return number_density_ang3 * 1.0e24 * (seg_L * 1.0e-8)


def chunk_line_yields(
    seg_E,
    seg_L,
    path_start,
    path_end,
    *,
    chunk,
    number_density_ang3,
    packed_shells,
    response,
    mu_by_line,
    shell_weight=None,
):
    """FP64 ``(n_line,)`` segment-summed line yields, or ``None`` to keep the chain.

    Arguments are those of :func:`.characteristic._accumulate_line_yields`.
    Declines unless selected by :data:`KERNEL_ENV`, on the CuPy float32 path
    with packed shell tables and at most :data:`MAX_SHELLS` subshells.
    """
    if not fused_requested() or packed_shells is None:
        return None
    if not _is_cupy(seg_E, seg_L) or seg_E.dtype != np.float32:
        return None
    n_shell, n_line = (int(n) for n in response.shape)
    if n_shell > MAX_SHELLS or n_shell != int(packed_shells[2].size) - 1:
        return None
    path_cm = _f64(path_lengths(seg_L, number_density_ang3))
    path_start, path_end = _f64(path_start), _f64(path_end)
    weight = None if shell_weight is None else _f64(shell_weight)
    table_E, table_sigma, offset = packed_shells
    kernel = _kernel(xp)
    n_rows = int(seg_E.size)
    n_layer = int(mu_by_line.shape[0])
    mu, response = _f64(mu_by_line), _f64(response)
    seg_E = xp.ascontiguousarray(seg_E)
    line_yields = xp.zeros(n_line, dtype=np.float64)
    for first in range(0, n_rows, chunk):
        rows = min(chunk, n_rows - first)
        blocks = -(-rows // BLOCK_ROWS)
        partial = xp.empty((blocks, n_line), dtype=np.float64)
        kernel(
            (blocks,),
            (BLOCK_ROWS,),
            (
                seg_E,
                path_cm,
                path_start,
                path_end,
                path_cm if weight is None else weight,
                table_E,
                table_sigma,
                offset,
                response,
                mu,
                np.int32(n_shell),
                np.int32(n_layer),
                np.int32(n_line),
                np.int32(weight is not None),
                np.int64(first),
                np.int64(rows),
                partial,
            ),
        )
        line_yields += partial.sum(axis=0)
    return line_yields


def host_chunk_line_yields(
    seg_E,
    seg_L,
    path_start,
    path_end,
    *,
    chunk,
    number_density_ang3,
    shell_sigma,
    response,
    mu_by_line,
    shell_weight=None,
):
    """NumPy twin of the fused route: same per-row terms and reduction tree.

    ``shell_sigma(energy_keV)`` returns the ``(rows, n_shell)`` masked
    interpolation (the scorer's :func:`._interpolate_shell_cross_sections`).
    Rows are summed per :data:`BLOCK_ROWS` block by the kernel's halving tree,
    then over blocks, per chunk; device transcendental functions may differ
    from NumPy's by an ulp, so the twin matches the kernel to FP64 rounding.
    """
    seg_E, seg_L = np.asarray(seg_E), np.asarray(seg_L)
    path_cm = np.asarray(path_lengths(seg_L, number_density_ang3), dtype=np.float64)
    start_paths = np.asarray(path_start, dtype=np.float64)
    end_paths = np.asarray(path_end, dtype=np.float64)
    mu = np.asarray(mu_by_line, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    weight = None if shell_weight is None else np.asarray(shell_weight, dtype=np.float64)
    n_shell, n_line = response.shape
    line_yields = np.zeros(n_line, dtype=np.float64)
    for first in range(0, int(seg_E.size), chunk):
        rows = slice(first, min(first + chunk, int(seg_E.size)))
        shell = np.asarray(shell_sigma(seg_E[rows]), dtype=np.float64)
        if weight is not None:
            shell = shell * weight[rows]
        sigma = np.zeros((shell.shape[0], n_line))
        for s in range(n_shell):
            sigma = sigma + shell[:, s, None] * response[s]
        tau0 = np.zeros_like(sigma)
        tau1 = np.zeros_like(sigma)
        for k in range(mu.shape[0]):
            tau0 = tau0 + start_paths[rows, k, None] * mu[k]
            tau1 = tau1 + end_paths[rows, k, None] * mu[k]
        values = _host_line_yield(tau0, tau1, path_cm[rows, None], sigma)
        line_yields += _block_tree_sum(values).sum(axis=0)
    return line_yields


def _host_line_yield(t0, t1, path, sigma):
    lo = np.minimum(t0, t1)
    d = np.abs(t1 - t0)
    big = d > 1.0e-4
    safe = np.where(big, d, 1.0)
    ratio = np.where(big, -np.expm1(-safe) / safe, (1.0 - 0.5 * d) + (d * d) / 6.0)
    return (path * sigma) * (np.exp(-lo) * ratio)


def _block_tree_sum(values):
    """``(blocks, n_line)`` partials of ``values`` summed as the kernel's tree."""
    rows, n_line = values.shape
    blocks = -(-rows // BLOCK_ROWS)
    padded = np.zeros((blocks * BLOCK_ROWS, n_line))
    padded[:rows] = values
    tree = padded.reshape(blocks, BLOCK_ROWS, n_line)
    stride = BLOCK_ROWS // 2
    while stride:
        tree[:, :stride] = tree[:, :stride] + tree[:, stride : 2 * stride]
        stride //= 2
    return tree[:, 0]
