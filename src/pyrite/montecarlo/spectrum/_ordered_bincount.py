"""Weighted ``bincount`` with a fixed summation order on every backend (#1292).

CuPy's weighted ``bincount`` adds float64 weights with ``atomicAdd`` in an
unspecified order, so the same inputs can round differently from run to run.
Per-electron sums feed the runner's adaptive statistics, which must replay
bitwise from the same seed.
"""

import numpy as np

from ..._backend import _to_cpu, array_namespace

_KERNELS: dict[int, object] = {}

_ORDERED_SUM_BODY = """
double s = 0.0;
for (long long k = bounds[i]; k < bounds[i + 1]; ++k) {
    s += w[order[k]];
}
out = s;
"""


def _kernel(xp):
    key = id(xp)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = xp.ElementwiseKernel(
            "raw int64 bounds, raw int64 order, raw float64 w",
            "float64 out",
            _ORDERED_SUM_BODY,
            "pyrite_ordered_bincount",
        )
    return kernel


def ordered_bincount(index, weights, minlength=0):
    """``np.bincount(index, weights, minlength)`` summed in input order.

    Each bin adds its weights sequentially in the order they appear in
    ``index``, starting from ``0.0``, exactly as NumPy does on the host, so
    the device result is bitwise the host result and does not vary between
    runs. On CuPy a stable sort groups the entries by bin and one thread per
    bin sums its run; a namespace without ``ElementwiseKernel`` sums on the
    host. Returns float64 in the namespace of the inputs.
    """
    xp = array_namespace(index, weights)
    if xp is np:
        return np.bincount(index, weights=np.asarray(weights, np.float64), minlength=minlength)
    if not hasattr(xp, "ElementwiseKernel"):
        host = np.bincount(
            _to_cpu(index), weights=_to_cpu(weights).astype(np.float64), minlength=minlength
        )
        return xp.asarray(host)
    index = index.astype(xp.int64, copy=False).ravel()
    weights = weights.astype(xp.float64, copy=False).ravel()
    if index.size != weights.size:
        raise ValueError("index and weights must have the same length")
    n_bins = int(minlength)
    if index.size:
        if int(index.min()) < 0:
            raise ValueError("bincount index must be non-negative")
        n_bins = max(n_bins, int(index.max()) + 1)
    order = xp.argsort(index, kind="stable").astype(xp.int64, copy=False)
    bounds = xp.searchsorted(index[order], xp.arange(n_bins + 1, dtype=xp.int64), side="left")
    out = xp.empty(n_bins, dtype=xp.float64)
    _kernel(xp)(bounds.astype(xp.int64, copy=False), order, weights, out)
    return out
