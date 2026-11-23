"""Weighted ``bincount`` with a fixed summation order on every backend (#1292).

CuPy's weighted ``bincount`` adds float64 weights with ``atomicAdd`` in an
unspecified order, so the same inputs can round differently from run to run.
Per-electron sums feed the runner's adaptive statistics, which must replay
bitwise from the same seed.

Each bin is a sequential chain of dependent float64 adds, so its length, not
the input size, sets the device time. A GeForce card adds float64 slowly (RTX
5080: ~30 ns per chained add on the device path, against 3-6 ns per entry for
the host round trip), and the runner's per-electron sums put few electrons,
often one, behind hundreds of thousands of lines. A device input whose longest
chain would cost more on the device than the whole input costs on the host is
therefore summed on the host (#1402).
"""

import numpy as np

from ..._backend import _to_cpu, array_namespace

_KERNELS: dict[int, object] = {}

# ``w`` is already gathered into sorted order, so each bin's run is a
# contiguous read.
_ORDERED_SUM_BODY = """
double s = 0.0;
for (long long k = bounds[i]; k < bounds[i + 1]; ++k) {
    s += w[k];
}
out = s;
"""

_DEVICE_TO_HOST_CHAIN_RATIO = 8
"""Device cost of one chained add over host cost of one entry (#1402).

Measured on an RTX 5080 (King 15601): the device sort path takes 26-34 ns
per add along a bin's run, the host round trip (copy out, ``np.bincount``,
copy back) 3-6 ns per entry; for lines grouped by bin, as the runner passes
them, the two break even at a longest bin of N/8 to N/12. The host route is
taken when ``longest_bin * ratio > n_entries``. A mis-set ratio only picks the
slower of two bounded routes; both results are bitwise equal.
"""


def _kernel(xp):
    key = id(xp)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = xp.ElementwiseKernel(
            "raw int64 bounds, raw float64 w",
            "float64 out",
            _ORDERED_SUM_BODY,
            "pyrite_ordered_bincount",
        )
    return kernel


def _host_bincount(xp, index, weights, minlength):
    host = np.bincount(
        _to_cpu(index), weights=_to_cpu(weights).astype(np.float64), minlength=minlength
    )
    return xp.asarray(host)


def _route(xp, index, n_bins, route):
    """``"host"`` or ``"device"`` for in-range ``index`` over ``n_bins`` bins."""
    if route is not None:
        if route not in ("host", "device"):
            raise ValueError(f"unknown ordered_bincount route {route!r}")
        return route
    # Integer counts are exact under atomics, so the occupancy is deterministic.
    longest = int(xp.bincount(index, minlength=n_bins).max())
    return "host" if longest * _DEVICE_TO_HOST_CHAIN_RATIO > index.size else "device"


def ordered_bincount(index, weights, minlength=0, *, route=None):
    """``np.bincount(index, weights, minlength)`` summed in input order.

    Each bin adds its weights sequentially in the order they appear in
    ``index``, starting from ``0.0``, exactly as NumPy does on the host, so
    the device result is bitwise the host result and does not vary between
    runs. On CuPy a stable sort groups the entries by bin and one thread per
    bin sums its run, unless the longest run would take longer on the device
    than the host round trip for the whole input, which then sums on the host
    (``route`` forces ``"host"`` or ``"device"``). A namespace without
    ``ElementwiseKernel`` sums on the host. Returns float64 in the namespace
    of the inputs.
    """
    xp = array_namespace(index, weights)
    if xp is np:
        return np.bincount(index, weights=np.asarray(weights, np.float64), minlength=minlength)
    if not hasattr(xp, "ElementwiseKernel"):
        return _host_bincount(xp, index, weights, minlength)
    index = index.astype(xp.int64, copy=False).ravel()
    weights = weights.astype(xp.float64, copy=False).ravel()
    if index.size != weights.size:
        raise ValueError("index and weights must have the same length")
    n_bins = int(minlength)
    if index.size:
        # One readback for both ends of the range.
        lo, hi = (int(v) for v in _to_cpu(xp.stack((index.min(), index.max()))))
        if lo < 0:
            raise ValueError("bincount index must be non-negative")
        n_bins = max(n_bins, hi + 1)
    if index.size and _route(xp, index, n_bins, route) == "host":
        return _host_bincount(xp, index, weights, n_bins)
    order = xp.argsort(index, kind="stable").astype(xp.int64, copy=False)
    bounds = xp.searchsorted(index[order], xp.arange(n_bins + 1, dtype=xp.int64), side="left")
    out = xp.empty(n_bins, dtype=xp.float64)
    _kernel(xp)(bounds.astype(xp.int64, copy=False), weights[order], out)
    return out
