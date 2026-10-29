"""Float64 CUDA evaluator for explicitly uploaded diagnostic moment tables.

This module is imported only on CUDA hosts. It does not alter transport.
Validation: radiative-moment-interpolation
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np

from .._cupy_jit import jit


@jit.rawkernel(device=True)
def _interpolate_moment(energy, logs, values, start, count, which, energy_eV):
    """One convex interpolated component; NaN signals unsupported coverage.

    Physical endpoints are checked before taking logs, with no extrapolation.
    Validation: radiative-moment-interpolation
    """
    end = start + count - np.int32(1)
    if not (energy_eV >= energy[start] and energy_eV <= energy[end]):
        return np.float64(np.nan)
    if energy_eV == energy[start]:
        return values[start * np.int32(3) + which]
    if energy_eV == energy[end]:
        return values[end * np.int32(3) + which]
    x = cp.log(energy_eV)
    lo = start
    hi = end
    while hi - lo > np.int32(1):
        mid = (lo + hi) // np.int32(2)
        if logs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - logs[lo]) / (logs[hi] - logs[lo])
    f = min(max(f, np.float64(0.0)), np.float64(1.0))
    return (np.float64(1.0) - f) * values[lo * np.int32(3) + which] + (
        f * values[hi * np.int32(3) + which]
    )


@jit.rawkernel()
def _evaluate(energy, logs, values, start, count, queries, output):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < queries.size:
        for which in range(3):
            output[i, which] = _interpolate_moment(
                energy, logs, values, start, count, np.int32(which), queries[i]
            )


@dataclass(frozen=True)
class UploadedRadiativeMoments:
    """Explicit caller/device ownership; reuse one upload for repeated calls."""

    energy: cp.ndarray
    logs: cp.ndarray
    values: cp.ndarray
    starts: np.ndarray
    counts: np.ndarray
    digest: str
    device: int

    @classmethod
    def from_host(cls, prepared):
        counts = np.array([g.size for g in prepared.energy_eV], dtype=np.int32)
        starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int32)
        return cls(
            cp.asarray(np.concatenate(prepared.energy_eV)),
            cp.asarray(np.concatenate(prepared.log_energy)),
            cp.asarray(np.concatenate(prepared.moments).ravel()),
            starts,
            counts,
            prepared.digest,
            cp.cuda.Device().id,
        )

    def evaluate(self, layer, queries):
        """Return (n_query, 3) device moments; invalid energies are explicit NaNs.

        The device sentinel avoids synchronization inside a diagnostic kernel.
        The caller must reject nonfinite results before using them.
        """
        if cp.cuda.Device().id != self.device:
            raise ValueError("moment tables belong to another CUDA device")
        if not 0 <= layer < self.counts.size:
            raise ValueError("invalid radiative moment layer")
        query = cp.asarray(queries, dtype=cp.float64)
        if query.ndim != 1:
            raise ValueError("moment queries must be one-dimensional")
        out = cp.empty((query.size, 3), dtype=cp.float64)
        if query.size:
            _evaluate(
                ((query.size + 127) // 128,),
                (128,),
                (
                    self.energy,
                    self.logs,
                    self.values,
                    self.starts[layer],
                    self.counts[layer],
                    query,
                    out,
                ),
            )
        return out
