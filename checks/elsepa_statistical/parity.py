"""Remote-only CPU/CUDA candidate node-law parity on analytic densities."""

import json
import types
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from samplers import install_cuda

from pyrite.montecarlo._cupy_jit import jit
from pyrite.montecarlo.transport.scattering import elsepa_guide_table

if TYPE_CHECKING:
    from pyrite.montecarlo.transport._jit_device import _elsepa_quantile_mu as _sample


def sample_kernel(xi, cdf, pdf, mu, guide, fraction, output):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if index < xi.size:
        output[index] = _sample(
            cdf, pdf, mu, np.int32(4), guide, np.int32(1024), np.int32(0), fraction, xi[index]
        )


def main():
    from pyrite.montecarlo.transport import _jit_device, _jit_precision

    install_cuda()
    mu = np.array([0.0, 0.1, 0.5, 1.0])
    cdf = np.array([mu, mu * mu])
    pdf = np.array([np.ones(4), 2 * mu])
    guide = cp.asarray(elsepa_guide_table(cdf).ravel())
    xi = (np.arange(4096) + 0.5) / 4096
    draws = cp.asarray(xi)
    output = cp.empty_like(draws)
    records = []
    for precision in ("fp64", "mixed"):
        helper = (
            _jit_device._elsepa_quantile_mu
            if precision == "fp64"
            else _jit_precision.fp32_twins()["_elsepa_quantile_mu"]
        )
        kernel = jit.rawkernel()(
            types.FunctionType(sample_kernel.__code__, {**globals(), "_sample": helper})
        )
        dtype = np.float64 if precision == "fp64" else np.float32
        tables = [
            cp.asarray(a.ravel(), dtype=dtype)
            for a in ((cdf if precision == "fp64" else 1 - cdf), pdf, mu)
        ]
        for f in (0.0, 0.125, 0.5, 0.875, 1.0):
            kernel((16,), (256,), (draws, *tables, guide, dtype(f), output))
            if f == 0:
                expected = xi
            elif f == 1:
                expected = np.sqrt(xi)
            else:
                expected = np.where(xi < f, np.sqrt(xi / f), (xi - f) / (1 - f))
            error = float(np.max(np.abs(cp.asnumpy(output) - expected)))
            tolerance = 2e-13 if precision == "fp64" else 2e-6
            if error > tolerance:
                raise AssertionError((precision, f, error, tolerance))
            records.append(
                dict(precision=precision, fraction=f, max_absolute_error=error, tolerance=tolerance)
            )
    print(json.dumps(records, indent=2))
    Path("parity895.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
