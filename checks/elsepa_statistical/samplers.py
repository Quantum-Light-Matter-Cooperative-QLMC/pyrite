"""Study-only node-mixture samplers; production imports none of this module.

Source: conditional-uniform change of variables in the #895 moment derivation.
Units: dimensionless. Detailed elastic transport only; no hard-tail conditioning.
Validation: elsepa-energy-mixture-moments
"""

import types
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from pyrite.montecarlo.transport.scattering import _elsepa_bracket, _elsepa_invert_row

if TYPE_CHECKING:
    from pyrite.montecarlo.transport._jit_device import _elsepa_invert_tail

ELSEPA_TAIL_INVERSION = np.int32(0)
I32_ONE = np.int32(1)
_random_real = np.float64


def node_uniform(row, fraction, xi):
    """Choose upper with probability f, remapping each branch to Uniform[0,1].

    For 0<f<1: xi<f selects upper with u=xi/f; otherwise lower with
    u=(xi-f)/(1-f). Endpoints preserve xi exactly. No additional stream draw.
    In exact arithmetic on the 53-bit input grid, each joint branch/quantile
    CDF discrepancy is at most one input-grid spacing. This count bound
    excludes remapping/inversion rounding and is conditional on supplied f.
    Validation: elsepa-energy-mixture-moments
    """
    weight = np.float64(fraction)
    if weight <= 0:
        return row, xi
    if weight >= 1:
        return row + 1, xi
    if xi < weight:
        return row + 1, xi / weight
    return row, (xi - weight) / (1.0 - weight)


_node_uniform_cpu = njit(cache=False)(node_uniform)
_node_uniform_device = node_uniform


@njit(cache=False)
def sample_cpu(E_keV, xi, logE_flat, cdf, pdf, mu, guide, start, length):
    """CPU candidate, same call signature as the production polar sampler."""
    row, fraction = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    selected, polar = _node_uniform_cpu(row, fraction, xi)
    return 1.0 - 2.0 * _elsepa_invert_row(cdf, pdf, mu, guide, selected, polar)


def sample_device(cdf, pdf, mu, n_mu, guide, guide_cells, row, fraction, xi):
    """Device candidate; rebound to the production constants and inversion helpers."""
    selected, polar = _node_uniform_device(row, fraction, xi)
    if ELSEPA_TAIL_INVERSION == I32_ONE:
        tail = np.float64(_random_real(1.0) - polar)
        return _elsepa_invert_tail(cdf, pdf, mu, n_mu, guide, guide_cells, selected, tail)
    else:
        return _elsepa_invert_row(cdf, pdf, mu, n_mu, guide, guide_cells, selected, polar)


def install_cuda():
    """Install only in a fresh study subprocess, before any kernel is compiled.

    FP32 twins use their normal tail tables; node selection and conditional
    remapping remain FP64. Runtime switches and hard-tail transport are refused
    by the caller. No production file or cached on-disk source is modified.
    """
    from pyrite.montecarlo._cupy_jit import jit
    from pyrite.montecarlo.transport import _jit_device, _jit_kernel, _jit_precision

    if _jit_precision.fp32_twins.cache_info().currsize:
        raise RuntimeError("install candidate before compiling any precision twins")
    choose = jit.rawkernel(device=True)(node_uniform)
    namespace = {
        **vars(_jit_device),
        "_node_uniform_device": choose,
        "_random_real": np.float64,
    }
    candidate = types.FunctionType(sample_device.__code__, namespace, "_elsepa_quantile_mu")
    helper = jit.rawkernel(device=True)(candidate)
    _jit_device._elsepa_quantile_mu = helper
    _jit_kernel._elsepa_quantile_mu = helper
