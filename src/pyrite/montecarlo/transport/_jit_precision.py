"""Mixed-precision build of the exact CUDA transport kernel (#838).

``transport_precision="mixed"`` keeps every accumulated quantity in FP64:
position, direction, energy, path, clock, optical depths and the elastic,
hard-inelastic and radiative rate sums. It moves into FP32 only the table
lookups and the sampling transcendentals whose result is a table query or a
bounded sample:

* the energy-grid lookups and interpolations (ELSEPA grid, SBETHE stopping,
  Mott screening, BremsLib moments and photon sampling) and ELSEPA polar-angle
  inversion, as the :data:`MIXED_HELPERS` device functions, and the tables
  they read (:data:`MIXED_TABLES`, passed as float32 copies). The ELSEPA
  inversion runs on the tail ``1 - xi`` of its FP64 deviate and the complement
  ``1 - F`` of its CDF (:data:`MIXED_COMPLEMENT_TABLES`), so a deflection near
  the top of the law keeps relative precision (#1013);
* mixed elastic hard-rate lookups (#894); its soft moment tables stay FP64, see
  :mod:`._jit_mixed_device`;
* the kernel body's query logarithms (``ln E``, ``log10 E``), the inline
  ELSEPA rate ``exp``, the Mott ``10**log_alpha``, ``sqrt(1 - cos^2)`` and the
  azimuth ``cos``/``sin``.

The free-path draws ``-ln u``, the direction renormalization, the straggling,
shell and stopping-power helpers and every RNG conversion stay FP64.

Like the #827 switches, the precision is a rebinding of module globals, not a
second source. ``cupyx.jit`` types a value by its operands, and in ``cuda``
mode a Python float literal is float32, so a helper called with float32
tables and a float32 query computes in float32 except where a module-level
``np.float64`` constant or an ``np.float64(...)`` cast widens it. The FP32
twin of a helper is the same code object under globals in which those
constants are ``np.float32`` and ``np`` resolves ``float64`` to ``float32``.
The kernel template narrows its helper arguments with ``table_real(...)`` and
its FP32 math with ``math_real(...)``. Both are ``np.float64`` in the default
build, and ``cupyx.jit`` drops a cast to a value's own type, so the FP64
kernel's generated source is unchanged.

Like its siblings, this module imports ``cupy`` at module scope (through its
helper imports) and stays out of the package ``__init__``.

Validation: gpu-mixed-precision-tables
Validation: gpu-mixed-precision-math
"""

import types
from functools import cache
from typing import Any

import numpy as np

from ..._numerics import TRANSPORT_PRECISIONS
from .._cupy_jit import jit
from . import (
    _jit_device,
    _jit_mixed_device,
    _jit_moment_lookup_device,
    _jit_radiative_device,
    _jit_shell_device,
)

# Device functions whose FP32 twin the mixed build calls. A helper not listed
# keeps its FP64 constants even when it receives float32 arguments; in
# particular `_stream_uniform` never narrows a draw.
MIXED_HELPERS = {
    _jit_shell_device: ("_log_grid_lower", "_log_grid_fraction"),
    _jit_device: (
        "_dEds_sbethe",
        "_interp_mott_log_alpha",
        "_elsepa_invert_tail",
        "_elsepa_quantile_mu",
    ),
    _jit_mixed_device: (
        "_mixed_hard_rate",
        "_mixed_layer_hard_bound",
        "_mixed_hard_element",
        "_mixed_has_soft_part",
    ),
    _jit_moment_lookup_device: ("_rad_transport_moment",),
    _jit_radiative_device: (
        "_rad_row",
        "_rad_fraction",
        "_rad_row_chi",
        "_rad_chi",
        "_rad_next_node",
        "_rad_photon",
        "_rad_cell_rate",
        "_rad_dielectric_cell",
        "_rad_moment",
        "_rad_layer_moment",
        "_rad_sample_photon_eV",
    ),
}

# Read-only kernel tables the mixed build reads as float32 copies: every
# table the helpers above read, plus the ELSEPA log-rate the body interpolates.
# The tables of MIXED_COMPLEMENT_TABLES are passed as float32(1 - table).
MIXED_TABLES = (
    "el_logE",
    "el_log_rate",
    "el_cdf",
    "el_pdf",
    "el_mu",
    "L_sbethe_logE",
    "L_sbethe_logS",
    "mott_logE_flat",
    "mott_logA_flat",
    "il_logE",
    "rad_ncm3",
    "rad_incident",
    "rad_nominal",
    "rad_top",
    "rad_chi",
    "rad_plasma",
    "mix_hf",
)

# The ELSEPA angular CDF reaches the FP32 twins as its complement 1 - F, formed
# in FP64 before rounding, so the top of the law keeps relative precision
# (#1013); see `_jit_device._elsepa_invert_tail`.
MIXED_COMPLEMENT_TABLES = ("el_cdf",)

# Module constants the FP32 twins rebind: the ELSEPA quantile inverts from
# the tail of its deviate on the complement table.
TWIN_CONSTANTS = {"ELSEPA_TAIL_INVERSION": np.int32(1)}


class _Float32Numpy:
    """``numpy`` as an FP32 twin sees it: ``np.float64`` is ``np.float32``."""

    float64 = np.float32

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


_FLOAT32_NUMPY = _Float32Numpy()


def _helpers():
    return {
        name: getattr(module, name) for module, names in MIXED_HELPERS.items() for name in names
    }


@cache
def fp32_twins() -> dict[str, Any]:
    """``{name: FP32 twin}`` of every :data:`MIXED_HELPERS` device function.

    Each twin shares its helper's code object. Its globals narrow every
    ``np.float64`` constant to ``np.float32``, resolve ``np.float64(...)`` to
    ``np.float32(...)``, set :data:`TWIN_CONSTANTS` and bind the other listed
    helpers to their twins, so a twin never calls an FP64 helper of the set.
    The ELSEPA quantile twin is the one that reads an FP64 argument, its
    deviate, and narrows only its tail ``1 - xi``.

    Validation: gpu-mixed-precision-tables
    """
    helpers = _helpers()
    twins: dict[str, Any] = {}
    namespaces = []
    for name, helper in helpers.items():
        func = helper._func
        namespace = {
            key: np.float32(value) if isinstance(value, np.float64) else value
            for key, value in func.__globals__.items()
        }
        namespace["np"] = _FLOAT32_NUMPY
        namespace.update({key: value for key, value in TWIN_CONSTANTS.items() if key in namespace})
        namespaces.append(namespace)
        twin = types.FunctionType(
            func.__code__, namespace, func.__name__, func.__defaults__, func.__closure__
        )
        twins[name] = jit.rawkernel(device=True)(twin)
    for namespace in namespaces:
        namespace.update({name: twin for name, twin in twins.items() if name in namespace})
    return twins


def precision_globals(precision: str) -> dict[str, Any]:
    """Kernel-template globals that select ``precision`` (empty for ``fp64``).

    Validation: gpu-mixed-precision-tables
    Validation: gpu-mixed-precision-math
    """
    if precision == "fp64":
        return {}
    if precision != "mixed":
        raise ValueError(f"transport precision must be one of {TRANSPORT_PRECISIONS}")
    return {**fp32_twins(), "table_real": np.float32, "math_real": np.float32}
