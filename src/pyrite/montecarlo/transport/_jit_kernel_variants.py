"""Exact CUDA transport kernel compiled per feature configuration (#827).

``_jit_kernel._transport_kernel`` is a template: it reads its feature switches
(the :class:`TransportKernelSwitches` fields) as module globals, not
parameters. Bound to the ``*_arg`` kernel parameters, as ``_jit_kernel`` binds
them, the template is the runtime-switched monolithic kernel. Rebound to
``np.int32`` constants, every ``if`` on a switch is folded by ``cupyx.jit``,
so a variant never emits the disabled paths and their registers (``sec_on``
excepted, which stays runtime; see the class). One Python
body serves both; see "Kernel specialization" in
``docs/repo-design/compute/gpu-transport-rawkernel.md``.

Like its siblings, this module imports ``cupy`` at module scope and stays out
of the package ``__init__``.

Validation: gpu-transport-core
"""

import types
from functools import cache
from typing import NamedTuple

import numpy as np

from .._cupy_jit import jit
from ._jit_kernel import _transport_kernel
from ._jit_precision import precision_globals


class TransportKernelSwitches(NamedTuple):
    """Feature switches the exact kernel takes as compile-time constants.

    ``elastic_model_code`` is 0/1/2 (screened Rutherford, Mott, ELSEPA),
    ``energy_model_code`` 0/1 (frozen, midpoint) and every other field 0/1.
    ``sec_on`` deliberately stays a runtime argument in every variant: as a
    constant it lets the compiler share ``2 pi u_phi`` between the primary and
    secondary azimuths, and the secondary azimuth then loses the fused
    multiply-add the monolithic kernel uses, which changes its last bit.
    """

    elastic_model_code: int
    energy_model_code: int
    finite_footprint: int
    sbethe_on: int
    straggle_on: int
    inelastic_on: int
    rad_on: int
    cont_on: int

    @classmethod
    def normalized(cls, **switches):
        """The switches as plain integers, e.g. from NumPy or bool values."""
        return cls(**{name: int(switches[name]) for name in cls._fields})


# The runtime-switched kernel: one compile serves every configuration, and its
# register allocation covers the union of all paths. Kept as the reference the
# variants are compared against bitwise, and as the fallback
# `_jit_launch._SPECIALIZE_TRANSPORT_KERNEL = False` selects.
monolithic_transport_kernel = jit.rawkernel()(_transport_kernel)


@cache
def transport_kernel_variant(switches: TransportKernelSwitches, precision: str = "fp64"):
    """Return the exact kernel compiled for ``switches``, built once per pair.

    The variant shares the template's code object, and so its one Python
    source, under a copy of its module globals in which every switch name is
    an ``np.int32`` constant; only ``sec_on_arg`` of its switch parameters is
    then read. ``precision`` ``"mixed"`` (#838) also rebinds the precision
    globals and FP32 helper twins of :mod:`._jit_precision`; ``"fp64"`` leaves
    the generated source unchanged. It is a separate key, not a switch field:
    switches mirror the case's physics features, while precision is a numerics
    choice recorded in run provenance (its model marker enters case identity).
    Transpilation happens on first launch in each process. CuPy's on-disk
    kernel cache keys the NVRTC compile by the generated source, so a later
    process with the same cache directory compiles nothing.

    Validation: gpu-mixed-precision-tables
    """
    constants = {name: np.int32(value) for name, value in switches._asdict().items()}
    variant = types.FunctionType(
        _transport_kernel.__code__,
        {**_transport_kernel.__globals__, **constants, **precision_globals(precision)},
        _transport_kernel.__name__,
        _transport_kernel.__defaults__,
        _transport_kernel.__closure__,
    )
    return jit.rawkernel()(variant)
