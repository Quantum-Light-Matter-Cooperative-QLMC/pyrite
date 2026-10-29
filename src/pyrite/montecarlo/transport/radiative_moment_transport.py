"""Caller-owned packed radiative moments for CPU/CUDA transport totals.

The leading physical plasma entries remain available to exact event samplers.
The tail is offsets, counts, physical energies, log energies and three moments.
No global host/device cache; batching uploads this buffer once per pass.
Validation: radiative-moment-interpolation
"""

import hashlib

import numpy as np

from ..._radiative_moment_policy import RADIATIVE_MOMENT_MODEL
from .radiative_moments import prepare_radiative_moments


def pack_transport_moments(compositions, tables, lo, hi, cutoff, plasma):
    prepared = prepare_radiative_moments(compositions, tables, lo, hi, cutoff)
    counts = np.asarray([g.size for g in prepared.energy_eV], dtype=np.int64)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    packed = np.ascontiguousarray(
        np.r_[
            plasma,
            starts,
            counts,
            *prepared.energy_eV,
            *prepared.log_energy,
            np.concatenate(prepared.moments).ravel(),
        ],
        dtype=np.float64,
    )
    packed.flags.writeable = False
    return packed


def add_moment_identity(result, arguments):
    """Persist exact/table policy and the actual numerical-buffer content pin."""
    layers = arguments[2].size
    packed = arguments[-1]
    table = packed.size > layers
    result["radiative"]["moment_model"] = RADIATIVE_MOMENT_MODEL if table else "exact"
    if table:
        result["radiative"]["moment_table_sha256"] = hashlib.sha256(packed.tobytes()).hexdigest()
