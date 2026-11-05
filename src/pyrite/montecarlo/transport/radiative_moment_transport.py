"""Caller-owned packed radiative moments for CPU/CUDA transport totals.

The leading physical plasma entries remain available to exact event samplers.
The tail is offsets, counts, physical energies, log energies and three moments.
A bounded process-local host cache reuses identical read-only buffers across
passes (e.g. runner blocks sharing one table range); it is keyed by the exact
quadrature input bytes, so a hit is bit-identical to a rebuild. There is no
device cache; batching uploads this buffer once per pass.
Validation: radiative-moment-interpolation
"""

import hashlib
import threading
from collections import OrderedDict

import numpy as np

from ..._radiative_moment_policy import RADIATIVE_MOMENT_MODEL
from .radiative_moments import prepare_radiative_moments

_CACHE_SIZE = 8
_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(compositions, tables, lo, hi, cutoff, exact):
    digest = hashlib.sha256(repr((RADIATIVE_MOMENT_MODEL, lo, hi, cutoff)).encode())
    # Source admission reads table endpoints; moments read only ``exact``.
    for element in sorted({element for comp in compositions for element, _ in comp}):
        table = tables[element]
        digest.update(
            repr(
                (
                    element,
                    table.key,
                    table.digest,
                    table.minimum_incident_energy_keV,
                    table.maximum_incident_energy_keV,
                )
            ).encode()
        )
    for array in exact:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def pack_transport_moments(compositions, tables, lo, hi, cutoff, exact):
    """Packed moment buffer for the exact ``pack_radiative_layer_tables`` output.

    ``exact`` must be that packer's tuple for the same compositions, tables
    and ``[lo, hi]`` eV range with dielectric suppression; its last entry is
    the per-layer plasma energy that leads the returned buffer.
    """
    key = _cache_key(compositions, tables, float(lo), float(hi), float(cutoff), exact)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    prepared = prepare_radiative_moments(
        compositions, tables, lo, hi, cutoff, source_admission="exact-reference"
    )
    counts = np.asarray([g.size for g in prepared.energy_eV], dtype=np.int64)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    packed = np.ascontiguousarray(
        np.r_[
            exact[-1],
            starts,
            counts,
            *prepared.energy_eV,
            *prepared.log_energy,
            np.concatenate(prepared.moments).ravel(),
        ],
        dtype=np.float64,
    )
    packed.flags.writeable = False
    with _cache_lock:
        _cache[key] = packed
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return packed


def add_moment_identity(result, arguments):
    """Persist exact/table policy and the actual numerical-buffer content pin."""
    layers = arguments[2].size
    packed = arguments[-1]
    table = packed.size > layers
    result["radiative"]["moment_model"] = RADIATIVE_MOMENT_MODEL if table else "exact"
    if table:
        result["radiative"]["moment_table_sha256"] = hashlib.sha256(packed.tobytes()).hexdigest()
