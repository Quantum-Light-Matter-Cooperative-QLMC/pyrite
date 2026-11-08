"""Lineage-addressed RNG keys of cascade daughters and their events (#923).

A cascade root's lineage key is its transport stream key: ``stream_keys(seed,
Ne)[e]`` of global primary ``e``, or the key an explicit launch supplies. A
daughter, an event or a draw namespace hangs off its parent's lineage key
through a process salt and a parent-local ordinal::

    key = mix(mix(parent ^ LINEAGE ^ process) + PHI * (ordinal + 1))

``ordinal`` counts the parent's events of that process in flight order (hard
collisions, or positive-energy hard photons), before any launch or energy
filter. The key therefore depends on neither the population size, the block a
history runs in, its position in the block, the generation or species
scheduling, nor dense track ids, which stay array indices only.

Every cascade stream (shell daughters, pair conversion and both charged
products, annihilation clocks and emission, hard-photon directions) uses this
scheme; :data:`CASCADE_RNG_MODEL` names it in a cascade's scientific identity.

Validation: shell-secondary-transport, photon-pair-first-interaction
"""

import numpy as np

from ..._cascade_rng_policy import CASCADE_RNG_MODEL
from .kinematics import (
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S11,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
    _U53_SCALE,
)

__all__ = [
    "CASCADE_RNG_MODEL",
    "PROCESS_ANNIHILATION_EMISSION",
    "PROCESS_ANNIHILATION_IN_FLIGHT",
    "PROCESS_HARD_PHOTON_DIRECTION",
    "PROCESS_PAIR_ELECTRON",
    "PROCESS_PAIR_PHOTON",
    "PROCESS_PAIR_POSITRON",
    "PROCESS_SHELL",
    "daughter_keys",
    "keyed_uniforms",
    "parent_ordinals",
]

# ``CASCADE_RNG_MODEL`` (``_cascade_rng_policy``) names this scheme; bump it
# on any change of the salts, the mixing or the ordinal definitions below.

_LINEAGE_SALT = np.uint64(0x6A09E667F3BCC909)
#: Process namespaces; distinct from each other and from every flat salt.
PROCESS_SHELL = np.uint64(0x0B5AD4ECEDA1CE2A)
PROCESS_PAIR_PHOTON = np.uint64(0x8E3C5B2F17A6D094)
PROCESS_PAIR_ELECTRON = np.uint64(0x2F9A61C7E4D83B05)
PROCESS_PAIR_POSITRON = np.uint64(0xC4E81B3A905F7D26)
PROCESS_ANNIHILATION_IN_FLIGHT = np.uint64(0x71D3F0A8B6C2E957)
PROCESS_ANNIHILATION_EMISSION = np.uint64(0xE5274C9D3A1B60F8)
PROCESS_HARD_PHOTON_DIRECTION = np.uint64(0x3D8B6E05F2C9A714)


def _mix(x):
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


def daughter_keys(parent_keys, process, ordinal):
    """Lineage keys of the ``ordinal``-th ``process`` event of each parent.

    ``parent_keys`` and ``ordinal`` broadcast elementwise.
    Validation: shell-secondary-transport, photon-pair-first-interaction
    """
    with np.errstate(over="ignore"):
        parent = np.asarray(parent_keys, dtype=np.uint64)
        n = np.asarray(ordinal, dtype=np.uint64)
        x = _mix(parent ^ _LINEAGE_SALT ^ np.uint64(process))
        return _mix(x + _SM64_GOLDEN * (n + _SM64_ONE))


def keyed_uniforms(keys, n_draws):
    """``u[i, c]`` in [0, 1): draw ``c`` of the counter stream keyed ``keys[i]``.

    The transport counter construction (53-bit mantissa), so each draw is a
    pure function of ``(key, c)`` and never of the array's order or length.
    """
    with np.errstate(over="ignore"):
        k = np.asarray(keys, dtype=np.uint64).reshape(-1, 1)
        c = np.arange(n_draws, dtype=np.uint64).reshape(1, -1)
        z = _mix(k + _SM64_GOLDEN * (c + _SM64_ONE))
        return (z >> _SM64_S11).astype(np.float64) * _U53_SCALE


def parent_ordinals(parent, *sort_keys):
    """Order and per-parent ordinals of events, ranked by ``sort_keys`` in flight order.

    ``sort_keys`` are the minor-to-major ``np.lexsort`` keys inside a parent
    (for example ``substep_id, flight_id``). Returns ``(order, ordinal)``:
    ``order`` sorts the events by parent then flight order, and
    ``ordinal[i]`` is the rank of event ``order[i]`` within its parent.
    """
    parent = np.asarray(parent, dtype=np.int64)
    order = np.lexsort((*sort_keys, parent))
    sorted_parent = parent[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = sorted_parent[1:] != sorted_parent[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
    return order, np.arange(order.size) - start
