"""Host admission for adaptive retained rows, their join and spectrum staging."""

import numpy as np

from ..transport import host_memory


def block_bytes(blocks, *, host_only=False):
    """Count array storage once per alias, including nested transport metadata."""
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(item) for item in value)
        if hasattr(value, "nbytes") and (not host_only or isinstance(value, np.ndarray)):
            return int(value.nbytes)
        return 0

    return size(blocks)


class AdaptiveHostAdmission:
    """Project observed bytes/history without changing histories or stopping.

    Three retained populations cover the blocks, concatenated result and a
    spectrum staging/finalization copy. Headroom excludes live host arrays, so
    charge only additional bytes. Recheck every growth and join against current
    system/cgroup headroom; a sampled projection is not a reservation.
    """

    def __init__(self, limit):
        self.limit = int(limit)

    def check(self, blocks, histories, *, force=False, phase="growth"):
        if not blocks or not histories:
            return
        total = block_bytes(blocks)
        host = block_bytes(blocks, host_only=True)
        host_rows = any(isinstance(block.get("L_ang"), np.ndarray) for block in blocks)
        if not force and not host_rows:
            return
        projected = (total * self.limit + histories - 1) // histories
        required = max(0, 3 * max(total, projected) - host)
        host_memory.admit_host_allocation(
            required,
            f"adaptive kept blocks {phase}; max_histories={self.limit}; "
            "lower max_histories or use streamed adaptive scoring (#1280)",
        )
