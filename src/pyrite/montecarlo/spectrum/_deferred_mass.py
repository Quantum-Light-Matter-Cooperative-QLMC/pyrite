"""Per-electron line mass summed off the main thread, bit for bit (#1481).

``_accumulate_edge_truncation`` adds each reduction unit's per-electron line
mass to ``audit["electron_mass"]``: a fixed-order bincount (#1292) of the
unit, then one add onto the running total. On the device the bincount took
the host route for the long per-electron chains (#1402), and its copies
blocked the main thread and drained the stream once per unit (~5 s of device
idle in a MoS2 300 keV case at 8192 electrons).

:class:`DeferredElectronMass` keeps the same additions in the same order but
moves them off the producing stream: each unit is copied asynchronously to
pinned host memory on a side stream, and one worker thread waits for the copy
and runs a ``nogil`` sequential bincount followed by the add onto the total.
Units are handled one at a time in enqueue order, so the total sees exactly
the serial sequence ``((seed + u_1) + u_2) + ...`` with each ``u_k`` summed
from ``0.0`` in input order, as ``np.bincount`` does. Bins a unit does not
touch are left as they are: the masses are positive, so ``x + 0.0 == x``.
:meth:`settle` drains the queue and writes the device total back.

Validation: line-grid-resonance-bandwidth
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from numba import njit

from ..._backend import _to_cpu, array_namespace, is_device_array

#: Transient audit key holding the pending accumulator; removed by settle.
PENDING_KEY = "_electron_mass_pending"


@njit(cache=True, nogil=True)
def _unit_into_total(index, weights, total, scratch, touched):
    """Add one unit's in-order bincount to ``total``; -1/-2 flag a bad index.

    ``scratch`` (zeros) and ``touched`` (False) have the length of ``total``
    and are returned to that state.
    """
    n = total.size
    bins = np.empty(index.size, dtype=np.int64)
    m = 0
    for k in range(index.size):
        b = np.int64(index[k])
        if b < 0:
            return -1
        if b >= n:
            return -2
        if not touched[b]:
            touched[b] = True
            bins[m] = b
            m += 1
        scratch[b] += np.float64(weights[k])
    for k in range(m):
        b = bins[k]
        total[b] = total[b] + scratch[b]
        scratch[b] = 0.0
        touched[b] = False
    return 0


@lru_cache(maxsize=1)
def _worker():
    """One worker: units of every accumulator run in enqueue order."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyrite-electron-mass")


class DeferredElectronMass:
    """Running per-electron mass of one spectrum call, summed off the main thread."""

    def __init__(self, cp, n_electrons, seed):
        self.cp = cp
        self.n = int(n_electrons)
        self.total = np.zeros(self.n, dtype=np.float64)
        if seed is not None:
            seed = np.asarray(_to_cpu(seed), dtype=np.float64)
            if seed.shape != (self.n,):
                raise ValueError(f"line electron id outside the {self.n} line electrons")
            self.total = seed.copy()
        self.scratch = np.zeros(self.n, dtype=np.float64)
        self.touched = np.zeros(self.n, dtype=np.bool_)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.futures = []

    def add(self, electron, mass):
        """Queue one unit; returns at once. Ids and masses stay referenced until copied."""
        cp = self.cp
        electron = cp.ascontiguousarray(electron)
        mass = cp.ascontiguousarray(mass)
        ready = cp.cuda.Event(block=False, disable_timing=True)
        ready.record()  # the producing (current) stream
        from cupyx import empty_pinned

        index_np = empty_pinned(electron.shape, electron.dtype)
        mass_np = empty_pinned(mass.shape, mass.dtype)
        with self.stream:
            self.stream.wait_event(ready)
            # Non-blocking: the main thread never waits for the producers.
            electron.get(stream=self.stream, out=index_np, blocking=False)
            mass.get(stream=self.stream, out=mass_np, blocking=False)
            copied = cp.cuda.Event(block=True, disable_timing=True)
            copied.record(self.stream)
        # The device arrays stay referenced until the copy is done, so the
        # pool cannot hand their memory to the producing stream before then.
        held = (electron, mass)
        self.futures.append(_worker().submit(self._sum, copied, index_np, mass_np, held))

    def _sum(self, copied, index, mass, held):
        copied.synchronize()
        status = _unit_into_total(index, mass, self.total, self.scratch, self.touched)
        del held
        if status == -1:
            raise ValueError("bincount index must be non-negative")
        if status == -2:
            raise ValueError(f"line electron id outside the {self.n} line electrons")

    def settle(self):
        """Wait for every queued unit and return the device total."""
        try:
            for future in self.futures:
                future.result()
        finally:
            self.futures = []
        return self.cp.asarray(self.total)

    def abandon(self):
        """Drain the queue after a failure without raising its errors."""
        for future in self.futures:
            future.exception()
        self.futures = []


#: Sum the per-electron line mass off the main thread on CuPy (#1481).
DEFER_ELECTRON_MASS = True


def defer_electron_mass(audit, electron, mass):
    """Queue ``(electron, mass)`` on the audit's accumulator; ``False`` if not deferred.

    Device (CuPy) arrays only, while :data:`DEFER_ELECTRON_MASS` holds. The
    first unit seeds the accumulator with any ``electron_mass`` already in
    the audit (an earlier call), so later adds land in the same order.
    """
    if not (DEFER_ELECTRON_MASS and is_device_array(mass) and is_device_array(electron)):
        return False
    xp = array_namespace(mass, electron)
    if getattr(xp, "__name__", "") != "cupy":
        return False
    pending = audit.get(PENDING_KEY)
    if pending is None:
        pending = audit[PENDING_KEY] = DeferredElectronMass(
            xp, audit["n_electrons"], audit.pop("electron_mass", None)
        )
    pending.add(electron, mass)
    return True


def settle_electron_mass(audit, *, failed=False):
    """Write a pending accumulator's total to ``audit["electron_mass"]`` (no-op without one)."""
    if audit is None:
        return
    pending = audit.pop(PENDING_KEY, None)
    if pending is None:
        return
    if failed:
        pending.abandon()
        return
    audit["electron_mass"] = pending.settle()
