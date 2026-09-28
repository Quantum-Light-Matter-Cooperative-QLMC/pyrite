"""Electron-aligned segment blocks for over-budget incoherent line sums (#192).

An incoherent line sum is a sum over segments normalized by a fixed electron
count, so disjoint electron subsets add. Splitting by whole electrons keeps
every flight and secondary track of an electron together. Cases whose segments
fit the device run as one block, unchanged.
"""

from math import ceil

import numpy as np

from ..._backend import BACKEND, _to_cpu
from ..spectrum.lines._kernels import _SEG_ARRAYS

#: Peak device bytes one segment costs in an incoherent line sum: resident rows
#: (~86 B at FP32) plus the per-segment setup arrays that the kernel's
#: ``(n_block, N_g)`` bound does not cover. SLURM 209 (h-BN 5 MeV, 1 mm,
#: 22-86 M segments, 11.4 GiB pool): blocks at ~1,000 B of headroom per
#: segment fit, blocks at ~500 B overran.
LINE_DEVICE_BYTES_PER_SEGMENT = 1024

#: Largest block count the OOM retry doubles to before re-raising to the
#: spectrum phase's own chunk-halving retry.
MAX_ELECTRON_BLOCKS = 256


def device_headroom_bytes():
    """Bytes the CuPy pool can still hand out, or ``None`` off a CUDA device."""
    cp = getattr(BACKEND, "cp", None)
    if cp is None:
        return None
    pool = cp.get_default_memory_pool()
    used = int(pool.used_bytes())
    free, _ = cp.cuda.runtime.memGetInfo()
    headroom = int(free) + int(pool.total_bytes()) - used
    limit = int(pool.get_limit())
    if limit > 0:
        headroom = min(headroom, limit - used)
    return max(0, headroom)


def electron_block_count(n_segments, headroom_bytes, bytes_per_segment=None):
    """Blocks needed so one block's segments fit ``headroom_bytes``."""
    per_segment = LINE_DEVICE_BYTES_PER_SEGMENT if bytes_per_segment is None else bytes_per_segment
    if headroom_bytes is None or n_segments <= 0:
        return 1
    if headroom_bytes <= 0:
        return MAX_ELECTRON_BLOCKS
    return min(MAX_ELECTRON_BLOCKS, max(1, ceil(n_segments * per_segment / headroom_bytes)))


def _electron_ids(segments):
    key = "electron_id" if "electron_id" in segments else "elec_id"
    return np.asarray(_to_cpu(segments[key]))


def _take(array, rows):
    if isinstance(rows, slice) or isinstance(array, np.ndarray):
        return array[rows]
    return array[BACKEND.xp.asarray(rows)]


def iter_electron_blocks(segments, n_blocks):
    """Partition ``segments`` into at most ``n_blocks`` electron-aligned blocks.

    Blocks hold about equal segment counts and never split an electron. Scalar
    fields (``Ne``, geometry) are shared, so per-electron normalization is
    unchanged. Electron-sorted rows yield zero-copy slices; otherwise each block
    gathers its rows in their original order.
    """
    ids = _electron_ids(segments)
    n = ids.size
    if n_blocks <= 1 or n == 0:
        yield segments
        return
    ordered = bool(np.all(ids[1:] >= ids[:-1]))
    order = None if ordered else np.argsort(ids, kind="stable")
    sorted_ids = ids if ordered else ids[order]
    cuts = [0]
    for k in range(1, n_blocks):
        target = (k * n) // n_blocks
        # Snap to the first row of the electron that owns the target row.
        cut = int(np.searchsorted(sorted_ids, sorted_ids[target], side="left"))
        if cut > cuts[-1]:
            cuts.append(cut)
    cuts.append(n)
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        rows = slice(start, stop) if ordered else np.sort(order[start:stop])
        block = dict(segments)
        for key in _SEG_ARRAYS:
            if key in block:
                block[key] = _take(block[key], rows)
        yield block


def snapshot_audit(audit):
    """State needed to undo a partially accumulated truncation audit."""
    if audit is None:
        return None
    return {k: v for k, v in audit.items() if k != "collect"}, len(audit.get("collect", ()))


def restore_audit(audit, snapshot):
    """Undo accumulations made since :func:`snapshot_audit`."""
    if audit is None:
        return
    values, collected = snapshot
    for key in [k for k in audit if k != "collect" and k not in values]:
        del audit[key]
    audit.update(values)
    if "collect" in audit:
        del audit["collect"][collected:]
