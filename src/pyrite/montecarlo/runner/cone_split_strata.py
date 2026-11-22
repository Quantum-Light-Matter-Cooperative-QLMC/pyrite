"""Spectrum tallies of detector-cone split copies, one weight stratum at a time (#379).

The weight ladder leaves every copy weight a power of two ``W``. A stratum's
copy rows go through the production segment scorers unchanged, normalized by
the block's ``n_c`` copies (electron IDs and per-electron fields stay as the
copy generation numbered them, as in every row view); rescaling by
``W n_c / N`` turns that into the stratum's share ``sum_c W x_c / N`` of the
per-primary tally, ``N`` the primaries' normalization. Copies of primaries
outside the scored population (``Ne_brem``) are dropped with their parents.
This is the #879 strata pattern.

Validation: detector-cone-variance-reduction
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np

from ..spectrum.cxr._kernels import _SEG_ARRAYS
from .electron_blocks import _electron_ids, _take


def copy_strata(
    segments: Mapping[str, Any], n_norm: int
) -> Iterator[tuple[float, dict[str, Any], int]]:
    """``(scale, stratum segments, n_c)`` per weight stratum of a split block.

    ``scale = W n_c / n_norm`` turns a score normalized by ``n_c`` copies into
    the stratum's per-primary contribution. Yields nothing without copies.
    ``_stream*`` accumulator keys of ``segments`` carry over.
    """
    record = segments.get("cone_split")
    if record is None or record.get("copies") is None:
        return
    copies = record["copies"]
    weight = np.asarray(record["weight"], dtype=np.float64)
    n_copies = int(weight.size)
    in_limit = int(record["electron_start"]) + np.asarray(record["parent"]) < int(n_norm)
    ids = np.asarray(_electron_ids(copies), dtype=np.int64)
    extra = {k: v for k, v in segments.items() if k.startswith("_stream")}
    for ladder in np.unique(weight[in_limit]):
        members = np.flatnonzero(in_limit & (weight == ladder))
        rows = np.flatnonzero(np.isin(ids, members))
        if rows.size == 0:
            continue
        view = dict(copies)
        for field in _SEG_ARRAYS:
            if field in view:
                view[field] = _take(view[field], rows)
        view.update(extra)
        yield float(ladder) * n_copies / int(n_norm), view, n_copies


def fold_copies(
    segments: Mapping[str, Any], n_norm: int, score: Callable[[dict[str, Any], int], Any]
) -> Any:
    """``sum_W scale_W score(stratum, n_c)``, or ``0.0`` without copies.

    ``score(segments, n)`` must be a linear tally normalized by ``n`` electrons.

    Validation: detector-cone-variance-reduction
    """
    total = None
    for scale, view, n_copies in copy_strata(segments, n_norm):
        part = score(view, n_copies)
        if part is None:
            continue
        part = part * scale
        total = part if total is None else total + part
    return 0.0 if total is None else total


__all__ = ["copy_strata", "fold_copies"]
