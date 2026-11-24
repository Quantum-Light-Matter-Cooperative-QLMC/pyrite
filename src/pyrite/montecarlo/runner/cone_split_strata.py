"""Spectrum tallies of detector-cone split copies, one weight stratum at a time (#379).

The weight ladder leaves every copy weight a power of two ``W``. A stratum's
copy rows go through the production segment scorers unchanged, normalized by
the block's ``n_c`` copies (electron IDs and per-electron fields stay as the
copy generation numbered them, as in every row view); rescaling by
``W n_c / N`` turns that into the stratum's share ``sum_c W x_c / N`` of the
per-primary tally, ``N`` the primaries' normalization. Copies of primaries
outside the scored population (``Ne_brem``) are dropped with their parents,
as are those outside a row view's ``parent_range`` (batch means).
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
    parent = int(record["electron_start"]) + np.asarray(record["parent"])
    low, high = record.get("parent_range", (0, n_norm))
    in_limit = (parent >= int(low)) & (parent < min(int(high), int(n_norm)))
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


def collect_copy_lines(audit, case, segments, n_hat, Ne, abs_layers, groove):
    """Append detector-cone copies' lines at their stratum weight ``W`` (#379).

    Collected line weights are un-normalized sums, like the primaries', so a
    stratum's lines scale by ``W`` alone (``scale N / n_c``).

    Validation: detector-cone-variance-reduction
    """
    from ..spectrum._device_sizing import collect_lines
    from . import _cxr_for_segments

    n_norm = int(segments["Ne"] if Ne is None else Ne)
    for scale, rows, n_copies in copy_strata(segments, n_norm):
        part = {"start_eV": audit["start_eV"], "stop_eV": audit["stop_eV"], "collect": []}
        grid = np.array([audit["start_eV"], audit["stop_eV"]])
        _cxr_for_segments(
            rows, grid, case, n_hat, abs_layers, groove, coherent=False, Ne=n_copies,
            truncation_audit=part,
        )  # fmt: skip
        for energy, width, weight in part["collect"]:
            collect_lines(audit, (energy, width, weight * (scale * n_norm / n_copies)))


def merge_split_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Join electron blocks' split records in electron order (#379).

    Parents shift to the first block's ``electron_start``; per-copy and
    per-primary arrays concatenate; the copies' own results join with their
    electron IDs offset by the preceding blocks' copy counts.

    Validation: detector-cone-variance-reduction
    """
    from .block_transport import _globalize, merge_electron_blocks

    start = int(records[0]["electron_start"])
    merged = dict(records[0])
    for key, value in records[0].items():
        if key == "parent":
            merged[key] = np.concatenate(
                [np.asarray(r[key]) + (int(r["electron_start"]) - start) for r in records]
            )
        elif isinstance(value, np.ndarray):
            merged[key] = np.concatenate([np.asarray(r[key]) for r in records])
    copies, offset = [], 0
    for record in records:
        if record.get("copies") is not None:
            copies.append(_globalize(dict(record["copies"]), offset))
        offset += int(np.asarray(record["parent"]).size)
    merged["copies"] = merge_electron_blocks(copies) if copies else None
    return merged


class SplitTally:
    """Provenance of one split case's copies, summed over its blocks (#379).

    Records the policy, the cone (axis, half-angle, target weight, copy cap)
    and the realized copies: per power-of-two weight, primaries killed by the
    in-cone roulette, and the largest copy count of one primary.

    Validation: detector-cone-variance-reduction
    """

    def __init__(self, case: Mapping[str, Any]):
        self.policy = dict(case["cone_split"])
        self.cone: dict[str, Any] | None = None
        self.primaries = self.killed = self.max_per_primary = 0
        self.strata: dict[float, int] = {}

    def add(self, segments: Mapping[str, Any]) -> None:
        record = segments.get("cone_split")
        if record is None:
            return
        if self.cone is None:
            self.cone = {
                "n_hat": [float(v) for v in record["n_hat"]],
                "half_angle_rad": float(record["half_angle_rad"]),
                "target_weight": float(record["target_weight"]),
            }
        killed = np.asarray(record["killed"])
        self.primaries += int(killed.size)
        self.killed += int(np.count_nonzero(killed))
        parent = np.asarray(record["parent"], dtype=np.int64)
        if parent.size:
            self.max_per_primary = max(self.max_per_primary, int(np.bincount(parent).max()))
        weights, counts = np.unique(np.asarray(record["weight"]), return_counts=True)
        for weight, count in zip(weights.tolist(), counts.tolist(), strict=True):
            self.strata[weight] = self.strata.get(weight, 0) + int(count)

    def snapshot(self) -> tuple[Any, ...]:
        """State :meth:`restore` rolls a later :meth:`add` back to."""
        return self.cone, self.primaries, self.killed, self.max_per_primary, dict(self.strata)

    def restore(self, state: tuple[Any, ...]) -> None:
        self.cone, self.primaries, self.killed, self.max_per_primary, strata = state
        self.strata = dict(strata)

    def record(self) -> dict[str, Any]:
        return {
            **self.policy,
            **(self.cone or {}),
            "primaries": self.primaries,
            "primaries_killed": self.killed,
            "copies": sum(self.strata.values()),
            "max_copies_of_one_primary": self.max_per_primary,
            "copy_weight_strata": [
                {"weight": weight, "copies": count} for weight, count in sorted(self.strata.items())
            ],
        }


def split_outputs(case: Mapping[str, Any], segments: Mapping[str, Any]) -> dict[str, Any]:
    """``{"cone_split": provenance}`` for a retained spectrum output, or ``{}``."""
    if case.get("cone_split") is None:
        return {}
    tally = SplitTally(case)
    tally.add(segments)
    return {"cone_split": tally.record()}


__all__ = [
    "SplitTally",
    "collect_copy_lines",
    "copy_strata",
    "fold_copies",
    "merge_split_records",
    "split_outputs",
]
