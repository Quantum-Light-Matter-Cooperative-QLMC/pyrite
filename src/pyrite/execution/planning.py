"""Streaming, deterministic work partitioning without a large task graph."""

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from .model import ResourceRequest, WorkUnit


def iter_work_units(
    cases: Sequence[Any],
    *,
    cases_per_unit: int = 1,
    resources: ResourceRequest | None = None,
    case_key: Callable[[Any], str] | None = None,
    estimate_cost: Callable[[Any], float] | None = None,
) -> Iterator[WorkUnit]:
    """Yield bounded contiguous work units without copying case payloads.

    ``case_key`` is deliberately injected: the existing CAS identity involves
    resolved physics and cross-section-table markers and must not be recomputed
    from incomplete scheduler metadata. The resulting descriptors carry only
    keys and indexes, never large arrays or materialized Dask graphs.

    ``cases_per_unit`` is an explicit starting point; adaptive bin-packing and
    resource estimation are future milestones.
    """
    if isinstance(cases_per_unit, bool) or not isinstance(cases_per_unit, int):
        raise ValueError("cases_per_unit must be a positive integer")
    if cases_per_unit < 1:
        raise ValueError("cases_per_unit must be a positive integer")
    request = ResourceRequest() if resources is None else resources
    for start in range(0, len(cases), cases_per_unit):
        stop = min(start + cases_per_unit, len(cases))
        batch = (cases[index] for index in range(start, stop))
        if case_key is not None or estimate_cost is not None:
            # Evaluate each case at most once for this unit; descriptors store
            # identities/cost only, not numerical input objects.
            selected = tuple(batch)
            keys = tuple(case_key(case) for case in selected) if case_key else None
            cost = sum(estimate_cost(case) for case in selected) if estimate_cost else None
        else:
            keys = None
            cost = None
        yield WorkUnit(
            unit_id=f"work-{start:08d}-{stop:08d}",
            start=start,
            stop=stop,
            resources=request,
            case_keys=keys,
            estimated_cost=cost,
        )
