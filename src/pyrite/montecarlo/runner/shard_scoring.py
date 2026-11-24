"""Shard-summed spectra: every history shard scores its own electrons (#1395).

Under #1387 the shards of one case transport in parallel, but the join scores
the whole population in one process. The line, characteristic and continuum
estimators are track-length sums over whole electrons normalized by fixed
counts (:class:`.streamed_scoring.BlockSpectrumScorer`), so once the
population-wide choices are frozen, each shard can score its own rows and the
spectra add. A case reduced this way runs in three steps:

1. **Plan** (:func:`resolve_shard_plan`). The population-wide choices, made
   once from the stored shard rows in electron order: the ``line_grid_policy``
   axis (streamed grid evidence, :mod:`.streamed_line_grid`) and the automatic
   reflection selection on the counter-addressed prefix
   (:mod:`.streamed_selection`). Only shards that reach into the selection
   prefix are read for it.
2. **Score** (:func:`score_history_shard`). One scorer per shard against the
   frozen plan; its additive state (:meth:`~.streamed_scoring.BlockSpectrumScorer.partial`)
   is the shard's spectrum payload, keyed by the plan digest.
3. **Merge** (:func:`merge_shard_spectra`). The partials are absorbed in shard
   order and finalized once (two-beam diffraction, truncation audit, records).

Numerical contract: the result is the streamed route's (:func:`.streamed_scoring.run_case_streamed`)
with ``block_electrons = shard_electrons``, bit for bit, because each shard
commits one block and the merge adds the blocks in the same order. It is
deterministic for a fixed shard size and differs from the joined route
(``run_case`` or the #1387 join) only in floating-point summation order. The
two population-wide transport steps the join redoes (bunch-centroid timing,
hard-photon direction completion) are skipped, as streaming skips them: no
eligible estimator reads them. A marked ``reduction="shard-sum"`` realization
records the shard size, so these numbers never share an identity with the
joined ones.

Supported cases: those live streaming supports (``streaming_refusal(live=True)``)
on the per-electron core with a fixed electron count, without line roulette.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .history_shards import SHARD_TRANSPORT_CORE

#: Private output key recording how a shard-summed result was reduced.
SHARD_SCORING_KEY = "_shard_scoring"
_PLAN_SCHEMA = b"pyrite.shard-scoring-plan.v1\x00"


@dataclass(frozen=True)
class ShardScoringPlan:
    """The population-wide choices every shard scores against.

    ``E_grid`` is the final line axis, ``grid_record``/``grid_pass`` the policy
    grid's provenance (``None`` for an explicit grid), ``selection`` the
    automatic reflection selection's ``(pruned_case, record)`` (``None`` when
    the case selects nothing), and ``digest`` a SHA-256 over all three that
    shard payloads carry.
    """

    E_grid: np.ndarray
    grid_record: dict | None
    grid_pass: dict | None
    selection: tuple[dict, dict] | None
    digest: str


def shard_scoring_refusal(case: Mapping[str, Any]) -> str | None:
    """Why ``case`` cannot be scored shard by shard, or ``None``."""
    from .streamed_scoring import streaming_refusal

    if case.get("adaptive_precision") is not None:
        return "shard-summed scoring needs a fixed electron count"
    if case.get("line_roulette") is not None:
        return "line roulette keeps per-electron line state across the population"
    return streaming_refusal(case, live=True)


def refuse_shard_scoring(case: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` before any work when ``case`` cannot be shard-scored."""
    reason = shard_scoring_refusal(case)
    if reason is not None:
        raise ValueError(f"cannot score this case shard by shard: {reason}")


def resolve_shard_plan(
    case: Mapping[str, Any], shards: Callable[[], Iterable[Mapping[str, Any]]]
) -> ShardScoringPlan:
    """Freeze ``case``'s line axis and reflection selection from its shard rows.

    ``shards()`` yields the stored shard rows in electron order; it is called
    once per population pass (the grid pass reads every shard, the selection
    only those that start inside its prefix), so it may load lazily.

    Validation: reflection-yield-tolerance-selector
    """
    from .artifacts import _case_geometry
    from .streamed_selection import _select_on_blocks, automatic_selection

    refuse_shard_scoring(case)
    E_grid, _, _, n_hat, groove = _case_geometry(case)
    record = grid_pass = None
    if case.get("line_grid_policy") is not None:
        from .streamed_line_grid import resolve_streamed_line_grid

        def feed(consumer):
            for rows in shards():
                consumer(rows)

        E_grid, record, grid_pass = resolve_streamed_line_grid(
            case, None, transport_core=SHARD_TRANSPORT_CORE, n_hat=n_hat, groove=groove, feed=feed
        )
    selection = None
    if automatic_selection(case):
        inputs = _inputs(case, E_grid)
        held = _prefix_shards(shards(), _selection_prefix(case))
        selection = _select_on_blocks(case, inputs, held)
    return ShardScoringPlan(
        E_grid, record, grid_pass, selection, _plan_digest(E_grid, record, selection)
    )


def score_history_shard(
    case: Mapping[str, Any], rows: Mapping[str, Any], plan: ShardScoringPlan
) -> dict[str, Any]:
    """Score one shard's rows against ``plan``; return its additive spectrum payload.

    The payload holds the scorer's :meth:`~.streamed_scoring.BlockSpectrumScorer.partial`
    state, the shard's line-coverage records and transport counts, and the
    plan digest. Coverage warnings are left to the merge.
    """
    from ..spectrum.cxr._coverage import collect_line_data_coverage
    from .block_transport import _ADDITIVE_COUNTS
    from .oom import _ensure_pool_limit

    refuse_shard_scoring(case)
    _ensure_pool_limit()
    scorer = _scorer(case, plan)
    with collect_line_data_coverage(warn=False) as coverage:
        scorer.add(rows)
    partial = scorer.partial()
    partial.update(
        coverage=list(coverage),
        counts={key: int(rows[key]) for key in _ADDITIVE_COUNTS if key in rows},
        plan=plan.digest,
    )
    return partial


def merge_shard_spectra(
    case: Mapping[str, Any], plan: ShardScoringPlan, partials: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The case's spectrum-phase output from its shard payloads, in electron order.

    Every payload must carry ``plan``'s digest. Totals, audits and depth
    weights add in the given order; line-coverage records are concatenated
    and warned about once.

    Validation: diffracted-bremsstrahlung, diffracted-characteristic-radiation
    """
    from ..spectrum.cxr._coverage import collect_line_data_coverage, publish_line_data_coverage

    if not partials:
        raise ValueError("no shard spectra to merge")
    stale = [i for i, partial in enumerate(partials) if partial["plan"] != plan.digest]
    if stale:
        raise ValueError(f"shard spectra {stale} were scored against another plan")
    scorer = _scorer(case, plan)
    counts: dict[str, int] = {}
    for partial in partials:
        scorer.absorb(partial)
        for key, value in partial["counts"].items():
            counts[key] = counts.get(key, 0) + int(value)
    with collect_line_data_coverage() as coverage:
        for partial in partials:
            for entry in partial["coverage"]:
                publish_line_data_coverage(entry)
    out = scorer.result(
        Ne=counts["Ne"],
        n_backscattered=counts.get("n_backscattered", 0),
        n_missed=counts.get("n_missed", 0),
        line_coverage=coverage,
        diagnostic_grid=plan.grid_record,
    )
    out[SHARD_SCORING_KEY] = {
        "plan_digest": plan.digest,
        "shards": len(partials),
        "grid_pass": plan.grid_pass,
    }
    return out


def run_case_shard_scored(
    case,
    shard_electrons: int | None = None,
    *,
    shards: Sequence[Mapping[str, Any]] | None = None,
    transport_core: str = SHARD_TRANSPORT_CORE,
) -> dict[str, Any]:
    """In-process shard-summed run: transport, plan, score and merge every shard.

    ``shard_electrons`` defaults to the case's ``reduction="shard-sum"``
    realization; ``shards`` (as :func:`.history_shards.transport_history_shard`
    returned them, in electron order) skips transport. The result is what
    the Dask executor produces for the same marked case.
    """
    from .history_shards import refuse_unshardable, shard_ranges, transport_history_shard

    size = _shard_size(case, shard_electrons)
    refuse_shard_scoring(case)
    core = refuse_unshardable(case, transport_core)
    with _scoring_scope(case):
        if shards is None:
            shards = [
                transport_history_shard(case, start, stop, transport_core=core)
                for start, stop in shard_ranges(case, size)
            ]
        plan = resolve_shard_plan(case, lambda: iter(shards))
        partials = [score_history_shard(case, rows, plan) for rows in shards]
        return merge_shard_spectra(case, plan, partials)


def _shard_size(case, shard_electrons):
    from ..realization import case_reduction

    realization = case_reduction(case)
    if realization is None:
        if shard_electrons is None:
            raise ValueError("shard_electrons is required for an unmarked case")
        return int(shard_electrons)
    if shard_electrons is not None and int(shard_electrons) != realization.shard_electrons:
        raise ValueError(
            f"shard_electrons={shard_electrons} conflicts with the case's shard-sum "
            f"realization ({realization.shard_electrons})"
        )
    size = realization.shard_electrons
    assert size is not None  # a shard-sum realization always records its size
    return size


def route_shard_sum(run_case):
    """Route a ``reduction="shard-sum"`` case's ``run_case`` here: its realization fixes the shards.

    Joined scoring would store joined numbers under the shard-sum identity.
    """
    from functools import wraps

    @wraps(run_case)
    def routed(case, *args, **kwargs):
        from ..realization import case_reduction

        if case_reduction(case) is None:
            return run_case(case, *args, **kwargs)
        # record_timing (positional or keyword) is accepted; shard runs keep no timing.
        if len(args) > 1 or any(
            kwargs.get(key) is not None for key in ("observation_directions", "trajectory_capture")
        ):
            raise ValueError("a shard-sum realization has no directions or trajectory capture")
        core = kwargs.get("transport_core", "auto")
        core = SHARD_TRANSPORT_CORE if core == "auto" else core
        return run_case_shard_scored(case, transport_core=core)

    return routed


def _scoring_scope(case):
    """The spectrum backend of a marked case's realization; CPU when unmarked."""
    from ..realization import case_realization, realization_context
    from .history_shards import _unmarked_cpu_scope

    return (
        realization_context(case) if case_realization(case) is not None else _unmarked_cpu_scope()
    )


def _inputs(case, E_grid) -> dict[str, Any]:
    from .artifacts import _case_geometry

    _, E_brem, _, n_hat, groove = _case_geometry(case)
    return dict(
        E_grid=E_grid,
        E_brem=E_brem,
        n_hat=n_hat,
        groove=groove,
        Ne_lines=case["Ne"],
        Ne_brem=case["Ne_brem"],
    )


def _scorer(case, plan: ShardScoringPlan):
    from .streamed_scoring import BlockSpectrumScorer

    return BlockSpectrumScorer(case, _inputs(case, plan.E_grid), selection=plan.selection)


def _selection_prefix(case) -> int:
    """Electrons the automatic selection reads (``PrefixSelectingScorer.prefix``)."""
    from .reflection_cutoff import SELECTION_HISTORIES

    return min(max(int(case["Ne"]), int(case["Ne_brem"])), SELECTION_HISTORIES)


def _prefix_shards(shards: Iterable[Mapping[str, Any]], prefix: int) -> list:
    """The leading shards :class:`.streamed_selection.PrefixSelectingScorer` would hold.

    It holds blocks until one reaches an electron at or past ``prefix`` and
    selects on all of them (their summed ``Ne`` is the selection header's).
    """
    from .electron_blocks import _electron_ids

    held = []
    for rows in shards:
        ids = _electron_ids(rows)
        if ids.size == 0:
            continue
        held.append(rows)
        if int(ids.max()) >= prefix:
            break
    return held


def _plan_digest(E_grid, record, selection) -> str:
    digest = hashlib.sha256(_PLAN_SCHEMA)
    digest.update(np.ascontiguousarray(E_grid, dtype=np.float64).tobytes())
    digest.update(_canonical(record).encode())
    if selection is not None:
        pruned, chosen = selection
        layers = pruned.get("layer_radiators") or []
        hkls = [pruned.get("hkl_list")] + [
            None if layer is None else layer.get("hkl_list") for layer in layers
        ]
        digest.update(_canonical({"hkls": hkls, "record": chosen}).encode())
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=_plain, separators=(",", ":"))


def _plain(value):
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    if isinstance(value, tuple | set | frozenset):
        return list(value)
    return repr(value)
