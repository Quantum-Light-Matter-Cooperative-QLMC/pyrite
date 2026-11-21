"""Automatic reflection selection on a streamed electron prefix (#1281).

Retained runs select reflections on the counter-addressed prefix
``elec_id < SELECTION_HISTORIES`` and then score the selected set on every
history (:func:`.reflection_cutoff.selected_batch_spectrum`). Streamed runs
(live blocks and artifact replay) see the population one whole-electron block
at a time, in electron order. :class:`PrefixSelectingScorer` holds the leading
blocks until a row of an electron at or past the prefix arrives, so the held
blocks contain every prefix electron. It then selects on those rows, builds the
:class:`.streamed_scoring.BlockSpectrumScorer` for the selected set, scores the
held blocks, and passes later blocks straight through. Held rows are bounded
by the prefix plus one block; no extra transport pass runs.

A streamed case is one whose rows do not fit the device, and the prefix can be
the whole population (``Ne <= SELECTION_HISTORIES``), so held blocks wait on
the host and the selection never merges them: each of its scoring calls
stages one held block at a time and sums the blocks (#1295 job 14426, where
merging and staging the 10 GiB prefix ran out of device memory). Device memory
then holds one block, as in the production pass.

Validation: reflection-yield-tolerance-selector
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


def automatic_selection(case: Mapping[str, Any]) -> bool:
    """Whether ``case`` carries an automatic measured reflection selection."""
    return bool((case.get("reflection_selection") or {}).get("automatic"))


def block_scorer(case: Mapping[str, Any], inputs: Mapping[str, Any]):
    """The streamed scorer for ``case``: prefix-selecting when selection is automatic."""
    from .streamed_scoring import BlockSpectrumScorer

    if automatic_selection(case):
        return PrefixSelectingScorer(case, inputs)
    return BlockSpectrumScorer(case, inputs)


class PrefixSelectingScorer:
    """Select on the streamed electron prefix, then score every block.

    Same ``add``/``result``/``n_segments`` contract as
    :class:`.streamed_scoring.BlockSpectrumScorer`. A replayed block (resident
    OOM recovery) replaces the held block with the same first electron, and a
    failed selection or held-block scoring discards the partial scorer, so a
    retried block is committed once.

    Validation: reflection-yield-tolerance-selector
    """

    def __init__(self, case: Mapping[str, Any], inputs: Mapping[str, Any]):
        from .reflection_cutoff import SELECTION_HISTORIES

        self.case, self.inputs = case, inputs
        lines, brem = int(inputs["Ne_lines"]), int(inputs["Ne_brem"])
        #: Electrons the selection reads: the larger of its two prefixes.
        self.prefix = min(max(lines, brem), SELECTION_HISTORIES)
        self._held: dict[int, Mapping[str, Any]] = {}
        self.scorer: Any = None
        self.record: dict | None = None

    @property
    def n_segments(self) -> int:
        held = sum(int(b["L_ang"].shape[0]) for b in self._held.values())
        return held + (0 if self.scorer is None else self.scorer.n_segments)

    def add(self, segs: Mapping[str, Any]) -> None:
        """Hold ``segs`` until the prefix is complete; afterwards score it."""
        if self.scorer is not None:
            self.scorer.add(segs)
            return
        from .electron_blocks import _electron_ids

        ids = _electron_ids(segs)
        if ids.size == 0:
            return
        self._held[int(ids.min())] = _held_copy(segs)
        if int(ids.max()) >= self.prefix:
            self._select()

    def _select(self):
        """Select on the held prefix, score the held blocks, and return the scorer."""
        from .streamed_scoring import BlockSpectrumScorer

        blocks = [self._held[key] for key in sorted(self._held)]
        pruned, record = _select_on_blocks(self.case, self.inputs, blocks)
        scorer = BlockSpectrumScorer(self.case, self.inputs, selection=(pruned, record))
        for block in blocks:
            scorer.add(block)
        # Commit only after every held block scored (a raised OOM retries all).
        self.scorer, self.record = scorer, record
        self._held.clear()
        return scorer

    def result(self, **counts) -> dict[str, Any]:
        """The scorer's output plus the selection record."""
        scorer = self.scorer if self.scorer is not None else self._select()
        out = scorer.result(**counts)
        out["reflection_selection"] = self.record
        return out


def _select_on_blocks(case, inputs, blocks):
    """Pruned case and record from the held prefix ``blocks`` (#1281).

    The selection scores the blocks one at a time (``parts``); its batch
    carries only the joined scalar header.
    """
    from .emission_adapters import EmissionBatch, select_emission
    from .reflection_cutoff import select_batch_reflections

    blocks = list(blocks) or [_empty_segments()]
    emission = select_emission(case)
    batch = EmissionBatch(
        _header(blocks),
        inputs["E_grid"],
        case,
        inputs["n_hat"],
        case.get("abs_layers"),
        inputs["groove"],
        electron_limit=int(inputs["Ne_lines"]),
        table_cache={},
    )
    return select_batch_reflections(
        emission, batch, int(inputs["Ne_brem"]), parts=lambda: iter(blocks)
    )


def _header(blocks) -> dict[str, Any]:
    """Scalar fields of the joined ``blocks`` (counts added) over zero rows."""
    from .block_transport import _ADDITIVE_COUNTS, _is_array

    header = {k: v for k, v in blocks[0].items() if not _is_array(v)}
    for key in _ADDITIVE_COUNTS:
        if key in header:
            header[key] = sum(int(b[key]) for b in blocks)
    return {**header, **{k: v for k, v in _empty_segments().items() if k != "Ne"}}


def _held_copy(segs: Mapping[str, Any]) -> Mapping[str, Any]:
    """``segs`` with device arrays downloaded, so held rows leave the device free."""
    from ..._backend import is_device_array
    from .recovery import _host_block

    if not any(is_device_array(value) for value in segs.values()):
        return segs
    return _host_block(segs)


def _empty_segments() -> dict[str, Any]:
    """A zero-row segment table for a population that left no row."""
    return {"L_ang": np.zeros(0), "elec_id": np.zeros(0, dtype=np.int64), "Ne": 0}
