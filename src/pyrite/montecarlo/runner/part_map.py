"""Ordered per-part reductions over staged electron blocks (#1281, #1395).

Streamed reflection selection scores every pilot over the held prefix blocks
one block at a time and adds the block results in order
(:mod:`.reflection_cutoff`, :class:`.lattice_share.SelectionContinuum`).
Each of those loops is :func:`map_parts` followed by an ordered sum. By default
the map runs in process. Inside :func:`part_mapper` the mapping runs
elsewhere, on Dask workers holding the history shards of a shard-summed
plan. The results come back in part order, so the sums, and with them the
selection, are bit for bit those of the in-process map.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

#: ``mapper(parts, fn) -> list`` for :class:`StagedParts`, or ``None`` (in process).
_MAPPER: ContextVar[Callable[[StagedParts, Callable], list] | None] = ContextVar(
    "part_mapper", default=None
)


@dataclass(frozen=True)
class StagedParts:
    """Device copies of ``parts()`` cut to ``elec_id < prefix``, one block at a time.

    Each block carries ``header``'s population ``Ne`` (as the merged population
    would) before the prefix cut, and blocks the cut leaves empty are skipped;
    a population with no prefix row yields one empty block, so a sum is never
    empty. Oversized held blocks are subdivided on the host using the scorer's
    device-headroom budget before staging (#1416); off a CUDA device a held
    block is staged whole. Consumers must release each yielded device block
    before requesting the next one.
    """

    parts: Callable[[], Iterable[Mapping[str, Any]]]
    header: Mapping[str, Any]
    prefix: int | None

    def cut(self, part: Mapping[str, Any]) -> dict[str, Any]:
        """``part`` with the population ``Ne``, cut to the prefix (host or device)."""
        from .electron_blocks import electron_prefix

        segments = {**part, "Ne": self.header["Ne"]}
        return segments if self.prefix is None else electron_prefix(segments, self.prefix)

    def empty_block(self, empty: Mapping[str, Any] | None) -> dict[str, Any]:
        """The one staged block of a population with no prefix row."""
        from . import _segments_on_device

        return _segments_on_device({**self.header, **(empty or {})})

    @staticmethod
    def staged(segments: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """Device copies of a non-empty cut ``segments``, sized to the device headroom."""
        from . import _segments_on_device
        from .electron_blocks import (
            device_headroom_bytes,
            electron_block_count,
            iter_electron_blocks,
        )

        # A transport block (or the whole short population) need not fit a
        # selection scorer's row setup. Partition on the host before uploading.
        count = electron_block_count(int(segments["L_ang"].shape[0]), device_headroom_bytes())
        for block in iter_electron_blocks(segments, count):
            yield _segments_on_device(block)

    def __call__(self) -> Iterator[dict[str, Any]]:
        staged = 0
        empty = None
        for part in self.parts():
            segments = self.cut(part)
            if int(segments["L_ang"].shape[0]) == 0:
                empty = segments if empty is None else empty
                continue
            staged += 1
            yield from self.staged(segments)
        if not staged:
            yield self.empty_block(empty)


def _mapped(blocks: Iterable[Any], fn: Callable[[Any], Any]) -> list:
    """``[fn(block) for block in blocks]``, each block released before the next is staged."""
    values = []
    for block in blocks:
        values.append(fn(block))
        del block  # release device rows before the generator uploads its next block
    return values


def map_parts(parts: Callable[[], Iterable[Any]], fn: Callable[[Any], Any]) -> list:
    """``[fn(part) for part in parts()]``, on another executor when one is installed."""
    mapper = _MAPPER.get()
    if mapper is not None and isinstance(parts, StagedParts):
        return mapper(parts, fn)
    return _mapped(parts(), fn)


@contextmanager
def part_mapper(mapper: Callable[[StagedParts, Callable], list]):
    """Route :func:`map_parts` over :class:`StagedParts` through ``mapper`` here."""
    token = _MAPPER.set(mapper)
    try:
        yield
    finally:
        _MAPPER.reset(token)


def staged_part_result(fn, part, header, prefix):
    """One part of a remote :func:`map_parts`: ``("values", [fn(block), ...])`` or ``("empty", cut)``.

    Runs where ``part`` lives, staging it as :meth:`StagedParts.staged` does in
    process. A mapper returns the values in part order and, when every part was
    empty, applies ``fn`` to :meth:`StagedParts.empty_block` of the first empty
    cut, as the in-process iteration does.
    """
    segments = StagedParts(lambda: (), header, prefix).cut(part)
    if int(segments["L_ang"].shape[0]) == 0:
        return "empty", segments
    return "values", _mapped(StagedParts.staged(segments), fn)


def ordered_values(parts: StagedParts, fn, results: Sequence[tuple[str, Any]]) -> list:
    """Assemble remote :func:`staged_part_result` outputs as :func:`map_parts` returns them."""
    values = [value for kind, block in results if kind == "values" for value in block]
    if values:
        return values
    first = next((value for kind, value in results if kind == "empty"), None)
    return [fn(parts.empty_block(first))]
