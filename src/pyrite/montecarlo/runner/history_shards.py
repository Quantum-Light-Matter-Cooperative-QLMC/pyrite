"""History shards: one fixed-count case's electrons split across processes (#1387).

A shard is an electron range ``[start, stop)`` of one case, transported in
another process by the electron-block driver (:mod:`.block_transport`) and
returned as raw, unfinalized segment rows. The joining process concatenates
the shards in electron order, redoes the two population-wide host steps once
(bunch centroid timing, hard-photon direction completion), and continues as
the fixed-count case: line grid, spectra and outputs are the single
``[0, N)`` run's, bit for bit.

Raw rows, not per-shard spectra, cross process boundaries: spectra add only to
summation order, and several spectrum-phase steps (automatic line grids,
reflection selection, coherent sums) read the whole population. Every random
input is counter-addressed by ``(seed, electron, draw)``, so no new stream or
seeding scheme exists here.

Supported: fixed-count cases on the per-electron CPU core. Refused before any
work: the lockstep core (one generator across electrons), the CUDA core (GPU
transport shards are out of scope; a ``scoring="gpu"`` realization scores the
join on the device instead), adaptive electron counts, secondary cascades,
channeling, transition radiation, photon transport and GDF beams.
"""

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

from .block_transport import ElectronShard, ShardJoin, electron_blocks

#: The only transport core history shards run on (CPU, counter-addressed).
SHARD_TRANSPORT_CORE = "per-electron"


def shard_ranges(case, shard_electrons: int) -> list[tuple[int, int]]:
    """Half-open electron ranges of ``case``'s transport population, ``shard_electrons`` each.

    The population is ``max(Ne, Ne_brem)``; the last shard is short when the
    size does not divide it.
    """
    return electron_blocks(max(int(case["Ne"]), int(case["Ne_brem"])), shard_electrons)


def refuse_unshardable(case, transport_core: str = SHARD_TRANSPORT_CORE) -> str:
    """Return the core ``case`` shards on, or raise before any transport.

    ``transport_core`` is resolved as ``run_case`` resolves it (including a
    ``PYRITE_MC_TRANSPORT_CORE`` pin); anything but the per-electron core is
    refused.
    """
    from .adaptive import case_transport_core
    from .transport_options import _check_opt_in_transport

    if case.get("adaptive_precision") is not None:
        raise ValueError("history shards need a fixed electron count, not adaptive precision")
    if case.get("secondary_threshold_eV") is not None:
        raise ValueError("history shards do not support secondary cascades")
    if "gdf_source" in case:
        raise ValueError("history shards do not support GDF beams")
    core = case_transport_core(case, transport_core)
    if core != SHARD_TRANSPORT_CORE:
        raise ValueError(
            f"history shards run on the per-electron CPU core; {transport_core!r} resolved to "
            f"{core!r}. Pin transport_core='per-electron' (lockstep shares one generator "
            "across electrons; GPU shards are not supported)"
        )
    # Channeling, transition radiation and photon transport refuse blocks here.
    _check_opt_in_transport(case, None, 1, transport_core=core)
    return core


def transport_history_shard(
    case,
    start: int,
    stop: int,
    *,
    transport_core: str = SHARD_TRANSPORT_CORE,
    block_electrons: int | None = None,
) -> dict[str, Any]:
    """Transport electrons ``[start, stop)`` of ``case``; return the shard's raw rows.

    ``block_electrons`` bounds the rows held at once inside the shard (default:
    the whole shard as one block); it never changes the rows.
    """
    from .. import runner

    core = refuse_unshardable(case, transport_core)
    rows: list[dict[str, Any]] = []
    # The consumer receives the shard's rows once; the spectrum inputs that
    # _transport_case still prepares skip the population-wide line grid.
    shard = ElectronShard(int(start), int(stop), int(block_electrons or (stop - start)))
    runner._transport_case(
        case, transport_core=core, block_electrons=shard, block_consumer=rows.append
    )
    (segments,) = rows
    return segments


def run_case_from_shards(
    case,
    shards: Sequence[dict[str, Any]],
    *,
    transport_core: str = SHARD_TRANSPORT_CORE,
    record_timing: bool = False,
) -> dict[str, Any]:
    """``run_case``'s output for ``case`` from its shards, in electron order.

    ``shards`` must cover the transport population ``[0, max(Ne, Ne_brem))``
    contiguously, as :func:`transport_history_shard` returned them. The spectrum
    phase runs as the serial scheduler runs it, on the backend a marked case's
    realization names (``scoring``: CPU by default, or the process's GPU);
    untimed private metrics are dropped exactly as there.
    """
    from .. import runner
    from ..realization import case_realization, realization_context
    from .recovery import _spectrum_with_recovery
    from .scheduling import _UNTIMED_OUTPUT_KEYS

    core = refuse_unshardable(case, transport_core)
    # A marked case's realization names the spectrum backend ("cpu" or "gpu").
    marked = case_realization(case) is not None
    with realization_context(case) if marked else _unmarked_cpu_scope():
        transport = runner._transport_case(
            case,
            record_timing,
            transport_core=core,
            block_electrons=ShardJoin(tuple(shards)),
        )
        out = _spectrum_with_recovery(case, transport, record_timing)
    if not record_timing:
        for key in _UNTIMED_OUTPUT_KEYS:
            out.pop(key, None)
    return out


@contextmanager
def _unmarked_cpu_scope():
    """An unmarked join scores on CPU, as the serial CPU scheduler does."""
    from .chunking import _RESOURCE_POLICY, _cpu_spectrum_backend
    from .recovery import recovery_engine

    with recovery_engine("cpu"), _cpu_spectrum_backend() if _RESOURCE_POLICY.gpu else nullcontext():
        yield
