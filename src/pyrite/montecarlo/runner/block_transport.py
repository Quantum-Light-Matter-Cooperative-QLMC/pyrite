"""Electron-block transport driver for adaptive electron counts (#361).

Transports electrons ``[0, N)`` as consecutive blocks ``[kB, (k+1)B)`` on the
counter-addressed per-electron/CUDA cores and joins the block results into the
result one call over ``[0, N)`` returns, bit for bit.

The equivalence rests on three properties:

* every random input is addressed by ``(seed, electron, draw)`` -- transport,
  hard-collision, radiative and straggling keys through ``stream_keys(...,
  start=)``, and spot, Twiss and energy-spread draws through the per-electron
  counter streams inside their child namespaces -- so a block draws exactly its
  slice of the larger run;
* the per-electron cores emit rows electron-major and step-minor, so
  concatenating blocks in order reproduces the single call's row order;
* the two population-wide host steps are redone once over the joined result:
  the bunch centroid subtraction (a property of the realized population) and
  the hard-photon direction completion (one sequential generator over rows).

The lockstep core shares one generator across electrons and is not supported.
This is internal API: the stopping rule, estimators, and the public ``"auto"``
electron count build on it.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import numpy as np

from ..._backend import BACKEND
from ..transport.beam_entry import face_arrival_delay_ang, initial_energies_keV
from ..transport.kinematics import _sample_bunch_offsets
from .case_tables import _case_radiative_kwargs
from .step_budget import retry_step_budget

#: Integer result fields that add across disjoint electron blocks.
_ADDITIVE_COUNTS = (
    "Ne",
    "n_backscattered",
    "n_transmitted",
    "n_side_exited",
    "n_missed",
    "n_cutoff_stopped",
    "n_step_limited",
    "n_stopped",
)
#: Fields holding a block-local electron index, shifted to the global one.
_ELECTRON_INDEX_FIELDS = ("electron_id", "elec_id", "vacuum_elec_id")
#: Bunch keyword arguments withheld from blocks and applied after the join.
BUNCH_KWARGS = ("bunch_length_fs", "long_shape", "long_offsets_fs", "longitudinal_distribution")


def electron_blocks(n_electrons: int, block_electrons: int) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` electron ranges of size ``block_electrons``.

    The last block is short when ``block_electrons`` does not divide
    ``n_electrons``; ``n_electrons == 0`` gives no blocks.
    """
    n, b = int(n_electrons), int(block_electrons)
    if n < 0:
        raise ValueError("n_electrons must be non-negative")
    if b < 1:
        raise ValueError("block_electrons must be a positive integer")
    return [(start, min(start + b, n)) for start in range(0, n, b)]


def _is_array(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and len(value.shape) >= 1


def _xp_of(value):
    return np if isinstance(value, np.ndarray) else BACKEND.xp


def _globalize(block: dict[str, Any], start: int) -> dict[str, Any]:
    """Shift a block result's electron indices by ``start``, keeping aliases."""
    if not start:
        return block
    shifted: dict[int, Any] = {}
    for key in _ELECTRON_INDEX_FIELDS:
        value = block.get(key)
        if value is None:
            continue
        if id(value) not in shifted:
            shifted[id(value)] = value + start
        block[key] = shifted[id(value)]
    return block


def merge_electron_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Join globalized block results in electron order.

    Arrays concatenate (per-electron and per-row alike, since both are
    electron-major), counts add, and metadata is taken from the first block.
    A field that aliases another in the first block aliases it in the join.
    """
    if not blocks:
        raise ValueError("no electron blocks to merge")
    first = blocks[0]
    merged: dict[str, Any] = {}
    joined: dict[int, Any] = {}
    for key, value in first.items():
        if key in _ADDITIVE_COUNTS:
            merged[key] = sum(int(b[key]) for b in blocks)
        elif _is_array(value):
            if id(value) not in joined:
                parts = [b[key] for b in blocks]
                joined[id(value)] = _xp_of(value).concatenate(parts)
            merged[key] = joined[id(value)]
        else:
            merged[key] = value
    return merged


def finalize_population_fields(
    result: dict[str, Any],
    *,
    seed: int,
    bunch: Mapping[str, Any],
    bremslib_tables=None,
) -> dict[str, Any]:
    """Redo the two population-wide host steps over the joined result.

    Bunch offsets are re-sampled for all ``result["Ne"]`` electrons (each raw
    draw is counter-addressed; only the centroid needs the population) and
    gathered onto rows exactly as ``simulate_trajectories`` does. Hard-photon
    directions are re-completed over the joined rows in their final order.

    Validation: longitudinal-bunch-sampling
    """
    n_electrons = int(result["Ne"])
    t0_electron = _sample_bunch_offsets(
        n_electrons,
        bunch.get("bunch_length_fs"),
        bunch.get("long_shape", "gaussian"),
        bunch.get("long_offsets_fs"),
        seed,
        bunch.get("longitudinal_distribution"),
    )
    # Blocks ran without bunch arguments; restore the face-arrival delay the
    # single call adds (zeros without tilt or analytic spot).
    t0_electron = t0_electron + face_arrival_delay_ang(
        result.get("beam_entry"), result["initial_r_ang"], result["initial_E_keV"]
    )
    elec_id = result["electron_id"]
    xp = _xp_of(elec_id)
    t0_by_electron = t0_electron if xp is np else xp.asarray(t0_electron)
    result["initial_t0_ang"] = t0_electron.copy()
    result["t0_ang"] = t0_by_electron[elec_id] if elec_id.size else xp.empty(0, dtype=float)
    vac_id = result["vacuum_elec_id"]
    result["vacuum_t0_ang"] = t0_electron[vac_id] if vac_id.size else np.empty(0, dtype=float)
    if "hard_radiative_k_eV" in result:
        from ..transport.hard_radiative import complete_hard_radiative_events

        complete_hard_radiative_events(result, bremslib_tables, seed)
    return result


def transport_electron_blocks(
    simulate_block: Callable[[int, int], dict[str, Any]],
    n_electrons: int,
    block_electrons: int,
    *,
    seed: int,
    bunch: Mapping[str, Any],
    bremslib_tables=None,
    on_block: Callable[[int, int, dict[str, Any]], None] | None = None,
    should_stop: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    """Transport ``[0, n_electrons)`` block by block and join the blocks.

    ``simulate_block(start, stop)`` must run ``simulate_trajectories`` for
    ``stop - start`` electrons with ``_electron_start=start``, the matching
    slice of any per-electron cutoffs, and no bunch arguments. ``on_block``
    (optional) sees each block's result as ``on_block(start, stop, block)``
    before the join, with block-local electron indices ``[0, stop - start)``
    exactly as ``simulate_block`` returned it -- the hook accumulators attach
    to; it must not mutate the block. ``should_stop(stop)`` (optional) is asked
    after each block's hook; ``True`` ends the run at ``stop`` electrons.

    Returns the result a single ``[0, n)`` call returns, bit for bit, where
    ``n`` is ``n_electrons`` or the earlier stop.
    """
    blocks = []
    for start, stop in electron_blocks(n_electrons, block_electrons):
        block = simulate_block(start, stop)
        if on_block is not None:
            on_block(start, stop, block)
        blocks.append(_globalize(block, start))
        if should_stop is not None and should_stop(stop):
            break
    merged = merge_electron_blocks(blocks)
    return finalize_population_fields(
        merged, seed=seed, bunch=bunch, bremslib_tables=bremslib_tables
    )


def transport_case_blocks(
    keep,
    *,
    simulate,
    case,
    E_cut_by_electrons,
    block_electrons,
    beam_kw,
    progress=None,
    monitor=None,
):
    """The runner's block path: ``_transport_case``'s transport in electron blocks.

    ``simulate(keep, max_steps, start=, stop=, block_kw=)`` is the case's own
    ``simulate_trajectories`` closure; each block runs under the step-budget
    retry, without bunch arguments, and over the population's table energy
    range, so the joined result equals the single-call transport bit for bit.

    ``monitor`` (optional) supplies ``on_block`` and ``should_stop`` hooks
    (:class:`.adaptive.StoppingMonitor`). The blocks then run over the table
    energy range of all ``E_cut_by_electrons.size`` electrons, the most the
    run may transport. When that range differs from the range of the realized
    population (an energy spread whose largest draw lies past the stop), the
    realized ``[0, n)`` is replayed over its own range -- the streams are
    counter-addressed, so the replay is the fixed-``n`` transport -- and the
    monitor recomputes its statistics. If the replay fails the stopping rule,
    grow by one block and replay again until convergence or the maximum.

    Validation: adaptive-sample-size-stopping
    """
    n_electrons = int(E_cut_by_electrons.size)
    no_bunch = {"bunch_length_fs": None, "long_shape": "gaussian"}
    no_bunch |= {"long_offsets_fs": None, "longitudinal_distribution": None}
    energies = initial_energies_keV(
        case["E0_keV"], n_electrons, case["seed"], beam_kw.get("energy_spread_frac")
    )

    def energy_range_of(n):
        return (float(np.min(E_cut_by_electrons[:n])), float(np.max(energies[:n])))

    energy_range = energy_range_of(n_electrons)

    def simulate_block(start, stop):
        block_kw: dict[str, Any] = {"_electron_start": start, "_energy_range_keV": energy_range}
        block_kw |= no_bunch
        if progress is not None:

            def block_progress(done, _n, start=start):
                progress(start + done, n_electrons)

            block_kw["transport_progress"] = block_progress
        return retry_step_budget(simulate, keep, start=start, stop=stop, block_kw=block_kw)

    run = partial(
        transport_electron_blocks,
        simulate_block,
        block_electrons=block_electrons,
        seed=case["seed"],
        bunch={k: beam_kw[k] for k in BUNCH_KWARGS},
        bremslib_tables=_case_radiative_kwargs(case).get("bremslib_tables"),
    )
    if monitor is None:
        return run(n_electrons)
    result = run(n_electrons, on_block=monitor.on_block, should_stop=monitor.should_stop)
    realized = int(result["Ne"])
    if energy_range_of(realized) != energy_range:
        monitor.replayed_transport = True
        while True:
            energy_range = energy_range_of(realized)
            # Replay changes interpolation tables and hence the measured
            # contributions. Recompute the moments and guards from the exact
            # transport we return; a stale pre-replay decision is insufficient.
            result = run(realized, on_block=monitor.on_block)
            if monitor.should_stop(realized):
                break
            realized = min(realized + block_electrons, n_electrons)
    return result
