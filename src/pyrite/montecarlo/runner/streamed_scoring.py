"""Additive spectrum scoring over whole-electron blocks (#186, #611).

:class:`BlockSpectrumScorer` sums the incoherent line, characteristic,
continuum and per-history transition-radiation estimators over disjoint
whole-electron row blocks. Each is a track-length sum normalized by a fixed
electron count, so blocks add; only the floating-point summation order differs
from scoring the joined population. Artifact replay
(:func:`.artifacts.stream_spectrum_from_artifact`) feeds it blocks read from
disk; :func:`run_case_streamed` feeds it live transport blocks and keeps none of
them, so peak memory follows the block size rather than the electron count.

Live streaming skips the two population-wide host steps of the joined block
path (:func:`.block_transport.finalize_population_fields`): bunch-centroid
timing, because every eligible estimator is invariant to a per-electron time
offset (it is a common phase inside one electron's flight groups), and the
re-completion of hard-photon directions, which no eligible estimator reads.

Secondary, pair and positron cascades stream too (#923): their RNG is
lineage-addressed, so a primary block runs each history's whole cascade
exactly as the joined run does. Pair cascades still complete hard-photon
directions inside each block, since conversion consumes them.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np

from ..._physics_fallback import FallbackAudit
from .fallbacks import measure_block_fallbacks

#: Case keys whose spectra need every segment at once. A measured reflection
#: selection (#1173) measures shell yields over every segment before it stops.
STREAM_REFUSED_KEYS = (
    "coherent_emission",
    "temporal_profile",
    "photon_transport",
    "reflection_selection",
)

#: Further case keys live streaming refuses (#611): the diagnostic line-grid
#: derivation, the channeling zone join, GDF beams, and transition radiation
#: (electron-block transport refuses it; artifact replay still streams it).
#: Cascades stream since #923; non-windowed ``line_grid_policy`` grids since
#: #1001; adaptive electron counts since #1280 (:func:`adaptive_refusal`).
LIVE_STREAM_REFUSED_KEYS = (
    "transition_radiation",
    "_diagnostic_line_grid",
    "channeling",
    "gdf_source",
)


def adaptive_refusal(case: Mapping[str, Any]) -> str | None:
    """Why an adaptive ``case`` keeps the retained route, or ``None`` (#1280).

    Streamed adaptive scoring is the fixed case at the realized pair. Automatic
    reflection selection streams on the electron prefix (#1281,
    :mod:`.streamed_selection`); per-bin batch means read every realized
    segment and keep the retained route.
    """
    precision = case.get("adaptive_precision")
    if precision is None:
        return None
    if precision.get("batch_means_band_eV") is not None:
        return "adaptive batch means need every segment at once"
    return None


def _streaming_case(case: Mapping[str, Any]) -> Mapping[str, Any]:
    """``case`` without its automatic selection, which the streamed scorer applies."""
    if (case.get("reflection_selection") or {}).get("automatic"):
        return {key: value for key, value in case.items() if key != "reflection_selection"}
    return case


def streaming_refusal(case: Mapping[str, Any], *, live: bool = False) -> str | None:
    """Why ``case`` cannot be scored in whole-electron blocks, or ``None``."""
    for key in STREAM_REFUSED_KEYS:
        if key == "reflection_selection" and (case.get(key) or {}).get("automatic"):
            continue
        if case.get(key):
            return f"{key} needs every segment at once"
    transition = case.get("transition_radiation")
    if transition is not None and transition["coherence"] == "bunch":
        return "bunch transition radiation needs every history at once"
    if live:
        for key in LIVE_STREAM_REFUSED_KEYS:
            if case.get(key) is not None:
                return f"{key} is not supported by live streamed scoring"
        if (reason := adaptive_refusal(case)) is not None:
            return reason
        from .streamed_line_grid import policy_streaming_refusal

        return policy_streaming_refusal(case)
    return None


class BlockSpectrumScorer:
    """Accumulate one case's additive spectra over whole-electron blocks.

    ``inputs`` holds the spectrum inputs: ``E_grid`` (final), ``E_brem``,
    ``n_hat``, ``groove``, ``Ne_lines`` and ``Ne_brem``. Every block passed to
    :meth:`add` must hold complete electrons with global electron indices;
    rows of electrons at or past an estimator's population are ignored by that
    estimator, exactly as in the joined scoring.

    ``selection`` (``(pruned_case, record)``, #1281) scores the lines of an
    automatic selection's set in admitted reflection chunks, and the continuum
    with that set's lattice share removed, as the retained route does.
    """

    def __init__(
        self,
        case: Mapping[str, Any],
        inputs: Mapping[str, Any],
        selection: tuple[Mapping[str, Any], dict] | None = None,
    ):
        from .cxr_grid import cxr_truncation_audit
        from .emission_adapters import select_emission
        from .lattice_share import continuum_case
        from .line_roulette import line_screen_for
        from .streamed_recovery import EscapeScoringRecovery, RowScoringRecovery

        self.case = _streaming_case(case)
        self.selection = None if selection is None else selection[1]
        self.line_case = self.case if selection is None else _streaming_case(selection[0])
        # The continuum removes the selected set's lattice share, as retained does.
        self.brem_case = self.case if selection is None else continuum_case(case, self.selection)
        case = self.case
        self.fallbacks = FallbackAudit(
            case.get("allow_physics_fallback", False), case.get("physics_fallbacks", ())
        )
        self.row_recovery = RowScoringRecovery()
        self.escape_recovery = EscapeScoringRecovery()
        self._restage_continuum = False
        self.emission = select_emission(case)
        self.emission.preflight(case, streaming=True)
        self.E_grid, self.E_brem = inputs["E_grid"], inputs["E_brem"]
        self.n_hat, self.groove = inputs["n_hat"], inputs["groove"]
        self.Ne_lines, self.Ne_brem = inputs["Ne_lines"], inputs["Ne_brem"]
        self.audit = cxr_truncation_audit(case, self.E_grid, n_electrons=self.Ne_lines)
        self.line_screen = line_screen_for(case, self.Ne_lines)
        self.table_cache: dict = {}
        from .diffracted_brem import DiffractedDepthAccumulator

        self.diffracted = DiffractedDepthAccumulator(case, self.E_grid, self.n_hat)
        self.totals: tuple | None = None
        self.n_segments = 0
        self.n_layers = 1

    @measure_block_fallbacks
    def add(self, segs: Mapping[str, Any]) -> None:
        """Score one whole-electron block and add it to the running totals.

        Validation: diffracted-characteristic-radiation
        """
        from . import _segments_on_device
        from .emission_adapters import EmissionBatch
        from .streamed_recovery import stage_continuum_rows

        rows = int(segs["L_ang"].shape[0])
        if rows == 0:
            return
        self.n_layers = int(segs.get("n_layers", 1))
        case = self.case
        block = _segments_on_device(segs)
        batch = EmissionBatch(
            block,
            self.E_grid,
            case,
            self.n_hat,
            case.get("abs_layers"),
            self.groove,
            electron_limit=self.Ne_lines,
            table_cache=self.table_cache,
            truncation_audit=self.audit,
            line_screen=self.line_screen,
        )
        brem_batch = replace(batch, electron_limit=self.Ne_brem)
        spec = self._lines(batch)
        # Diffracted brem (#1166) and Kossel lines (#1176) are track-length sums
        # over a fixed electron count: blocks add their depth weights, and the
        # two-beam mode is solved once in :meth:`result`.
        self.diffracted.add(block, self.Ne_brem)
        characteristic = self.row_recovery.evaluate(
            self.emission.characteristic, brem_batch, "spec_characteristic"
        )
        brem = None
        fallback_snapshot = self.fallbacks.snapshot()
        if not self._restage_continuum:
            from .. import runner
            from .oom import _drop_traceback_locals

            try:
                brem = self.emission.continuum.evaluate(
                    replace(
                        brem_batch,
                        case=self.brem_case,
                        energy_eV=self.E_brem,
                        segments={**block, "_stream_escape_recovery": self.escape_recovery},
                    ),
                    "brem_wide",
                ).values
            except Exception as error:
                if not runner._is_gpu_oom(error):
                    raise
                _drop_traceback_locals(error)
                error.__traceback__ = None
                self.fallbacks.restore(fallback_snapshot)
                self._restage_continuum = True
        if self._restage_continuum:
            # Drop line staging before retrying BremsLib's full-background
            # interpolation state; do not change the continuum reduction order.
            from .. import runner

            del batch, brem_batch, block
            runner.BACKEND.release_memory()
            continuum = EmissionBatch(
                {
                    **stage_continuum_rows(segs, self.Ne_brem),
                    "_stream_escape_recovery": self.escape_recovery,
                },
                self.E_brem,
                self.brem_case,
                self.n_hat,
                case.get("abs_layers"),
                self.groove,
                electron_limit=self.Ne_brem,
            )
            brem = self.emission.continuum.evaluate(continuum, "brem_wide").values
        parts = (spec, characteristic, brem, self._transition(segs))
        if self.totals is None:  # FP64 block sums, like the retained route (#1313)
            self.totals = tuple(np.asarray(part, dtype=np.float64) for part in parts)
        else:
            self.totals = tuple(
                total + part for total, part in zip(self.totals, parts, strict=True)
            )

        self.n_segments += rows

    def _lines(self, batch):
        """The block's incoherent lines; a selected set scores in admitted chunks.

        Both routes return host arrays, like the other block totals: the
        result adds host diffracted spectra to them (#1281, CUDA job 13652).
        """
        if self.selection is None:
            return self.emission.cxr.evaluate(batch, "spec").values
        from .reflection_cutoff import _selected_spectrum

        evaluate = self.emission.cxr.evaluate
        return _selected_spectrum(
            self.line_case,
            lambda pilot: evaluate(replace(batch, case=pilot, table_cache={}), "spec").values,
            self.E_grid,
        )

    def _transition(self, segs):
        """Per-history TR of one block, normalized by ``Ne_brem``; zeros when off."""
        if self.case.get("transition_radiation") is None:
            # Off: skip the optical-response import (optional dependencies).
            return np.zeros(self.E_grid.shape)
        from .transition_emission import block_transition_spectrum

        return block_transition_spectrum(self.case, segs, self.E_grid, self.n_hat, self.Ne_brem)

    def result(
        self,
        *,
        Ne: int,
        n_backscattered: int,
        n_missed: int,
        line_coverage,
        diagnostic_grid=None,
    ) -> dict[str, Any]:
        """The spectrum-phase output from the accumulated totals and counts.

        Validation: diffracted-bremsstrahlung, diffracted-characteristic-radiation
        """
        from .cxr_grid import check_cxr_truncation

        case, E_grid, E_brem = self.case, self.E_grid, self.E_brem
        if self.totals is None:  # no segment at all: every estimator is zero
            spec, spec_characteristic = np.zeros(E_grid.shape), np.zeros(E_grid.shape)
            brem_wide, spec_transition = np.zeros(E_brem.shape), None
        else:
            spec, spec_characteristic, brem_wide, spec_transition = self.totals
        dbr, kossel = self.diffracted.spectra()
        spec, spec_characteristic = spec + dbr, spec_characteristic + kossel
        truncation = None if self.audit is None else check_cxr_truncation(case, self.audit)
        out = dict(
            line_data_coverage=line_coverage,
            E_grid=E_grid,
            spec=spec,
            spec_characteristic=spec_characteristic,
            brem=np.interp(E_grid, E_brem, brem_wide),
            E_grid_brem=E_brem,
            brem_wide=brem_wide,
            eta=n_backscattered / Ne,
            hit_frac=1.0 - n_missed / Ne,
            n_segments=self.n_segments,
            crystal=case["crystal"],
            E0_keV=case["E0_keV"],
        )
        if case.get("channeling") is not None:
            from .channeling_emission import rescore_channeling_spectrum

            # CR is one deterministic ensemble source per case, never a block sum.
            out["spec_channeling"] = rescore_channeling_spectrum(case, E_grid, self.n_hat)
        if case.get("transition_radiation") is not None:
            out["spec_transition"] = (
                np.zeros(E_grid.shape) if spec_transition is None else spec_transition
            )
        if diagnostic_grid is not None:
            out["line_grid_diagnostic"] = diagnostic_grid
            if case.get("line_grid_policy") is not None:
                out["line_grid_resolved"] = (
                    diagnostic_grid
                    if truncation is None
                    else {**diagnostic_grid, "truncation_audit": truncation}
                )
        if self.line_screen is not None:
            out["line_roulette"] = self.line_screen.record()
        from .lattice_share import lattice_share_outputs

        out.update(lattice_share_outputs(self.brem_case, self.n_layers))
        if self.selection is not None:
            out["reflection_selection"] = self.selection
        if events := self.fallbacks.events():
            out["physics_fallbacks"] = events
        return out


def run_case_streamed(
    case,
    block_electrons: Any,
    *,
    transport_core: str = "auto",
    spill: Any = None,
) -> dict[str, Any]:
    """Opt-in live streamed form of :func:`.run_case` for additive spectra (#611).

    Transports ``[0, max(Ne, Ne_brem))`` in electron blocks of
    ``block_electrons`` (a count, or a :class:`.stream_sizing.MemoryBlockSizer`)
    on the per-electron/CUDA core, scores each block as soon
    as it finishes and releases it. A ``line_grid_policy`` grid is first
    resolved by a transport pass of its own
    (:func:`.streamed_line_grid.resolve_streamed_line_grid`, spilling under
    ``spill``, a :class:`.line_grid_spill.GridSpillPolicy` or ``None`` for its
    default) unless the speed cache holds it. The spectra equal :func:`.run_case`'s up to
    floating-point summation order; transport counts are exact. Cases outside
    :func:`streaming_refusal` (``live=True``) raise ``ValueError`` before any
    transport.

    An adaptive case (#1280) first runs a measure-only monitored transport
    that keeps no chunk, then the fixed case at the realized ``(Ne, Ne_brem)``
    as above. Counter-addressed streams make that the population the retained
    adaptive route scores; ``adaptive_sampling`` is the monitor's record.

    Validation: adaptive-sample-size-stopping
    """
    from ..spectrum.cxr._coverage import collect_line_data_coverage
    from . import _transport_case
    from .artifacts import _case_geometry
    from .oom import _ensure_pool_limit
    from .streamed_selection import block_scorer

    reason = streaming_refusal(case, live=True)
    if reason is not None:
        raise ValueError(f"cannot stream this case's spectra: {reason}")
    sampling = None
    if case.get("adaptive_precision") is not None:
        case, sampling, transport_core = _measure_adaptive_case(case, transport_core)
    E_grid, E_brem, _, n_hat, groove = _case_geometry(case)
    grid_record = grid_pass = None
    # Same allocator cap as the joined spectrum phase.
    _ensure_pool_limit()

    sized = not isinstance(block_electrons, int)
    with block_electrons.pool_limit() if sized else nullcontext():
        if case.get("line_grid_policy") is not None:
            from .streamed_line_grid import resolve_streamed_line_grid

            # Pass 1 of 2: the grid needs the whole line population (#1001).
            E_grid, grid_record, grid_pass = resolve_streamed_line_grid(
                case,
                block_electrons,
                transport_core=transport_core,
                n_hat=n_hat,
                groove=groove,
                spill=spill,
            )
        inputs = dict(
            E_grid=E_grid,
            E_brem=E_brem,
            n_hat=n_hat,
            groove=groove,
            Ne_lines=case["Ne"],
            Ne_brem=case["Ne_brem"],
        )
        scorer = block_scorer(case, inputs)
        with collect_line_data_coverage() as line_coverage:
            tp = _transport_case(
                case,
                transport_core=transport_core,
                block_electrons=block_electrons,
                block_consumer=scorer.add,
                keep_segments_on_device=True,
            )
    counts = tp["segs"]
    out = scorer.result(
        Ne=int(counts["Ne"]),
        n_backscattered=int(counts["n_backscattered"]),
        n_missed=int(counts["n_missed"]),
        line_coverage=line_coverage,
        diagnostic_grid=grid_record,
    )
    if grid_pass is not None:
        out["_streamed_line_grid_pass"] = grid_pass
    from .recovery import RECOVERY_KEYS

    out.update({key: tp[key] for key in RECOVERY_KEYS if key in tp})
    if sampling is not None:
        out["adaptive_sampling"] = sampling
    return out


def _measure_adaptive_case(case, transport_core):
    """The realized fixed case, sampling record and core of an adaptive ``case`` (#1280).

    Runs the monitored adaptive transport measure-only: the monitor reads each
    chunk and no chunk is kept, so memory follows the chunk size.

    Validation: adaptive-sample-size-stopping
    """
    from . import _transport_case
    from .block_transport import _discard

    tp = _transport_case(
        case,
        transport_core=transport_core,
        keep_segments_on_device=True,
        block_consumer=_discard,
    )
    sampling = dict(tp["adaptive_sampling"])
    fixed = {key: value for key, value in case.items() if key != "adaptive_precision"}
    fixed.update(Ne=int(tp["Ne_lines"]), Ne_brem=int(tp["Ne_brem"]))
    return fixed, sampling, sampling["transport_core"]
