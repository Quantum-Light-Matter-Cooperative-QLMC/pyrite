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
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

#: Case keys whose spectra need every segment at once.
STREAM_REFUSED_KEYS = ("coherent_emission", "temporal_profile", "photon_transport")

#: Further case keys live streaming refuses in its first version (#611): a
#: population-resolved line grid, adaptive electron counts, the channeling zone
#: join, GDF beams, secondary/pair/positron cascades, and transition radiation
#: (electron-block transport refuses it; artifact replay still streams it).
LIVE_STREAM_REFUSED_KEYS = (
    "transition_radiation",
    "line_grid_policy",
    "_diagnostic_line_grid",
    "adaptive_precision",
    "channeling",
    "gdf_source",
    "secondary_threshold_eV",
    "pair_production_model",
    "positron_transport",
)


def streaming_refusal(case: Mapping[str, Any], *, live: bool = False) -> str | None:
    """Why ``case`` cannot be scored in whole-electron blocks, or ``None``."""
    for key in STREAM_REFUSED_KEYS:
        if case.get(key):
            return f"{key} needs every segment at once"
    transition = case.get("transition_radiation")
    if transition is not None and transition["coherence"] == "bunch":
        return "bunch transition radiation needs every history at once"
    if live:
        for key in LIVE_STREAM_REFUSED_KEYS:
            if case.get(key) is not None:
                return f"{key} is not supported by live streamed scoring"
    return None


class BlockSpectrumScorer:
    """Accumulate one case's additive spectra over whole-electron blocks.

    ``inputs`` holds the spectrum inputs: ``E_grid`` (final), ``E_brem``,
    ``n_hat``, ``groove``, ``Ne_lines`` and ``Ne_brem``. Every block passed to
    :meth:`add` must hold complete electrons with global electron indices;
    rows of electrons at or past an estimator's population are ignored by that
    estimator, exactly as in the joined scoring.
    """

    def __init__(self, case: Mapping[str, Any], inputs: Mapping[str, Any]):
        from .cxr_grid import cxr_truncation_audit
        from .emission_adapters import select_emission

        self.case = case
        self.emission = select_emission(case)
        self.emission.preflight(case, streaming=True)
        self.E_grid, self.E_brem = inputs["E_grid"], inputs["E_brem"]
        self.n_hat, self.groove = inputs["n_hat"], inputs["groove"]
        self.Ne_lines, self.Ne_brem = inputs["Ne_lines"], inputs["Ne_brem"]
        self.audit = cxr_truncation_audit(case, self.E_grid, n_electrons=self.Ne_lines)
        self.table_cache: dict = {}
        self.totals: tuple | None = None
        self.n_segments = 0

    def add(self, segs: Mapping[str, Any]) -> None:
        """Score one whole-electron block and add it to the running totals."""
        from . import _segments_on_device
        from .emission_adapters import EmissionBatch

        rows = int(segs["L_ang"].shape[0])
        self.n_segments += rows
        if rows == 0:
            return
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
        )
        brem_batch = replace(batch, electron_limit=self.Ne_brem)
        parts = (
            self.emission.cxr.evaluate(batch, "spec").values,
            self.emission.characteristic.evaluate(brem_batch, "spec_characteristic").values,
            self.emission.continuum.evaluate(
                replace(brem_batch, energy_eV=self.E_brem), "brem_wide"
            ).values,
            self._transition(segs),
        )
        if self.totals is None:
            self.totals = parts
        else:
            self.totals = tuple(
                total + part for total, part in zip(self.totals, parts, strict=True)
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
        """The spectrum-phase output from the accumulated totals and counts."""
        from .cxr_grid import check_cxr_truncation

        case, E_grid, E_brem = self.case, self.E_grid, self.E_brem
        if self.totals is None:  # no segment at all: every estimator is zero
            spec, spec_characteristic = np.zeros(E_grid.shape), np.zeros(E_grid.shape)
            brem_wide, spec_transition = np.zeros(E_brem.shape), None
        else:
            spec, spec_characteristic, brem_wide, spec_transition = self.totals
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
        return out


def run_case_streamed(
    case,
    block_electrons: int,
    *,
    transport_core: str = "auto",
) -> dict[str, Any]:
    """Opt-in live streamed form of :func:`.run_case` for additive spectra (#611).

    Transports ``[0, max(Ne, Ne_brem))`` in electron blocks of
    ``block_electrons`` on the per-electron/CUDA core, scores each block as soon
    as it finishes and releases it. The spectra equal :func:`.run_case`'s up to
    floating-point summation order; transport counts are exact. Cases outside
    :func:`streaming_refusal` (``live=True``) raise ``ValueError`` before any
    transport.
    """
    from ..._backend import BACKEND
    from ..spectrum.cxr._coverage import collect_line_data_coverage
    from . import _transport_case
    from .artifacts import _case_geometry
    from .oom import _ensure_pool_limit

    reason = streaming_refusal(case, live=True)
    if reason is not None:
        raise ValueError(f"cannot stream this case's spectra: {reason}")
    E_grid, E_brem, _, n_hat, groove = _case_geometry(case)
    inputs = dict(
        E_grid=E_grid,
        E_brem=E_brem,
        n_hat=n_hat,
        groove=groove,
        Ne_lines=case["Ne"],
        Ne_brem=case["Ne_brem"],
    )
    scorer = BlockSpectrumScorer(case, inputs)
    # Same allocator cap as the joined spectrum phase.
    _ensure_pool_limit()

    def consume(block):
        scorer.add(block)
        # Return the finished block's cached device blocks: per-block staging
        # and scoring sizes vary, so a caching pool would otherwise ratchet
        # reserved VRAM with the electron count (#611, job 2937).
        BACKEND.release_memory()

    with collect_line_data_coverage() as line_coverage:
        tp = _transport_case(
            case,
            transport_core=transport_core,
            block_electrons=block_electrons,
            block_consumer=consume,
        )
    counts = tp["segs"]
    return scorer.result(
        Ne=int(counts["Ne"]),
        n_backscattered=int(counts["n_backscattered"]),
        n_missed=int(counts["n_missed"]),
        line_coverage=line_coverage,
    )
