"""Opt-in CPU reference: weighted BremsLib sources -> photon hits (#341).

This is a diagnostic runner, not a ``Physics`` mode.  It does not alter
``run_case`` or the existing attenuated directional bremsstrahlung scorer.
The only source in this runner is the continuous BremsLib *expected-value*
estimator; physical hard-radiative events must not also be tallied for the
same radiation.  PXR/CBS and primary characteristic radiation are excluded.

All geometry and flight coordinates are sample-frame Angstroms.  Cross
sections have their existing BremsLib convention.  Counts are integrated
over the physical aperture, per primary electron and per eV (NOT per sr).

Validation: photon-continuous-source-scoring (reference implementation).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..transport.photon_coupling import ChargedStage, CoupledHistories
    from ..transport.photon_histories import PhotonAperture, PhotonHistories
    from ..transport.photon_navigation import PhotonBox
    from ..transport.photon_sources import WeightedPhotonSources


@dataclass(frozen=True, slots=True)
class PhotonContinuumScore:
    """Aperture-integrated spectra and the underlying weighted histories.

    ``total_per_primary_eV = direct_per_primary_eV + redistributed_per_primary_eV``.
    Direct means no collision and no charged-stage re-emission. Redistributed
    means one or more collisions (or charged-stage re-emission): it includes
    scattered photons and fluorescence, not just scattered source photons.
    ``unbinned_detected_per_primary`` counts hits outside the supplied bins;
    the energy grid may intentionally be a restricted region of interest.

    ``histories.ledger`` is the energy account of weighted *scoring* photons,
    not the physical energy loss of the original transport electrons.
    """

    sources: "WeightedPhotonSources"
    histories: "PhotonHistories | CoupledHistories"
    edges_eV: np.ndarray
    total_per_primary_eV: np.ndarray
    direct_per_primary_eV: np.ndarray
    redistributed_per_primary_eV: np.ndarray
    detected_per_primary: float
    unbinned_detected_per_primary: float
    n_primaries: int
    n_proposals: int
    charged_feedback: bool


def _edges(edges_eV) -> np.ndarray:
    edges = np.asarray(edges_eV, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("detector energy-bin edges must be finite and strictly increasing")
    return edges.copy()


def _score_exits(histories, edges: np.ndarray, n_primaries: int):
    """Split ordinary physical aperture hits, using one common normalization.

    ``round`` is present only for charged-coupled histories; a photon made by
    a secondary electron can be generation zero in its *new* photon queue,
    but must never be reported as a direct photon from the primary source.

    Source: weighted histogram integration (ADR-0020). Each hit contributes
    w/(N_primary * bin_width), or w/N_primary to the outside-band count.
    Assumes an absorbing aperture and positive source weights; no hits gives
    zero in every tally. Validation: photon-continuous-source-scoring.
    """
    from ..transport.photon_histories import detected_spectrum

    exits = histories.exits
    hit = np.asarray(exits["detected"], dtype=bool)
    energy = np.asarray(exits["energy_eV"], dtype=float)
    weight = np.asarray(exits["weight"], dtype=float)
    direct = hit & (np.asarray(exits["generation"]) == 0)
    if "round" in exits:
        direct &= np.asarray(exits["round"]) == 0
    norm = n_primaries * np.diff(edges)
    total = detected_spectrum(histories, edges, n_primaries)
    direct_counts, _ = np.histogram(energy[direct], edges, weights=weight[direct])
    other = hit & ~direct
    other_counts, _ = np.histogram(energy[other], edges, weights=weight[other])
    # Use exactly the official scorer for the total; the other categories
    # partition hits, so a failure of this invariant is a scoring bug.
    direct_spec = direct_counts / norm
    other_spec = other_counts / norm
    if not np.allclose(total, direct_spec + other_spec, atol=1e-12, rtol=1e-12):
        raise RuntimeError("photon hit categories do not partition the detector score")
    total_hits = float(np.sum(weight[hit], dtype=np.float64)) / n_primaries
    # Sum the outside hits directly: total minus binned loses small weights
    # when an in-band source history carries a much larger importance weight.
    outside = hit & ((energy < edges[0]) | (energy > edges[-1]))
    unbinned = float(np.sum(weight[outside], dtype=np.float64)) / n_primaries
    return total, direct_spec, other_spec, total_hits, unbinned


def score_bremslib_continuum(
    segments: "Mapping",
    layers: "Sequence",
    tables: "Mapping",
    aperture: "PhotonAperture",
    edges_eV,
    *,
    n_primaries: int,
    n_samples: int,
    seed: int,
    photon_cutoff_eV: float,
    source_ceiling_eV: float,
    electron_cutoff_eV: float | None = None,
    charged_stage: "ChargedStage | None" = None,
    boxes: "Sequence[PhotonBox]" = (),
    width_ang: float | None = None,
    height_ang: float | None = None,
    segment_layer=None,
    max_generations: int = 1000,
    max_rounds: int = 64,
    source_kind: int = 0,
) -> PhotonContinuumScore:
    """Run the complete *reference* weighted photon source and ordinary-hit tally.

    The source samples each electron segment's unattenuated BremsLib DDCS
    over the entire sphere and full requested photon-energy interval, then
    follows its descendants through layers and finite filter boxes.  All
    transport uses lineage-keyed streams; the separate source proposal uses
    ``seed`` as a local Philox seed.  There is no second Beer--Lambert pass.

    The caller owns a previously computed charged trajectory and must use an
    appropriate radiative-loss model for it: this estimator does not recoil
    or debit that trajectory.  ``n_primaries`` is the original electron
    population, never the number of photon proposals or detected hits.

    ``charged_stage=None`` is the explicit photon-only diagnostic: charged
    daughters are *recorded* but not transported, and an electron cutoff must
    be supplied.  With ``charged_stage``, photon-born electron feedback is
    followed by ``track_coupled_histories``; its own threshold sets the charged
    cutoff. Positrons presently remain pending, so this is not a validated
    complete MeV shower or a production simulation mode.

    ``layers`` are the actual contiguous/gapped photon scene and provide one
    composition per electron segment layer.  For multilayers the segment's
    ``layer`` index from transport is used unless ``segment_layer`` is
    explicitly given.  Source histories are *never* summed with physical
    hard-photon event histograms.  No PXR/CBS source conversion is made.

    Validation: photon-continuous-source-scoring.
    """
    from ..transport.photon_coupling import track_coupled_histories
    from ..transport.photon_histories import track_photon_histories
    from ..transport.photon_sources import sample_bremslib_continuum

    edges = _edges(edges_eV)
    if aperture is None:
        raise ValueError("an explicit physical aperture is required for ordinary-hit scoring")
    if not isinstance(n_primaries, int) or n_primaries <= 0:
        raise ValueError("n_primaries must be a positive integer")
    if len(layers) == 0:
        raise ValueError("bremsstrahlung sources require at least one material layer")
    if charged_stage is None:
        if electron_cutoff_eV is None:
            raise ValueError("photon-only scoring requires electron_cutoff_eV")
        if max_rounds != 64:
            raise ValueError("max_rounds requires a charged_stage")
    elif electron_cutoff_eV is not None:
        raise ValueError("charged_stage owns its electron threshold; omit electron_cutoff_eV")

    if segment_layer is None and len(layers) > 1:
        if "layer" not in segments:
            raise ValueError("multilayer source requires segments['layer'] or segment_layer")
        segment_layer = segments["layer"]

    sources = sample_bremslib_continuum(
        segments,
        [layer[2] for layer in layers],
        tables,
        n_samples=n_samples,
        n_primaries=n_primaries,
        seed=seed,
        cutoff_eV=photon_cutoff_eV,
        ceiling_eV=source_ceiling_eV,
        segment_layer=segment_layer,
        source_kind=source_kind,
    )
    if charged_stage is None:
        assert electron_cutoff_eV is not None  # checked before sampling
        histories = track_photon_histories(
            *sources.track_args,
            layers,
            seed=seed,
            photon_cutoff_eV=photon_cutoff_eV,
            electron_cutoff_eV=electron_cutoff_eV,
            source_kind=sources.source_kind,
            aperture=aperture,
            boxes=boxes,
            width_ang=width_ang,
            height_ang=height_ang,
            max_generations=max_generations,
        )
    else:
        histories = track_coupled_histories(
            *sources.track_args,
            layers,
            charged_stage=charged_stage,
            max_rounds=max_rounds,
            seed=seed,
            photon_cutoff_eV=photon_cutoff_eV,
            source_kind=sources.source_kind,
            aperture=aperture,
            boxes=boxes,
            width_ang=width_ang,
            height_ang=height_ang,
            max_generations=max_generations,
        )
    total, direct, other, hits, outside = _score_exits(histories, edges, n_primaries)
    return PhotonContinuumScore(
        sources,
        histories,
        edges,
        total,
        direct,
        other,
        hits,
        outside,
        n_primaries,
        n_samples,
        charged_stage is not None,
    )
