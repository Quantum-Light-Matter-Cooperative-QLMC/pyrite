"""One physical-detector observation of one transport case, before and after transport.

Every producer -- :func:`pyrite.simulate`, :func:`pyrite.runs.observe.produce_observation`,
and ``pyrite run`` sweeps -- goes through this module, so pixel sampling,
factor assembly, and layered identity have exactly one implementation.

Before transport a plan knows its pixel rays, angular tiles, and the
sample-frame directions the runner must evaluate, and can recognise a stored
observation whose true-spatial factors it would reproduce. After transport it
assembles the factorized :class:`~pyrite.results.SpatialResult` and the
persistable :class:`~pyrite.observations.StoredObservation` from the runner's
directional output.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from ..instrument import (
    FilterPlate,
    PlanarDetector,
    ResolvedObservation,
    observation_identity,
)
from ..instrument.attenuation import attenuation_matrix
from ..instrument.geometry import (
    PixelRays,
    angular_tiles,
    filter_path_lengths,
    planar_detector_rays,
)
from ..instrument.observation import payload_digest, true_spatial_payload
from ..montecarlo.geometry import directions_to_sample_frame
from ..results.model import PixelRayMap, SpatialResult, SpectralFactors
from ..results.store import DEFAULT_BUNCH_CHARGE_PC, DEFAULT_REP_RATE_HZ
from .store import ObservationStore, ObservationStoreError, StoredObservation


@dataclass(frozen=True)
class PixelSampling:
    """Exact pixel rays and their coarse angular tiles for one detector.

    Parameters
    ----------
    rays
        Centre ray, distance, and solid angle of every pixel.
    tile_index
        ``(ny, nx)`` pixel-to-tile map.
    directions_lab
        ``(n_tile, 3)`` solid-angle-weighted representative lab directions.
    """

    rays: PixelRays
    tile_index: np.ndarray
    directions_lab: np.ndarray

    @classmethod
    def of(cls, detector: PlanarDetector, angular_shape: tuple[int, int]) -> PixelSampling:
        """Sample ``detector`` on an ``angular_shape`` grid of representative directions."""
        rays = planar_detector_rays(detector)
        tile_index, directions_lab = angular_tiles(rays, angular_shape)
        return cls(rays, tile_index, directions_lab)

    def directions_sample(self, *, tilt_deg: float, tilt_azim_deg: float) -> np.ndarray:
        """Representative directions in the sample frame of a tilted target."""
        return directions_to_sample_frame(
            self.directions_lab, np.deg2rad(tilt_deg), np.deg2rad(tilt_azim_deg)
        )

    def spatial(
        self,
        output: Mapping[str, Any],
        *,
        detector: PlanarDetector,
        filters: tuple[FilterPlate, ...],
        brem_source: str = "mc",
    ) -> SpatialResult:
        """Assemble factorized spatial data from runner directional output.

        ``output`` is :func:`~pyrite.montecarlo.runner.run_case_directions`
        output or a scalar run's ``out["directional"]``. A non-``"mc"``
        ``brem_source`` zeroes the continuum, as the scalar path does.
        """
        energy = np.asarray(output["E_grid"])
        background_energy = np.asarray(output["E_grid_brem"])
        line_mu = attenuation_matrix(filters, energy)
        background_mu = attenuation_matrix(filters, background_energy)
        background = np.asarray(output["brem_wide_by_direction"])
        if brem_source != "mc":
            background = np.zeros_like(background)
        return SpatialResult(
            ray_map=PixelRayMap(
                tile_index=self.tile_index,
                solid_angle_sr=self.rays.solid_angle_sr,
                path_length_mm=filter_path_lengths(self.rays, filters),
            ),
            line=SpectralFactors(energy, np.asarray(output["spec_by_direction"]), line_mu),
            background=SpectralFactors(background_energy, background, background_mu),
            detector=detector,
            coherent_line=(
                None
                if "spec_coherent_by_direction" not in output
                else SpectralFactors(
                    energy, np.asarray(output["spec_coherent_by_direction"]), line_mu
                )
            ),
            characteristic_line=SpectralFactors(
                energy, np.asarray(output["spec_characteristic_by_direction"]), line_mu
            ),
            tile_directions_lab=self.directions_lab,
        )


@dataclass(frozen=True)
class ObservationPlan:
    """A counting observation of one source case.

    Parameters
    ----------
    observation
        Physical detector, response, scorer, ordered filters, and acquisition.
    source_identity_digest
        Content key of the transport case the observation is produced from.
    tilt_deg, tilt_azim_deg
        Target orientation of that case, which fixes the sample-frame directions.
    rep_rate_hz, bunch_charge_pc
        Beam cadence and charge for expected-count normalization.
    emission
        Source emission policy; selects the default line component.
    brem_source
        ``"mc"`` keeps the Monte Carlo continuum; anything else zeroes it.
    """

    observation: ResolvedObservation
    source_identity_digest: str
    tilt_deg: float
    tilt_azim_deg: float
    rep_rate_hz: float
    bunch_charge_pc: float
    emission: str
    brem_source: str = "mc"
    #: Pixel rays and angular tiles of the detector. Derived when omitted; a
    #: sweep passes one shared instance because it depends only on the
    #: detector and scorer, never on the case.
    sampling: PixelSampling | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.sampling is None:
            object.__setattr__(
                self,
                "sampling",
                PixelSampling.of(self.observation.detector, self.observation.scorer.angular_shape),
            )

    @property
    def _sampling(self) -> PixelSampling:
        assert self.sampling is not None
        return self.sampling

    @property
    def directions_sample(self) -> np.ndarray:
        """Sample-frame directions the runner evaluates for this case."""
        return self._sampling.directions_sample(
            tilt_deg=self.tilt_deg, tilt_azim_deg=self.tilt_azim_deg
        )

    def _true_payload(self, attenuation: tuple[np.ndarray, ...]) -> dict[str, Any]:
        return true_spatial_payload(
            self.source_identity_digest,
            self.observation,
            representative_directions_lab=self._sampling.directions_lab,
            attenuation_arrays=attenuation,
        )

    def find_reusable(self, store: ObservationStore) -> StoredObservation | None:
        """Return a stored observation this plan would reproduce without transport.

        A candidate indexed under this source matches when its true-spatial
        digest equals the one recomputed from this plan's geometry, filters,
        scorer, representative directions, and attenuation on the candidate's
        own stored energy grids. The match is rescored with this plan's
        response, acquisition, and normalization. Damaged candidates are
        skipped, so the caller's transported ``put`` repairs them.
        """
        filters = self.observation.filters
        for digest in store.digests(self.source_identity_digest):
            try:
                stored = store.load(digest)
            except ObservationStoreError:
                continue
            spatial = stored.spatial
            payload = self._true_payload(
                (
                    attenuation_matrix(filters, spatial.line.energy_eV),
                    attenuation_matrix(filters, spatial.background.energy_eV),
                )
            )
            if payload_digest(payload) == stored.identity.true_spatial_digest:
                return stored.rescore(
                    acquisition=self.observation.acquisition,
                    response=self.observation.detector.response,
                    rep_rate_hz=self.rep_rate_hz,
                    bunch_charge_pc=self.bunch_charge_pc,
                )
        return None

    def assemble(
        self, output: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> StoredObservation:
        """Build the persistable observation from runner directional output.

        ``provenance`` holds JSON-compatible software/model provenance of the
        producing run.
        """
        observation = self.observation
        spatial = self._sampling.spatial(
            output,
            detector=observation.detector,
            filters=observation.filters,
            brem_source=self.brem_source,
        )
        identity = observation_identity(
            self.source_identity_digest,
            observation,
            rep_rate_hz=self.rep_rate_hz,
            bunch_charge_pc=self.bunch_charge_pc,
            representative_directions_lab=self._sampling.directions_lab,
            attenuation_arrays=(
                spatial.line.mu_by_filter_inv_mm,
                spatial.background.mu_by_filter_inv_mm,
            ),
        )
        return StoredObservation(
            identity=identity,
            spatial=spatial,
            acquisition=observation.acquisition,
            rep_rate_hz=float(self.rep_rate_hz),
            bunch_charge_pc=float(self.bunch_charge_pc),
            emission=self.emission,
            provenance=provenance,
        )


@dataclass(frozen=True)
class SweepObservation:
    """The counting observation a profile sweep produces for each of its cases.

    Parameters
    ----------
    observation
        Resolved physical detector, response, scorer, filters, and acquisition.
    store
        Destination store for this sweep's dataset stem.
    content_key_fn
        The sweep's own case content key, so every observation is indexed under
        the key its intrinsic checkpoint record is stored under.
    emission
        Profile emission policy.
    provenance_fn
        ``case -> mapping`` of JSON-compatible software/model provenance.
    """

    observation: ResolvedObservation
    store: ObservationStore
    content_key_fn: Callable[[Mapping[str, Any]], str]
    emission: str
    provenance_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    @cached_property
    def sampling(self) -> PixelSampling:
        """Pixel sampling shared by every case of the sweep."""
        return PixelSampling.of(self.observation.detector, self.observation.scorer.angular_shape)

    def plan(self, case: Mapping[str, Any]) -> ObservationPlan:
        """Plan this observation for one sweep case."""
        return ObservationPlan(
            self.observation,
            source_identity_digest=self.content_key_fn(case),
            tilt_deg=float(case.get("tilt_deg", 0.0)),
            tilt_azim_deg=float(case.get("tilt_azim_deg", 0.0)),
            rep_rate_hz=float(case.get("rep_rate_hz", DEFAULT_REP_RATE_HZ)),
            bunch_charge_pc=float(case.get("bunch_charge_pc", DEFAULT_BUNCH_CHARGE_PC)),
            emission=self.emission,
            sampling=self.sampling,
        )


__all__ = ["ObservationPlan", "PixelSampling", "SweepObservation"]
