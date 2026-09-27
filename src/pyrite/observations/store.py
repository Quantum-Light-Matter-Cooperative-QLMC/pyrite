"""Immutable observation objects, their JSON records, and the mutable index.

Layout under ``<root>/<stem>/``::

    index.json                          source content key -> observation digests
    objects/true/<true_spatial_digest>.h5
    objects/obs/<observation_digest>.json

A true object holds only factorized arrays -- pixel tile map, solid angles,
filter paths, representative tile directions, per-tile intrinsic spectra, and
attenuation coefficients -- never an ``(ny, nx, energy)`` cube. It is shared by
every response/acquisition choice over the same transport and geometry. An
observation record holds the three canonical layer payloads that name it, from
which the detector response, acquisition, and beam normalization are rebuilt.

Every file is written to a sibling temporary and moved into place with
``os.replace``, in dependency order (true object, record, index), so an
interrupted run leaves at worst an unindexed object that the next ``put``
verifies and reuses. Readers verify schema, array checksums, and digests, and
raise :class:`ObservationStoreError` rather than return a corrupt observation.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np

from ..console.config import workspace_root
from ..detectors import IdealPhotonCounter, Timepix3
from ..detectors.spec import DetectorResponse
from ..instrument import (
    Acquisition,
    AcquisitionBatch,
    ObservationIdentity,
    PixelGrid,
    PlanarDetector,
    PlanarPose,
)
from ..instrument.observation import (
    acquisition_payload,
    link_identity,
    payload_digest,
    response_payload,
)
from ..results.model import PixelRayMap, Result, SpatialResult, SpectralFactors

INDEX_SCHEMA = "pyrite.observation-index.v1"
TRUE_SCHEMA = "pyrite.observation-true.v1"
RECORD_SCHEMA = "pyrite.observation-record.v1"

#: HDF5 group name -> ``SpatialResult`` attribute for each spectral factor.
_FACTORS = {
    "line": "line",
    "background": "background",
    "coherent_line": "coherent_line",
    "characteristic_line": "characteristic_line",
}
_RAY_ARRAYS = ("tile_index", "solid_angle_sr", "path_length_mm")
_FACTOR_ARRAYS = ("energy_eV", "intrinsic_by_tile", "mu_by_filter_inv_mm")
_PROVENANCE_KEYS = (
    "versions",
    "backend",
    "device",
    "stopping_model",
    "characteristic_model",
    "bremsstrahlung_model",
    "xsgen_tables",
)
_RESPONSES: dict[str, type] = {
    f"{kind.__module__}.{kind.__qualname__}": kind for kind in (IdealPhotonCounter, Timepix3)
}
_DIGEST = re.compile(r"[0-9a-f]{64}")


class ObservationStoreError(RuntimeError):
    """A stored observation is missing, corrupt, or of an unknown schema."""


def default_observation_root() -> Path:
    """Return ``<workspace>/observations``, the sibling of ``checkpoints``."""
    return workspace_root() / "observations"


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    # Same encoding as the identity payloads' attenuation/direction entries.
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"provenance value {type(value).__name__} is not JSON-serializable")


def _check_digest(name: str, digest: str) -> str:
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True)
class StoredObservation:
    """One factorized observation that can be scored without transport.

    Parameters
    ----------
    identity
        Layered digests and the canonical payloads that produced them.
    spatial
        Factorized true-spatial result whose detector carries the response.
    acquisition
        Exposure, reporting edges, cut, and realization mode.
    rep_rate_hz, bunch_charge_pc
        Beam cadence and charge used for expected-count normalization.
    emission
        Source emission policy; selects the default line component.
    provenance
        JSON-compatible software/model provenance of the producing run.
    """

    identity: ObservationIdentity
    spatial: SpatialResult
    acquisition: Acquisition
    rep_rate_hz: float
    bunch_charge_pc: float
    emission: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def digest(self) -> str:
        """Observation identity digest."""
        return self.identity.observation_digest

    @property
    def source_identity_digest(self) -> str:
        """Content key of the intrinsic source case."""
        return self.identity.payload["true_spatial"]["source_identity_digest"]

    def _components(self, components: tuple[str, ...] | None) -> tuple[str, ...]:
        if components is not None:
            return components
        line = "coherent" if self.emission == "coherent" else "line"
        return (line, "background")

    def acquire(
        self,
        *,
        pixels=None,
        region=None,
        components: tuple[str, ...] | None = None,
    ) -> AcquisitionBatch:
        """Score selected pixels with this observation's frozen acquisition."""
        return self.spatial.acquire(
            self.acquisition,
            rep_rate_hz=self.rep_rate_hz,
            bunch_charge_pc=self.bunch_charge_pc,
            observation_digest=self.digest,
            pixels=pixels,
            region=region,
            components=self._components(components),
        )

    def acquisition_image(
        self,
        energy_range_eV: tuple[float, float] | None = None,
        *,
        components: tuple[str, ...] | None = None,
        pixel_chunk: int = 1024,
    ) -> np.ndarray:
        """Return the total or reporting-window count image in bounded chunks."""
        return self.spatial.acquisition_image(
            acquisition=self.acquisition,
            rep_rate_hz=self.rep_rate_hz,
            bunch_charge_pc=self.bunch_charge_pc,
            observation_digest=self.digest,
            energy_range_eV=energy_range_eV,
            components=self._components(components),
            pixel_chunk=pixel_chunk,
        )

    def rescore(
        self,
        *,
        acquisition: Acquisition | None = None,
        response: DetectorResponse | None = None,
        rep_rate_hz: float | None = None,
        bunch_charge_pc: float | None = None,
    ) -> StoredObservation:
        """Return a new observation over the same true-spatial factors.

        Only read-time layers change: the detector response, the acquisition,
        and beam normalization. Geometry, filters, and angular sampling are
        fixed by the true factors and require a new transport run.
        """
        new_acquisition = self.acquisition if acquisition is None else acquisition
        new_rate = self.rep_rate_hz if rep_rate_hz is None else rep_rate_hz
        new_charge = self.bunch_charge_pc if bunch_charge_pc is None else bunch_charge_pc
        detector = self.spatial.detector
        new_response = detector.response if response is None else response
        identity = link_identity(
            self.identity.payload["true_spatial"],
            response_payload(new_response),
            acquisition_payload(
                new_acquisition, rep_rate_hz=new_rate, bunch_charge_pc=new_charge
            ),
        )
        spatial = (
            self.spatial
            if new_response is detector.response
            else replace(self.spatial, detector=replace(detector, response=new_response))
        )
        return StoredObservation(
            identity=identity,
            spatial=spatial,
            acquisition=new_acquisition,
            rep_rate_hz=float(new_rate),
            bunch_charge_pc=float(new_charge),
            emission=self.emission,
            provenance=self.provenance,
        )


def observation_from_result(result: Result) -> StoredObservation:
    """Extract the persistable observation from an acquisition-enabled result.

    Raises
    ------
    ValueError
        If ``result`` has no layered observation identity, spatial factors, or
        representative tile directions.
    """
    observation = result.provenance.get("observation")
    if not isinstance(observation, Mapping) or observation.get("schema") != "pyrite.observation.v2":
        raise ValueError("result has no layered (acquisition-enabled) observation identity")
    spatial = result.spatial
    if spatial is None or spatial.tile_directions_lab is None:
        raise ValueError("result has no persistable spatial factors")
    scene = result.provenance["scene"]
    identity = link_identity(
        observation["true_spatial"], observation["response"], observation["acquisition"]
    )
    if identity.observation_digest != result.provenance["observation_identity_digest"]:
        raise ValueError("result observation payload does not reproduce its digest")
    return StoredObservation(
        identity=identity,
        spatial=spatial,
        acquisition=scene.acquisition,
        rep_rate_hz=float(scene.beam.rep_rate_hz),
        bunch_charge_pc=float(scene.beam.bunch_charge_pc),
        emission=str(scene.emission),
        provenance=_jsonable(
            {key: result.provenance[key] for key in _PROVENANCE_KEYS if key in result.provenance}
        ),
    )


def _decode_detector(
    true_payload: Mapping[str, Any], response: DetectorResponse
) -> PlanarDetector:
    geometry = true_payload["detector_geometry"]
    pose = geometry["pose"]
    pixels = geometry["pixels"]
    size = geometry["size_mm"]
    return PlanarDetector(
        pose=PlanarPose(
            center_mm=tuple(pose["center_mm"]),
            normal=tuple(pose["normal"]),
            x_axis=tuple(pose["x_axis"]),
        ),
        size_mm=tuple(size) if size else None,
        pixels=PixelGrid(shape=tuple(pixels["shape"]), pitch_mm=tuple(pixels["pitch_mm"])),
        response=response,
    )


def _decode_response(payload: Mapping[str, Any]) -> DetectorResponse:
    kind = _RESPONSES.get(payload["type"])
    if kind is None:
        raise ObservationStoreError(f"unsupported stored detector response {payload['type']!r}")
    return kind(**payload["config"])


def _decode_acquisition(payload: Mapping[str, Any]) -> tuple[Acquisition, float, float]:
    config = dict(payload["config"])
    config["measured_edges_eV"] = tuple(config["measured_edges_eV"])
    normalization = payload["normalization"]
    return (
        Acquisition(**config),
        float(normalization["rep_rate_hz"]),
        float(normalization["bunch_charge_pc"]),
    )


class ObservationStore:
    """Content-addressed observation store for one dataset stem.

    Parameters
    ----------
    stem
        Dataset stem, matching the intrinsic checkpoint stem it links to.
    root
        Observation root; ``None`` uses :func:`default_observation_root`.
    """

    def __init__(self, stem: str, root: str | os.PathLike[str] | None = None) -> None:
        if not stem or stem in {".", ".."} or "/" in stem or os.sep in stem:
            raise ValueError(f"invalid observation stem {stem!r}")
        self.root = default_observation_root() if root is None else Path(root)
        self.path = self.root / stem

    # -- layout -------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.path / "index.json"

    def true_path(self, true_spatial_digest: str) -> Path:
        digest = _check_digest("true_spatial_digest", true_spatial_digest)
        return self.path / "objects" / "true" / f"{digest}.h5"

    def record_path(self, observation_digest: str) -> Path:
        digest = _check_digest("observation_digest", observation_digest)
        return self.path / "objects" / "obs" / f"{digest}.json"

    @staticmethod
    def _temporary(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")

    @contextlib.contextmanager
    def _index_lock(self) -> Iterator[None]:
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "index.lock", "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    # -- index --------------------------------------------------------------

    def _read_index(self) -> dict[str, list[str]]:
        if not self.index_path.exists():
            return {}
        try:
            index = json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ObservationStoreError(f"unreadable observation index {self.index_path}") from exc
        if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
            raise ObservationStoreError(f"unknown observation index schema in {self.index_path}")
        return {key: list(value) for key, value in index["sources"].items()}

    def sources(self) -> tuple[str, ...]:
        """Source content keys with at least one indexed observation."""
        return tuple(sorted(self._read_index()))

    def digests(self, source_identity_digest: str) -> tuple[str, ...]:
        """Indexed observation digests produced from one source case."""
        return tuple(self._read_index().get(source_identity_digest, ()))

    def _add_to_index(self, source: str, observation_digest: str) -> None:
        with self._index_lock():
            sources = self._read_index()
            entries = sources.setdefault(source, [])
            if observation_digest in entries:
                return
            entries.append(observation_digest)
            entries.sort()
            tmp = self._temporary(self.index_path)
            tmp.write_text(
                json.dumps({"schema": INDEX_SCHEMA, "sources": sources}, indent=1, sort_keys=True)
            )
            os.replace(tmp, self.index_path)

    # -- true objects -------------------------------------------------------

    def _write_true(self, observation: StoredObservation) -> None:
        spatial = observation.spatial
        path = self.true_path(observation.identity.true_spatial_digest)
        tmp = self._temporary(path)
        try:
            with h5py.File(tmp, "w") as handle:
                handle.attrs["schema"] = TRUE_SCHEMA
                handle.attrs["true_spatial_digest"] = observation.identity.true_spatial_digest
                handle.attrs["true_payload"] = json.dumps(
                    observation.identity.payload["true_spatial"], sort_keys=True
                )
                arrays: dict[str, np.ndarray] = {
                    f"ray_map/{name}": getattr(spatial.ray_map, name) for name in _RAY_ARRAYS
                }
                assert spatial.tile_directions_lab is not None
                arrays["ray_map/tile_directions_lab"] = spatial.tile_directions_lab
                for group, attribute in _FACTORS.items():
                    factor = getattr(spatial, attribute)
                    if factor is not None:
                        for name in _FACTOR_ARRAYS:
                            arrays[f"factors/{group}/{name}"] = getattr(factor, name)
                for name, array in arrays.items():
                    dataset = handle.create_dataset(name, data=np.ascontiguousarray(array))
                    dataset.attrs["sha256"] = _array_identity(array)["sha256"]
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _read_true(
        self, true_spatial_digest: str
    ) -> tuple[Mapping[str, Any], PixelRayMap, np.ndarray, dict[str, SpectralFactors]]:
        path = self.true_path(true_spatial_digest)
        if not path.exists():
            raise ObservationStoreError(f"missing true-spatial object {path}")
        try:
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("schema") != TRUE_SCHEMA:
                    raise ObservationStoreError(f"unknown true-spatial schema in {path}")
                payload = json.loads(handle.attrs["true_payload"])
                arrays: dict[str, np.ndarray] = {}

                def collect(name: str, node: object) -> None:
                    if isinstance(node, h5py.Dataset):
                        array = node[()]
                        if _array_identity(array)["sha256"] != node.attrs["sha256"]:
                            raise ObservationStoreError(f"checksum mismatch for {name} in {path}")
                        arrays[name] = array

                handle.visititems(collect)
        except ObservationStoreError:
            raise
        except (OSError, KeyError, ValueError) as exc:
            raise ObservationStoreError(f"corrupt true-spatial object {path}") from exc
        if payload_digest(payload) != true_spatial_digest:
            raise ObservationStoreError(f"true-spatial payload does not match digest for {path}")
        try:
            directions = arrays["ray_map/tile_directions_lab"]
            ray_map = PixelRayMap(*(arrays[f"ray_map/{name}"] for name in _RAY_ARRAYS))
            factors = {
                group: SpectralFactors(
                    *(arrays[f"factors/{group}/{name}"] for name in _FACTOR_ARRAYS)
                )
                for group in _FACTORS
                if f"factors/{group}/energy_eV" in arrays
            }
            identities = [
                _array_identity(factors["line"].mu_by_filter_inv_mm),
                _array_identity(factors["background"].mu_by_filter_inv_mm),
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservationStoreError(f"incomplete true-spatial object {path}") from exc
        if (
            identities != payload["attenuation_arrays"]
            or _array_identity(directions) != payload["representative_directions_lab"]
        ):
            raise ObservationStoreError(f"true-spatial arrays do not match their identity in {path}")
        return payload, ray_map, directions, factors

    # -- records ------------------------------------------------------------

    def _write_record(self, observation: StoredObservation) -> None:
        path = self.record_path(observation.digest)
        record = {
            "schema": RECORD_SCHEMA,
            "observation_identity_digest": observation.digest,
            "identity": _jsonable(observation.identity.payload),
            "emission": observation.emission,
            "provenance": _jsonable(observation.provenance),
        }
        tmp = self._temporary(path)
        try:
            tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _read_record(self, observation_digest: str) -> Mapping[str, Any]:
        path = self.record_path(observation_digest)
        if not path.exists():
            raise ObservationStoreError(f"missing observation record {path}")
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ObservationStoreError(f"corrupt observation record {path}") from exc
        if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
            raise ObservationStoreError(f"unknown observation record schema in {path}")
        return record

    # -- public -------------------------------------------------------------

    def put(self, observation: StoredObservation) -> str:
        """Persist ``observation`` idempotently and index it; return its digest.

        An existing true object or record is verified and reused. A corrupt
        one, such as a file left by an interrupted writer, is replaced.
        """
        true_digest = observation.identity.true_spatial_digest
        if observation.spatial.tile_directions_lab is None:
            raise ValueError("observation spatial factors lack tile_directions_lab")
        try:
            self._read_true(true_digest)
        except ObservationStoreError:
            self._write_true(observation)
        try:
            layers = self._read_record(observation.digest)["identity"]
            valid = (
                link_identity(
                    layers["true_spatial"], layers["response"], layers["acquisition"]
                ).observation_digest
                == observation.digest
            )
        except (ObservationStoreError, KeyError, TypeError):
            valid = False
        if not valid:
            self._write_record(observation)
        self._add_to_index(observation.source_identity_digest, observation.digest)
        return observation.digest

    def load(self, observation_digest: str) -> StoredObservation:
        """Reopen one stored observation without transport."""
        record = self._read_record(observation_digest)
        layers = record["identity"]
        identity = link_identity(layers["true_spatial"], layers["response"], layers["acquisition"])
        if identity.observation_digest != observation_digest:
            raise ObservationStoreError(
                f"observation record does not reproduce digest {observation_digest}"
            )
        try:
            response = _decode_response(layers["response"])
            acquisition, rep_rate_hz, bunch_charge_pc = _decode_acquisition(layers["acquisition"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservationStoreError(
                f"undecodable observation record {observation_digest}"
            ) from exc
        if response_payload(response) != layers["response"] or acquisition_payload(
            acquisition, rep_rate_hz=rep_rate_hz, bunch_charge_pc=bunch_charge_pc
        ) != _jsonable(layers["acquisition"]):
            raise ObservationStoreError(
                f"observation record {observation_digest} does not round-trip its configuration"
            )
        _, ray_map, directions, factors = self._read_true(identity.true_spatial_digest)
        spatial = SpatialResult(
            ray_map=ray_map,
            line=factors["line"],
            background=factors["background"],
            detector=_decode_detector(layers["true_spatial"], response),
            coherent_line=factors.get("coherent_line"),
            characteristic_line=factors.get("characteristic_line"),
            tile_directions_lab=directions,
        )
        return StoredObservation(
            identity=identity,
            spatial=spatial,
            acquisition=acquisition,
            rep_rate_hz=rep_rate_hz,
            bunch_charge_pc=bunch_charge_pc,
            emission=record["emission"],
            provenance=record["provenance"],
        )

    def verify(self) -> tuple[str, ...]:
        """Return human-readable problems; empty when every indexed entry loads."""
        problems: list[str] = []
        try:
            sources = self._read_index()
        except ObservationStoreError as exc:
            return (str(exc),)
        for source, digests in sources.items():
            for digest in digests:
                try:
                    stored = self.load(digest)
                except ObservationStoreError as exc:
                    problems.append(str(exc))
                    continue
                if stored.source_identity_digest != source:
                    problems.append(f"observation {digest} is indexed under the wrong source")
        problems.extend(
            f"leftover temporary file {path}" for path in sorted(self.path.rglob(".*.tmp"))
        )
        return tuple(problems)


__all__ = [
    "INDEX_SCHEMA",
    "RECORD_SCHEMA",
    "TRUE_SCHEMA",
    "ObservationStore",
    "ObservationStoreError",
    "StoredObservation",
    "default_observation_root",
    "observation_from_result",
]
