"""Public Python simulation entry points."""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np

from . import __version__
from ._backend import BACKEND
from ._numerics import Physics
from .campaign.kinematic_validity import case_kinematic_validity
from .campaign.lowering import build_case, build_sweep_cases, lower_simulation
from .campaign.lowering import scene_physics as scene_physics
from .campaign.model import (
    Numerics,
    Scene,
)
from .campaign.profiles import (
    case_bremsstrahlung_marker,
    case_characteristic_marker,
    case_content_key,
)
from .campaign.resolution import UNSET, ResolvedSimulation, _Unset, resolve_simulation
from .detectors import Detector
from .instrument import (
    Acquisition,
    Filter,
    FilterPlate,
    NativeAcquisition,
    PixelScorer,
    PlanarDetector,
    ResolvedObservation,
)
from .materials import CATALOG, MediumSpec
from .materials._composition import resolve_composition
from .materials.photon_cross_sections import ATTENUATION_MODEL
from .montecarlo import Case, run_case
from .montecarlo.runner import cached_case_table_markers, case_table_markers, run_case_directions
from .montecarlo.spectrum import (
    BREM_ENDF_PARSERPY_VERSION,
)
from .montecarlo.transport import STOPPING_MODEL
from .observations.plan import ObservationPlan, PixelSampling
from .observations.spec import Observation
from .quantum import QuantumSetup
from .results.model import CxrSpectra, Result, SpectralDensity


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _canonical_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise TypeError(f"cannot canonicalize observation value {type(value).__name__}")


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _resolved_composition(material: str | MediumSpec) -> tuple[tuple[str, float], ...]:
    return resolve_composition(material, catalog=CATALOG)


def _observation_provenance(
    source_digest: str,
    scene: Scene,
    scorer: PixelScorer,
    coefficients: tuple[np.ndarray, ...],
) -> tuple[str, dict[str, Any]]:
    detector = scene.detector
    assert isinstance(detector, PlanarDetector)
    detector_payload = {
        "pose": _canonical_value(detector.pose),
        "size_mm": list(detector.size_mm or ()),
        "pixels": None if detector.pixels is None else _canonical_value(detector.pixels),
        "response": (
            None
            if detector.response is None
            else {
                "type": f"{type(detector.response).__module__}.{type(detector.response).__qualname__}",
                "config": _canonical_value(detector.response),
            }
        ),
    }
    filters_payload = [
        {
            "composition": _canonical_value(_resolved_composition(plate.material)),
            "thickness_mm": plate.thickness_mm,
            "size_mm": list(plate.size_mm),
            "pose": _canonical_value(plate.pose),
        }
        for plate in scene.filter_plates
    ]
    payload = {
        "schema": "pyrite.observation.v1",
        "source_identity_digest": source_digest,
        "detector": detector_payload,
        "filters": filters_payload,
        "angular_shape": list(scorer.angular_shape),
        "attenuation_arrays": [_array_identity(array) for array in coefficients],
        "approximations": {
            "emission_source": "point-source-target-reference",
            "pixel_sampling": "centre-ray",
            "filter_interactions": "primary-attenuation-only",
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@cache
def source_git_state() -> tuple[str | None, bool | None]:
    """Return ``(git_sha, git_dirty)`` of the source checkout, or ``(None, None)``.

    Only a checkout whose top level is this package's own repository counts, so
    an installed wheel sitting inside some unrelated project's git tree reports
    nulls instead of that project's revision. Cached per process: the code does
    not change under a running interpreter.
    """
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != root:
        return None, None
    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    if sha is None or status is None:
        return None, None
    return sha, bool(status)


def run_provenance(case: Case | Mapping[str, Any], xsgen_tables: Mapping[str, str] | None):
    """Software, backend, and physics-model provenance of one transport case.

    Shared by :func:`simulate` results and persisted observations, so every
    producer records the same keys. ``git_sha``/``git_dirty`` identify the
    source checkout and are ``None`` outside one (for example an installed wheel).
    """
    from .campaign.physics_view import resolved_physics_view

    git_sha, git_dirty = source_git_state()
    return {
        "resolved_physics": json.loads(
            resolved_physics_view(case, table_markers=xsgen_tables or {}).to_json()
        ),
        "kinematic_validity": case_kinematic_validity(case),
        "xsgen_tables": xsgen_tables,
        "stopping_model": STOPPING_MODEL,
        "characteristic_model": case_characteristic_marker(case),
        "attenuation_model": ATTENUATION_MODEL,
        "bremsstrahlung_model": case_bremsstrahlung_marker(case),
        "backend": BACKEND.name,
        "device": BACKEND.device,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "versions": {
            "pyrite": __version__,
            "numpy": np.__version__,
            "xraydb": _package_version("xraydb"),
            "endf-parserpy": BREM_ENDF_PARSERPY_VERSION,
        },
    }


def build_configured_cases(old_sweep: Any, settings: Any) -> list[Case]:
    """Lower a configured `campaign.sweep.Sweep`/`Settings` pair to cases.

    This is the production path `pyrite scan` and `pyrite material blaze` take,
    not a compatibility shim: `runs.scan` and `runs.blaze` both default their
    `build_cases` hook to it. The pair it accepts is the config-driven campaign
    description assembled from the `data/catalog/` material catalog, which is why the lowering
    lives behind one named function rather than in each runner.
    """
    from .campaign.legacy import adapt_configured_sweep

    return build_sweep_cases(adapt_configured_sweep(old_sweep, settings))


def simulate(
    beam: Any,
    target: Any = None,
    detector: Detector | PlanarDetector | None = None,
    *,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    filters: tuple[FilterPlate | Filter, ...] | None = None,
    acquisition: Acquisition | NativeAcquisition | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> Result:
    """Simulate one scene without reading or writing a checkpoint.

    A scalar :class:`Detector` returns response-free source photon densities
    per incident electron per eV per sr. With a :class:`PlanarDetector`, the
    scalar arrays are filter-attenuated, solid-angle-weighted observation
    averages in the same per-sr units. Selected spatial spectra include pixel
    solid angle and are accepted flux per incident electron per eV. Detector
    response remains an explicit read-time operation.

    Parameters
    ----------
    beam
        Scalar :class:`~pyrite.Beam` description, or a complete
        :class:`~pyrite.Scene`. With a scene, omit target, detector, filters
        and acquisition; physics, numerics and quantum may override its settings.
    target
        Scalar :class:`~pyrite.Target`, :class:`~pyrite.Slab` or :class:`~pyrite.Stack`.
    detector
        Scalar detector model or physical :class:`~pyrite.PlanarDetector`.
    physics
        Emission and interaction models. ``None`` uses the scene's settings,
        or :class:`~pyrite.Physics` for a component call.
    numerics
        Sampling and execution controls. ``None`` uses the scene's settings,
        or :class:`~pyrite.Numerics` for a component call.
    filters
        Ordered ``Filter`` or ``FilterPlate`` values; require a physical planar detector.
    acquisition
        Concrete reporting edges or an ``Acquisition.native(...)`` request;
        requires a physical pixel detector with native response scoring.
    quantum
        Independent :class:`~pyrite.quantum.QuantumSetup`. Omit to inherit
        ``Scene.quantum``; explicit ``None`` disables the scene's setup.
        Unsupported setups and conflicting legacy channeling fields refuse
        before transport. Component calls default to no quantum setup.
    observation
        Independent response, acquisition and numerical sampling request.
        References the scene's detector or supplies a virtual angular detector.
        Conflicting embedded legacy settings raise before transport.

    Returns
    -------
    Result
        In-memory spectra, resolved case, provenance, and optional spatial data.

    Raises
    ------
    TypeError
        If components or numerics have incompatible types.
    ValueError
        If scene geometry or requested backend is inconsistent.

    Notes
    -----
    This function performs no checkpoint I/O.
    """
    request = resolve_simulation(
        beam,
        target,
        detector,
        physics=physics,
        numerics=numerics,
        filters=filters,
        acquisition=acquisition,
        quantum=quantum,
        observation=observation,
        require_detector=True,
    )
    scene, resolved_numerics = request.scene, request.numerics
    case, xsgen_tables = _source_case(request)
    if isinstance(scene.detector, PlanarDetector):
        return _simulate_planar(request, case, xsgen_tables)
    output = run_case(case, transport_core=resolved_numerics.transport_core)
    background_energy = output.get("E_grid_brem", output["E_grid"])
    background = output.get("brem_wide", output["brem"])
    if scene.physics.brem_source != "mc":
        background = np.zeros_like(background)
    coherent = output.get("spec_coherent")
    characteristic = output.get("spec_characteristic")
    channeling = output.get("spec_channeling")
    transition = output.get("spec_transition")
    return Result(
        cxr=CxrSpectra(
            SpectralDensity(output["E_grid"], output["spec"]),
            None if coherent is None else SpectralDensity(output["E_grid"], coherent),
        ),
        brem=SpectralDensity(background_energy, background),
        characteristic=(
            None if characteristic is None else SpectralDensity(output["E_grid"], characteristic)
        ),
        channeling=(None if channeling is None else SpectralDensity(output["E_grid"], channeling)),
        transition=(None if transition is None else SpectralDensity(output["E_grid"], transition)),
        case=case,
        provenance={
            **request.composition_provenance,
            "scene": scene,
            "numerics": resolved_numerics,
            "identity_digest": case_content_key(case, xsgen_tables=xsgen_tables),
            # Automatic case-local line grids (issue #101): the policy is what
            # identity hashes, the resolved record is what the run actually
            # sampled. Both absent for an explicit or stored grid.
            **_cxr_grid_provenance(case, output),
            **(
                {"adaptive_sampling": output["adaptive_sampling"]}
                if "adaptive_sampling" in output
                else {}
            ),
            # Channeling-zone fates and angles (#506); absent when off.
            **({"channeling": output["channeling"]} if "channeling" in output else {}),
            **run_provenance(case, xsgen_tables),
        },
    )


def _source_case(request: ResolvedSimulation) -> tuple[Case, dict[str, str]]:
    """Lower a resolved request and prepare its cross-section table markers."""
    case = lower_simulation(request)
    return case, case_table_markers(case)


def source_content_key_fn():
    """Per-case source content key for a batch of cases, e.g. a sweep.

    The key :func:`simulate` records as ``identity_digest`` for the same case:
    both hash the tables the case itself reads (:func:`case_table_markers`),
    never a dataset's union of table markers (#191). Table resolution is
    memoized per call, so take one per run.
    """
    table_markers = cached_case_table_markers()

    def content_key(case: Case | Mapping[str, Any]) -> str:
        return case_content_key(case, xsgen_tables=table_markers(case))

    return content_key


def observation_plan(
    scene: Scene,
    numerics: Numerics,
    *,
    physics: Physics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> ObservationPlan:
    """Plan ``scene``'s counting observation without running transport.

    The plan carries the same source content key, sample-frame directions,
    and identity inputs :func:`simulate` uses, so a stored observation it
    recognises is exactly one ``simulate`` would produce.

    Raises
    ------
    ValueError
        If ``scene`` lacks a pixelated detector or an acquisition.
    """
    request = resolve_simulation(
        scene, numerics=numerics, physics=physics, quantum=quantum, observation=observation
    )
    scene = request.scene
    detector = scene.detector
    if not isinstance(detector, PlanarDetector) or scene.acquisition is None:
        raise ValueError("observation production requires a pixelated detector and acquisition")
    case, xsgen_tables = _source_case(request)
    return ObservationPlan(
        ResolvedObservation(
            detector=detector,
            scorer=_detector_scorer(detector),
            filters=scene.filter_plates,
            acquisition=scene.acquisition,
        ),
        source_identity_digest=case_content_key(case, xsgen_tables=xsgen_tables),
        tilt_deg=float(case.get("tilt_deg", 0.0)),
        tilt_azim_deg=float(case.get("tilt_azim_deg", 0.0)),
        rep_rate_hz=scene.beam.rep_rate_hz,
        bunch_charge_pc=scene.beam.bunch_charge_pc,
        emission=request.physics.emission_policy,
        brem_source=cast(str, request.physics.brem_source),
    )


def _detector_scorer(detector: PlanarDetector) -> PixelScorer:
    """The detector's scorer; a pixel-free detector scores one whole-face tile."""
    return PixelScorer() if detector.scorer is None else detector.scorer


def _cxr_grid_provenance(case, output) -> dict[str, object]:
    """Line-data coverage and any automatically resolved grid provenance.

    Records the policy, the per-observable tolerances, the configuration source
    of each, and the resolved coordinate set -- the provenance half of #101's
    "record resolved policy, tolerance, coordinate set, and configuration
    source". The identity half is that the policy sits in the case payload, so
    ``case_content_key`` already moves when any of it moves.
    """
    provenance: dict[str, object] = {}
    if "line_data_coverage" in output:
        provenance["line_data_coverage"] = output["line_data_coverage"]
    policy = case.get("line_grid_policy")
    if policy is None:
        return provenance
    provenance["line_grid_policy"] = policy
    resolved = output.get("line_grid_resolved")
    if resolved is not None:
        provenance["line_grid_resolved"] = resolved
    return provenance


def _simulate_planar(
    request: ResolvedSimulation,
    case: Case,
    xsgen_tables: Mapping[str, str],
) -> Result:
    scene, numerics = request.scene, request.numerics
    detector = scene.detector
    assert isinstance(detector, PlanarDetector)
    scorer = _detector_scorer(detector)
    source_digest = case_content_key(case, xsgen_tables=xsgen_tables)
    tilt_deg = float(case.get("tilt_deg", 0.0))
    tilt_azim_deg = float(case.get("tilt_azim_deg", 0.0))
    identity_provenance: dict[str, Any] = {}
    if scene.acquisition is None:
        sampling = PixelSampling.of(detector, scorer.angular_shape)
        output = run_case_directions(
            case,
            sampling.directions_sample(tilt_deg=tilt_deg, tilt_azim_deg=tilt_azim_deg),
            transport_core=numerics.transport_core,
        )
        spatial = sampling.spatial(
            output,
            detector=detector,
            filters=scene.filter_plates,
            brem_source=cast(str, scene.physics.brem_source),
            reconstruction=scorer.reconstruction,
        )
        observation_digest, observation = _observation_provenance(
            source_digest,
            scene,
            scorer,
            (spatial.cxr.mu_by_filter_inv_mm, spatial.brem.mu_by_filter_inv_mm),
        )
    else:
        plan = ObservationPlan(
            ResolvedObservation(
                detector=detector,
                scorer=scorer,
                filters=scene.filter_plates,
                acquisition=scene.acquisition,
            ),
            source_identity_digest=source_digest,
            tilt_deg=tilt_deg,
            tilt_azim_deg=tilt_azim_deg,
            rep_rate_hz=scene.beam.rep_rate_hz,
            bunch_charge_pc=scene.beam.bunch_charge_pc,
            emission=scene.physics.emission_policy,
            brem_source=cast(str, scene.physics.brem_source),
        )
        output = run_case_directions(
            case, plan.directions_sample, transport_core=numerics.transport_core
        )
        stored = plan.assemble(output, provenance={})
        spatial = stored.spatial
        observation_digest = stored.digest
        observation = dict(stored.identity.payload)
        identity_provenance = {
            **stored.provenance,
            "true_spatial_digest": stored.identity.true_spatial_digest,
            "response_digest": stored.identity.response_digest,
            "acquisition_digest": stored.identity.acquisition_digest,
        }
    energy = spatial.cxr.energy_eV
    background_energy = spatial.brem.energy_eV
    coherent = spatial.coherent_cxr
    coherent_average = None if coherent is None else spatial.average_density("coherent")
    characteristic_average = spatial.average_density("characteristic")
    channeling_average = (
        None if spatial.channeling is None else spatial.average_density("channeling")
    )
    transition_average = (
        None if spatial.transition is None else spatial.average_density("transition")
    )
    return Result(
        cxr=CxrSpectra(
            SpectralDensity(energy, spatial.average_density("cxr"), "observation-average"),
            None
            if coherent_average is None
            else SpectralDensity(energy, coherent_average, "observation-average"),
        ),
        brem=SpectralDensity(
            background_energy, spatial.average_density("brem"), "observation-average"
        ),
        characteristic=SpectralDensity(energy, characteristic_average, "observation-average"),
        channeling=(
            None
            if channeling_average is None
            else SpectralDensity(energy, channeling_average, "observation-average")
        ),
        transition=(
            None
            if transition_average is None
            else SpectralDensity(energy, transition_average, "observation-average")
        ),
        case=case,
        provenance={
            **request.composition_provenance,
            "scene": scene,
            "numerics": numerics,
            "identity_digest": source_digest,
            **_cxr_grid_provenance(case, output),
            "observation_identity_digest": observation_digest,
            "observation": observation,
            **identity_provenance,
            **(
                {
                    "photon_aperture_identity_digest": output["photon_aperture"]["metadata"][
                        "observation_identity_digest"
                    ]
                }
                if "photon_aperture" in output
                else {}
            ),
            **run_provenance(case, xsgen_tables),
        },
        spatial=(spatial if detector.pixels is not None or scene.acquisition is not None else None),
        aperture=_aperture_result(output),
    )


def _aperture_result(output):
    from .results.aperture import aperture_from_payload

    return aperture_from_payload(output.get("photon_aperture"))


__all__ = ["build_case", "build_configured_cases", "build_sweep_cases", "simulate"]
