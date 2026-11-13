"""Score the spectra of an existing transport: the second step of two (#1162).

:func:`pyrite.transport.simulate_trajectories` transports a scene once;
:func:`score_spectra` runs only the production spectrum phase (lines,
characteristic, bremsstrahlung, transition radiation, automatic line-grid
resolution) on those segments, as often as needed, under any detector,
observation, acquisition, filters or spectral grid. It never transports.
``score_spectra(simulate_trajectories(...), detector)`` is
``pyrite.simulate(...)`` for the same components, bit for bit on one backend.

A request whose lowered case differs from the transported one in any field the
transport read is refused before spectrum work, naming the field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .campaign.trajectory_product import Trajectories

if TYPE_CHECKING:
    from ._numerics import Physics
    from ._simulation_progress import SimulationProgress
    from .campaign.model import Numerics
    from .detectors import Detector, PlanarDetector
    from .instrument import Acquisition, FilterPlate, NativeAcquisition
    from .instrument.filter import Filter
    from .observations.spec import Observation
    from .results.model import Result

__all__ = ["ScoringRefusedError", "score_spectra"]

#: Case fields only the spectrum phase reads; they may differ from the transport.
SCORING_ONLY_FIELDS = frozenset(
    {
        "name",
        "E_grid",
        "E_grid_line",
        "E_grid_brem",
        "line_grid_policy",
        "line_quadrature",
        "sinc_cutoff",
        "brem_step_eV",
        "coherent_flat_omission_limit",
        "spec_chunk",
        "brem_chunk",
        "hkl_list",
        "coherent_emission",
        "temporal_profile",
        "line_roulette",
        "theta_obs_rad",
        "dtheta_obs_rad",
        "domega_sr",
    }
)
#: Detector placement fields; fixed for coherent and temporal scoring.
_DETECTOR_FIELDS = ("theta_obs_rad", "dtheta_obs_rad", "domega_sr")

#: User-facing owner of the case fields most often refused.
_FIELD_OWNERS = {
    "seed": "numerics.seed",
    "Ne": "numerics.sampling (line history population)",
    "Ne_brem": "numerics.sampling (continuum history population)",
    "adaptive_precision": "numerics.sampling",
    "E0_keV": "beam energy",
    "thickness_ang": "target thickness",
    "composition": "target material",
    "crystal": "target material",
    "abs_layers": "target layers",
    "tilt_deg": "target tilt",
    "tilt_azim_deg": "target tilt azimuth",
    "groove_spacing_ang": "target grooves",
    "crystal_width_mm": "target footprint",
    "crystal_height_mm": "target footprint",
    "beam_fwhm_mm": "beam spot",
    "beam_fwhm_y_mm": "beam spot",
    "E_cut_lines_keV": "line electron cutoff",
    "E_cut_brem_keV": "continuum electron cutoff",
}


class ScoringRefusedError(ValueError):
    """A scoring request would have changed the transport it scores.

    ``field`` names the offending input.
    """

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def score_spectra(
    trajectories: Trajectories,
    detector: Detector | PlanarDetector | None = None,
    *,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    filters: tuple[FilterPlate | Filter, ...] | None = None,
    acquisition: Acquisition | NativeAcquisition | None = None,
    observation: Observation | None = None,
    progress: Callable[[SimulationProgress], None] | None = None,
) -> Result:
    """Score spectra of an existing transport; never transports.

    Returns the :class:`~pyrite.Result` that :func:`pyrite.simulate` returns
    for the transported beam and target with these scoring inputs: response-free
    source photon densities per incident electron per eV per sr for a scalar
    :class:`~pyrite.Detector`, and observation averages plus optional spatial
    data for a :class:`~pyrite.detectors.PlanarDetector`.

    Parameters
    ----------
    trajectories
        A :class:`~pyrite.transport.Trajectories` from
        :func:`~pyrite.transport.simulate_trajectories` (default cutoff and step
        budget) or :func:`~pyrite.storage.load_trajectories`.
    detector
        Scalar or planar detector. ``None`` uses the capture detector.
        Any detector is allowed for incoherent, non-temporal scoring; coherent
        or temporal scoring requires the capture detector's placement, and
        grooved targets the capture observation angle.
    physics, numerics
        ``None`` keeps the transport's policies. Only spectrum-phase settings
        may change: line grids and their policy, quadrature, reflections,
        emission mode, temporal output, line roulette, spectral chunks and CXR
        launch controls.
    filters, acquisition, observation
        As :func:`pyrite.simulate`; filters and acquisition need a planar
        detector.
    progress
        Optional callback receiving one
        :class:`~pyrite.phases.SimulationProgress` per spectrum stage.

    Returns
    -------
    Result
        Spectra, resolved scoring case and provenance; identical to
        ``pyrite.simulate`` on the same backend.

    Raises
    ------
    TypeError
        ``trajectories`` is not a :class:`~pyrite.transport.Trajectories`.
    ScoringRefusedError
        Before any spectrum work, when an input would have changed transport
        (target, beam, transport physics or numerics, seed, population,
        explicit ``E_cut_keV``/``max_steps``), the coherent/temporal detector
        differs, or a loaded artifact fails its fingerprint. ``field`` names it.
    ValueError
        Unsupported scoring requests, as for :func:`pyrite.simulate`.
    """
    from ._mixed_elastic_policy import refuse_mixed_cxr
    from .api import _scalar_result, _simulate_planar, _source_case
    from .campaign.resolution import resolve_simulation
    from .detectors import PlanarDetector
    from .montecarlo.runner.transport_options import run_progress, transport_tuning

    if not isinstance(trajectories, Trajectories):
        raise TypeError("trajectories must be a pyrite.transport.Trajectories")
    _refuse_unscorable(trajectories)
    scene = trajectories.scene
    assert scene is not None
    detector = scene.detector if detector is None else detector
    if detector is None:
        raise ValueError(
            "score_spectra requires a detector: these trajectories were transported "
            "without one, so pass the detector to score"
        )
    request = resolve_simulation(
        scene.beam,
        scene.target,
        detector,
        physics=trajectories.physics if physics is None else physics,
        numerics=trajectories.numerics if numerics is None else numerics,
        filters=filters,
        acquisition=acquisition,
        observation=observation,
        require_detector=True,
    )
    case, xsgen_tables = _source_case(request)
    refuse_mixed_cxr(case)
    check_transport_identity(trajectories, request.numerics, case, xsgen_tables)
    transport = _scoring_transport(trajectories, case)
    if isinstance(request.scene.detector, PlanarDetector):
        if request.numerics.temporal_profile:
            raise ValueError("temporal_profile currently requires a scalar Detector")
        from .montecarlo.runner.case_driver import _directional_outputs

        with transport_tuning(request.numerics), run_progress(progress):
            return _simulate_planar(
                request,
                case,
                xsgen_tables,
                lambda directions: _directional_outputs(case, transport, directions),
            )
    from .montecarlo.runner import _spectrum_with_recovery

    with transport_tuning(request.numerics), run_progress(progress):
        output = _spectrum_with_recovery(case, transport, segs_owner=trajectories._transport)
    return _scalar_result(request, case, xsgen_tables, output)


def _refuse_unscorable(trajectories: Trajectories) -> None:
    for name, value in trajectories._engine_overrides.items():
        raise ScoringRefusedError(
            name,
            f"score_spectra refuses trajectories transported with explicit {name}={value!r}; "
            "pyrite.simulate uses the production default, so transport again with "
            f"simulate_trajectories(..., {name}=None)",
        )
    if trajectories.scene is None:
        raise ScoringRefusedError(
            "scene",
            "these trajectories carry no typed scene; score them with "
            "`pyrite trajectories score` or recapture with simulate_trajectories",
        )
    stored = trajectories._stored_fingerprint
    if stored is not None and stored != trajectories.fingerprint:
        raise ScoringRefusedError(
            "fingerprint",
            f"trajectory artifact {trajectories._source} segments do not match their "
            f"recorded fingerprint ({trajectories.fingerprint} != {stored}); it is corrupt",
        )


def check_transport_identity(
    trajectories: Trajectories, numerics: Numerics, case: Mapping[str, Any], xsgen_tables
) -> None:
    """Refuse a scoring case that differs from the transport in a transport input."""
    captured = trajectories.case
    for key in sorted(set(captured) | set(case)):
        if key in SCORING_ONLY_FIELDS:
            continue
        before, after = captured.get(key), case.get(key)
        if not _same(before, after):
            owner = _FIELD_OWNERS.get(key, f"case field {key!r}")
            raise ScoringRefusedError(
                key,
                f"score_spectra refuses a change of {owner} ({key}: transported "
                f"{_short(before)}, requested {_short(after)}); it changes transport, "
                "so transport again with simulate_trajectories",
            )
    fixed = trajectories.numerics
    if fixed is not None:
        for name, before, after in (
            ("numerics.transport_core", fixed.transport_core, numerics.transport_core),
            (
                "numerics.kernel_tuning.transport",
                fixed.kernel_tuning.transport,
                numerics.kernel_tuning.transport,
            ),
        ):
            if before != after:
                raise ScoringRefusedError(
                    name,
                    f"score_spectra refuses a change of {name} ({before!r} -> {after!r}); "
                    "transport settings were fixed by simulate_trajectories",
                )
    if dict(xsgen_tables) != dict(trajectories._xsgen_tables):
        raise ScoringRefusedError(
            "xsgen_tables",
            "score_spectra refuses: the cross-section tables changed since transport",
        )
    _check_geometry(captured, case)


def _check_geometry(captured: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    """Observation-dependent transport geometry and coherent/temporal placement."""
    from .montecarlo.runner.artifacts import _case_geometry

    *_, beam_before, _n, groove_before = _case_geometry(captured)
    *_, beam_after, _n, groove_after = _case_geometry(case)
    if not np.array_equal(beam_before, beam_after) or not _same(groove_before, groove_after):
        raise ScoringRefusedError(
            "detector",
            "score_spectra refuses this detector: the grooved target's transport geometry "
            "depends on the observation angle, which differs from the capture detector",
        )
    if case.get("coherent_emission") or case.get("temporal_profile"):
        for key in _DETECTOR_FIELDS:
            if not _same(captured.get(key), case.get(key)):
                raise ScoringRefusedError(
                    "detector",
                    f"coherent and temporal scoring require the capture detector's "
                    f"placement ({key} differs)",
                )


def _scoring_transport(trajectories: Trajectories, case) -> dict[str, Any]:
    """The kept transport with this case's spectrum inputs, as the live run builds them."""
    from .montecarlo.runner.case_driver import case_spectrum_inputs

    segs = trajectories.segments
    return {
        **trajectories._transport,
        **case_spectrum_inputs(case, segs, case["Ne"], case["Ne_brem"]),
    }


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    try:
        result = a == b
    except (TypeError, ValueError):
        return False
    if isinstance(result, np.ndarray):
        return bool(result.all()) and np.shape(a) == np.shape(b)
    return bool(result)


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."
