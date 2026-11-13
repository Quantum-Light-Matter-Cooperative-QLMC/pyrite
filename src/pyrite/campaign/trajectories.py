"""Typed public trajectory transport over canonical scene lowering (#936).

``simulate_trajectories`` lowers ``Beam``/``Target``/``Physics``/``Numerics``
through :func:`pyrite.campaign.lowering.build_case` and the runner's
production case adapters, then calls the private flat engine
:func:`pyrite.montecarlo.transport._simulate_trajectories`. It lives above
``pyrite.montecarlo`` because the physics core may not import campaign.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..montecarlo.transport import _simulate_trajectories
from ..montecarlo.transport.lut import TransportLUTConfig
from .trajectory_product import Trajectories

if TYPE_CHECKING:
    from .._numerics import Numerics, Physics
    from ..detectors import Detector
    from .beam import Beam
    from .geometry import Slab, Stack
    from .target import Target

__all__ = ["Trajectories", "simulate_trajectories"]


def simulate_trajectories(
    beam: Beam,
    target: Target | Slab | Stack,
    *,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    detector: Detector | None = None,
    E_cut_keV: float | None = None,
    max_steps: int | None = None,
    keep_segments_on_device: bool = False,
    collect_diagnostics: bool = False,
    transport_progress: Callable[[int, int], None] | None = None,
) -> Trajectories:
    """Transport a fixed population from typed physical and numerical inputs.

    Parameters
    ----------
    beam, target
        Scalar :class:`pyrite.Beam` and :class:`pyrite.geometry.Target`,
        :class:`pyrite.geometry.Slab`, or :class:`pyrite.geometry.Stack`.
        Energies are in keV; target lengths follow their constructor units.
    physics, numerics
        Existing :class:`pyrite.Physics` and :class:`pyrite.Numerics` policies;
        omitted values use their constructor defaults. The population is
        ``max(cxr_histories, brem_histories)`` of ``numerics.sampling``. ``Numerics.seed``
        uses the canonical case seed when omitted. Integration, transport core,
        precision and kernel tuning come from Numerics. Models and source
        policies resolve through the same scene lowering as ``pyrite.simulate``.
        Adaptive sampling, channeling, quantum entry states, streaming scoring
        and photon aperture tallies require ``pyrite.simulate`` instead.
    detector
        Optional scalar observation geometry for target-dependent longitudinal
        timing. None uses canonical detector-free source lowering; no detector
        response or spectrum is evaluated.
    E_cut_keV
        ``None`` (default) uses the production per-history cutoffs of
        :func:`pyrite.simulate`: ``E_cut_lines_keV`` (5 keV) for line-only
        histories, ``E_cut_brem_keV`` (1 keV) for continuum-only ones and the
        lower of the two for shared histories. A positive value in keV applies
        to every primary; the result is then not scorable.
    max_steps
        ``None`` (default) uses the production retrying step budget. An
        explicit per-electron limit makes the result not scorable; incomplete
        histories raise.
    keep_segments_on_device, collect_diagnostics, transport_progress
        Device-resident segment output, fixed-size post-transport diagnostics,
        and an optional ``callback(electrons_done, total)``. Default off.
        Device output requires the CUDA core.

    Returns
    -------
    Trajectories
        The segment arrays, incident phase space, vacuum diagnostics and fate
        counts (:attr:`Trajectories.segments`, the transport-output contract),
        with the resolved scene, physics, numerics and case. With default
        ``E_cut_keV`` and ``max_steps`` the transport is the one
        :func:`pyrite.simulate` performs for the same scene, and
        :func:`pyrite.score_spectra` reproduces its spectra. Dict-style access
        to the object is deprecated (#1162).

    Raises
    ------
    TypeError
        Inputs have the wrong types or use removed flat transport keywords.
    ValueError
        Policies, geometry, table coverage or execution options are unsupported.
    RuntimeError
        Any history remains incomplete at the step limit.

    Notes
    -----
    This replaces the former flat transport arguments (#936). Material tables,
    beam phase space, target geometry and model controls belong to their typed
    owners; internal cascade and electron-block controls are private. Before
    #1162 the default was one 5 keV cutoff for every primary.

    Validation: electron-transport
    """
    request, case, numerics = trajectory_request(beam, target, physics, numerics, detector)
    overrides = {
        name: value
        for name, value in (("E_cut_keV", E_cut_keV), ("max_steps", max_steps))
        if value is not None
    }
    if overrides:
        segments = _engine_transport(
            case,
            numerics,
            E_cut_keV=E_cut_keV,
            max_steps=400_000 if max_steps is None else max_steps,
            keep_segments_on_device=keep_segments_on_device,
            collect_diagnostics=collect_diagnostics,
            transport_progress=transport_progress,
        )
        transport: dict[str, Any] = {"segs": segments}
    else:
        transport = _production_transport(
            case,
            numerics,
            keep_segments_on_device=keep_segments_on_device,
            collect_diagnostics=collect_diagnostics,
            transport_progress=transport_progress,
        )
    from ..montecarlo.runner.case_tables import case_table_markers

    return Trajectories(
        request.scene,
        request.physics,
        numerics,
        case,
        transport,
        _xsgen_tables=case_table_markers(case),
        _engine_overrides=overrides,
    )


def _production_transport(
    case, numerics, *, keep_segments_on_device, collect_diagnostics, transport_progress
) -> dict[str, Any]:
    """The transport phase :func:`pyrite.simulate` runs, with its spectrum inputs."""
    from ..montecarlo.runner import _TRANSPORT_PROGRESS, _transport_case
    from ..montecarlo.runner.transport_options import transport_tuning

    token = None if transport_progress is None else _TRANSPORT_PROGRESS.set(transport_progress)
    try:
        with transport_tuning(numerics):
            transport = _transport_case(
                case,
                transport_core=numerics.transport_core,
                keep_segments_on_device=keep_segments_on_device,
                engine_options={"collect_diagnostics": True} if collect_diagnostics else None,
            )
    finally:
        if token is not None:
            _TRANSPORT_PROGRESS.reset(token)
    return {key: value for key, value in transport.items() if not key.startswith("_t_")}


def _engine_transport(
    case,
    numerics,
    *,
    E_cut_keV,
    max_steps,
    keep_segments_on_device,
    collect_diagnostics,
    transport_progress,
) -> dict[str, Any]:
    """One flat-engine call with explicit cutoff or step limit (not scorable)."""
    from ..montecarlo.runner.adaptive import case_transport_core
    from ..montecarlo.runner.case_driver import electron_cutoffs
    from ..montecarlo.runner.transport_options import per_electron_config

    tuning = numerics.resolved_execution.transport
    kwargs = case_transport_kwargs(case)
    core = case_transport_core(case, numerics.transport_core)
    # Mirror the case driver: the CUDA LUT kernel lacks explicit inelastic
    # modes, so those runs select the exact CUDA kernel.
    if core == "cuda" and case.get("inelastic_model") is not None:
        kwargs["transport_lut_config"] = TransportLUTConfig(enabled=False)
    cutoff = (
        {"E_cut_by_electrons": electron_cutoffs(case)}
        if E_cut_keV is None
        else {"E_cut_keV": E_cut_keV}
    )
    return _simulate_trajectories(
        **kwargs,
        **cutoff,
        max_steps=max_steps,
        transport_core=core,
        per_electron_config=per_electron_config(tuning),
        transport_precision=tuning.precision or "auto",
        keep_segments_on_device=keep_segments_on_device,
        collect_diagnostics=collect_diagnostics,
        transport_progress=transport_progress,
    )


def trajectory_request(
    beam: Beam,
    target: Target | Slab | Stack,
    physics: Physics | None,
    numerics: Numerics | None,
    detector: Detector | None,
):
    """Resolve supported fixed-count transport through the canonical scene owner.

    Returns ``(request, case, numerics)``; the case is the one
    :func:`pyrite.simulate` lowers for the same components.
    """
    from .._numerics import Numerics, Physics
    from .._sampling import AdaptiveSampling
    from .lowering import lower_simulation
    from .model import Scene
    from .resolution import resolve_simulation

    numerics = Numerics() if numerics is None else numerics
    physics = Physics() if physics is None else physics
    if not isinstance(numerics, Numerics):
        raise TypeError("numerics must be a Numerics")
    if not isinstance(physics, Physics):
        raise TypeError("physics must be a Physics")
    if isinstance(numerics.sampling, AdaptiveSampling):
        raise ValueError(
            "simulate_trajectories requires fixed sampling; adaptive sampling scores spectra"
        )
    execution = numerics.resolved_execution
    if execution.stream_block_electrons is not None or execution.stream_memory_mib is not None:
        from ..montecarlo.runner.transport_options import STREAMING_ENTRY_ERROR

        raise ValueError(f"stream_block_electrons and stream_memory_mib: {STREAMING_ENTRY_ERROR}")
    scene = Scene(beam, target, detector=detector)
    if scene.beam.state is not None:
        raise ValueError("channeling and Beam.state require pyrite.simulate")
    if physics.photon_transport is not None:
        raise ValueError("photon aperture scoring requires pyrite.simulate")
    request = resolve_simulation(scene, physics=physics, numerics=numerics)
    return request, lower_simulation(request), request.numerics


def case_transport_kwargs(case) -> dict[str, Any]:
    """Share production geometry, source and identity-matched table adapters."""
    from ..montecarlo.runner.artifacts import _case_geometry
    from ..montecarlo.runner.case_tables import (
        _case_elastic_kwargs,
        _case_inelastic_kwargs,
        _case_radiative_kwargs,
        _case_stopping_tables,
    )
    from ..montecarlo.runner.transport_options import _beam_kwargs

    polar = np.deg2rad(case.get("tilt_deg", 0.0))
    azimuth = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    _, _, direction, _, groove = _case_geometry(case)
    return dict(
        E0_keV=case["E0_keV"],
        Ne=max(case["Ne"], case["Ne_brem"]),
        thickness_ang=case["thickness_ang"],
        composition=case["composition"],
        layers=case.get("abs_layers"),
        seed=case["seed"],
        beam_dir=direction,
        crystal_width_mm=case.get("crystal_width_mm"),
        crystal_height_mm=case.get("crystal_height_mm"),
        tilt_polar_rad=polar,
        tilt_azim_rad=azimuth,
        groove=groove,
        energy_model=case.get("energy_model", "frozen"),
        max_dE_frac=case.get("max_dE_frac", 0.0),
        straggling=bool(case.get("straggling", False)),
        stopping_tables=_case_stopping_tables(case),
        **_beam_kwargs(case),
        **_case_inelastic_kwargs(case),
        **_case_elastic_kwargs(case),
        **_case_radiative_kwargs(case),
    )


def scene_trajectories(scene, *, numerics=None, progress=None) -> Trajectories:
    """Transport a scalar-detector :class:`~pyrite.Scene` (keeps CUDA rows on device).

    The scene-level form behind the deprecated ``pyrite.phases.simulate_transport``
    and kernel tuning; ``progress`` receives ``SimulationProgress`` events.
    """
    from .._simulation_progress import SimulationProgress
    from ..detectors import PlanarDetector
    from .model import Scene

    if not isinstance(scene, Scene):
        raise TypeError("scene must be a pyrite.Scene")
    if isinstance(scene.detector, PlanarDetector):
        raise ValueError("scene transport supports scalar detectors only")

    def transport_progress(done, total):
        if progress is not None:
            progress(SimulationProgress("transport", "electrons", int(done), int(total)))

    return simulate_trajectories(
        scene.beam,
        scene.target,
        physics=scene.physics,
        numerics=scene.numerics if numerics is None else numerics,
        detector=scene.detector,
        keep_segments_on_device=True,
        transport_progress=None if progress is None else transport_progress,
    )
