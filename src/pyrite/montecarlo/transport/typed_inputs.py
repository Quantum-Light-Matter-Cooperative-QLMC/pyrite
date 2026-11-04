"""Scene-owned lowering for the public trajectory-only operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..._numerics import Numerics, Physics
    from ...campaign.beam import Beam
    from ...campaign.geometry import Slab, Stack
    from ...campaign.target import Target
    from ...detectors import Detector


def trajectory_case(
    beam: Beam,
    target: Target | Slab | Stack,
    physics: Physics | None,
    numerics: Numerics | None,
    detector: Detector | None,
):
    """Resolve supported fixed-count transport through the canonical scene owner."""
    from ..._numerics import Numerics, Physics
    from ...campaign.lowering import build_case
    from ...campaign.model import Scene

    numerics = Numerics() if numerics is None else numerics
    physics = Physics() if physics is None else physics
    if not isinstance(numerics, Numerics):
        raise TypeError("numerics must be a Numerics")
    if not isinstance(physics, Physics):
        raise TypeError("physics must be a Physics")
    if numerics.precision is not None:
        raise ValueError("simulate_trajectories requires fixed sampling; adaptive sampling scores spectra")
    if numerics.resolved_execution.stream_block_electrons is not None:
        raise ValueError("stream_block_electrons requires pyrite.simulate")
    scene = Scene(beam, target, detector=detector)
    if physics.channeling_model != "off" or scene.beam.state is not None:
        raise ValueError("channeling and Beam.state require pyrite.simulate")
    if physics.photon_transport is not None:
        raise ValueError("photon aperture scoring requires pyrite.simulate")
    return build_case(scene, physics=physics, numerics=numerics), numerics


def case_transport_kwargs(case) -> dict[str, Any]:
    """Share production geometry, source and identity-matched table adapters."""
    from ..runner.artifacts import _case_geometry
    from ..runner.case_tables import (
        _case_elastic_kwargs,
        _case_inelastic_kwargs,
        _case_radiative_kwargs,
        _case_stopping_tables,
    )
    from ..runner.transport_options import _beam_kwargs

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
