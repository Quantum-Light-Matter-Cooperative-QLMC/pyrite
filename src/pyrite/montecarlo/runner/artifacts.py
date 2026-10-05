"""Case-determined spectrum inputs and spectrum-phase replay from artifacts (#186).

:func:`_case_geometry` is the part of the transport phase's setup that the case
alone determines; :func:`spectrum_from_artifact` recomputes it to check a
trajectory artifact's recorded spectrum-phase inputs before replaying them.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..._energy_grid_encoding import decode_energy_grid
from ..case import Case
from ..geometry import tilted_geometry
from ..groove import blazed_groove_spec
from ..trajectories import (
    TrajectoryArtifact,
    TrajectoryArtifactError,
    artifact_spectrum_inputs,
    case_digest,
)


def _case_geometry(case):
    """Case-determined grids and geometry shared by transport and spectrum.

    Returns ``(E_grid, E_brem, beam_dir, n_hat, groove)``; ``E_grid`` is the
    case's line grid before any automatic resolution.
    """
    if "E_grid_line" in case:
        E_grid = decode_energy_grid(case["E_grid_line"])
        E_brem = decode_energy_grid(case["E_grid_brem"])
    else:
        E_grid = decode_energy_grid(case["E_grid"])
        step_b = case.get("brem_step_eV", 10.0)
        E_brem = np.arange(E_grid[0], E_grid[-1] + step_b, step_b)
    tilt_polar_rad = np.deg2rad(case.get("tilt_deg", 0.0))
    tilt_azim_rad = np.deg2rad(case.get("tilt_azim_deg", 0.0))
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad)
    # blazed sawtooth entrance-face grooves (docs/superpowers/plans/
    # 2026-07-23-blazed-groove-geometry.md): built once per case, then threaded
    # into both electron-entry transport calls and the line-spectrum escape
    # model. None -> a strict no-op (the flat-face slab, unchanged).
    groove = None
    if case.get("groove_spacing_ang") is not None:
        groove = blazed_groove_spec(
            case["groove_spacing_ang"], case["theta_obs_rad"], tilt_polar_rad, tilt_azim_rad
        )
    return E_grid, E_brem, beam, n_hat, groove


def spectrum_from_artifact(
    artifact: TrajectoryArtifact,
    case: Case | Mapping[str, Any] | None = None,
    record_timing: bool = False,
) -> dict[str, Any]:
    """Run only the spectrum phase of :func:`run_case` on a captured transport.

    Never transports. The artifact's recorded spectrum-phase inputs are used as
    stored, so on the same spectrum backend the result is the live
    :func:`run_case` output for its case; the inputs the case alone determines
    are recomputed and must agree exactly.

    Parameters
    ----------
    artifact
        Schema-2 artifact from :func:`~pyrite.montecarlo.trajectories.
        read_trajectory_artifact` that recorded spectrum-phase inputs.
    case
        Case the caller expects; ``None`` uses the artifact's own case.
    record_timing
        Include internal spectrum timing metrics.

    Raises
    ------
    TrajectoryArtifactError
        Schema 1, no recorded inputs, or a case, input, or segment count that
        disagrees with the artifact.
    """
    inputs = artifact_spectrum_inputs(artifact)
    stored = artifact.case
    if case_digest(stored) != artifact.attrs.get("case_sha256"):
        raise TrajectoryArtifactError(f"trajectory artifact {artifact.path} case is corrupt")
    if case is not None and case_digest(case) != case_digest(stored):
        raise TrajectoryArtifactError(
            f"trajectory artifact {artifact.path} records a different case than requested"
        )
    case = stored
    E_grid, E_brem, _beam, n_hat, groove = _case_geometry(case)
    expected: dict[str, Any] = {
        "E_brem": E_brem,
        "n_hat": n_hat,
        "groove": groove,
        "Ne_lines": case["Ne"],
        "Ne_brem": case["Ne_brem"],
    }
    if case.get("line_grid_policy") is None and case.get("_diagnostic_line_grid") is None:
        expected["E_grid"] = E_grid
    for key, value in expected.items():
        if not _same_input(inputs.get(key), value):
            raise TrajectoryArtifactError(
                f"trajectory artifact {artifact.path} spectrum input {key!r} disagrees with "
                "its case; it was written by an incompatible PyRITE version"
            )
    segs = artifact.transport
    if int(segs["L_ang"].size) != int(artifact.attrs["segment_count"]):
        raise TrajectoryArtifactError(f"trajectory artifact {artifact.path} segments are truncated")
    from . import _spectrum_case

    return _spectrum_case(case, dict(inputs, segs=segs), record_timing)


def _same_input(actual, expected) -> bool:
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        return (
            isinstance(actual, np.ndarray)
            and isinstance(expected, np.ndarray)
            and actual.dtype == expected.dtype
            and np.array_equal(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected
