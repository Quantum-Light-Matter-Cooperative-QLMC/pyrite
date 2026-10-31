"""Case-determined spectrum inputs and spectrum-phase replay from artifacts (#186).

:func:`_case_geometry` is the part of the transport phase's setup that the case
alone determines; :func:`spectrum_from_artifact` recomputes it to check a
trajectory artifact's recorded spectrum-phase inputs before replaying them.
"""

from collections.abc import Mapping
from os import PathLike
from typing import Any

import numpy as np

from ..._energy_grid_encoding import decode_energy_grid
from ..case import Case, refuse_axial_channeling
from ..geometry import tilted_geometry
from ..groove import blazed_groove_spec
from ..trajectories import (
    TrajectoryArtifact,
    TrajectoryArtifactError,
    artifact_spectrum_inputs,
    case_digest,
    iter_electron_row_blocks,
    read_trajectory_artifact,
)
from .streamed_scoring import STREAM_REFUSED_KEYS as STREAM_REFUSED_KEYS
from .streamed_scoring import BlockSpectrumScorer, streaming_refusal

#: Default segment rows one streamed block holds. Measured on CPU for a 7.1 M
#: segment hopg shell + coupled-radiative artifact (#186): peak host memory
#: above baseline was ~1.1 GiB at 2**20 rows, ~0.7 GiB at 2**18, ~0.4 GiB at
#: 2**16 (a ~0.3 GiB fixed cost plus the block), against ~4.9 GiB for the
#: whole-artifact read. Smaller blocks were not slower on CPU.
STREAM_MAX_SEGMENTS = 1 << 20


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
    case, inputs = _checked_case_and_inputs(artifact, case)
    segs = artifact.transport
    if int(segs["L_ang"].size) != int(artifact.attrs["segment_count"]):
        raise TrajectoryArtifactError(f"trajectory artifact {artifact.path} segments are truncated")
    from . import _spectrum_case

    return _spectrum_case(case, dict(inputs, segs=segs), record_timing)


def _check_streamable(case, path) -> None:
    """Refuse a case whose spectra need every segment or history at once."""
    reason = streaming_refusal(case)
    if reason is not None:
        raise TrajectoryArtifactError(f"{reason}; score {path} with spectrum_from_artifact instead")


def stream_spectrum_from_artifact(
    path: str | PathLike[str],
    case: Case | Mapping[str, Any] | None = None,
    *,
    max_segments: int = STREAM_MAX_SEGMENTS,
) -> dict[str, Any]:
    """Score an artifact's spectra in whole-electron segment blocks.

    The bounded-memory form of :func:`spectrum_from_artifact`: never
    transports, never holds more than about ``max_segments`` segment rows
    (plus per-electron and scalar fields) on the host, and sums the
    incoherent line, characteristic, continuum and per-history transition
    radiation estimators over disjoint electron blocks. Each is a track-length sum normalized by a fixed electron
    count, so blocks add; only the floating-point summation order differs from
    the live run.

    Raises
    ------
    TrajectoryArtifactError
        As :func:`spectrum_from_artifact`, or a case that needs every segment
        at once (coherent emission, temporal profile, bunch transition radiation).
    """
    from ..spectrum.cxr._coverage import collect_line_data_coverage

    artifact = read_trajectory_artifact(path, load_transport=False)
    case, inputs = _checked_case_and_inputs(artifact, case)
    _check_streamable(case, artifact.path)
    try:
        scorer = BlockSpectrumScorer(case, inputs)
    except ValueError as error:
        raise TrajectoryArtifactError(str(error)) from error
    segs = {}
    with collect_line_data_coverage() as line_coverage:
        for segs in iter_electron_row_blocks(path, max_segments):
            scorer.add(segs)
    if scorer.n_segments != int(artifact.attrs["segment_count"]):
        raise TrajectoryArtifactError(f"trajectory artifact {artifact.path} segments are truncated")
    return scorer.result(
        Ne=segs["Ne"],
        n_backscattered=segs["n_backscattered"],
        n_missed=segs["n_missed"],
        line_coverage=line_coverage,
        diagnostic_grid=inputs.get("diagnostic_grid"),
    )


def _checked_case_and_inputs(artifact, case):
    """The artifact's own case and its spectrum inputs, after every identity check."""
    try:
        refuse_axial_channeling(artifact.case)
        if case is not None:
            refuse_axial_channeling(case)
    except ValueError as error:
        raise TrajectoryArtifactError(str(error)) from error
    inputs = artifact_spectrum_inputs(artifact)
    stored = artifact.case
    if case_digest(stored) != artifact.attrs.get("case_sha256"):
        raise TrajectoryArtifactError(f"trajectory artifact {artifact.path} case is corrupt")
    if case is not None and case_digest(case) != case_digest(stored):
        raise TrajectoryArtifactError(
            f"trajectory artifact {artifact.path} records a different case than requested"
        )
    case = stored
    _check_physics_provenance(artifact)
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
    return case, inputs


def _check_physics_provenance(artifact) -> None:
    """Optional model/data check shared with batch planning before any writes."""
    snapshot = artifact.settings.get("resolved_physics")
    if snapshot is not None:
        from ..._physics_contract import ResolvedPhysicsView
        from ..._physics_provenance import resolved_physics_view
        from .case_tables import case_table_markers

        try:
            recorded = ResolvedPhysicsView.from_json(snapshot)
            current = resolved_physics_view(
                artifact.case, table_markers=case_table_markers(artifact.case)
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            raise TrajectoryArtifactError(
                f"trajectory artifact {artifact.path} physics provenance cannot be verified: {error}"
            ) from error
        if (
            recorded.descriptor_ids,
            recorded.model_markers,
            recorded.table_markers,
            recorded.parameters_json,
        ) != (
            current.descriptor_ids,
            current.model_markers,
            current.table_markers,
            current.parameters_json,
        ):
            raise TrajectoryArtifactError(
                f"trajectory artifact {artifact.path} physics model or table identity changed; "
                "capture new trajectories before replay"
            )


def _same_input(actual, expected) -> bool:
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        return (
            isinstance(actual, np.ndarray)
            and isinstance(expected, np.ndarray)
            and actual.dtype == expected.dtype
            and np.array_equal(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected
