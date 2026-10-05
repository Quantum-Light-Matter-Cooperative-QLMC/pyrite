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
from ..case import Case
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
    incoherent line, characteristic, and continuum estimators over disjoint
    electron blocks. Each is a track-length sum normalized by a fixed electron
    count, so blocks add; only the floating-point summation order differs from
    the live run.

    Raises
    ------
    TrajectoryArtifactError
        As :func:`spectrum_from_artifact`, or a case that needs every segment
        at once (coherent emission, temporal profile).
    """
    from . import (
        _brem_wide_from_segments,
        _characteristic_from_segments,
        _lines_for_segments,
        _segments_on_device,
    )
    from .line_grid import check_line_truncation, line_truncation_audit

    artifact = read_trajectory_artifact(path, load_transport=False)
    case, inputs = _checked_case_and_inputs(artifact, case)
    for key in ("coherent_emission", "temporal_profile"):
        if case.get(key):
            raise TrajectoryArtifactError(
                f"{key} needs every segment at once; score {artifact.path} with "
                "spectrum_from_artifact instead"
            )
    E_grid, E_brem, n_hat = inputs["E_grid"], inputs["E_brem"], inputs["n_hat"]
    groove, abs_layers = inputs["groove"], case.get("abs_layers")
    Ne_lines, Ne_brem = inputs["Ne_lines"], inputs["Ne_brem"]
    audit = line_truncation_audit(case, E_grid, n_electrons=Ne_lines)
    table_cache: dict = {}
    spec = spec_characteristic = brem_wide = None
    n_segments = 0
    segs = {}
    for segs in iter_electron_row_blocks(path, max_segments):
        n_segments += int(segs["L_ang"].shape[0])
        if segs["L_ang"].shape[0] == 0:
            continue
        block = _segments_on_device(segs)
        parts = (
            _lines_for_segments(
                block,
                E_grid,
                case,
                n_hat,
                abs_layers,
                groove,
                coherent=False,
                Ne=Ne_lines,
                table_cache=table_cache,
                truncation_audit=audit,
            ),
            _characteristic_from_segments(
                block, E_grid, case, n_hat, abs_layers, groove=groove, Ne=Ne_brem
            ),
            _brem_wide_from_segments(
                block, E_brem, case, n_hat, abs_layers, groove=groove, Ne=Ne_brem
            ),
        )
        if spec is None:
            spec, spec_characteristic, brem_wide = parts
        else:
            spec, spec_characteristic, brem_wide = (
                total + part
                for total, part in zip((spec, spec_characteristic, brem_wide), parts, strict=True)
            )
    if n_segments != int(artifact.attrs["segment_count"]):
        raise TrajectoryArtifactError(f"trajectory artifact {artifact.path} segments are truncated")
    if spec is None:  # no segment at all: every estimator is zero
        spec = np.zeros(E_grid.shape)
        spec_characteristic = np.zeros(E_grid.shape)
        brem_wide = np.zeros(E_brem.shape)
    assert spec_characteristic is not None and brem_wide is not None
    truncation = None if audit is None else check_line_truncation(case, audit)
    out = dict(
        E_grid=E_grid,
        spec=spec,
        spec_characteristic=spec_characteristic,
        brem=np.interp(E_grid, E_brem, brem_wide),
        E_grid_brem=E_brem,
        brem_wide=brem_wide,
        eta=segs["n_backscattered"] / segs["Ne"],
        hit_frac=1.0 - segs["n_missed"] / segs["Ne"],
        n_segments=n_segments,
        crystal=case["crystal"],
        E0_keV=case["E0_keV"],
    )
    diagnostic = inputs.get("diagnostic_grid")
    if diagnostic is not None:
        out["line_grid_diagnostic"] = diagnostic
        if case.get("line_grid_policy") is not None:
            out["line_grid_resolved"] = (
                diagnostic if truncation is None else {**diagnostic, "truncation_audit": truncation}
            )
    return out


def _checked_case_and_inputs(artifact, case):
    """The artifact's own case and its spectrum inputs, after every identity check."""
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
    return case, inputs


def _same_input(actual, expected) -> bool:
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        return (
            isinstance(actual, np.ndarray)
            and isinstance(expected, np.ndarray)
            and actual.dtype == expected.dtype
            and np.array_equal(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected
