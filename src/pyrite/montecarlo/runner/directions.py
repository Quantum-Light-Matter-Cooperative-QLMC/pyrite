"""Physical-detector directions evaluated on one runner transport.

Split out of ``runner/__init__`` to keep that module inside the source-size
budget; it is runner-internal. Directions travel beside a case, never inside
it, so case content keys and scalar outputs are unchanged by them.
"""

from typing import Any

import numpy as np

from .cxr_grid import resolve_observation_cxr_grid


def validated_directions(n_hats) -> np.ndarray:
    """Return ``n_hats`` as a checked ``(N, 3)`` array of unit vectors."""
    directions = np.asarray(n_hats, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not directions.shape[0]:
        raise ValueError("n_hats must have shape (N, 3) with N positive")
    if not np.all(np.isfinite(directions)):
        raise ValueError("n_hats must contain only finite values")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("n_hats must contain unit vectors")
    return directions


def directional_outputs(case, transport, n_hats, spectrum) -> dict[str, Any]:
    """Evaluate every observation direction on one already-run transport.

    The line grid covers all directions jointly
    (:func:`~.line_grid.resolve_observation_cxr_grid`); it is independent of
    the scalar case grid, so adding directions never changes scalar output.
    ``spectrum(case, transport)`` is the runner's spectrum phase (the GPU
    pipeline passes its recovery policy).
    """
    from ..case import refuse_axial_directions
    from .photon_aperture import cached_photon_aperture

    refuse_axial_directions(case)
    cached_photon_aperture(case, transport)
    directions = validated_directions(n_hats)
    E_grid, grid_record = resolve_observation_cxr_grid(
        case,
        transport["segs"],
        directions,
        transport["Ne_lines"],
        transport["E_grid"],
        case.get("abs_layers"),
        transport.get("groove"),
    )
    outputs, E_grid, grid_record = _joint_directional_spectra(
        case, transport, directions, spectrum, E_grid, grid_record
    )
    first = outputs[0]
    result = {
        key: value
        for key, value in first.items()
        if not key.startswith("_")
        and key
        not in {
            "spec",
            "spec_coherent",
            "spec_characteristic",
            "spec_channeling",
            "spec_transition",
            "brem",
            "brem_wide",
            "temporal_intensity",
            "temporal_intensity_coherent",
        }
    }
    result.pop("line_grid_diagnostic", None)
    if grid_record is not None:
        audits = [
            (output.get("line_grid_resolved") or {}).get("truncation_audit") for output in outputs
        ]
        result["line_grid_resolved"] = (
            grid_record if audits[0] is None else {**grid_record, "truncation_audits": audits}
        )
    else:
        result.pop("line_grid_resolved", None)
    result["line_data_coverage"] = [
        record for output in outputs for record in output.get("line_data_coverage", [])
    ]
    result["spec_by_direction"] = np.stack([np.asarray(output["spec"]) for output in outputs])
    result["spec_characteristic_by_direction"] = np.stack(
        [np.asarray(output["spec_characteristic"]) for output in outputs]
    )
    result["brem_by_direction"] = np.stack([np.asarray(output["brem"]) for output in outputs])
    result["brem_wide_by_direction"] = np.stack(
        [np.asarray(output["brem_wide"]) for output in outputs]
    )
    for key in ("spec_channeling", "spec_transition", "spec_coherent"):
        if first.get(key) is not None:
            result[f"{key}_by_direction"] = np.stack(
                [np.asarray(output[key]) for output in outputs]
            )
    for key in ("temporal_intensity", "temporal_intensity_coherent"):
        if key in first:
            result[f"{key}_by_direction"] = np.stack(
                [np.asarray(output[key]) for output in outputs]
            )
    return result


def _joint_directional_spectra(case, transport, directions, spectrum, E_grid, grid_record):
    """Every direction's spectrum on one axis, extended jointly after a refusal.

    A defaulted measured axis that any direction's audit refuses is rebuilt
    for all directions from the observation resolver at the extended stop, and
    every direction is rescored and re-audited on it (#1243). Returns the
    outputs, the scored axis and its record.

    Validation: line-grid-resonance-bandwidth
    """
    from ..._cxr_grid_policy import CxrGridToleranceError
    from .bandwidth_fallback import fallback_axis_payload, fallback_record

    joint = case if grid_record is None else {**case, "_joint_bandwidth_fallback": True}
    while True:
        temporal_grid = None
        if case.get("temporal_profile"):
            from ..spectrum.cxr import temporal_profile_for

            # One time grid for every direction so the profiles stack (#292).
            temporal_grid = temporal_profile_for(transport["segs"], E_grid, directions)
        try:
            outputs = []
            for direction in directions:
                directional_transport = dict(transport)
                if temporal_grid is not None:
                    directional_transport["temporal_grid"] = temporal_grid
                directional_transport["n_hat"] = direction
                directional_transport["E_grid"] = E_grid
                directional_transport["diagnostic_grid"] = grid_record
                if grid_record is None:
                    directional_transport.pop("diagnostic_grid")
                outputs.append(spectrum(joint, directional_transport))
            return outputs, E_grid, grid_record
        except CxrGridToleranceError as refusal:
            earlier = (grid_record or {}).get("bandwidth_fallback")
            fallback, stop, ceiling, refusals = fallback_axis_payload(case, refusal, earlier)
            if not stop > float(E_grid[-1]):
                raise refusal from None
            E_grid, resolved = resolve_observation_cxr_grid(
                {**case, "line_grid_policy": fallback},
                transport["segs"],
                directions,
                transport["Ne_lines"],
                transport["E_grid"],
                case.get("abs_layers"),
                transport.get("groove"),
            )
            grid_record = fallback_record(resolved, case, stop, ceiling, refusals)
