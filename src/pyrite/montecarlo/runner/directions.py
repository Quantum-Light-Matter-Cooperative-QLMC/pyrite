"""Physical-detector directions evaluated on one runner transport.

Split out of ``runner/__init__`` to keep that module inside the source-size
budget; it is runner-internal. Directions travel beside a case, never inside
it, so case content keys and scalar outputs are unchanged by them.
"""

from typing import Any

import numpy as np

from .line_grid import resolve_observation_line_grid


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
    (:func:`~.line_grid.resolve_observation_line_grid`); it is independent of
    the scalar case grid, so adding directions never changes scalar output.
    ``spectrum(case, transport)`` is the runner's spectrum phase (the GPU
    pipeline passes its OOM-retrying variant).
    """
    directions = validated_directions(n_hats)
    E_grid, grid_record = resolve_observation_line_grid(
        case,
        transport["segs"],
        directions,
        transport["Ne_lines"],
        transport["E_grid"],
        case.get("abs_layers"),
        transport.get("groove"),
    )
    outputs = []
    for direction in directions:
        directional_transport = dict(transport)
        directional_transport["n_hat"] = direction
        directional_transport["E_grid"] = E_grid
        directional_transport.pop("diagnostic_grid", None)
        outputs.append(spectrum(case, directional_transport))

    first = outputs[0]
    result = {
        key: value
        for key, value in first.items()
        if key not in {"spec", "spec_coherent", "spec_characteristic", "brem", "brem_wide"}
    }
    if grid_record is not None:
        result["line_grid_resolved"] = grid_record
    else:
        result.pop("line_grid_resolved", None)
    result["spec_by_direction"] = np.stack([np.asarray(output["spec"]) for output in outputs])
    result["spec_characteristic_by_direction"] = np.stack(
        [np.asarray(output["spec_characteristic"]) for output in outputs]
    )
    result["brem_by_direction"] = np.stack([np.asarray(output["brem"]) for output in outputs])
    result["brem_wide_by_direction"] = np.stack(
        [np.asarray(output["brem_wide"]) for output in outputs]
    )
    if "spec_coherent" in first:
        result["spec_coherent_by_direction"] = np.stack(
            [np.asarray(output["spec_coherent"]) for output in outputs]
        )
    return result
