"""Real CUDA parity of the explicit optical response and bounded TR seam.

Stage with ``pyrite remote sync`` and run in a bounded SLURM GPU allocation
on that host: ``uv run --extra nvidia python checks/optical_response_cuda.py``.
Missing CUDA fails; no skip or CPU backend fallback. Prints JSON evidence.
Validation: optical-material-response (measurement only).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from pyrite.energy_grid.component_support import ComponentSupport, component_grid
from pyrite.materials.optical import gold_rakic_response
from pyrite.montecarlo.spectrum.transition_radiation import planar_transition_radiation


def main() -> None:
    """Check device response upload, optical TR parity and fail-closed support."""
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("actual CUDA hardware is required")
    response = gold_rakic_response()
    energy = np.geomspace(*response.bands_eV[0], 513)
    host = response.epsilon(energy)
    device = response.epsilon(energy, xp=cp)
    if not isinstance(device, cp.ndarray) or device.dtype != cp.complex128:
        raise AssertionError("optical device upload must produce complex128 CuPy data")
    # The interpolation is intentionally NumPy-side, so device upload must be
    # lossless, not merely within a loose float32 tolerance.
    np.testing.assert_array_equal(cp.asnumpy(device), host)
    cp.cuda.get_current_stream().synchronize()
    refusals = []
    for query in ([0.01], [6.0], [np.nan]):
        try:
            response.epsilon(query, xp=cp)
        except ValueError:
            refusals.append(str(query))
        else:
            raise AssertionError(f"unsupported optical query accepted: {query}")
    nodes, values = response.segments[0]
    gapped = replace(response, segments=((nodes[:20], values[:20]), (nodes[40:], values[40:])))
    try:
        gapped.epsilon([float(nodes[30])], xp=cp)
    except ValueError:
        refusals.append("declared gap")
    else:
        raise AssertionError("gap was silently interpolated")
    try:
        replace(response, symmetry="tensor")
    except ValueError:
        refusals.append("tensor")
    else:
        raise AssertionError("unsupported tensor response accepted")
    capability = ComponentSupport(response.medium, "transition_radiation", response.bands_eV)
    try:
        component_grid(energy, required=((response.medium, "eaglexo_qe"),), support=(capability,))
    except ValueError as error:
        if "Au/eaglexo_qe" not in str(error):
            raise AssertionError("refusal must name the responsible component") from error
        refusals.append("Au/eaglexo_qe")
    else:
        raise AssertionError("optical TR support must not enable detector response")
    direction, velocity = np.array([0.6, 0.0, 0.8]), np.array([0.0, 0.0, 0.8])
    cpu_tr = planar_transition_radiation(energy, direction, velocity, [(host, 1000.0)])
    gpu_tr = cp.asnumpy(
        planar_transition_radiation(energy, direction, velocity, [(device, 1000.0)], xp=cp)
    )
    # Float64 LAPACK/cuSOLVER paths round differently; the same tolerance used
    # by the established TR CUDA parity checks includes a peak-scaled floor
    # for cancelled fields. Material-response upload itself is bit-exact.
    peak = float(np.max(np.abs(cpu_tr)))
    np.testing.assert_allclose(gpu_tr, cpu_tr, rtol=1e-10, atol=1e-12 * peak)
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print(
        json.dumps(
            {
                "validation_id": "optical-material-response",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "device": device_name,
                "compute_capability": cp.cuda.Device().compute_capability,
                "cupy": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "source_sha256": response.source_sha256,
                "response_digest": response.digest,
                "bands_eV": response.bands_eV,
                "nodes": int(energy.size),
                "response_max_abs_difference": float(np.max(np.abs(cp.asnumpy(device) - host))),
                "tr_peak_normalized_max_abs_difference": float(
                    np.max(np.abs(gpu_tr - cpu_tr)) / peak
                ),
                "tr_rtol": 1e-10,
                "tr_atol_over_peak": 1e-12,
                "refusals": refusals,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
