"""Real-CUDA first-flight parity and bounded end-to-end timing for #341.

Use ``uv run --extra nvidia python checks/photon_first_flight_cuda.py`` in a
bounded remote GPU allocation after ``pyrite remote sync``. No CUDA means
failure. Outputs one JSON record; this is backend parity, not a new physics
validation or evidence for the full photon history queue.
"""

import hashlib
import json
import platform
import resource
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from pyrite._backend import select_backend
from pyrite.materials.photon_cross_sections import EPDL_TABLE_SHA256
from pyrite.montecarlo.transport import photon_navigation as nav
from pyrite.montecarlo.transport.photon_cuda import scene_first_interactions_cuda

SEED = 341_701
N = 16_384
REPEATS = 5
WARMUPS = 2


def _workload(n=N):
    rng = np.random.default_rng(SEED)
    origins = rng.uniform(-2e7, 2e7, (n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins[: n // 4] = [4e7, 0, -2e7]
    directions[: n // 4] = [0, 0, 1]
    energies, uniforms = np.geomspace(1e3, 1e7, n), rng.random((n, 2))
    layers = [(0, 1e7, [("C", 0.08), ("Pb", 0.02)]), (2e7, 3e7, [("Si", 0.05)])]
    angle = np.pi / 7
    boxes = [
        nav.PhotonBox(
            (4e7, 0, 0),
            ((np.cos(angle), 0, np.sin(angle)), (0, 1, 0), (-np.sin(angle), 0, np.cos(angle))),
            (5e6, 2e7, 1e7),
            (("Pb", 0.03),),
        )
    ]
    kwargs = {"width_ang": 4e7, "height_ang": 4e7}
    nav.validate_photon_regions(layers, boxes, **kwargs)
    return (origins, directions, energies, uniforms, layers, boxes), kwargs


def main():
    backend = select_backend("cuda")
    cp = backend.xp
    args, kwargs = _workload()

    def cpu():
        return nav.scene_first_interactions(*args, **kwargs)

    def cuda():
        out = scene_first_interactions_cuda(*args, **kwargs)
        return {key: cp.asnumpy(value) for key, value in out.items()}

    # Both arms are warm, including GPU compilation, table loads and pool caches.
    for _ in range(WARMUPS):
        expected, actual = cpu(), cuda()
    for key in expected:
        if key == "distance_ang":
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert np.any(actual["layer"] == 2), "workload must exercise oriented-box collisions"
    times = {"cpu": [], "cuda": []}
    for repeat in range(REPEATS):
        # Alternate arm order to avoid always favoring one warmed host cache.
        arms = (("cpu", cpu), ("cuda", cuda))
        for name, run in arms if repeat % 2 == 0 else reversed(arms):
            start = perf_counter()
            run()
            times[name].append(perf_counter() - start)
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/pyrite/montecarlo/transport/photon_navigation.py",
        "src/pyrite/montecarlo/transport/photon_cuda.py",
        "checks/photon_first_flight_cuda.py",
    ]
    finite = np.isfinite(expected["distance_ang"])
    delta = np.abs(actual["distance_ang"][finite] - expected["distance_ang"][finite])
    print(
        json.dumps(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "scope": "standalone first-flight parity; no full-track validation",
                "device": backend.device.name,
                "compute_capability": cp.cuda.Device().compute_capability,
                "cupy": cp.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "epdl_table_sha256": EPDL_TABLE_SHA256,
                "source_sha256": {
                    path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
                },
                "input_arrays_sha256": hashlib.sha256(
                    b"".join(a.tobytes() for a in args[:4])
                ).hexdigest(),
                "seed": SEED,
                "rays": N,
                "regions": 3,
                "warmups": WARMUPS,
                "repeats": REPEATS,
                "end_to_end_seconds": times,
                "cuda_over_cpu_median": float(np.median(times["cuda"]) / np.median(times["cpu"])),
                "distance_rtol": 2e-13,
                "distance_atol_ang": 1e-13,
                "max_distance_abs_difference_ang": float(delta.max(initial=0)),
                "interactions": int(finite.sum()),
                "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "device_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
                "memory_scope": "combined process peak RSS and retained CuPy pool; not per-arm peaks",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
