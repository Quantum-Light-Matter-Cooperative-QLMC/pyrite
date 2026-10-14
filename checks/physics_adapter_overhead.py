"""Small matched-revision CPU transport/scoring benchmark for ADR-0017.

Run through uv in each checkout, using identical locked dependencies and data.
No sweep, checkpoint I/O, downloads, or device execution. Reports cold process
initialization/first calls separately from warmed phase medians and all samples.
"""

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time


def benchmark(repeats: int, electrons: int) -> dict:
    started = time.perf_counter()
    import numpy as np

    import pyrite as pr
    from pyrite.api import build_case
    from pyrite.campaign.profiles import case_content_key
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.case_tables import case_table_markers

    if runner._RESOURCE_POLICY.gpu:
        raise ValueError("this bounded benchmark requires PYRITE_MC_BACKEND=cpu")
    case = build_case(
        pr.Scene(
            pr.Beam(energy_keV=30.0, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
            pr.Slab("hopg", thickness_ang=1000.0, tilt_deg=30.0, footprint=pr.Footprint(5.0, 5.0)),
            pr.Detector(),
        ),
        pr.Numerics(
            cxr_trials=electrons,
            brem_trials=electrons,
            energy_bins=pr.EnergyBins(
                cxr=np.linspace(1500.0, 4500.0, 31), brem=np.linspace(1000.0, 10000.0, 31)
            ),
        ),
    )
    initialization_s = time.perf_counter() - started

    def phases():
        start = time.perf_counter()
        transport = runner._transport_case(case, transport_core="lockstep")
        transported = time.perf_counter()
        output = runner._spectrum_case(case, transport)
        finished = time.perf_counter()
        return transported - start, finished - transported, transport, output

    first_transport, first_spectrum, transport, reference = phases()
    for _ in range(3):
        phases()
    samples = []
    for _ in range(repeats):
        transport_s, spectrum_s, transport, output = phases()
        for slot in ("spec", "spec_characteristic", "brem_wide", "brem"):
            np.testing.assert_array_equal(output[slot], reference[slot])
        samples.append({"transport_s": transport_s, "spectrum_s": spectrum_s})
    markers = case_table_markers(case)
    return {
        "schema": "pyrite.physics-adapter-overhead.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "backend": "cpu",
        "core": "lockstep",
        "electrons": electrons,
        "seed": case["seed"],
        "repeats": repeats,
        "warmup_calls": 3,
        "segments": int(transport["segs"]["L_ang"].size),
        "grid_nodes": 31,
        "initialization_s": initialization_s,
        "first_transport_s": first_transport,
        "first_spectrum_s": first_spectrum,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "source_content_key": case_content_key(case, xsgen_tables=markers),
        "table_markers": markers,
        "array_sha256": {
            key: hashlib.sha256(np.ascontiguousarray(reference[key]).tobytes()).hexdigest()
            for key in ("spec", "spec_characteristic", "brem_wide", "brem")
        },
        "samples": samples,
        "medians_s": {
            key: statistics.median(row[key] for row in samples)
            for key in ("transport_s", "spectrum_s")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--electrons", type=int, default=32)
    args = parser.parse_args()
    if args.repeats < 3 or not 1 <= args.electrons <= 128:
        parser.error("need repeats >= 3 and 1 <= electrons <= 128")
    print(json.dumps(benchmark(args.repeats, args.electrons), sort_keys=True))
