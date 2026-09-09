"""Bit-for-bit goldens for the five CPU transport cores (issue #66 contract).

``capture`` runs every CPU core x energy model x straggling combination through
``simulate_trajectories`` at a fixed seed and stores every returned array and
counter under ``--dir``. ``check`` reruns the same matrix and compares against
the stored goldens with exact (bit-for-bit) equality -- these runs are
deterministic on a given CPU/build, so any drift is a real change, not noise.

Goldens must be captured from the pre-collapse code; ``--dir`` data is a local
working artifact and is not committed.

Usage:
    uv run python checks/transport_core_goldens.py capture --dir DIR
    uv run python checks/transport_core_goldens.py check --dir DIR
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pyrite.montecarlo.groove import blazed_groove_spec
from pyrite.montecarlo.transport.api import simulate_trajectories
from pyrite.montecarlo.transport.lut import TransportLUTConfig

CARBON = [("C", 0.1136)]
E0_KEV = 30.0
NE = 300
THICKNESS_ANG = 3.0e4
SEED = 66

GROOVE = blazed_groove_spec(
    spacing_ang=2.0e4,
    theta_obs_rad=np.pi / 2,
    tilt_polar_rad=np.deg2rad(45.0),
    tilt_azim_rad=np.pi,
)

# One entry per CPU kernel under test. Grooved only exists as a lockstep core;
# the LUT is only built for ungrooved runs.
KERNELS = {
    "lockstep": dict(transport_core="lockstep", lut=False),
    "lockstep_lut": dict(transport_core="lockstep", lut=True),
    "grooved": dict(transport_core="lockstep", lut=False, groove=True),
    "perelectron": dict(transport_core="per-electron", lut=False),
    "perelectron_lut": dict(transport_core="per-electron", lut=True),
}

ENERGY_MODELS = ("frozen", "midpoint")
STRAGGLING = (False, True)

# Keys compared bit-for-bit. Scalars go to a JSON sidecar instead.
_SKIP_KEYS = {"Ne", "thickness_ang", "crystal_width_ang", "crystal_height_ang", "n_layers"}


def _run(kernel, energy_model, straggling, ne=NE):
    spec = KERNELS[kernel]
    kwargs = dict(
        composition=CARBON,
        E_cut_keV=5.0,
        seed=SEED,
        transport_core=spec["transport_core"],
        transport_lut_config=TransportLUTConfig(enabled=spec["lut"]),
        energy_model=energy_model,
        # Substepping on the midpoint runs exercises the (flight, substep)
        # stream addressing; frozen forbids it.
        max_dE_frac=0.05 if energy_model == "midpoint" else 0.0,
        straggling=straggling,
    )
    if spec.get("groove"):
        kwargs["groove"] = GROOVE
    start = time.perf_counter()
    result = simulate_trajectories(E0_KEV, ne, THICKNESS_ANG, **kwargs)
    return result, time.perf_counter() - start


def _iter_runs(ne=NE):
    for kernel in KERNELS:
        for energy_model in ENERGY_MODELS:
            for straggling in STRAGGLING:
                name = _case_name(kernel, energy_model, straggling)
                result, elapsed = _run(kernel, energy_model, straggling, ne)
                yield name, result, elapsed


def _case_name(kernel, energy_model, straggling):
    return f"{kernel}_{energy_model}_{'straggle' if straggling else 'nostraggle'}"


def _split(result):
    arrays, scalars = {}, {}
    for key, value in result.items():
        if key in _SKIP_KEYS:
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalars[key] = value
    return arrays, scalars


def capture(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name, result, elapsed in _iter_runs():
        arrays, scalars = _split(result)
        np.savez(out_dir / f"{name}.npz", **arrays)
        (out_dir / f"{name}.json").write_text(json.dumps(scalars, indent=2))
        timings[name] = elapsed
        print(f"captured {name}: nseg={arrays['L_ang'].size} in {elapsed:.2f}s")
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))


BENCH_NE = 4000
BENCH_VARIANTS = (("frozen", False), ("midpoint", True))


def bench(out_dir):
    """Warm per-kernel runtime at a measurable scale (no goldens involved)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    for kernel in KERNELS:
        for energy_model, straggling in BENCH_VARIANTS:
            name = _case_name(kernel, energy_model, straggling)
            _, elapsed = _run(kernel, energy_model, straggling, BENCH_NE)
            timings[name] = elapsed
            print(f"bench {name}: {elapsed:.2f}s")
    (out_dir / "timings_bench.json").write_text(json.dumps(timings, indent=2))


def check(golden_dir):
    failures = []
    timings = {}
    for name, result, elapsed in _iter_runs():
        timings[name] = elapsed
        arrays, scalars = _split(result)
        golden = np.load(golden_dir / f"{name}.npz")
        golden_scalars = json.loads((golden_dir / f"{name}.json").read_text())
        if set(golden.files) != set(arrays):
            failures.append(f"{name}: array keys {sorted(arrays)} != {sorted(golden.files)}")
            continue
        for key in golden.files:
            if not np.array_equal(arrays[key], golden[key], equal_nan=True):
                failures.append(f"{name}: array {key} differs")
        for key, value in golden_scalars.items():
            if scalars.get(key) != value:
                failures.append(f"{name}: scalar {key} {scalars.get(key)!r} != {value!r}")
        print(f"checked {name} in {elapsed:.2f}s")
    (golden_dir / "timings_check.json").write_text(json.dumps(timings, indent=2))
    if failures:
        print(f"\n{len(failures)} mismatch(es):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nall goldens reproduce bit-for-bit")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "check", "bench"))
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.dir)
        return 0
    if args.mode == "bench":
        bench(args.dir)
        return 0
    return check(args.dir)


if __name__ == "__main__":
    sys.exit(main())
