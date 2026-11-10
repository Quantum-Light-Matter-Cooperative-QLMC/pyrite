"""Pinned CPU vs CUDA source-spectrum parity probe for issue #344.

The three synthetic, seeded transport records are replayed unchanged by three
fresh processes (CPU FP64, CUDA FP64, CUDA FP32). Four independently scored
channels are checked: incoherent line, bremsstrahlung, characteristic and
coherent line sums. No GPU simulator or CPU fallback can pass for CUDA.

Run on the lab box after ``pyrite remote sync`` with its pinned tables fetched::

    uv run --extra nvidia python checks/scheduled_cuda_parity.py --output-dir evidence/parity-344

The numerical bounds are initial regression guards, NOT an uncertainty model
or statistical validation of real transport. They require hardware calibration
before being treated as stable physics acceptance thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Seed, beam energy (keV), transverse spread (angstrom), longitudinal scale.
CASES = ((344, 30.0, 2000.0, 1.0), (345, 60.0, 3500.0, 1.4), (346, 100.0, 4800.0, 2.0))
CHANNELS = ("line", "brem", "characteristic", "coherent")
# Same histories => no Monte Carlo sampling discrepancy between backends.
# FP32 phase summation is less stable; #298 reported an approximately 1.9%
# difference. These provisional peak-relative limits intentionally cover it.
BOUNDS = {"cuda64": 1e-5, "cuda32": 3e-2}


def segments(seed: int, energy: float, transverse: float, scale: float) -> dict:
    from pyrite.montecarlo.transport import beta_from_keV

    rng = np.random.default_rng(seed)
    ne, per_electron = 12, 16
    count = ne * per_electron
    v_hat = np.column_stack((rng.normal(0, 0.015, (count, 2)), np.ones(count)))
    v_hat /= np.linalg.norm(v_hat, axis=1, keepdims=True)
    r_mid = np.column_stack(
        (
            rng.uniform(-transverse, transverse, (count, 2)),
            rng.uniform(100, 5000 * scale, count),
        )
    )
    length = np.full(count, 250.0 * scale)
    return {
        "r_mid": r_mid,
        "v_hat": v_hat,
        "L_ang": length,
        "E_keV": np.full(count, energy),
        "t_ang": np.tile(np.arange(per_electron) * 250.0 * scale / beta_from_keV(energy), ne),
        "elec_id": np.repeat(np.arange(ne), per_electron),
        "layer": np.zeros(count, dtype=int),
        "Ne": ne,
        "thickness_ang": 1e6,
        "crystal_width_ang": 1e4,
        "crystal_height_ang": 1e4,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def snapshot(case_id: int, output: Path, backend: str) -> None:
    from pyrite._backend import BACKEND
    from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
    from pyrite.montecarlo.spectrum.characteristic import mc_characteristic_spectrum
    from pyrite.montecarlo.spectrum.cxr import mc_cxr_spectrum

    if backend == "cpu":
        if BACKEND.name != "cpu":
            raise RuntimeError(f"expected CPU, got {BACKEND.name}")
    else:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1 or BACKEND.name != "cuda":
            raise RuntimeError(f"real CUDA backend required, got {BACKEND.name}")
    seed, energy, transverse, scale = CASES[case_id]
    rows = segments(seed, energy, transverse, scale)
    # HOPG (002) at ~90 deg sits near hc*beta/d: ~1.2 keV at 30 keV and
    # ~2.0 keV at 100 keV, so the grid must reach past 2 keV (4 eV spacing).
    grid = np.linspace(70.0, 2510.0, 611)
    n_hat = np.array([1.0, 0.0, 0.01])
    n_hat /= np.linalg.norm(n_hat)
    line_kwargs = {
        "crystal": "hopg",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.8,
        "n_hat": n_hat,
    }

    def to_host(value):
        return np.asarray(value.get() if hasattr(value, "get") else value, dtype=np.float64)

    spectra = {
        "line": to_host(mc_cxr_spectrum(rows, grid, coherent=False, **line_kwargs)),
        "brem": to_host(
            mc_brem_spectrum(
                rows,
                grid,
                composition=[("C", 0.1)],
                n_hat=n_hat,
                cross_section_model="eedl",
            )
        ),
        "characteristic": to_host(
            mc_characteristic_spectrum(rows, grid, composition=[("C", 0.1)], n_hat=n_hat)
        ),
        "coherent": to_host(mc_cxr_spectrum(rows, grid, coherent=True, **line_kwargs)),
    }
    for name, values in spectra.items():
        if values.shape != grid.shape or not np.all(np.isfinite(values)) or np.any(values < -1e-30):
            raise RuntimeError(f"invalid {name} output on {backend}")
        if not np.any(values > 0):
            raise RuntimeError(f"{name} spectrum is identically zero on {backend}; widen the grid")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, grid=grid, **spectra)


def compare(
    reference: np.ndarray, candidate: np.ndarray, *, bound: float
) -> dict[str, float | bool]:
    """Peak-scaled absolute maximum error and integrated positive yield error."""
    if reference.shape != candidate.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("shape mismatch or nonfinite candidate")
    peak = float(np.max(np.abs(reference)))
    if peak <= 0 or not np.isfinite(peak):
        raise ValueError("zero/nonfinite reference spectrum; cannot establish parity")
    peak_error = float(np.max(np.abs(candidate - reference)) / peak)
    sum_ref = float(np.sum(reference))
    integrated_error = (
        float(abs(np.sum(candidate) - sum_ref) / abs(sum_ref)) if sum_ref else float("inf")
    )
    return {
        "peak_relative_error": peak_error,
        "integrated_relative_error": integrated_error,
        "bound": bound,
        "pass": peak_error <= bound and integrated_error <= bound,
    }


def run(output_dir: Path) -> int:
    from pyrite.validation.check_records import current_revision

    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "schema": 1,
        "issue": 344,
        "revision": current_revision(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "cases": [],
    }
    failed = False
    for index, (seed, energy, transverse, scale) in enumerate(CASES):
        data: dict = {
            "seed": seed,
            "energy_keV": energy,
            "transverse_ang": transverse,
            "scale": scale,
            "channels": {},
        }
        paths: dict[str, Path] = {}
        for backend in ("cpu", "cuda64", "cuda32"):
            output = output_dir / f"case{index}-{backend}.npz"
            env = dict(os.environ)
            env["PYRITE_MC_BACKEND"] = "cpu" if backend == "cpu" else "cuda"
            env.pop("PYRITE_TEST_BACKEND", None)
            env.pop("NUMBA_DISABLE_JIT", None)
            env.pop("PYRITE_FP64", None)
            if backend == "cuda64":
                env["PYRITE_FP64"] = "1"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--snapshot",
                str(index),
                "--backend",
                env["PYRITE_MC_BACKEND"],
                "--out",
                str(output.resolve()),
            ]
            with (output_dir / f"case{index}-{backend}.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                data["error"] = f"{backend} exited {result.returncode}; inspect its .log"
                failed = True
                break
            paths[backend] = output
        if len(paths) == 3:
            with np.load(paths["cpu"], allow_pickle=False) as base:
                for backend in ("cuda64", "cuda32"):
                    with np.load(paths[backend], allow_pickle=False) as candidate:
                        if not np.array_equal(base["grid"], candidate["grid"]):
                            data["error"] = f"{backend} grid changed"
                            failed = True
                            continue
                        for channel in CHANNELS:
                            try:
                                metrics = compare(
                                    base[channel], candidate[channel], bound=BOUNDS[backend]
                                )
                            except ValueError as exc:
                                metrics = {"pass": False, "error": str(exc)}
                            data["channels"][f"{channel}_{backend}"] = metrics
                            failed |= not metrics["pass"]
        report["cases"].append(data)
    report["result"] = "fail" if failed else "pass"
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot", type=int, choices=range(len(CASES)))
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.snapshot is not None:
        if args.backend is None or args.out is None:
            parser.error("snapshot requires --backend and --out")
        snapshot(args.snapshot, args.out, args.backend)
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    return run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
