"""Measure the observable effect of Urban energy-loss straggling (slice H).

The comparison is paired by transport seed: every ``straggling=False`` run is
matched to a ``straggling=True`` run with the same free-path and scattering
streams.  Independent seeds, rather than electrons within one run, are the
replicates used for uncertainty estimates.

Measured at the motivating HOPG point (25 keV electrons, 1 um traversal,
1 keV clock phase):

* backscatter, transmission, and cutoff-stop fractions;
* the mean and standard deviation of the path range among cutoff-stopped
  electrons in a 5 um slab;
* the transmitted-electron terminal clock and its 1 keV phase;
* bremsstrahlung yield and normalized spectral-shape distance; and
* the production coherent-line peak and integrated yield for HOPG (002).

The 1 keV clock phase is ``phi = E_gamma * t / (hbar c)`` because transport
stores ``t`` in Angstrom with ``c=1``.  It is a clock-only diagnostic, not the
coherent kernel's full phase and not a Debye--Waller exponent: the latter also
contains position, reciprocal-lattice, and in-medium propagation terms.

The corrected analytic target from slices A/B is a 9.8--19 rad mean clock
bias.  ``phase_residual_rad`` is the signed distance of the measured magnitude
from that interval (zero inside it), not a claim that the outstanding
mean-stopping uncertainty has closed.

Validation: energy-loss-straggling

Run:
  uv run python checks/energy_loss_straggling_observables.py --quick
  uv run python checks/energy_loss_straggling_observables.py --output REPORT.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from pyrite.montecarlo.spectrum import mc_brem_spectrum, mc_spectrum
from pyrite.montecarlo.transport import beta_from_keV, simulate_trajectories

E0_KEV = 25.0
E_CUT_KEV = 1.0
THIN_ANG = 1.0e4
THICK_ANG = 5.0e4
PHOTON_E_EV = 1.0e3
COMPOSITION = [("C", 0.1136)]
SEEDS = (7, 101, 2024, 31337, 11, 523, 90210, 4243)

FULL_NE = 600
QUICK_NE = 120
QUICK_SEEDS = 3

BREM_GRID_EV = np.linspace(250.0, 25_000.0, 121)
THETA_OBS_RAD = np.deg2rad(119.0)
HKL = (0, 0, 2)
LINE_HALF_WINDOW_EV = 200.0
PHASE_TARGET_RAD = (9.8, 19.0)


def _last_rows(segments: dict) -> np.ndarray:
    """Final emitted row for each represented electron, ordered by id."""
    electron_id = np.asarray(segments["electron_id"], dtype=np.int64)
    t_start = np.asarray(segments["t_start_ang"], dtype=float)
    if not electron_id.size:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((t_start, electron_id))
    return order[np.flatnonzero(np.diff(electron_id[order], append=-1))]


def _terminal_observables(segments: dict) -> dict[str, float]:
    """Per-realization transport observables, including stopped path range."""
    Ne = int(segments["Ne"])
    electron_id = np.asarray(segments["electron_id"], dtype=np.int64)
    length = np.asarray(segments["L_ang"], dtype=float)
    path = np.bincount(electron_id, weights=length, minlength=Ne)
    last = _last_rows(segments)
    last_id = electron_id[last]

    r_mid = np.asarray(segments["r_mid"], dtype=float)[last]
    v_hat = np.asarray(segments["v_hat"], dtype=float)[last]
    endpoint = r_mid + 0.5 * length[last, None] * v_hat
    thickness = float(segments["thickness_ang"])
    tolerance = max(1.0e-6, 1.0e-8 * thickness)
    back = endpoint[:, 2] <= tolerance
    trans = endpoint[:, 2] >= thickness - tolerance
    stopped = ~(back | trans)

    expected = {
        "back": int(segments["n_backscattered"]),
        "trans": int(segments["n_transmitted"]),
        "stop": int(segments["n_cutoff_stopped"]),
    }
    measured = {"back": int(back.sum()), "trans": int(trans.sum()), "stop": int(stopped.sum())}
    if measured != expected:
        raise RuntimeError(f"terminal classification disagrees: {measured=} {expected=}")

    t_end = np.asarray(segments["t_end_ang"], dtype=float)[last]
    trans_clock = t_end[trans]
    stopped_path = path[last_id[stopped]]
    phase = PHOTON_E_EV * trans_clock / HBARC_EV_ANG
    fixed_path_clock = _clock_at_path(segments, THIN_ANG)
    fixed_path_phase = PHOTON_E_EV * fixed_path_clock / HBARC_EV_ANG

    def _mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else float("nan")

    def _std(values: np.ndarray) -> float:
        return float(values.std(ddof=1)) if values.size > 1 else float("nan")

    return {
        "backscatter_fraction": expected["back"] / Ne,
        "transmission_fraction": expected["trans"] / Ne,
        "cutoff_stop_fraction": expected["stop"] / Ne,
        "stopped_range_mean_ang": _mean(stopped_path),
        "stopped_range_std_ang": _std(stopped_path),
        "transmitted_clock_mean_ang": _mean(trans_clock),
        "transmitted_clock_std_ang": _std(trans_clock),
        "transmitted_phase_mean_rad": _mean(phase),
        "transmitted_phase_std_rad": _std(phase),
        "fixed_path_fraction": fixed_path_clock.size / Ne,
        "fixed_path_clock_mean_ang": _mean(fixed_path_clock),
        "fixed_path_clock_std_ang": _std(fixed_path_clock),
        "fixed_path_phase_mean_rad": _mean(fixed_path_phase),
        "fixed_path_phase_std_rad": _std(fixed_path_phase),
    }


def _clock_at_path(segments: dict, path_ang: float) -> np.ndarray:
    """Electron clock when cumulative material path first reaches ``path_ang``.

    This removes geometric path-length variation from the Jensen-bias
    measurement.  The row containing the crossing uses the propagator's own
    midpoint representative energy, matching its clock quadrature.
    """
    electron_id = np.asarray(segments["electron_id"], dtype=np.int64)
    t_start = np.asarray(segments["t_start_ang"], dtype=float)
    length = np.asarray(segments["L_ang"], dtype=float)
    energy = np.asarray(segments["E_repr_keV"], dtype=float)
    order = np.lexsort((t_start, electron_id))
    clocks = []
    for current_id in np.unique(electron_id):
        rows = order[electron_id[order] == current_id]
        cumulative = np.cumsum(length[rows])
        crossing = int(np.searchsorted(cumulative, path_ang, side="left"))
        if crossing == rows.size:
            continue
        row = rows[crossing]
        before = 0.0 if crossing == 0 else cumulative[crossing - 1]
        remaining = path_ang - before
        beta = float(beta_from_keV(np.asarray(energy[row])))
        clocks.append(t_start[row] + remaining / beta)
    return np.asarray(clocks, dtype=float)


def _line_grid() -> np.ndarray:
    crystal = CRYSTALS["hopg"]
    g_vec, _ = reciprocal_g_vector(HKL, crystal["lattice"])
    n_hat = np.array([np.sin(THETA_OBS_RAD), 0.0, np.cos(THETA_OBS_RAD)])
    beta = float(beta_from_keV(np.asarray(E0_KEV)))
    resonance = HBARC_EV_ANG * beta * g_vec[2] / (1.0 - beta * n_hat[2])
    return np.linspace(
        max(1.0, resonance - LINE_HALF_WINDOW_EV),
        resonance + LINE_HALF_WINDOW_EV,
        161,
    )


def _spectrum_observables(segments: dict, line_grid: np.ndarray) -> dict[str, float]:
    with np.errstate(all="ignore"):
        brem = np.asarray(
            mc_brem_spectrum(segments, BREM_GRID_EV, composition=COMPOSITION), dtype=float
        )
        line = np.asarray(
            mc_spectrum(
                segments,
                line_grid,
                crystal="hopg",
                hkl_list=[HKL],
                B_ang2=0.8,
                theta_obs_rad=THETA_OBS_RAD,
                composition=COMPOSITION,
                coherent=True,
            ),
            dtype=float,
        )
    return {
        "brem_yield": float(np.trapezoid(brem, BREM_GRID_EV)),
        "line_yield": float(np.trapezoid(line, line_grid)),
        "line_peak": float(line.max()),
        "line_peak_energy_eV": float(line_grid[int(np.argmax(line))]),
        "_brem": brem,
    }


def _shape_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Total-variation distance between spectra normalized to unit area."""
    left_area = float(np.trapezoid(left, BREM_GRID_EV))
    right_area = float(np.trapezoid(right, BREM_GRID_EV))
    if left_area <= 0.0 or right_area <= 0.0:
        return float("nan")
    return float(
        0.5 * np.trapezoid(np.abs(left / left_area - right / right_area), BREM_GRID_EV)
    )


def _mean_sem(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"mean": None, "sem": None, "n": 0}
    sem = float(finite.std(ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else None
    return {"mean": float(finite.mean()), "sem": sem, "n": int(finite.size)}


def _paired_summary(off: list[dict[str, float]], on: list[dict[str, float]]) -> dict:
    keys = sorted(set(off[0]) & set(on[0]) - {"_brem"})
    result = {}
    for key in keys:
        off_values = [row[key] for row in off]
        on_values = [row[key] for row in on]
        differences = [b - a for a, b in zip(off_values, on_values, strict=True)]
        relative = [
            (b - a) / a if np.isfinite(a) and np.isfinite(b) and a != 0.0 else float("nan")
            for a, b in zip(off_values, on_values, strict=True)
        ]
        result[key] = {
            "off": _mean_sem(off_values),
            "on": _mean_sem(on_values),
            "paired_on_minus_off": _mean_sem(differences),
            "paired_relative_change": _mean_sem(relative),
        }
    return result


def _git_revision() -> str:
    supplied = os.environ.get("PYRITE_MEASUREMENT_REVISION")
    if supplied:
        return supplied
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "source-snapshot"


def run(*, quick: bool) -> dict:
    Ne = QUICK_NE if quick else FULL_NE
    seeds = SEEDS[:QUICK_SEEDS] if quick else SEEDS
    line_grid = _line_grid()
    thin: dict[bool, list[dict[str, float]]] = {False: [], True: []}
    thick: dict[bool, list[dict[str, float]]] = {False: [], True: []}
    shape_distances = []
    started = time.perf_counter()

    for seed in seeds:
        spectra = {}
        for straggling in (False, True):
            common = {
                "E0_keV": E0_KEV,
                "Ne": Ne,
                "composition": COMPOSITION,
                "E_cut_keV": E_CUT_KEV,
                "seed": seed,
                "energy_model": "midpoint",
                "straggling": straggling,
            }
            thin_segments = simulate_trajectories(thickness_ang=THIN_ANG, **common)
            thin_metrics = _terminal_observables(thin_segments)
            spectra[straggling] = _spectrum_observables(thin_segments, line_grid)
            thin_metrics.update(spectra[straggling])
            thin[straggling].append(thin_metrics)

            thick_segments = simulate_trajectories(thickness_ang=THICK_ANG, **common)
            thick[straggling].append(_terminal_observables(thick_segments))

        shape_distances.append(
            _shape_distance(spectra[False]["_brem"], spectra[True]["_brem"])
        )

    thin_summary = _paired_summary(thin[False], thin[True])
    thick_summary = _paired_summary(thick[False], thick[True])
    phase_shift = abs(thin_summary["fixed_path_phase_mean_rad"]["paired_on_minus_off"]["mean"])
    if phase_shift < PHASE_TARGET_RAD[0]:
        residual = phase_shift - PHASE_TARGET_RAD[0]
    elif phase_shift > PHASE_TARGET_RAD[1]:
        residual = phase_shift - PHASE_TARGET_RAD[1]
    else:
        residual = 0.0

    return {
        "schema_version": 1,
        "revision": _git_revision(),
        "quick": quick,
        "elapsed_seconds": time.perf_counter() - started,
        "workload": {
            "E0_keV": E0_KEV,
            "E_cut_keV": E_CUT_KEV,
            "thin_thickness_ang": THIN_ANG,
            "thick_thickness_ang": THICK_ANG,
            "photon_energy_eV": PHOTON_E_EV,
            "electrons_per_seed": Ne,
            "seeds": list(seeds),
            "energy_model": "midpoint",
            "max_dE_frac": 0.0,
            "transport_core": "auto (CPU lockstep in this run)",
        },
        "phase_target_rad": list(PHASE_TARGET_RAD),
        "phase_residual_rad": residual,
        "thin_1um": thin_summary,
        "thick_5um": thick_summary,
        "brem_normalized_shape_tv": _mean_sem(shape_distances),
        "interpretation": {
            "phase": "clock-only at 1 keV; not the full coherent-emission phase",
            "line": "coherent=True pure-geometry limit (all bunch offsets zero)",
            "uncertainty": "SEM over independent seed replicas; shifts are paired by seed",
            "mean_stopping": "residual mean-stopping uncertainty is outside this task",
            "cuda": "not exercised by this bounded CPU measurement",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="120 electrons x 3 seeds")
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    args = parser.parse_args()
    report = run(quick=args.quick)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
