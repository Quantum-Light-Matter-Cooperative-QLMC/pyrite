"""Slice-F ensemble evidence: does substep refinement move collision statistics?

The energy-controlled propagator draws ONE optical depth ``tau = -log(U)`` per
physical flight and consumes it as ``tau -= ds / lambda(E_substep)`` across the
flight's numerical substeps.  Tightening ``max_dE_frac`` therefore changes the
energies at which the elastic hazard is evaluated, which moves the sampled
collision point, which decorrelates the whole downstream trajectory.  A
single-seed flight count is consequently meaningless as an acceptance measure:
the slice-F notes record 2568 / 2882 / 2599 / 2553 / 2605 distinct flights down
the ladder for one seed, a spread that is pure realization noise.

This script measures the criterion the way slice E measured its Part A --
ensemble means over independent seed replicates with Monte Carlo standard
errors.  Two rungs agree when their shift is inside the combined error.

What is being tested is not "refinement changes nothing".  Evaluating
``lambda(E)`` at substep energies instead of freezing it at ``E_start`` is
strictly more accurate: the Browning hazard rises as ``E`` falls, so the
unrefined draw understates the collision rate and overstates the mean flight
length by an amount first order in the step.  The claim under test is that this
bias is (a) inside Monte Carlo error at production step sizes and (b)
convergent in ``f`` where it is resolvable, so no rung can be reached at which
collision statistics are still moving.

Observables, per electron unless noted:

- ``flights/e``   physical flights (rows with ``substep_id == 0``).  In an
                  ungrooved single-layer slab a flight closes only on a
                  collision or on the electron's terminal event, so this is the
                  collision count plus one.
- ``L_flight``    ensemble mean physical flight length [Ang], the intensive
                  view of the same statistic: the sampled elastic mean free
                  path, which is what the optical-depth budget controls
                  directly.
- ``path/e``      total path length [Ang]
- ``E_ret``       energy at the last row [keV]
- ``clock``       transit clock at the last row [Ang, c=1]
- ``trans``/``back``/``stop``  exit-channel fractions

Numbers from this script back
``docs/validation/beam-transport/energy-controlled-propagation.md``.

Validation: energy-controlled-propagation

Run:  uv run python checks/collision_statistics_refinement.py
      uv run python checks/collision_statistics_refinement.py --quick
"""

import argparse
import time

import numpy as np

from pyrite.montecarlo.transport import simulate_trajectories

CARBON = [("C", 0.1136)]
TUNGSTEN = [("W", 0.06305)]

# (label, composition, E0_keV, thickness_ang).  The first case is the one the
# slice-F notes reported the single-seed flight-count spread for; the rest add a
# stopping-dominated low-Z case, a backscatter-dominated high-Z case, and a
# high-energy case where per-flight fractional loss is smallest.
CASES = [
    ("C   25 keV 4000A", CARBON, 25.0, 4.0e3),
    ("C   25 keV thick", CARBON, 25.0, 2.0e4),
    ("W   25 keV thick", TUNGSTEN, 25.0, 5.0e3),
    ("C  100 keV thick", CARBON, 100.0, 2.0e5),
]

# ``None`` is one row per physical flight: the hazard is held at the flight's
# start energy for the whole flight, which is the unrefined production draw.
LADDER = [None, 1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3, 5.0e-4]

NE = 1000
SEEDS = (7, 101, 2024, 31337, 11, 523, 90210, 4243, 61, 1729, 8675309, 314159)
QUICK_NE = 250
QUICK_SEEDS = 4
E_CUT_KEV = 1.0

# Flagged in the shift table.  Two-sided, so ~0.3% of comparisons trip by
# chance; the table prints every shift so a systematic sign pattern is visible
# whether or not it trips.
SIGMA_FLAG = 3.0

_KEYS = ("flights/e", "L_flight", "path/e", "E_ret", "clock", "trans", "back", "stop")
_WIDTHS = (
    ("f", 8),
    ("flights/e", 17),
    ("L_flight", 17),
    ("path/e", 19),
    ("E_ret", 17),
    ("clock", 19),
    ("trans", 19),
    ("back", 19),
    ("stop", 19),
)


def _row(cols, widths):
    for key, width in widths:
        if len(str(cols[key])) > width:
            raise ValueError(f"cell {key}={cols[key]!r} overflows width {width}")
    return " ".join(f"{cols[key]:>{width}}" for key, width in widths)


def _seed_means(segments):
    """Ensemble means of one realization.  One number per observable."""
    Ne = int(segments["Ne"])
    electron_id = np.asarray(segments["electron_id"]).astype(np.int64, copy=False)
    length = np.asarray(segments["L_ang"]).astype(float, copy=False)
    substep_id = np.asarray(segments["substep_id"]).astype(np.int64, copy=False)
    t_start = np.asarray(segments["t_start_ang"]).astype(float, copy=False)
    E_end = np.asarray(segments["E_end_keV"]).astype(float, copy=False)
    t_end = np.asarray(segments["t_end_ang"]).astype(float, copy=False)

    # Electrons that never enter the target contribute no row; they stay in the
    # denominator with zero path so every rung divides by the same Ne.
    path = np.bincount(electron_id, weights=length, minlength=Ne)
    n_flights = np.bincount(electron_id, weights=(substep_id == 0).astype(float), minlength=Ne)

    retained = np.zeros(Ne)
    clock = np.zeros(Ne)
    if electron_id.size:
        # Rows of one electron are emitted in flight order, but the lockstep
        # core interleaves electrons, so the last row is selected by clock.
        order = np.lexsort((t_start, electron_id))
        last = order[np.flatnonzero(np.diff(electron_id[order], append=-1))]
        retained[electron_id[last]] = E_end[last]
        clock[electron_id[last]] = t_end[last]

    total_flights = float(n_flights.sum())
    return {
        "flights/e": float(n_flights.mean()),
        "L_flight": float(path.sum() / total_flights) if total_flights else float("nan"),
        "path/e": float(path.mean()),
        "E_ret": float(retained.mean()),
        "clock": float(clock.mean()),
        "trans": int(segments["n_transmitted"]) / Ne,
        "back": int(segments["n_backscattered"]) / Ne,
        "stop": int(segments["n_cutoff_stopped"]) / Ne,
    }


def _rung(composition, E0_keV, thickness_ang, max_dE_frac, Ne, seeds):
    """Per-seed ensemble means, one array of length ``len(seeds)`` per observable."""
    per_seed = []
    for seed in seeds:
        segments = simulate_trajectories(
            E0_keV,
            Ne,
            thickness_ang,
            composition=composition,
            E_cut_keV=E_CUT_KEV,
            seed=seed,
            energy_model="midpoint",
            max_dE_frac=0.0 if max_dE_frac is None else max_dE_frac,
        )
        per_seed.append(_seed_means(segments))
    return {key: np.array([s[key] for s in per_seed], dtype=float) for key in _KEYS}


def _mean_pm(values):
    """Mean and standard error of the mean over independent seed replicates.

    The replicates are independent realizations, so the seed-to-seed spread
    already carries every source of sampling error; no per-observable variance
    model is assumed.
    """
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _paired_shift_sigma(values, reference):
    """Per-seed difference against the reference rung, in units of its own error.

    Two rungs sharing a seed are not independent: refinement decorrelates a
    trajectory only after enough energy has been lost for the substep grid to
    move a collision point, so the early history and the incident sampling are
    common. Differencing seed by seed cancels that common variance, which makes
    this test strictly more sensitive to a systematic refinement bias than
    combining the two rungs' standard errors. It degrades to the unpaired test
    when the rungs are fully decorrelated.
    """
    difference = values - reference
    error = float(difference.std(ddof=1) / np.sqrt(difference.size))
    if error == 0.0:
        return 0.0
    return float(difference.mean()) / error


def _fmt(pair, digits=4):
    value, error = pair
    return f"{value:.{digits}g}+-{error:.2g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer electrons and seeds")
    args = parser.parse_args()

    Ne = QUICK_NE if args.quick else NE
    seeds = SEEDS[:QUICK_SEEDS] if args.quick else SEEDS

    print(f"Collision statistics vs substep refinement   Ne={Ne} x {len(seeds)} seeds")
    print(f"E_cut = {E_CUT_KEV} keV, energy_model='midpoint', lockstep core")
    print("Errors are the standard error of the mean over seed replicates.\n")

    for label, composition, E0_keV, thickness_ang in CASES:
        started = time.time()
        rungs = [(f, _rung(composition, E0_keV, thickness_ang, f, Ne, seeds)) for f in LADDER]
        print(f"{label}  ({thickness_ang:.3g} Ang)   {time.time() - started:.1f} s")
        print(_row({key: key for key, _ in _WIDTHS}, _WIDTHS))
        for f, samples in rungs:
            cols = {"f": "none" if f is None else f"{f:.0e}"}
            cols.update({key: _fmt(_mean_pm(samples[key])) for key in _KEYS})
            print(_row(cols, _WIDTHS))

        # Every rung against the finest one.  A refinement bias shows up as a
        # monotone sign pattern shrinking down the ladder; realization noise
        # does not.
        reference = rungs[-1][1]
        print(f"  paired shift vs f={LADDER[-1]:.0e}, in sigma:")
        for f, samples in rungs[:-1]:
            cells = []
            for key in _KEYS:
                shift = _paired_shift_sigma(samples[key], reference[key])
                mark = "*" if abs(shift) > SIGMA_FLAG else " "
                cells.append(f"{key}={shift:+6.2f}{mark}")
            print(f"    {'none' if f is None else f'{f:.0e}':>6}  " + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
