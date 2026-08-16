"""Slice-G evidence: is emitted radiation invariant to numerical substepping?

Slices C--F made a physical flight's energy and clock controllable by splitting
it into numerical substeps.  Those substeps are integration detail.  The
radiation kernels must therefore converge as the substep grid is refined, and
must not read a substep as an independent emitter -- otherwise tightening a
numerical tolerance would change the predicted spectrum, which is the failure
mode this slice exists to close.

The physical flights are held FIXED here.  ``subdivide_flights`` splits one
transport run's flights along their own straight rays, refining only the
sampling of the energy and clock, so nothing in the measurement is confounded
by the trajectory decorrelation that a second transport run at a tighter
``max_dE_frac`` would introduce (slice F,
``docs/validation/beam-transport/energy-controlled-propagation.md``).

Four L1 columns per rung, each a difference over the spectral grid against the
finest rung of the same column, so a converging reduction drives its column to
zero down the ladder:

- ``brem``      production bremsstrahlung.  Each row is a one-point quadrature
                of its own path integral, evaluated at ``E_repr_keV``.
- ``brem_lep``  the same with the representative energy withheld, i.e. the
                historical left-endpoint evaluation.  The pair isolates what
                the representative energy buys.
- ``cxr``       production (incoherent) CXR with flight grouping: substeps of
                one flight add coherently, whole flights add incoherently.
- ``cxr_ungr``  the same rows with ``flight_id`` withheld, i.e. every substep
                treated as an independent emitter.  This is the regression the
                grouping prevents.

Two peak columns read the line height, which is the sharpest single number:

- ``peak``      grouped line peak over the finest rung's grouped peak.  This is
                the convergence claim: it approaches 1 from the coarse end.
- ``ungr/gr``   ungrouped line peak over the grouped peak AT THE SAME RUNG, so
                the two share every row.  It is 1 by construction on the
                unsplit rung (one row per flight leaves nothing to group) and
                falls roughly as 1/N_substep afterwards.

Numbers from this script back
``docs/validation/beam-transport/substep-radiation-invariance.md``.

Validation: substep-radiation-invariance

Run:  uv run python checks/substep_invariance.py
      uv run python checks/substep_invariance.py --quick
"""

import argparse
import time

import numpy as np

from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from pyrite.montecarlo.spectrum import (
    _observation_direction,
    mc_brem_spectrum,
    mc_spectrum,
    subdivide_flights,
)
from pyrite.montecarlo.transport import beta_from_keV, simulate_trajectories

CARBON = [("C", 0.1136)]
TUNGSTEN = [("W", 0.06305)]

# CXR needs the crystal to BE the transported material, so the hopg columns are
# carbon-only; tungsten contributes bremsstrahlung alone.
CASES = [
    ("C   25 keV thin", CARBON, 25.0, 2.0e3),
    ("C   25 keV thick", CARBON, 25.0, 2.0e4),
    ("C  100 keV thick", CARBON, 100.0, 2.0e5),
    ("W   25 keV thick", TUNGSTEN, 25.0, 5.0e3),
    ("W  100 keV thick", TUNGSTEN, 100.0, 5.0e4),
]

# ``None`` is one row per physical flight: the production input.
LADDER = [None, 5.0e-3, 1.0e-3]
LADDER_REFERENCE = 2.0e-4
# A cutoff-stopped flight can lose most of its energy, so the substep count is
# bounded; the reference rung needs the highest bound of the ladder.
MAX_SUBSTEPS = 256

BREM_GRID = np.linspace(2.0e2, 2.0e4, 401)
CXR_THETA_OBS_RAD = np.deg2rad(119.0)
CXR_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "theta_obs_rad": CXR_THETA_OBS_RAD,
}
# The resonance rides beta, so the line window is built per beam energy.  The L1
# window matches the slice-E matrix; the peak is read from a narrow band around
# the resonance because the wide window's own maximum sits on the soft edge of
# the spectrum, not on the line.
CXR_HALF_WINDOW_EV = 400.0
CXR_PEAK_BAND_EV = 40.0

NE = 200
QUICK_NE = 40
SEED = 7
E_CUT_KEV = 1.0

_WIDTHS = (
    ("f", 8),
    ("rows", 8),
    ("brem", 10),
    ("brem_lep", 10),
    ("cxr", 10),
    ("cxr_ungr", 10),
    ("peak", 8),
    ("ungr/gr", 9),
)


def _row(cols):
    return " ".join(f"{cols[key]:>{width}}" for key, width in _WIDTHS)


def _cxr_grid(E0_keV, g_vec):
    """1 eV grid centred on the forward-beam resonance, plus the line-peak mask.

    ``E_res = hbar_c (beta v.g) / (1 - beta v.n)`` at the incident direction +z,
    the kernel's own resonance condition evaluated before any scattering.
    """
    unit = _observation_direction(CXR_THETA_OBS_RAD, None)
    v_hat = np.array([0.0, 0.0, 1.0])
    beta = float(beta_from_keV(np.asarray(E0_keV, dtype=float)))
    E_res = HBARC_EV_ANG * beta * float(v_hat @ g_vec) / (1.0 - beta * float(v_hat @ unit))
    grid = np.arange(max(1.0, E_res - CXR_HALF_WINDOW_EV), E_res + CXR_HALF_WINDOW_EV, 1.0)
    return grid, np.abs(grid - E_res) <= CXR_PEAK_BAND_EV


def _grid_l1(value, reference):
    total = float(np.abs(reference).sum())
    if total == 0.0:
        return float("nan")
    return float(np.abs(value - reference).sum() / total)


def _reductions(rows, composition, cxr_grid):
    """Production kernel reductions plus their withheld-field counterparts."""
    without_repr = {key: value for key, value in rows.items() if key != "E_repr_keV"}
    out = {
        "brem": mc_brem_spectrum(rows, BREM_GRID, composition=composition),
        "brem_lep": mc_brem_spectrum(without_repr, BREM_GRID, composition=composition),
    }
    if cxr_grid is not None:
        without_flight = {key: value for key, value in rows.items() if key != "flight_id"}
        out["cxr"] = mc_spectrum(rows, cxr_grid, composition=composition, **CXR_KWARGS)
        out["cxr_ungr"] = mc_spectrum(
            without_flight, cxr_grid, composition=composition, **CXR_KWARGS
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer electrons")
    args = parser.parse_args()
    Ne = QUICK_NE if args.quick else NE

    g_vec, _ = reciprocal_g_vector(CXR_KWARGS["hkl_list"][0], CRYSTALS["hopg"]["lattice"])

    print(f"Radiation vs numerical substep refinement at FIXED physical flights   Ne={Ne}")
    print(f"E_cut = {E_CUT_KEV} keV, seed {SEED}, energy_model='midpoint', lockstep core")
    print(f"L1 and peak columns are against the f={LADDER_REFERENCE:.0e} rung of the same")
    print("reduction; ungr/gr compares the two CXR reductions on identical rows.\n")

    for label, composition, E0_keV, thickness_ang in CASES:
        started = time.time()
        segments = simulate_trajectories(
            E0_keV,
            Ne,
            thickness_ang,
            composition=composition,
            E_cut_keV=E_CUT_KEV,
            seed=SEED,
            energy_model="midpoint",
        )
        cxr_grid, line_band = _cxr_grid(E0_keV, g_vec) if composition is CARBON else (None, None)

        rungs = []
        for max_dE_frac in [*LADDER, LADDER_REFERENCE]:
            rows, _ = subdivide_flights(
                segments,
                composition=composition,
                max_dE_frac=0.0 if max_dE_frac is None else max_dE_frac,
                max_substeps=MAX_SUBSTEPS,
            )
            rungs.append(
                (max_dE_frac, int(rows["L_ang"].size), _reductions(rows, composition, cxr_grid))
            )

        reference = rungs[-1][2]
        print(f"{label}  ({thickness_ang:.3g} Ang)   {time.time() - started:.1f} s")
        print(_row({key: key for key, _ in _WIDTHS}))
        for max_dE_frac, n_rows, values in rungs:
            cols = {
                "f": "none" if max_dE_frac is None else f"{max_dE_frac:.0e}",
                "rows": str(n_rows),
            }
            for key in ("brem", "brem_lep", "cxr", "cxr_ungr"):
                cols[key] = f"{_grid_l1(values[key], reference[key]):.2e}" if key in values else "-"
            if "cxr" in values:
                peak = values["cxr"][line_band].max()
                cols["peak"] = f"{peak / reference['cxr'][line_band].max():.4f}"
                cols["ungr/gr"] = f"{values['cxr_ungr'][line_band].max() / peak:.4f}"
            else:
                cols["peak"] = cols["ungr/gr"] = "-"
            print(_row(cols))
        print()


if __name__ == "__main__":
    main()
