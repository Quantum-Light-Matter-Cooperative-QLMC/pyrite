"""Calibrate the slice-D radiation error estimator warning thresholds.

For a low-Z/high-Z x thin/thick case matrix this measures, per case:

- the estimator summaries (CXR endpoint resonance drift in linewidths, brem
  endpoint quadrature relative error) on a frozen-energy transport run; and
- the ACTUAL spectral change when every flight's emission coefficient is
  evaluated at the flight's midpoint energy instead of its start energy --
  the same segments, only ``E_keV`` swapped for ``(E_start + E_end)/2``, so
  trajectory divergence does not confound the quadrature/evaluation error.

The threshold question is: at what estimator value does the actual spectrum
move materially? Numbers from this script are recorded in
``docs/validation/beam-transport/radiation-error-estimators.md`` and the
chosen defaults live in ``montecarlo/spectrum/diagnostics.py``.

Run:  uv run python checks/radiation_error_estimator_calibration.py

Validation: radiation-error-estimators
"""

import numpy as np

from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.montecarlo.spectrum import (
    brem_endpoint_quadrature_error,
    cxr_endpoint_resonance_drift,
    mc_brem_spectrum,
    mc_spectrum,
)
from pyrite.montecarlo.spectrum.diagnostics import _flight_E_end_keV
from pyrite.montecarlo.transport import simulate_trajectories

CARBON = [("C", 0.1136)]
TUNGSTEN = [("W", 0.06305)]

BREM_GRID = np.linspace(1.0e3, 3.0e4, 501)
# hopg (0,0,2) line window, mirroring tests/montecarlo/test_coherent_emission.py.
CXR_GRID = np.arange(700.0, 1500.0)
CXR_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": np.array([1.0, 0.0, 0.01]),
}

CASES = [
    ("C  25 keV  2e3 Ang", CARBON, 25.0, 2.0e3),
    ("C  25 keV  2e4 Ang", CARBON, 25.0, 2.0e4),
    ("C 100 keV  2e4 Ang", CARBON, 100.0, 2.0e4),
    ("W  25 keV  5e2 Ang", TUNGSTEN, 25.0, 5.0e2),
    ("W  25 keV  5e3 Ang", TUNGSTEN, 25.0, 5.0e3),
    ("W 100 keV  5e3 Ang", TUNGSTEN, 100.0, 5.0e3),
]

_WIDTHS = (
    ("case", 22),
    ("flights", 8),
    ("drift p50", 10),
    ("drift p99", 10),
    ("cxr L1", 9),
    ("cxr max", 9),
    ("quad p50", 9),
    ("quad p99", 9),
    ("brem L1", 9),
    ("brem max", 9),
)


def _rel_l1(a, b):
    return float(np.sum(np.abs(a - b)) / np.sum(np.abs(b)))


def _rel_max_bin(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def main():
    g002, g002_mag = reciprocal_g_vector((0, 0, 2), CRYSTALS["hopg"]["lattice"])
    print(f"hopg (0,0,2) |g| = {g002_mag:.4f} 1/Ang")
    print(" ".join(f"{key:>{w}}" for key, w in _WIDTHS))
    for name, comp, E0, thickness in CASES:
        segments = simulate_trajectories(
            E0_keV=E0,
            Ne=2000,
            thickness_ang=thickness,
            composition=comp,
            seed=7,
            transport_core="lockstep",
        )
        E_end = _flight_E_end_keV(segments, comp, None)
        mid = dict(segments)
        mid["E_keV"] = 0.5 * (np.asarray(segments["E_keV"]) + E_end)

        with np.errstate(all="ignore"):
            quad = brem_endpoint_quadrature_error(
                segments, BREM_GRID, composition=comp, warn_threshold=np.inf
            )
        spec_start = mc_brem_spectrum(segments, BREM_GRID, composition=comp)
        spec_mid = mc_brem_spectrum(mid, BREM_GRID, composition=comp)

        cols = {
            "case": name,
            "flights": str(segments["L_ang"].size),
            "drift p50": "--",
            "drift p99": "--",
            "cxr L1": "--",
            "cxr max": "--",
            "quad p50": f"{quad['integrated_relative_error']['p50']:.2e}",
            "quad p99": f"{quad['integrated_relative_error']['p99']:.2e}",
            "brem L1": f"{_rel_l1(spec_start, spec_mid):.2e}",
            "brem max": f"{_rel_max_bin(spec_start, spec_mid):.2e}",
        }

        if comp is CARBON:
            with np.errstate(all="ignore"):
                drift = cxr_endpoint_resonance_drift(
                    segments,
                    g002[None, :],
                    n_hat=CXR_KWARGS["n_hat"],
                    composition=comp,
                    warn_threshold=np.inf,
                )
            line_start = mc_spectrum(segments, CXR_GRID, **CXR_KWARGS)
            line_mid = mc_spectrum(mid, CXR_GRID, **CXR_KWARGS)
            cols["drift p50"] = f"{drift['resonance_drift_linewidths']['p50']:.2e}"
            cols["drift p99"] = f"{drift['resonance_drift_linewidths']['p99']:.2e}"
            cols["cxr L1"] = f"{_rel_l1(line_start, line_mid):.2e}"
            cols["cxr max"] = f"{_rel_max_bin(line_start, line_mid):.2e}"
        print(" ".join(f"{cols[key]:>{w}}" for key, w in _WIDTHS))


if __name__ == "__main__":
    main()
