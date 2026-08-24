"""The emitted line's response to beam energy, end to end through the MC.

`energy_grid.bounds.line_shift_fraction` is what decides whether an
energy-spread-broadened line still fits the catalog `E_grid_line` window (task
step H). That bound is a first-order derivative of the PXR resonance, so it is
only trustworthy if the full kernel actually moves the line by as much as it
claims -- which is what this pins.

Validation: beam-energy-spread-injection
"""

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.energy_grid.bounds import coverage_energy, line_shift_fraction
from pyrite.montecarlo.runner import run_case

# Small on purpose: the peak bin is stable well below the count where the
# spectrum shape is, so the whole module costs ~1 s.
_NE = 200


def _line_spectrum(energy_keV, *, energy_spread_frac=None):
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=energy_keV, energy_spread_frac=energy_spread_frac),
        tilt_deg=30.0,
        n_electrons=_NE,
        n_electrons_brem=5,
    )
    out = run_case(build_cases(sweep)[0])
    # This regression is specifically about the PXR resonance derivative. The
    # new atomic characteristic component is beam-energy stationary over this
    # step and can dominate the combined peak, so remove its auditable array.
    pxr = np.asarray(out["spec"], dtype=float) - np.asarray(
        out["spec_characteristic"],
        dtype=float,
    )
    return np.asarray(out["E_grid"], dtype=float), pxr


def test_simulated_line_moves_by_the_predicted_sensitivity():
    """A 5% beam energy step must move the line by S*0.05 of its energy.

    Done with a deterministic energy offset rather than a random spread: the
    intrinsic sinc^2 line width is several times the spread-induced broadening
    at any realistic delta, so the *shift* is the observable that isolates the
    sensitivity factor.
    """
    grid, mono = _line_spectrum(30.0)
    peak_eV = grid[int(np.argmax(mono))]
    shifted_grid, shifted = _line_spectrum(30.0 * 1.05)
    shifted_peak_eV = shifted_grid[int(np.argmax(shifted))]

    measured = (shifted_peak_eV - peak_eV) / peak_eV
    # theta_obs is 90 deg here, so the Doppler denominator is exactly 1.
    assert measured == pytest.approx(line_shift_fraction(30.0, 0.05), rel=0.1)


def test_spread_line_stays_inside_the_catalog_window():
    """The step H question itself: nothing is clipped at the grid ceiling.

    A 1% RMS spread is already generous for a photoinjector, and the catalog
    grid is cut with 15% headroom, so the 90% coverage energy has to resolve
    without `coverage_energy` hitting its final bin.
    """
    grid, spec = _line_spectrum(30.0, energy_spread_frac=0.01)
    # Raises CoverageGridTooNarrow if coverage is only reached in the last bin.
    covered_eV = coverage_energy(grid, spec, 0.9)
    assert covered_eV < grid[-1]
