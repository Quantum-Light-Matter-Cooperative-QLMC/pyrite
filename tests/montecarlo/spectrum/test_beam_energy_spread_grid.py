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
from pyrite.energy_grid.bounds import CoverageGridTooNarrow, coverage_energy, line_shift_fraction
from pyrite.montecarlo.runner import run_case

# Keep the slow peak-shift sibling at its established statistical gate.
_NE = 200


def _line_spectrum(energy_keV, *, energy_spread_frac=None, n_electrons=_NE):
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=energy_keV, energy_spread_frac=energy_spread_frac),
        tilt_deg=30.0,
        n_electrons=n_electrons,
        n_electrons_brem=5,
        # The (002) resonance derivative is the observable; pinned dominant
        # families keep measured shell selection out of a beam-energy check.
        n_families=4,
    )
    # Continuous stopping: loss fluctuations (hard inelastic events, the
    # default shell mode) send a few electrons to low energy, and their broad
    # low-photon-energy emission can outrank the resonance at this electron
    # count. The resonance derivative is independent of the loss law.
    out = run_case(build_cases(sweep, inelastic_model="continuous", straggling=False)[0])
    # The runner's separate-component contract already excludes characteristic
    # emission from spec. Subtracting spec_characteristic again makes the
    # integral negative and turns coverage_energy into a first-bin no-op.
    return np.asarray(out["E_grid"], dtype=float), np.asarray(out["spec"], dtype=float)


@pytest.mark.slow
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


def _line_tail_edge(grid, spec, ceiling):
    """Use the complete reference tail so a clipped grid cannot renormalize it."""
    peak = int(np.argmax(spec))
    assert 0 < peak < grid.size - 1, "reference line peak must be interior"
    assert np.all(spec >= 0), "line component must be a nonnegative density"
    assert np.trapezoid(spec[peak:], grid[peak:]) > 0, "reference tail must emit"
    edge = coverage_energy(grid[peak:], spec[peak:], 0.9)
    if edge >= ceiling:
        raise CoverageGridTooNarrow()
    return edge


def test_spread_line_stays_inside_the_catalog_window():
    """The upper 90% tail of a 1% RMS spread line clears the catalog ceiling.

    Normalize above the line peak on the complete production grid. A candidate
    ceiling inside that tail must fail against the same reference, rather than
    normalizing away its missing emission on a shortened grid.
    """
    grid, spec = _line_spectrum(30.0, energy_spread_frac=0.01, n_electrons=50)
    edge = _line_tail_edge(grid, spec, grid[-1])
    peak_eV = grid[int(np.argmax(spec))]
    assert peak_eV < edge < grid[-1]

    # Cut through the upper side of the broadened line. Reuse the full
    # simulation: no extra transport or stochastic negative-control noise.
    clipped_ceiling = peak_eV + 0.5 * (edge - peak_eV)
    with pytest.raises(CoverageGridTooNarrow):
        _line_tail_edge(grid, spec, clipped_ceiling)
