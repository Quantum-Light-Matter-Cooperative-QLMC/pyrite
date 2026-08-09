"""Trajectory plots show the beam the run actually used.

A finite emittance, bunch length or energy spread is invisible in a picture
drawn from a point source, so `_trajectory_data` forwards the case's beam block
into transport and the incident bundle is drawn per electron (task step I).
"""

import numpy as np
import pytest
from matplotlib.collections import LineCollection

from cxr_mc.campaign.sweep import BeamSpec, Sweep, build_cases
from cxr_mc.campaign.transverse import TransverseDistribution
from cxr_mc.plots.mpl.trajectories import (
    _beam_phase_space,
    _draw_incident_bundle,
    _trajectory_data,
)

_POLICY = TransverseDistribution(
    normalized_emittance_x_mm_mrad=50.0,
    beta_twiss_x_m=0.5,
    alpha_twiss_x=-0.8,
)


def _case(beam):
    return build_cases(Sweep(material="hopg", thickness_ang=2.0e4, beam=beam))[0]


def test_phase_space_kwargs_are_inert_without_a_beam_block():
    """No policy on the case, no change to the transport call."""
    forwarded = _beam_phase_space(_case(BeamSpec()))
    assert forwarded == {
        "bunch_length_fs": None,
        "long_shape": "gaussian",
        "long_offsets_fs": None,
        "longitudinal_distribution": None,
        "transverse_distribution": None,
        "energy_spread_frac": None,
    }
    # The spot stays the caller's business: a 2D cross-section wants a point
    # source, and the 3D view supplies its own display width.
    assert "beam_fwhm_mm" not in forwarded


def test_trajectory_data_is_unchanged_when_no_phase_space_is_set():
    baseline = _trajectory_data(_case(BeamSpec()), 40, 3)
    repeat = _trajectory_data(_case(BeamSpec(energy_spread_frac=0.0)), 40, 3)
    np.testing.assert_array_equal(repeat["initial_v_hat"], baseline["initial_v_hat"])
    np.testing.assert_array_equal(repeat["px"], baseline["px"])


def test_transverse_policy_spreads_the_plotted_entry_points_and_directions():
    data = _trajectory_data(_case(BeamSpec.with_transverse(_POLICY)), 40, 3)
    directions = np.asarray(data["initial_v_hat"], dtype=float)
    entries = np.asarray(data["initial_r_ang"], dtype=float)
    assert directions[:, 0].std() > 0.0  # divergence, not one shared axis
    assert entries[:, :2].std() > 0.0  # a spot, not the origin
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_energy_spread_reaches_the_plotted_initial_energies():
    data = _trajectory_data(_case(BeamSpec(energy_spread_frac=0.05)), 40, 3)
    energies = np.asarray(data["initial_E_keV"], dtype=float)
    assert energies.std() > 0.0
    assert energies.mean() == pytest.approx(30.0, rel=0.1)


def test_twiss_policy_overrides_a_display_spot_width():
    """Both spellings at once is an error in transport; the policy wins here.

    The 3D view passes a display FWHM unconditionally, so without this a case
    carrying a Twiss policy could not be plotted at all.
    """
    data = _trajectory_data(_case(BeamSpec.with_transverse(_POLICY)), 24, 3, beam_fwhm_mm=1.0)
    assert np.asarray(data["initial_v_hat"], dtype=float)[:, 0].std() > 0.0


def _panel_data(spread):
    n = 8
    directions = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    entries = np.zeros((n, 3))
    if spread:
        directions[:, 0] = np.linspace(-1e-3, 1e-3, n)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        entries[:, 0] = np.linspace(-1.0, 1.0, n)
    return {
        "initial_r_ang": entries,
        "initial_v_hat": directions,
        "u": 1.0,
        "beam": np.array([0.0, 0.0, 1.0]),
        "detector": np.array([1.0, 0.0, 0.0]),
    }


def test_incident_bundle_is_drawn_only_for_a_finite_phase_space():
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        _draw_incident_bundle(ax, _panel_data(spread=False), 1.0)
        assert not [c for c in ax.collections if isinstance(c, LineCollection)]

        _draw_incident_bundle(ax, _panel_data(spread=True), 1.0)
        bundles = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(bundles) == 1
        assert len(bundles[0].get_segments()) == 8
    finally:
        plt.close(figure)


def test_incident_bundle_tolerates_a_payload_without_transport():
    """Hand-built panel dicts (tests, notebooks) must not raise."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        _draw_incident_bundle(ax, {"beam": np.zeros(3), "detector": np.zeros(3)}, 1.0)
        assert not ax.collections
    finally:
        plt.close(figure)


def test_plotly_incident_stubs_follow_each_electron_direction():
    from cxr_mc.plots.plotly.trajectories import _incident_beam_lines

    data = _trajectory_data(_case(BeamSpec.with_transverse(_POLICY)), 40, 3)
    trace = _incident_beam_lines(data, 1.0)
    xs = np.asarray(trace.x, dtype=float)
    # start, entry, NaN per electron: the upstream ends must not be collinear
    # with a single nominal axis, which is what a shared direction would give.
    upstream = xs[0::3]
    entry = xs[1::3]
    assert np.nanstd(upstream - entry) > 0.0
