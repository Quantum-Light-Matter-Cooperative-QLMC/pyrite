"""Research-only accounting and coordinate conversion checks for #623."""

from dataclasses import dataclass

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_zone_handoff import (
    AxialDepthEvidence,
    AxialFluxAccount,
    propose_correlated_outgoing,
    require_axial_depth_budget,
)


@dataclass(frozen=True)
class MinimalGrid:
    extent_ang: tuple[float, float] = (2000.0, 1000.0)
    shape: tuple[int, int] = (2, 4)

    @property
    def cell_area_ang2(self):
        return np.prod(self.extent_ang) / np.prod(self.shape)

    @property
    def x(self):
        return np.arange(self.shape[0]) * self.extent_ang[0] / self.shape[0]

    @property
    def y(self):
        return np.arange(self.shape[1]) * self.extent_ang[1] / self.shape[1]

    @property
    def kx(self):
        return 2 * np.pi * np.fft.fftfreq(self.shape[0], self.extent_ang[0] / self.shape[0])

    @property
    def ky(self):
        return 2 * np.pi * np.fft.fftfreq(self.shape[1], self.extent_ang[1] / self.shape[1])


def test_account_incident_closure_and_independent_unresolved_sink():
    a = AxialFluxAccount(0.78, 0.12, 0.10)
    assert a.physical_outgoing == 0.12
    assert a.unresolved_numerical == 0.10
    with pytest.raises(ValueError, match="equal incident"):
        AxialFluxAccount(0.8, 0.2, 0.01)
    with pytest.raises(ValueError, match="nonnegative"):
        AxialFluxAccount(1.1, 0.0, -0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        AxialFluxAccount(np.nan, 0.0, 0.0)


def test_depth_guard_refuses_missing_mismatch_overbudget_and_noisy_measurement():
    entry = dict(case_key="si5mev/001/32x32/tds-electronic", requested_depth_ang=1000.0)
    with pytest.raises(ValueError, match="no matching"):
        require_axial_depth_budget(**entry, evidence=None)
    good = AxialDepthEvidence(
        entry["case_key"], "regression-2026-10-21", 2000.0, 0.001, 0.002, 0.003
    )
    require_axial_depth_budget(**entry, evidence=good)
    with pytest.raises(ValueError, match="no matching"):
        require_axial_depth_budget(**{**entry, "case_key": "other-case"}, evidence=good)
    with pytest.raises(ValueError, match="exceeds calibrated"):
        require_axial_depth_budget(**{**entry, "requested_depth_ang": 3000.0}, evidence=good)
    noisy = AxialDepthEvidence(entry["case_key"], "pilot", 2000.0, 0.01, 0.01, 0.0)
    with pytest.raises(ValueError, match="upper error"):
        require_axial_depth_budget(**entry, evidence=noisy)
    with pytest.raises(ValueError, match="upper_return_fraction"):
        AxialDepthEvidence("case", "evidence", 20.0, -0.01, 0, 0)


def _params():
    g = MinimalGrid()
    probability = 0.3
    # Physical phi uses inverse Angstrom units; discrete amplitude has norm p.
    state = np.ones(g.shape, complex) * np.sqrt(probability / np.prod(g.extent_ang))
    win = np.ones(g.shape, complex) / np.sqrt(np.prod(g.shape))
    return dict(
        state_per_ang=state,
        window=win,
        grid=g,
        seed=5,
        incident_id=31,
        account=AxialFluxAccount(0.6, probability, 0.1),
        depth_ang=500.0,
        entrance_origin_lab_ang=(10.0, 20.0, 30.0),
        frame_lab=np.eye(3),
        kinetic_energy_eV=5e6,
        clock_at_surface_ang=600.0,
        t0_ang=100.0,
        native_stream_key=2**63 + 19,
    )


def test_correlated_candidate_preserves_frame_energy_clock_and_native_identity():
    args = _params()
    a = propose_correlated_outgoing(**args)
    b = propose_correlated_outgoing(**args)
    assert a.incident_id == 31 and a.stream_key == 2**63 + 19
    assert a.state_weight == pytest.approx(0.3)
    assert a.t_ang == 600 and a.t0_ang == 100 and a.depth_ang == 500
    assert a.E_keV == pytest.approx(5000)
    assert np.linalg.norm(a.v_hat) == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(a.r_ang, b.r_ang) and np.allclose(a.v_hat, b.v_hat)
    assert a.momentum_index == (0, 0)  # constant wavefunction/window picks zero FFT order
    np.testing.assert_allclose(a.v_hat, [0, 0, 1], atol=1e-14)
    assert a.r_ang[2] == pytest.approx(530.0)
    assert a.r_ang[0] in (10.0, 1010.0)
    assert a.r_ang[1] in (20.0, 270.0, 520.0, 770.0)


def test_rotated_frame_and_carrier_map_to_onshell_lab_unit_vector():
    args = _params()
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    args.update(
        frame_lab=rotation, carrier_inv_ang=(2.0, 1.0), transverse_origin_ang=(-500.0, -250.0)
    )
    event = propose_correlated_outgoing(**args)
    p = np.sqrt(5e6 * (5e6 + 2 * 510998.95)) / 1973.269804
    reference = rotation @ [2 / p, 1 / p, np.sqrt(1 - 5 / p**2)]
    np.testing.assert_allclose(event.v_hat, reference, atol=1e-14)
    np.testing.assert_allclose(
        rotation.T @ (event.r_ang - np.array([10.0, 20.0, 30.0])),
        [
            args["grid"].x[event.position_index[0]] - 500,
            args["grid"].y[event.position_index[1]] - 250,
            500,
        ],
    )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"native_stream_key": -1}, "native_stream_key"),
        ({"native_stream_key": 2**64}, "native_stream_key"),
        ({"depth_ang": -1.0}, "depth/energy"),
        ({"kinetic_energy_eV": 0.0}, "depth/energy"),
        ({"frame_lab": np.diag([1, 1, -1])}, "right-handed"),
        ({"clock_at_surface_ang": 99.0}, "predates"),
        ({"account": AxialFluxAccount(0.5, 0.4, 0.1)}, "outgoing state norm"),
        ({"carrier_inv_ang": (10000, 0)}, "paraxial"),
    ],
)
def test_handoff_fails_closed_on_unphysical_input(mutation, message):
    args = _params()
    args.update(mutation)
    with pytest.raises(ValueError, match=message):
        propose_correlated_outgoing(**args)


def test_constant_state_delta_window_remains_positive_and_incident_addressed():
    # The actual phase-space sampler is tested separately against a direct Weyl
    # oracle. Here the same incident must pick the same coupled position/momentum.
    args = _params()
    args["window"] = np.zeros(args["grid"].shape, complex)
    args["window"][0, 0] = 1.0
    out = propose_correlated_outgoing(**args)
    assert out.momentum_index in tuple(np.ndindex(args["grid"].shape))
    assert out.state_weight == pytest.approx(args["account"].physical_outgoing)
    assert out.incident_id == 31
