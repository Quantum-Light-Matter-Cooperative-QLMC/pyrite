import numpy as np
import pytest

from pyrite.montecarlo.transport import beta_from_keV_scalar, simulate_trajectories

CARBON = [("C", 0.1136)]


@pytest.mark.parametrize("transport_core", ["lockstep", "per-electron"])
def test_flight_diagnostics_summarize_frozen_state_error_without_changing_transport(
    transport_core,
):
    common = dict(
        E0_keV=5.01,
        Ne=1,
        thickness_ang=1.0e8,
        composition=CARBON,
        E_cut_keV=5.0,
        elastic_model="sr",
        seed=4,
        max_steps=20,
        transport_core=transport_core,
    )
    baseline = simulate_trajectories(**common)
    diagnosed = simulate_trajectories(**common, collect_diagnostics=True)

    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer"):
        np.testing.assert_array_equal(diagnosed[key], baseline[key])
    assert "transport_diagnostics" not in baseline

    summary = diagnosed["transport_diagnostics"]
    assert summary["n_flights"] == diagnosed["E_keV"].size
    params = {"Z": 6.0, "J_keV": 0.078}
    k = 0.731 + 0.0688 * np.log10(params["Z"])
    coeff = CARBON[0][1] / 0.602214076 * params["Z"]
    starts = diagnosed["E_keV"]
    stopping = (
        7.85e-4 / starts * coeff * np.log(1.166 * (starts + k * params["J_keV"]) / params["J_keV"])
    )
    ends = starts - stopping * diagnosed["L_ang"]
    expected_loss = (starts - ends) / starts
    assert summary["fractional_energy_loss"]["p50"] == pytest.approx(
        np.percentile(expected_loss, 50.0)
    )
    assert summary["fractional_energy_loss"]["max"] == pytest.approx(expected_loss.max())
    assert summary["relative_hazard_change"]["max"] > 0.0

    beta_start = np.array([beta_from_keV_scalar(value) for value in starts])
    beta_mid = np.array([beta_from_keV_scalar(value) for value in 0.5 * (starts + ends)])
    expected_clock_error = np.abs(1.0 / beta_mid - 1.0 / beta_start) / (1.0 / beta_mid)
    assert summary["relative_clock_error_estimate"]["max"] == pytest.approx(
        expected_clock_error.max()
    )
    assert summary["cutoff_overshoot_keV"]["max"] == pytest.approx(0.0, abs=2e-15)


def test_flight_diagnostics_are_seed_deterministic():
    kwargs = dict(
        E0_keV=30.0,
        Ne=8,
        thickness_ang=100.0,
        composition=CARBON,
        elastic_model="sr",
        seed=1729,
        max_steps=10,
        transport_core="lockstep",
        collect_diagnostics=True,
    )
    first = simulate_trajectories(**kwargs)["transport_diagnostics"]
    second = simulate_trajectories(**kwargs)["transport_diagnostics"]
    assert first == second
