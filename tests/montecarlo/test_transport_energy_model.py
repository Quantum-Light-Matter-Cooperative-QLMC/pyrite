import numpy as np
import pytest

from pyrite.montecarlo.transport import simulate_trajectories

CARBON = [("C", 0.1136)]
_Z = 6.0
_J_KEV = 0.078
_K = 0.731 + 0.0688 * np.log10(_Z)
_COEFF = (CARBON[0][1] / 0.602214076) * _Z


def _dEds(E_keV):
    """Joy--Luo continuous slowing down [keV/Ang], independent of the core."""
    return -7.85e-4 / E_keV * _COEFF * np.log(1.166 * (E_keV + _K * _J_KEV) / _J_KEV)


def _beta(E_keV):
    gamma = 1.0 + E_keV / 510.99895
    return np.sqrt(1.0 - 1.0 / (gamma * gamma))


def _reference_flight(E_start_keV, length_ang, n_sub=20_000):
    """RK4 reference for dE/ds and the accumulated L/beta clock over one flight."""
    h = length_ang / n_sub
    E = float(E_start_keV)
    t = 0.0
    for _ in range(n_sub):
        k1 = _dEds(E)
        k2 = _dEds(E + 0.5 * h * k1)
        k3 = _dEds(E + 0.5 * h * k2)
        k4 = _dEds(E + h * k3)
        t += h / _beta(E + 0.5 * h * k1)
        E += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return E, t


def _single_flight(thickness_ang, energy_model="midpoint", E0_keV=25.0):
    """One electron whose first flight is truncated by the exit face, not a collision.

    The free path is sampled before any truncation and depends only on the start
    energy and the seed, so all thicknesses below it yield exactly one row of
    length ``thickness_ang``.
    """
    result = simulate_trajectories(
        E0_keV=E0_keV,
        Ne=1,
        thickness_ang=thickness_ang,
        composition=CARBON,
        elastic_model="sr",
        seed=11,
        max_steps=200,
        transport_core="lockstep",
        energy_model=energy_model,
    )
    assert result["L_ang"].size == 1
    assert result["L_ang"][0] == pytest.approx(thickness_ang)
    return result


def test_frozen_energy_model_is_the_bit_for_bit_default():
    common = dict(
        E0_keV=40.0,
        Ne=32,
        thickness_ang=8000.0,
        composition=CARBON,
        seed=7,
        transport_core="lockstep",
    )
    default = simulate_trajectories(**common)
    frozen = simulate_trajectories(**common, energy_model="frozen")

    assert default.keys() == frozen.keys()
    assert "E_end_keV" not in default
    assert "t_end_ang" not in default
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "t0_ang", "elec_id", "layer"):
        np.testing.assert_array_equal(default[key], frozen[key])
    for key in ("n_backscattered", "n_transmitted", "n_side_exited", "n_cutoff_stopped"):
        assert default[key] == frozen[key]


def test_canonical_start_fields_alias_the_compatibility_spellings():
    result = simulate_trajectories(
        E0_keV=40.0,
        Ne=8,
        thickness_ang=8000.0,
        composition=CARBON,
        seed=7,
        transport_core="lockstep",
    )
    assert result["E_start_keV"] is result["E_keV"]
    assert result["t_start_ang"] is result["t_ang"]


def test_midpoint_adds_end_state_without_adding_rows():
    common = dict(
        E0_keV=40.0,
        Ne=32,
        thickness_ang=8000.0,
        composition=CARBON,
        seed=7,
        transport_core="lockstep",
    )
    frozen = simulate_trajectories(**common)
    midpoint = simulate_trajectories(**common, energy_model="midpoint")

    assert set(midpoint) - set(frozen) == {"E_end_keV", "t_end_ang"}
    for key in ("E_end_keV", "t_end_ang"):
        assert midpoint[key].shape == midpoint["L_ang"].shape
    # One radiating row per physical flight either way: the controlled rule
    # changes each flight's end state, never the flight decomposition itself.
    assert midpoint["E_end_keV"].size == midpoint["E_start_keV"].size
    assert np.all(midpoint["E_end_keV"] < midpoint["E_start_keV"])
    assert np.all(midpoint["t_end_ang"] > midpoint["t_start_ang"])


def test_midpoint_end_energy_and_clock_beat_the_frozen_rule():
    result = _single_flight(600.0)
    E_start = float(result["E_start_keV"][0])
    length = float(result["L_ang"][0])
    E_ref, t_ref = _reference_flight(E_start, length)

    # A representative flight at the ~0.5% fractional loss the step limit targets.
    assert 1e-3 < (E_start - E_ref) / E_start < 1e-2

    frozen_E = E_start + _dEds(E_start) * length
    frozen_t = length / _beta(E_start)

    assert abs(float(result["E_end_keV"][0]) - E_ref) < 0.01 * abs(frozen_E - E_ref)
    assert abs(float(result["t_end_ang"][0]) - t_ref) < 0.01 * abs(frozen_t - t_ref)


@pytest.mark.parametrize("field", ["E_end_keV", "t_end_ang"])
def test_midpoint_end_state_error_is_third_order_in_flight_length(field):
    """One flight is one step, so its local truncation error is O(s^3).

    Halving the flight length must cut that error by ~8. The frozen rule is one
    order lower and drops by ~4, which is what
    `test_midpoint_end_energy_and_clock_beat_the_frozen_rule` pins in amplitude.
    """
    errors = []
    for length in (600.0, 300.0, 150.0):
        result = _single_flight(length)
        E_start = float(result["E_start_keV"][0])
        E_ref, t_ref = _reference_flight(E_start, float(result["L_ang"][0]))
        reference = E_ref if field == "E_end_keV" else t_ref
        errors.append(abs(float(result[field][0]) - reference))

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert 7.0 < coarse / fine < 9.0


def test_midpoint_cutoff_flight_stops_exactly_on_the_energy_floor():
    result = simulate_trajectories(
        E0_keV=6.0,
        Ne=64,
        thickness_ang=1.0e8,
        composition=CARBON,
        E_cut_keV=5.0,
        elastic_model="sr",
        seed=5,
        transport_core="lockstep",
        energy_model="midpoint",
    )
    # The truncation distance solves the midpoint rule for E_end == E_cut, so a
    # cutoff-stopped flight lands on the floor exactly rather than through it.
    assert np.all(result["E_end_keV"] >= 5.0)
    on_floor = result["E_end_keV"] == 5.0
    assert int(on_floor.sum()) == result["n_cutoff_stopped"] > 0


def test_midpoint_cutoff_distance_is_shorter_than_the_frozen_extrapolation():
    common = dict(
        E0_keV=6.0,
        Ne=64,
        thickness_ang=1.0e8,
        composition=CARBON,
        E_cut_keV=5.0,
        elastic_model="sr",
        seed=5,
        transport_core="lockstep",
    )
    frozen = simulate_trajectories(**common)
    midpoint = simulate_trajectories(**common, energy_model="midpoint")
    # |dE/ds| grows as E falls, so the frozen rule overshoots the true range.
    assert midpoint["L_ang"].sum() < frozen["L_ang"].sum()


def test_row_transforms_keep_the_new_fields_in_step_with_the_rows():
    """An unregistered per-row array would survive a mask at full length."""
    from pyrite.montecarlo.spectrum import _clip_segments_to_cutoff, _segments_in_layer

    layers = [(0.0, 4000.0, CARBON), (4000.0, 8000.0, CARBON)]
    segments = simulate_trajectories(
        E0_keV=40.0,
        Ne=24,
        thickness_ang=8000.0,
        layers=layers,
        seed=7,
        transport_core="lockstep",
        energy_model="midpoint",
    )

    for layer in (0, 1):
        sliced = _segments_in_layer(segments, layer)
        mask = segments["layer"] == layer
        assert sliced["L_ang"].size == int(mask.sum()) > 0
        for key in ("E_start_keV", "t_start_ang", "E_end_keV", "t_end_ang"):
            np.testing.assert_array_equal(sliced[key], segments[key][mask])

    clipped = _clip_segments_to_cutoff(segments, 12.0, CARBON, layers=layers)
    keep = segments["E_keV"] >= 12.0
    np.testing.assert_array_equal(clipped["E_start_keV"], segments["E_keV"][keep])
    # Shortened flights have no reconstructable end state, so it is dropped
    # rather than left stale.
    assert "E_end_keV" not in clipped
    assert "t_end_ang" not in clipped


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"energy_model": "left"}, "energy_model must be"),
        ({"energy_model": "midpoint", "transport_core": "per-electron"}, "lockstep core"),
    ],
)
def test_unsupported_energy_model_requests_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_trajectories(
            E0_keV=40.0,
            Ne=8,
            thickness_ang=8000.0,
            composition=CARBON,
            seed=7,
            **kwargs,
        )
