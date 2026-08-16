import numpy as np
import pytest

from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories

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

    The energy LUT is disabled so the core integrates the same analytic
    Joy--Luo derivative as the RK4 reference: these tests pin the scheme's own
    truncation error, not the LUT interpolation error, which is a separate
    discretization shared by both energy models and covered by the LUT tests.
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
        transport_lut_config=TransportLUTConfig(enabled=False),
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

    assert set(midpoint) - set(frozen) == {
        "E_end_keV",
        "E_repr_keV",
        "t_end_ang",
        "flight_id",
        "substep_id",
    }
    for key in ("E_end_keV", "E_repr_keV", "t_end_ang", "flight_id", "substep_id"):
        assert midpoint[key].shape == midpoint["L_ang"].shape
    np.testing.assert_allclose(
        midpoint["E_repr_keV"],
        0.5 * (midpoint["E_start_keV"] + midpoint["E_end_keV"]),
        rtol=0,
        atol=0,
    )
    # Without a step cap every row is its flight's only substep.
    assert np.all(midpoint["substep_id"] == 0)
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
        for key in ("E_start_keV", "t_start_ang", "E_end_keV", "t_end_ang", "E_repr_keV"):
            np.testing.assert_array_equal(sliced[key], segments[key][mask])

    # 39 keV binds on this 40 keV run; a lower floor clips nothing here.
    clipped = _clip_segments_to_cutoff(segments, 39.0, CARBON, layers=layers)
    keep = segments["E_keV"] >= 39.0
    np.testing.assert_array_equal(clipped["E_start_keV"], segments["E_keV"][keep])
    # The clip reapplies the transport core's own midpoint cutoff solve, so a
    # shortened flight ends exactly on the floor and its representative energy
    # is the midpoint of the shortened flight -- reconstructed, never stale.
    shortened = clipped["L_ang"] < segments["L_ang"][keep]
    assert shortened.any()
    np.testing.assert_allclose(clipped["E_end_keV"][shortened], 39.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        clipped["E_repr_keV"],
        0.5 * (clipped["E_start_keV"] + clipped["E_end_keV"]),
        rtol=1e-12,
    )
    untouched = ~shortened
    np.testing.assert_array_equal(
        clipped["E_end_keV"][untouched], segments["E_end_keV"][keep][untouched]
    )
    np.testing.assert_array_equal(
        clipped["t_end_ang"][untouched], segments["t_end_ang"][keep][untouched]
    )


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


_STEP_COMMON = dict(
    E0_keV=25.0,
    Ne=200,
    thickness_ang=4000.0,
    composition=CARBON,
    seed=7,
    E_cut_keV=1.0,
    transport_core="lockstep",
    energy_model="midpoint",
)


def _flight_keys(result):
    return set(zip(result["electron_id"].tolist(), result["flight_id"].tolist(), strict=True))


def test_substeps_subdivide_flights_without_scattering_or_redrawing():
    coarse = simulate_trajectories(**_STEP_COMMON)
    fine = simulate_trajectories(**_STEP_COMMON, max_dE_frac=2e-3)

    assert coarse["substep_id"].max() == 0
    assert fine["substep_id"].max() > 0
    assert fine["L_ang"].size > coarse["L_ang"].size

    # Substeps are integration detail: they add rows inside a flight, never
    # flights, and every row of a flight keeps the flight's direction.
    for result in (coarse, fine):
        order = np.lexsort((result["substep_id"], result["flight_id"], result["electron_id"]))
        keys = np.stack([result["electron_id"], result["flight_id"]])[:, order]
        substep = result["substep_id"][order]
        same_flight = np.all(keys[:, 1:] == keys[:, :-1], axis=0)
        # substep_id restarts at 0 per flight and increments by one within it.
        assert np.all(substep[1:][same_flight] == substep[:-1][same_flight] + 1)
        assert np.all(substep[1:][~same_flight] == 0)
        assert substep[0] == 0

    for eid, fid in _flight_keys(fine):
        rows = (fine["electron_id"] == eid) & (fine["flight_id"] == fid)
        if rows.sum() < 2:
            continue
        directions = fine["v_hat"][rows]
        np.testing.assert_array_equal(directions, np.broadcast_to(directions[0], directions.shape))
        break


def test_substep_rows_tile_their_flight_in_length_energy_and_clock():
    fine = simulate_trajectories(**_STEP_COMMON, max_dE_frac=2e-3)
    split = 0
    for eid, fid in _flight_keys(fine):
        rows = np.flatnonzero((fine["electron_id"] == eid) & (fine["flight_id"] == fid))
        if rows.size < 2:
            continue
        rows = rows[np.argsort(fine["substep_id"][rows])]
        split += 1
        # Each substep starts exactly where the previous one ended.
        np.testing.assert_allclose(
            fine["E_start_keV"][rows][1:], fine["E_end_keV"][rows][:-1], rtol=1e-12
        )
        np.testing.assert_allclose(
            fine["t_start_ang"][rows][1:], fine["t_end_ang"][rows][:-1], rtol=1e-12
        )
        # and no substep exceeds the requested fractional energy loss.
        # The cap is applied to the left-endpoint loss prediction; the realized
        # midpoint loss is slightly larger because |dE/ds| grows as E falls.
        loss = fine["E_start_keV"][rows] - fine["E_end_keV"][rows]
        assert np.all(loss[:-1] / fine["E_start_keV"][rows][:-1] <= 2e-3 * 1.01)
    assert split > 0


def test_step_cap_requires_the_controlled_propagator():
    with pytest.raises(ValueError, match="max_dE_frac > 0 requires"):
        simulate_trajectories(**{**_STEP_COMMON, "energy_model": "frozen"}, max_dE_frac=1e-2)
    with pytest.raises(ValueError, match="max_dE_frac must be non-negative"):
        simulate_trajectories(**_STEP_COMMON, max_dE_frac=-1.0)
