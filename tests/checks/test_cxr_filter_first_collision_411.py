"""Pure geometry and first-event checks for the research-only #411 filter study."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks import cxr_filter_first_collision_411 as c


def _plate():
    return c.CircularPlate((0, 0, 10), (0, 0, 1), 4, 0.0005)


def _detector():
    return c.IdealDisk((0, 0, 20), (0, 0, 1), 2.922)


def _rates(n, *, photo=0.0, rayleigh=0.0, compton=0.0):
    return {
        "photoelectric": np.full(n, photo),
        "coherent": np.full(n, rayleigh),
        "incoherent": np.full(n, compton),
        "pair_nuclear": np.zeros(n),
        "pair_electron": np.zeros(n),
    }


def test_benchmark_disk_solid_angle_and_geometry():
    assert c.disk_solid_angle_sr(2.922, 20) == pytest.approx(0.066, abs=2e-4)
    assert c.detector_distance_mm((0, 0, 0), (0, 0, 1), _detector()) == pytest.approx(20)
    assert np.isinf(c.detector_distance_mm((0, 0, 0), (1, 0, 1), _detector()))
    assert np.isinf(c.detector_distance_mm((0, 0, 0), (0, 0, -1), _detector()))
    assert c.plate_path_mm((0, 0, 0), (0, 0, 1), _plate()) == pytest.approx(0.0005)


def test_plate_thickness_obliquity_sidewall_and_limit():
    plate = _plate()
    direction = (0.2, 0, np.sqrt(1 - 0.2**2))
    assert c.plate_path_mm((0, 0, 0), direction, plate) == pytest.approx(
        plate.thickness_mm / direction[2]
    )
    # A ray with large transverse excursion cannot be assigned t/cos(theta).
    assert c.plate_path_mm((0, 0, 0), (1, 0, 0), plate) == 0
    assert c.plate_path_mm((5, 0, 0), (0, 0, 1), plate) == 0
    assert c.plate_path_mm((0, 0, 0), (0, 0, 1), plate, limit_mm=9.0) == 0
    assert c.plate_path_mm((0, 0, 0), (0, 0, 1), plate, limit_mm=10.0) == pytest.approx(
        0.00025
    )
    # A side-entry path is set by radius, not thickness. Ray starts inside the slab.
    assert c.plate_path_mm((3, 0, 10), (1, 0, 0), plate) == pytest.approx(1.0)
    # A grazing tangent contributes no nonzero length.
    assert c.plate_path_mm((4, 0, 10), (0, 1, 0), plate) == 0


def test_first_event_partition_is_positive_conservative_and_stable():
    path = np.array([0.0, 0.5, 1.0, 1000.0])
    probs = c.first_collision_probabilities(path, _rates(4, photo=2, rayleigh=1, compton=1))
    combined = probs.uncollided + sum(probs.by_channel.values())
    np.testing.assert_allclose(combined, 1.0, atol=1e-15)
    np.testing.assert_allclose(probs.uncollided, np.exp(-4 * path))
    np.testing.assert_allclose(probs.by_channel["coherent"], -np.expm1(-4 * path) / 4)
    assert np.all(probs.by_channel["coherent"] >= 0)
    assert probs.by_channel["coherent"][-1] == pytest.approx(0.25)
    assert probs.by_channel["pair_nuclear"].sum() == 0
    zero = c.first_collision_probabilities(path, _rates(4))
    np.testing.assert_array_equal(zero.uncollided, np.ones(4))
    assert all(np.count_nonzero(v) == 0 for v in zero.by_channel.values())


def test_bounds_include_off_cone_and_controls_do_not_double_count(monkeypatch):
    # Intentionally supplied SYNTHETIC boundary cells, not extracted from the
    # CXR angular scorer or electron segment positions. Positions are Å.
    source = SimpleNamespace(
        position_ang=np.zeros((3, 3)),
        direction=np.array([[0, 0, 1], [0.3, 0, np.sqrt(1 - 0.3**2)], [0, 0, -1]]),
        energy_eV=np.array([500.0, 500.0, 500.0]),
        weight_per_primary=np.array([2.0, 3.0, 5.0]),
    )
    monkeypatch.setattr(
        c, "carbon_epdl_coefficients_inv_mm", lambda e: _rates(len(e), rayleigh=2000)
    )
    result = c.boundary_controls(source, _plate(), _detector())
    assert result.vacuum_detector_per_primary == 2.0
    assert result.collisions_disabled_detector_per_primary == result.vacuum_detector_per_primary
    assert result.uncollided_detector_per_primary == pytest.approx(2 * np.exp(-1))
    # Both a direct ray and an initially off-cone ray traverse the plate. The
    # latter's first-scatter probability is only a *potential*, not a hit.
    off_length = c.plate_path_mm((0, 0, 0), source.direction[1], _plate())
    expected_first = 2 * (-np.expm1(-1)) + 3 * (-np.expm1(-2000 * off_length))
    assert result.first_collision_per_primary["coherent"] == pytest.approx(expected_first)
    assert (
        result.uncollided_detector_per_primary
        + sum(result.first_collision_per_primary.values())
    ) <= 10.0


def test_source_positions_are_not_replaced_by_sample_origin(monkeypatch):
    source = SimpleNamespace(
        position_ang=np.array([[5.0e7, 0, 0], [0, 0, 0]]),  # 5 mm and 0 mm
        direction=np.array([[0, 0, 1], [0, 0, 1]]),
        energy_eV=np.array([500.0, 500.0]),
        weight_per_primary=np.array([7.0, 2.0]),
    )
    monkeypatch.setattr(c, "carbon_epdl_coefficients_inv_mm", lambda e: _rates(len(e), photo=2000))
    result = c.boundary_controls(source, _plate(), _detector())
    assert result.vacuum_detector_per_primary == 2.0  # 5 mm offset misses detector
    assert result.first_collision_per_primary["photoelectric"] == pytest.approx(2 * -np.expm1(-1))


@pytest.mark.parametrize(
    "bad,match",
    [
        (np.array([-1.0]), "nonnegative"),
        (np.array([np.nan]), "nonnegative"),
        (np.array([[1.0]]), "1D"),
    ],
)
def test_first_collision_rejects_unphysical_paths(bad, match):
    with pytest.raises(ValueError, match=match):
        c.first_collision_probabilities(bad, _rates(1))


def test_reject_undefined_epdl_coefficients_and_geometry(monkeypatch):
    with pytest.raises(ValueError, match="support"):
        c.carbon_epdl_coefficients_inv_mm(np.array([0.9]))
    with pytest.raises(ValueError, match="positive"):
        c.carbon_epdl_coefficients_inv_mm(np.array([500.0]), density_g_cm3=0)
    with pytest.raises(ValueError, match="plate radius"):
        c.CircularPlate((0, 0, 0), (0, 0, 1), -1, 0.001)
    with pytest.raises(ValueError, match="nonzero"):
        c.IdealDisk((0, 0, 0), (0, 0, 0), 1)
    with pytest.raises(ValueError, match="five"):
        c.first_collision_probabilities(np.array([1.0]), {"coherent": np.ones(1)})
    with pytest.raises(ValueError, match="coefficients"):
        c.first_collision_probabilities(np.array([1.0]), _rates(1, photo=-1))
    # Mock only the data boundary. Catch a missing or undefined EPDL value.
    from unittest.mock import patch
    fake = {name: np.array([np.nan]) for name in c.CHANNELS}
    with patch.dict("sys.modules", {"pyrite.materials.photon_cross_sections": SimpleNamespace(
        photon_cross_sections_ang2=lambda el, e: fake
    )}):
        with pytest.raises(ValueError, match="undefined"):
            c.carbon_epdl_coefficients_inv_mm(np.array([500.0]))


def test_empty_boundary_is_conservative_without_epdl_data():
    source = SimpleNamespace(
        position_ang=np.empty((0, 3)), direction=np.empty((0, 3)),
        energy_eV=np.empty(0), weight_per_primary=np.empty(0),
    )
    result = c.boundary_controls(source, _plate(), _detector())
    assert result.vacuum_detector_per_primary == 0
    assert result.uncollided_detector_per_primary == 0
    assert all(value == 0 for value in result.first_collision_per_primary.values())


def test_unscaled_boundary_directions_fail_closed():
    source = SimpleNamespace(
        position_ang=np.zeros((1, 3)), direction=np.array([[0, 0, 2]]),
        energy_eV=np.array([500.0]), weight_per_primary=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="unit vectors"):
        c.boundary_controls(source, _plate(), _detector())
