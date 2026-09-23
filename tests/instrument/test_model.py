import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.materials import MediumSpec


def _detector() -> pr.PlanarDetector:
    return pr.PlanarDetector(
        pose=pr.PlanarPose.from_observation(100.0, 0.0),
        pixels=pr.PixelGrid(shape=(10, 10), pitch_mm=(1.0, 1.0)),
    )


def _scene(*, filters=(), pixel_scorer=None) -> pr.Scene:
    return pr.Scene(
        beam=pr.Beam(energy_keV=30.0),
        target=pr.Slab("hopg", thickness_ang=1_000.0, tilt_deg=30.0),
        detector=_detector(),
        filters=filters,
        pixel_scorer=pixel_scorer,
    )


def test_from_observation_defines_downstream_normal_and_source_facing_side() -> None:
    pose = pr.PlanarPose.from_observation(400.0, 90.0)

    np.testing.assert_allclose(pose.center_mm, [400.0, 0.0, 0.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.normal, [1.0, 0.0, 0.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.x_axis, [0.0, 0.0, -1.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.y_axis, [0.0, 1.0, 0.0], atol=1.0e-13)
    assert np.dot(-np.asarray(pose.center_mm), pose.normal) < 0.0


def test_from_observation_positive_roll_rotates_x_toward_y() -> None:
    pose = pr.PlanarPose.from_observation(10.0, 0.0, roll_deg=90.0)

    np.testing.assert_allclose(pose.x_axis, [0.0, 1.0, 0.0], atol=1.0e-13)
    np.testing.assert_allclose(pose.y_axis, [-1.0, 0.0, 0.0], atol=1.0e-13)


def test_pose_normalizes_axes_but_rejects_ambiguous_nonorthogonal_axes() -> None:
    pose = pr.PlanarPose((0.0, 0.0, 10.0), normal=(0.0, 0.0, 2.0), x_axis=(3.0, 0.0, 0.0))
    assert pose.normal == (0.0, 0.0, 1.0)
    assert pose.x_axis == (1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="must be orthogonal"):
        pr.PlanarPose((0.0, 0.0, 10.0), normal=(0.0, 0.0, 1.0), x_axis=(1.0, 0.0, 1.0))


def test_timepix_preset_is_one_physical_chip_not_a_response() -> None:
    grid = pr.PixelGrid.timepix3_chip()
    detector = pr.PlanarDetector.timepix3_chip(pr.PlanarPose((0.0, 0.0, 400.0)))

    assert grid.shape == (256, 256)
    assert grid.pitch_mm == (0.055, 0.055)
    assert grid.active_size_mm == pytest.approx((14.08, 14.08))
    assert detector.size_mm == pytest.approx((14.08, 14.08))
    assert detector.response is None


def test_filter_accepts_catalog_or_explicit_number_density_material() -> None:
    pose = pr.PlanarPose((0.0, 0.0, 50.0))
    catalog = pr.FilterPlate("silicon", 0.1, (5.0, 5.0), pose)
    custom = pr.FilterPlate(
        MediumSpec("al-foil", (("Al", 0.0602),)),
        0.1,
        (5.0, 5.0),
        pose,
    )

    assert catalog.material == "silicon"
    assert isinstance(custom.material, MediumSpec)


def test_off_centre_filter_is_valid_without_central_ray_intersection() -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((4.0, 0.0, 50.0)),
        name="right-edge",
    )

    scene = _scene(filters=(plate,), pixel_scorer=pr.PixelScorer())

    assert scene.filters == (plate,)


@pytest.mark.parametrize("z", [0.04, 99.96, 101.0])
def test_full_filter_volume_must_lie_between_source_and_detector_plane(z: float) -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((0.0, 0.0, z)),
    )

    with pytest.raises(ValueError, match="full FilterPlate volume"):
        _scene(filters=(plate,))


def test_reversed_public_normal_is_rejected_instead_of_hidden_by_absolute_dot() -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 2.0),
        pose=pr.PlanarPose((0.0, 0.0, 50.0), normal=(0.0, 0.0, -1.0)),
    )

    with pytest.raises(ValueError, match=r"local \+z must point downstream"):
        _scene(filters=(plate,))


def test_pixel_scorer_requires_physical_pixels_and_bounded_angular_shape() -> None:
    detector = pr.PlanarDetector(
        pose=pr.PlanarPose((0.0, 0.0, 100.0)),
        size_mm=(10.0, 10.0),
    )
    with pytest.raises(ValueError, match="requires PlanarDetector.pixels"):
        pr.Scene(
            beam=pr.Beam(30.0),
            target=pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
            detector=detector,
            pixel_scorer=pr.PixelScorer(),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        _scene(pixel_scorer=pr.PixelScorer((11, 1)))


def test_physical_detector_projects_to_unchanged_case_schema() -> None:
    scene = _scene()
    case = api.build_case(scene, pr.Numerics(n_electrons=1, n_electrons_brem=1))

    assert case.theta_obs_rad == pytest.approx(0.0)
    assert "filters" not in case.to_dict()
    assert "pixel_scorer" not in case.to_dict()


def test_simulate_returns_partial_filter_spatial_result_and_observation_identity(
    monkeypatch,
) -> None:
    plate = pr.FilterPlate(
        "silicon",
        thickness_mm=0.1,
        size_mm=(2.0, 10.0),
        pose=pr.PlanarPose((0.0, 0.0, 50.0)),
    )

    def fake_directional(case, n_hats, *, transport_core):
        n_tile = len(n_hats)
        return {
            "E_grid": np.array([5_000.0, 6_000.0]),
            "E_grid_brem": np.array([5_000.0, 6_000.0]),
            "spec_by_direction": np.ones((n_tile, 2)),
            "spec_characteristic_by_direction": np.full((n_tile, 2), 0.25),
            "brem_wide_by_direction": np.full((n_tile, 2), 0.5),
        }

    monkeypatch.setattr(api, "run_case_directions", fake_directional)
    kwargs = dict(
        filters=(plate,),
        pixel_scorer=pr.PixelScorer(),
        numerics=pr.Numerics(n_electrons=1, n_electrons_brem=1),
    )
    result = pr.simulate(
        pr.Beam(30.0),
        pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
        _detector(),
        **kwargs,
    )

    assert result.spatial is not None
    np.testing.assert_allclose(result.characteristic_spectrum, 0.25 * result.spectrum)
    paths = result.spatial.ray_map.path_length_mm[..., 0]
    assert np.any(paths == 0.0)
    assert np.any(paths > 0.0)
    _, pixel_spectra = result.spatial.spectra(region=(slice(None), slice(None)), component="line")
    np.testing.assert_allclose(
        result.spectrum,
        np.sum(pixel_spectra, axis=0) / np.sum(result.spatial.ray_map.solid_angle_sr),
        rtol=1.0e-15,
    )
    assert result.provenance["identity_digest"] == api.case_content_key(
        result.case,
        xsgen_tables=result.provenance["xsgen_tables"],
    )
    assert len(result.provenance["observation_identity_digest"]) == 64

    moved = pr.FilterPlate(
        "silicon",
        thickness_mm=plate.thickness_mm,
        size_mm=plate.size_mm,
        pose=pr.PlanarPose((1.0, 0.0, 50.0)),
    )
    moved_result = pr.simulate(
        pr.Beam(30.0),
        pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
        _detector(),
        **{**kwargs, "filters": (moved,)},
    )
    assert moved_result.provenance["identity_digest"] == result.provenance["identity_digest"]
    assert (
        moved_result.provenance["observation_identity_digest"]
        != result.provenance["observation_identity_digest"]
    )

    labeled = pr.FilterPlate(
        "silicon",
        thickness_mm=plate.thickness_mm,
        size_mm=plate.size_mm,
        pose=plate.pose,
        name="operator label only",
    )
    labeled_result = pr.simulate(
        pr.Beam(30.0),
        pr.Slab("hopg", 1_000.0, tilt_deg=30.0),
        _detector(),
        **{**kwargs, "filters": (labeled,)},
    )
    assert (
        labeled_result.provenance["observation_identity_digest"]
        == result.provenance["observation_identity_digest"]
    )
