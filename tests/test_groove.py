import numpy as np
import pytest

from cxr_mc.montecarlo.groove import (
    blazed_groove_spec,
    entry_points,
    escape_distance_ang,
)

TP = np.deg2rad(45.0)
SPEC = blazed_groove_spec(
    spacing_ang=2.0e4,
    theta_obs_rad=np.pi / 2,
    tilt_polar_rad=TP,
    tilt_azim_rad=np.pi,
)


def _z_surf(x, spec):
    """Brute-force sawtooth profile height (depth of the surface) at x."""
    tp = spec.tilt_polar_rad
    lam, h = spec.spacing_ang, spec.depth_ang
    u = np.mod(x, lam)
    x_valley = h * np.tan(tp)
    return np.where(u <= x_valley, u / np.tan(tp), (lam - u) * np.tan(tp))


def _march_escape(x, z, spec, ds=0.05):
    """Reference ray march along n_hat until the point leaves the material."""
    tp = spec.tilt_polar_rad
    n = np.array([np.cos(tp), -np.sin(tp)])
    p = np.array([x, z], dtype=float)
    s = 0.0
    while p[1] > 0 and p[1] >= _z_surf(p[0], spec) - 1e-12:
        p += ds * n
        s += ds
    return s


def test_depth_closes_unit_cell():
    assert SPEC.depth_ang == pytest.approx(SPEC.spacing_ang * np.sin(TP) * np.cos(TP))


def test_escape_matches_ray_march():
    rng = np.random.default_rng(7)
    lam, h = SPEC.spacing_ang, SPEC.depth_ang
    x = rng.uniform(0.0, 3 * lam, 200)
    z = rng.uniform(0.0, 4 * h, 200)
    inside = z >= _z_surf(x, SPEC) + 1e-6
    x, z = x[inside], z[inside]
    L = escape_distance_ang(x, z, SPEC)
    ref = np.array([_march_escape(xi, zi, SPEC) for xi, zi in zip(x, z, strict=False)])
    assert np.all(np.abs(L - ref) < 0.2)  # ray-march step tolerance


def test_escape_flat_limit_small_depth():
    tiny = blazed_groove_spec(
        spacing_ang=1.0,
        theta_obs_rad=np.pi / 2,
        tilt_polar_rad=TP,
        tilt_azim_rad=np.pi,
    )
    z = np.array([5.0e4])
    L = escape_distance_ang(np.array([1234.5]), z, tiny)
    assert L[0] == pytest.approx(z[0] / np.sin(TP), rel=1e-3)


def test_escape_never_exceeds_flat_path_and_nonnegative():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 5 * SPEC.spacing_ang, 500)
    z = rng.uniform(SPEC.depth_ang, 10 * SPEC.depth_ang, 500)
    L = escape_distance_ang(x, z, SPEC)
    assert np.all(L >= 0.0)
    assert np.all(L <= z / np.sin(TP) + 1e-9)


def test_entry_points_land_on_relief_facet():
    rng = np.random.default_rng(11)
    x0 = rng.uniform(-3 * SPEC.spacing_ang, 3 * SPEC.spacing_ang, 300)
    xe, ze = entry_points(x0, SPEC)
    assert np.all((ze >= 0.0) & (ze < SPEC.depth_ang))
    # every entry point lies ON the surface profile
    assert np.allclose(ze, _z_surf(xe, SPEC), atol=1e-6)
    # displacement is along the beam direction
    assert np.allclose((xe - x0) / np.sin(TP) * np.cos(TP), ze, atol=1e-6)


def test_blazed_groove_spec_validates_geometry():
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.deg2rad(119.0), TP, np.pi)  # theta_obs != 90
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.pi / 2, TP, 0.0)  # azim != 180
    with pytest.raises(ValueError):
        blazed_groove_spec(1e4, np.pi / 2, 0.0, np.pi)  # tp == 0
    with pytest.raises(ValueError):
        blazed_groove_spec(-1.0, np.pi / 2, TP, np.pi)  # spacing <= 0
