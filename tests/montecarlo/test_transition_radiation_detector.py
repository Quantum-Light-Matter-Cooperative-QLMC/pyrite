"""TR integrated through planar-detector acceptance and energy channels (#103)."""

import numpy as np
import pytest

from pyrite._planar_geometry import PixelGrid, PlanarPose, planar_rays
from pyrite.materials.crystal import M_E_EV
from pyrite.montecarlo.spectrum.transition_radiation import (
    ginzburg_frank_interface,
    planar_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import BoundaryCrossings
from pyrite.montecarlo.spectrum.transition_radiation_detector import (
    detected_transition_radiation,
)


def _track(planes, beta, faces):
    """Crossings of a straight +z charge through the origin at t = 0."""
    planes = np.asarray(planes, dtype=float)
    faces = np.asarray(faces)
    z = planes[faces]
    k = faces.size
    return BoundaryCrossings(
        electron_id=np.zeros(k, int), track_id=np.zeros(k, int), charge=-np.ones(k, np.int8),
        position_ang=np.outer(z, [0.0, 0.0, 1.0]), direction=np.tile([0.0, 0.0, 1.0], (k, 1)),
        kinetic_energy_keV=np.full(k, M_E_EV * (1 / np.sqrt(1 - beta**2) - 1) / 1e3),
        time_ang=z / beta, interface=faces, normal=np.tile([0.0, 0.0, 1.0], (k, 1)),
        layer_before=faces - 1, layer_after=np.where(faces == planes.size - 1, -1, faces),
        z_interfaces_ang=planes, n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


def _pose(theta, distance_mm=100.0):
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    x_axis = (np.cos(theta), 0.0, -np.sin(theta))
    return PlanarPose(
        center_mm=tuple(distance_mm * direction), normal=tuple(direction), x_axis=x_axis
    )


def _reference(density, pose, size_mm, edges, n_energy, s):
    """Gauss-Legendre in energy and a fine point sub-pixel grid over the plane."""
    rays = planar_rays(pose, pixels=PixelGrid((s, s), (size_mm[1] / s, size_mm[0] / s)))
    n = rays.directions_lab.reshape(-1, 3)
    omega = rays.solid_angle_sr.reshape(-1)
    x, w = np.polynomial.legendre.leggauss(n_energy)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        energy = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        out.append(0.5 * (hi - lo) * np.sum(w * (density(energy[:, None], n) @ omega)))
    return np.array(out)


def drude(wp, loss=0.0):
    return lambda energy: 1.0 - (wp / energy) ** 2 * (1.0 - 1j * loss)


def test_single_exit_interface_matches_the_closed_form_through_the_acceptance():
    beta, wp = 0.995, 31.0
    eps = drude(wp, 0.05)
    # One exit crossing of a thick absorbing slab: Ginzburg-Frank into vacuum.
    crossings = _track([0.0, 1e7], beta, [1])
    pose, size, edges = _pose(0.1), (4.0, 4.0), np.array([200.0, 600.0, 2000.0])
    result = detected_transition_radiation(
        crossings, lambda e: [eps(e)], pose, edges, size_mm=size, n_electrons=1, rtol=1e-4
    )
    assert result.converged and result.photons.shape == (1, 1, 2)

    def density(energy, n):
        theta = np.arccos(n[:, 2])
        return ginzburg_frank_interface(eps(energy), beta, theta[None], energy)

    expected = _reference(density, pose, size, edges, 48, 48)
    np.testing.assert_allclose(result.photons[0, 0], expected, rtol=1e-3)
    assert np.all(result.error_estimate <= 1e-4 * result.photons.max())


def test_formation_zone_fringes_are_resolved_by_refinement():
    # A 50 um transparent film at gamma ~ 200 and theta = 0.03: about twenty
    # formation-zone fringes per channel, unresolved by the starting grid.
    beta, wp, foil = 1 - 1.25e-5, 31.0, 5e5
    eps = drude(wp)
    crossings = _track([0.0, foil], beta, [0, 1])
    pose, size, edges = _pose(0.03, 1000.0), (0.2, 0.2), np.array([1000.0, 1500.0, 2000.0])
    coarse = detected_transition_radiation(
        crossings, lambda e: [eps(e)], pose, edges, size_mm=size, n_electrons=1,
        max_energy_intervals=4, max_angular_subdivisions=1,
    )  # fmt: skip
    result = detected_transition_radiation(
        crossings, lambda e: [eps(e)], pose, edges, size_mm=size, n_electrons=1,
        rtol=1e-4, max_energy_intervals=1024, max_angular_subdivisions=32,
    )  # fmt: skip
    assert result.converged and result.energy_intervals >= 64

    def density(energy, n):
        return planar_transition_radiation(energy, n[None], [0.0, 0.0, beta], [(eps(energy), foil)])

    expected = _reference(density, pose, size, edges, 200, 16)
    np.testing.assert_allclose(result.photons[0, 0], expected, rtol=5e-4)
    assert np.max(np.abs(coarse.photons[0, 0] / expected - 1)) > 0.05


def test_pixel_tables_sum_to_the_whole_plane():
    beta, eps = 0.99, drude(31.0, 0.05)
    crossings = _track([0.0, 1e7], beta, [1])
    pose, edges = _pose(0.15, 50.0), np.array([300.0, 900.0])
    plane = detected_transition_radiation(
        crossings, lambda e: [eps(e)], pose, edges, size_mm=(3.0, 2.0), n_electrons=1, rtol=1e-4
    )
    pixels = detected_transition_radiation(
        crossings, lambda e: [eps(e)], pose, edges, pixels=PixelGrid((4, 6), (0.5, 0.5)),
        n_electrons=1, rtol=1e-4,
    )  # fmt: skip
    assert pixels.photons.shape == (4, 6, 1)
    np.testing.assert_allclose(pixels.photons.sum((0, 1)), plane.photons[0, 0], rtol=5e-4)
    # The forward cone is centred on the beam: the pixel row nearest it is brightest.
    assert np.argmax(pixels.photons[:, :, 0].sum(0)) == 0


def test_tilted_sample_and_rejections():
    beta, eps = 0.99, drude(31.0, 0.05)
    crossings = _track([0.0, 1e7], beta, [1])
    edges = np.array([300.0, 900.0])
    pose = _pose(0.0, 50.0)
    straight = detected_transition_radiation(
        crossings, lambda e: [eps(e)], _pose(0.2, 50.0), edges, size_mm=(2.0, 2.0),
        n_electrons=1, rtol=1e-4,
    )  # fmt: skip
    # Tilting the slab normal by 0.2 rad toward +x puts the same slab-frame
    # direction on the lab axis.
    tilted = detected_transition_radiation(
        crossings, lambda e: [eps(e)], pose, edges, size_mm=(2.0, 2.0), n_electrons=1,
        rtol=1e-4, tilt_polar_rad=-0.2,
    )  # fmt: skip
    np.testing.assert_allclose(tilted.photons, straight.photons, rtol=2e-3)
    side = PlanarPose(center_mm=(50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), x_axis=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="one slab face"):
        detected_transition_radiation(
            crossings, lambda e: [eps(e)], side, edges, size_mm=(20.0, 2.0), n_electrons=1
        )
    with pytest.raises(ValueError, match="channel edges"):
        detected_transition_radiation(
            crossings, lambda e: [eps(e)], pose, [900.0, 300.0], size_mm=(2.0, 2.0), n_electrons=1
        )


def test_transported_bilayer_on_a_tilted_sample():
    from pyrite.montecarlo.spectrum.transition_radiation_crossings import boundary_crossings
    from pyrite.montecarlo.transport import simulate_trajectories

    layers = [(0.0, 1500.0, [("C", 0.1136)]), (1500.0, 3000.0, [("W", 0.0632)])]
    rows = simulate_trajectories(
        E0_keV=25.0, Ne=40, seed=3, E_cut_keV=1.0, energy_model="midpoint", thickness_ang=3000.0,
        layers=layers, tilt_polar_rad=0.3,
    )  # fmt: skip
    crossings = boundary_crossings(rows, layers)
    # Backward detector on the entrance side of the tilted slab.
    pose = PlanarPose(center_mm=(0.0, 0.0, -40.0), normal=(0.0, 0.0, -1.0), x_axis=(1.0, 0.0, 0.0))
    result = detected_transition_radiation(
        crossings, lambda e: [drude(31.0, 0.05)(e), drude(80.0, 0.3)(e)], pose,
        np.array([100.0, 200.0, 400.0]), size_mm=(4.0, 4.0), n_electrons=rows["Ne"],
        tilt_polar_rad=0.3, rtol=1e-2,
    )  # fmt: skip
    assert result.converged
    assert np.all(np.isfinite(result.photons)) and np.all(result.photons > 0.0)
