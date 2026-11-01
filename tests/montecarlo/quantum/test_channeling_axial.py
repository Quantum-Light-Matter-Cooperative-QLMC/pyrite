"""Axial string potential and real-space split-step backend (#508)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.special import airy as scipy_airy
from scipy.special import eval_genlaguerre

from pyrite.materials.channeling import longitudinal_kinematics, planar_profile
from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.materials.crystal import CRYSTALS
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.axial_emission import axial_line_catalog
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid, SplitStepPropagator


def _planar_average(profile, x):
    y = np.linspace(0.0, profile.period_ang[1], 512, endpoint=False)
    return profile.potential(x, y).mean(axis=1)


@pytest.mark.parametrize(
    ("crystal", "uvw", "x_plane", "y_plane"),
    [
        ("silicon", (0, 0, 1), (1, 1, 0), (1, -1, 0)),
        ("silicon", (1, 1, 0), (1, -1, 0), (0, 0, 1)),
        ("diamond", (1, 1, 0), (1, -1, 0), (0, 0, 1)),
        ("silicon", (1, 1, 1), (1, 0, -1), (1, -2, 1)),
    ],
)
def test_string_potential_averages_to_planar(crystal, uvw, x_plane, y_plane):
    profile = axial_profile(crystal, uvw)
    x = np.linspace(0.0, profile.period_ang[0], 64, endpoint=False)
    planar_x = planar_profile(crystal, x_plane)
    ratio = profile.period_ang[0] / planar_x.period_ang
    assert ratio == pytest.approx(round(ratio), abs=1e-9)
    np.testing.assert_allclose(_planar_average(profile, x), planar_x.potential(x), atol=1e-10)

    # The same along y: average over x, compare with the plane normal to y.
    y = np.linspace(0.0, profile.period_ang[1], 64, endpoint=False)
    xs = np.linspace(0.0, profile.period_ang[0], 512, endpoint=False)
    planar_y = planar_profile(crystal, y_plane)
    np.testing.assert_allclose(
        profile.potential(xs, y).mean(axis=0), planar_y.potential(y), atol=1e-10
    )


def test_cell_and_strings_silicon():
    a = CRYSTALS["silicon"]["lattice"]["a"]
    p001 = axial_profile("silicon", (0, 0, 1))
    np.testing.assert_allclose(p001.period_ang, (a / np.sqrt(8.0),) * 2, rtol=1e-12)
    assert len(p001.strings) == 1
    p110 = axial_profile("silicon", (1, 1, 0))
    np.testing.assert_allclose(p110.period_ang, (a / np.sqrt(2.0), a), rtol=1e-12)
    assert len(p110.strings) == 4
    # Dumbbell: string pairs a/4 apart along y.
    ys = sorted(s.position_ang[1] for s in p110.strings if abs(s.position_ang[0]) < 1e-9)
    assert np.isclose(ys[1] - ys[0], a / 4.0)
    for profile in (p001, p110, axial_profile("silicon", (1, 1, 1))):
        total = sum(sum(s.linear_density.values()) for s in profile.strings)
        assert np.isclose(total, 8.0 * profile.area_ang2 / a**3, rtol=1e-12)
        assert np.isclose(profile.nuclear_density[profile.max_order].real, 8.0 / a**3)
        # Wells sit on the strings.
        u_min, _ = profile.extrema()
        for s in profile.strings:
            u = profile.potential([s.position_ang[0]], [s.position_ang[1]])[0, 0]
            assert u == pytest.approx(u_min, abs=0.5)


def test_displacement_override():
    a = CRYSTALS["silicon"]["lattice"]["a"]
    default = axial_profile("silicon", (0, 0, 1))
    u2 = default.displacements_ang2[0][0, 0]
    iso = axial_profile("silicon", (0, 0, 1), displacements_ang2={"Si": u2})
    np.testing.assert_allclose(iso.potential_eV, default.potential_eV, atol=1e-14)
    # Displacement along the axis does not smear the string (g . z = 0).
    along = u2 * np.eye(3) + 0.05 * np.outer(default.axes[2], default.axes[2])
    np.testing.assert_allclose(
        axial_profile("silicon", (0, 0, 1), displacements_ang2=[along] * 8).potential_eV,
        default.potential_eV,
        atol=1e-14,
    )
    # In-plane anisotropy breaks the x <-> y symmetry of <001>.
    across = u2 * np.eye(3) + 0.02 * np.outer(default.axes[0], default.axes[0])
    tilted = axial_profile("silicon", (0, 0, 1), displacements_ang2={"Si": across})
    x = np.linspace(0.0, a / np.sqrt(8.0), 16, endpoint=False)
    u = tilted.potential(x, x)
    assert np.abs(u - u.T).max() > 0.1
    with pytest.raises(ValueError, match="symmetric 3x3"):
        axial_profile("silicon", (0, 0, 1), displacements_ang2={"Si": np.ones((3, 3, 1))})


def test_on_grid_band_limited_and_tiles():
    profile = axial_profile("silicon", (1, 1, 0))
    single = profile.on_grid((1, 1), (32, 48))
    tiled = profile.on_grid((2, 3), (64, 144))
    np.testing.assert_allclose(tiled[:32, :48], single, atol=1e-10)
    np.testing.assert_allclose(tiled[32:, 96:], single, atol=1e-10)
    # Within the band the grid equals the direct synthesis of the kept harmonics.
    fine = profile.on_grid((1, 1), (256, 256))
    x = np.arange(256) * profile.period_ang[0] / 256
    y = np.arange(256) * profile.period_ang[1] / 256
    np.testing.assert_allclose(fine, profile.potential(x, y), atol=1e-8)


def test_axial_states_symmetry_and_norm():
    gamma, _ = longitudinal_kinematics(10e6)
    profile = axial_profile("silicon", (0, 0, 1))
    result = axial_states(profile, gamma, max_order=10)
    e = result.energies_eV
    assert e[0] < e[1]
    # C4v string: the first excited level is the doubly degenerate 2p pair.
    assert e[1] == pytest.approx(e[2], abs=1e-9)
    assert e[3] - e[2] > 1.0
    x = np.linspace(0.0, profile.period_ang[0], 64, endpoint=False)
    psi = result.wavefunction(x, x, 0)
    area = (x[1] - x[0]) ** 2
    assert np.sum(np.abs(psi) ** 2) * area == pytest.approx(1.0, rel=1e-12)


@pytest.mark.slow
def test_split_step_holds_eigenstates_stationary():
    """Propagator and plane-wave eigensolver agree; the phase error is O(dz^2)."""
    kinetic = 10e6
    gamma, _ = longitudinal_kinematics(kinetic)
    profile = axial_profile("silicon", (0, 0, 1))
    result = axial_states(profile, gamma, max_order=14)
    grid = RealSpaceGrid(profile.period_ang, (64, 64))
    prop = SplitStepPropagator(grid, kinetic, potential_eV=profile.on_grid((1, 1), grid.shape))
    depth = 2000.0
    errors = []
    for step in (5.0, 2.5):
        for state in (0, 3):
            phi = result.wavefunction(grid.x, grid.y, state)
            out = prop.evolve(phi, depth, step_ang=step)
            overlap = np.vdot(phi, out) * grid.cell_area_ang2
            assert grid.norm(out) == pytest.approx(1.0, abs=1e-12)
            assert abs(overlap) > 1.0 - 1e-6
            phase = np.exp(-1j * result.energies_eV[state] * depth / prop.hbar_v)
            errors.append(abs(np.angle(overlap / phase)))
    assert max(errors[:2]) < 1e-2
    assert max(errors[2:]) < 3e-3
    assert errors[0] / errors[2] == pytest.approx(4.0, rel=0.1)


def _lg_mode(ell, p, z, x, y, waist, k):
    z_r = k * waist**2 / 2.0
    w = waist * np.hypot(1.0, z / z_r)
    r2 = x**2 + y**2
    curvature = np.exp(0.5j * k * r2 * z / (z**2 + z_r**2))
    a = abs(ell)
    return (
        (np.sqrt(2.0 * r2) / w) ** a
        * eval_genlaguerre(p, a, 2.0 * r2 / w**2)
        * np.exp(-r2 / w**2 + 1j * ell * np.arctan2(y, x))
        * curvature
        / w
    )


@pytest.mark.parametrize(("ell", "p"), [(0, 0), (1, 0), (-2, 1), (3, 2)])
def test_free_space_laguerre_gauss_gouy_phase(ell, p):
    kinetic = 200e3
    k = states.momentum_inv_ang(kinetic)
    waist = 3.0
    z_r = k * waist**2 / 2.0
    grid = RealSpaceGrid((120.0, 120.0), (256, 256))
    x, y = np.meshgrid(
        (grid.x + 60.0) % 120.0 - 60.0, (grid.y + 60.0) % 120.0 - 60.0, indexing="ij"
    )
    phi = grid.sample(states.laguerre_gauss(ell, p, waist_ang=waist))
    prop = SplitStepPropagator(grid, kinetic)
    start = _lg_mode(ell, p, 0.0, x, y, waist, k)
    c0 = np.vdot(start, phi)
    for z in (0.5 * z_r, 2.0 * z_r):
        out = prop.evolve(phi, z, step_ang=z)
        mode = _lg_mode(ell, p, z, x, y, waist, k)  # no Gouy factor
        cz = np.vdot(mode, out) / np.sqrt(np.vdot(mode, mode).real / np.vdot(start, start).real)
        assert abs(cz) == pytest.approx(abs(c0), rel=1e-10)
        gouy = -(2 * p + abs(ell) + 1) * np.arctan(z / z_r)
        assert np.angle(cz / c0 * np.exp(-1j * gouy)) == pytest.approx(0.0, abs=1e-10)


def test_free_space_airy_self_acceleration():
    """Finite-energy 2D Airy follows Siviloglou & Christodoulides (2007), eq. 4."""
    kinetic = 200e3
    k = states.momentum_inv_ang(kinetic)
    x0, a = 1.0, 0.3
    grid = RealSpaceGrid((160.0, 160.0), (512, 512))
    phi = grid.sample(states.airy(scale_ang=x0, truncation=a, dimension=2))
    prop = SplitStepPropagator(grid, kinetic)
    x = (grid.x + 80.0) % 160.0 - 80.0
    norm = (x0 * np.exp(2.0 * a**3 / 3.0) / np.sqrt(8.0 * np.pi * a)) ** -0.5

    def field(xi):
        s = x / x0
        return (
            norm
            * scipy_airy(s - xi**2 / 4.0 + 1j * a * xi)[0]
            * np.exp(a * s - a * xi**2 / 2.0 - 1j * xi**3 / 12.0 + 0.5j * a**2 * xi + 0.5j * s * xi)
        )

    lobes = []
    for xi in (1.0, 2.0, 4.0):
        out = prop.evolve(phi, xi * k * x0**2, step_ang=xi * k * x0**2)
        ref = np.outer(field(xi), field(xi))
        assert np.abs(out - ref).max() < 1e-6 * np.abs(ref).max()
        lobes.append(x[np.argmax(np.abs(out).max(axis=1))])
    # The main lobe accelerates sideways (ballistic parabola, shifted by the truncation).
    assert lobes[0] < lobes[1] < lobes[2]
    assert lobes[2] - lobes[1] > 2.0 * (lobes[1] - lobes[0])


def test_absorption_decays_norm():
    grid = RealSpaceGrid((10.0, 10.0), (16, 16))
    w = np.full(grid.shape, 0.5)
    prop = SplitStepPropagator(grid, 10e6, absorptive_eV=w)
    phi = np.full(grid.shape, 0.1 + 0j)
    out = prop.evolve(phi, 100.0, step_ang=10.0)
    expected = np.exp(-2.0 * 0.5 * 100.0 / prop.hbar_v)
    assert grid.norm(out) == pytest.approx(grid.norm(phi) * expected, rel=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        SplitStepPropagator(grid, 10e6, absorptive_eV=-w)


def test_tilt_along_plane_converges_to_planar():
    """A beam tilted far from <001> but aligned with (110) sees the planar potential."""
    kinetic = 10e6
    profile = axial_profile("silicon", (0, 0, 1))
    planar = planar_profile("silicon", (1, 1, 0))
    grid = RealSpaceGrid(profile.period_ang, (48, 48))
    string = profile.on_grid((1, 1), grid.shape)
    sheet = np.broadcast_to(planar.potential(grid.x)[:, None], grid.shape).copy()
    entry = np.full(grid.shape, grid.area_ang2**-0.5, dtype=complex)
    depths = np.linspace(500.0, 1e4, 20)

    def x_density(potential, tilt):
        prop = SplitStepPropagator(grid, kinetic, potential_eV=potential, tilt_rad=(0.0, tilt))
        return np.array(
            [(np.abs(f) ** 2).sum(1) for _, f in prop.trajectory(entry, depths, step_ang=2.0)]
        )

    reference = x_density(sheet, 0.02)
    # Tilt within the plane does not change planar evolution.
    np.testing.assert_allclose(x_density(sheet, 0.1), reference, atol=1e-10 * reference.max())
    errors = [
        np.abs(x_density(string, t) - reference).max() / reference.max() for t in (0.02, 0.05, 0.1)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01
    # Approaches as 1/theta^2.
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.25)


def test_grid_sampling_rejects_planar_states_and_mixes():
    grid = RealSpaceGrid((40.0, 40.0), (64, 64))
    with pytest.raises(ValueError, match="2D"):
        grid.sample(states.gaussian(waist_ang=2.0, dimension=1))
    mix = states.mixture(
        [(0.25, states.gaussian(waist_ang=2.0)), (0.75, states.laguerre_gauss(1, waist_ang=2.0))]
    )
    parts = grid.ensemble(mix)
    assert [w for w, _ in parts] == [0.25, 0.75]
    assert all(grid.norm(phi) == pytest.approx(1.0, abs=1e-9) for _, phi in parts)


_WATSON = json.loads(
    (Path(__file__).parents[2] / "data/channeling/watson-koehler-1982-si-axial.json").read_text()
)


@cache
def _main_line_keV(uvw, kinetic_energy_MeV):
    """Forward energy of the strongest dipole line onto the deepest Si string level.

    Upper-level states are equally populated (``rho`` = identity). The 0.5 eV
    flat-band gate admits the Si <111> 2p band (0.06-0.2 eV wide at 2.4-3 MeV);
    lines inside the split deepest level (< 5 eV) are not channeling lines.
    ``max_order`` 16 is converged to 0.4% (<110>) and 0.03% (<111>) against 24.
    """
    catalog = axial_line_catalog(
        axial_profile("silicon", uvw), kinetic_energy_MeV * 1e6, max_order=16, max_bandwidth_eV=0.5
    )
    deepest = catalog.levels[0].energy_eV
    candidates = [
        line
        for line in catalog.lines
        if line.delta_eV > 5.0 and catalog.levels[line.final].energy_eV - deepest < 1.0
    ]
    line = max(
        candidates,
        key=lambda ln: catalog.yield_per_ang(ln, np.eye(len(catalog.levels[ln.initial].states))),
    )
    return float(catalog.photon_energy_eV(line)) / 1e3


@pytest.mark.slow
@pytest.mark.parametrize(
    ("axis", "peak", "other"), [("111", 1, "andersen_laegsgaard"), ("110", 0, "cue_et_al")]
)
def test_silicon_axial_main_line_matches_watson_koehler(axis, peak, other):
    # Watson & Koehler, PRB 25, 3079 (1982), Tables I-II: forward-equivalent
    # peaks of 2.2-3.1 MeV electrons along Si <111> (3d->2p, 2p->1s, 3p->1s)
    # and <110>. The model's main line onto the deepest level is 2p->1s on
    # <111> and the lowest <110> peak. The other two peaks per axis need
    # states above the inter-string saddle; this flat-level model has no
    # strong line onto 1s from them, so they are not tested.
    data = _WATSON["axes"][axis]
    gammas, model = [], []
    for row in data["rows"]:
        measured = row["peaks_keV"]
        if measured[peak] is None:
            continue
        energy = _main_line_keV(tuple(data["uvw"]), row["kinetic_energy_MeV"])
        known = [(abs(energy - value), index) for index, value in enumerate(measured) if value]
        assert min(known)[1] == peak, row  # nearest measured peak is the assigned one
        # The model sits 6-15% below Watson, inside the 20-30% spread between
        # Watson and Andersen & Laegsgaard on the same lines, and within the
        # 10% combined error Watson quotes of the other lab's Table I fit.
        assert energy == pytest.approx(measured[peak], rel=0.2)
        fit = data["fits"][other][peak]
        assert energy == pytest.approx(fit["E0_keV"] * row["gamma"] ** fit["x"], rel=0.1)
        gammas.append(longitudinal_kinematics(row["kinetic_energy_MeV"] * 1e6)[0])
        model.append(energy)
    exponent = np.polyfit(np.log(gammas), np.log(model), 1)[0]
    assert exponent == pytest.approx(data["fits"]["this_work"][peak]["x"], abs=0.25)


@pytest.mark.slow
def test_silicon_axial_main_lines_order_like_watson_koehler():
    # At equal electron energy the <110> main line lies above the <111> 2p->1s
    # line (Watson & Koehler Table II: 3.03 vs 3.44 keV at 2.78 MeV).
    rows = {
        axis: {r["kinetic_energy_MeV"]: r["peaks_keV"] for r in _WATSON["axes"][axis]["rows"]}
        for axis in ("111", "110")
    }
    for energy in sorted(set(rows["111"]) & set(rows["110"])):
        assert rows["110"][energy][0] > rows["111"][energy][1]
        assert _main_line_keV((1, 1, 0), energy) > _main_line_keV((1, 1, 1), energy)
