"""Helicity- and angular-momentum-resolved axial channeling radiation (#508)."""

from __future__ import annotations

import numpy as np
import pytest

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.axial_emission import (
    AxialLine,
    LevelDensity,
    axial_line_catalog,
    axis_stokes,
    helicity_pattern,
    photon_angular_momentum,
)
from pyrite.montecarlo.channeling.dipole import boosted_dipole_pattern
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


@pytest.fixture(scope="module")
def tungsten():
    """W <001> at 300 keV: 1s and the 2p pair, the 2p -> 1s line, a grid about one string."""
    profile = axial_profile("tungsten", (0, 0, 1))
    catalog = axial_line_catalog(profile, 300e3, max_order=10)
    cell = profile.period_ang[0]
    grid = RealSpaceGrid((5 * cell, 5 * cell), (100, 100))
    center = (2 * cell, 2 * cell)
    return catalog, grid, center


def _circular_basis(grid, center, orbitals):
    """``L_z`` eigenvectors of a degenerate level (columns ordered ``l = +1, -1``)."""
    x = grid.x[:, None] - center[0]
    y = grid.y[None, :] - center[1]
    spectra = np.fft.fft2(orbitals)
    dx = np.fft.ifft2(1j * grid.kx[:, None] * spectra)
    dy = np.fft.ifft2(1j * grid.ky[None, :] * spectra)
    lz_orbitals = -1j * (x * dy - y * dx)
    matrix = np.einsum("sij,tij->st", orbitals.conj(), lz_orbitals) * grid.cell_area_ang2
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return values[::-1], vectors[:, ::-1]


def test_catalog_levels_and_dipole_forms_agree(tungsten):
    catalog, grid, center = tungsten
    assert [len(level.states) for level in catalog.levels] == [1, 2]  # 1s, 2p pair
    assert all(level.bandwidth_eV < 0.1 for level in catalog.levels)
    (line,) = catalog.lines
    assert line.delta_eV == pytest.approx(250.1, abs=0.5)
    assert catalog.photon_energy_eV(line) == pytest.approx(
        line.delta_eV / (1.0 - np.sqrt(1.0 - catalog.gamma**-2)), rel=1e-9
    )
    # Velocity form against the length form of the cut on-column orbitals.
    orbitals = catalog.orbitals(grid, center)
    x = grid.x[:, None] - center[0]
    y = grid.y[None, :] - center[1]
    length = (
        np.einsum(
            "fij,aij,sij->afs",
            orbitals[line.final].conj(),
            np.stack([x + 0 * y, y + 0 * x]),
            orbitals[line.initial],
        )
        * grid.cell_area_ang2
    )
    np.testing.assert_allclose(line.dipole_ang, length, rtol=5e-3, atol=1e-5)


def test_helicity_and_photon_jz_scale_with_the_ell_imbalance(tungsten):
    # Korotchenko & Kunashenko (arXiv:2403.01930): 2p(l) -> 1s emits J_z = l;
    # net on-axis helicity and photon J_z follow the 2p l-imbalance linearly,
    # and an l-balanced 2p population emits neither.
    catalog, grid, center = tungsten
    (line,) = catalog.lines
    lz, basis = _circular_basis(grid, center, catalog.orbitals(grid, center)[line.initial])
    np.testing.assert_allclose(lz, [1.0, -1.0], atol=2e-2)
    for p_plus in (0.0, 0.2, 0.5, 0.9, 1.0):
        rho = basis @ np.diag([p_plus, 1.0 - p_plus]) @ basis.conj().T
        polarization = line.polarization(rho)
        i, q, u, v = axis_stokes(polarization)
        imbalance = 2.0 * p_plus - 1.0
        assert v / i == pytest.approx(imbalance, abs=1e-3)
        assert abs(q) < 1e-6 * i and abs(u) < 1e-6 * i
        cone = photon_angular_momentum(polarization, catalog.gamma, 1.0 / catalog.gamma)
        assert cone["J_z"] == pytest.approx(imbalance, abs=1e-3)
        assert cone["helicity"] == pytest.approx(0.7951 * imbalance, abs=2e-3)
    # Coherence between the 2p states without imbalance: linear, not circular.
    rho = 0.5 * np.ones((2, 2))
    _, q, u, v = axis_stokes(line.polarization(basis @ rho @ basis.conj().T))
    assert abs(v) < 1e-6 * np.hypot(q, u)


def test_centred_vortex_probe_emits_its_helicity(tungsten):
    catalog, grid, center = tungsten
    (line,) = catalog.lines
    observable = LevelDensity(catalog, grid, center)
    for ell in (1, -1):
        phi = grid.sample(states.laguerre_gauss(ell, 0, waist_ang=0.35, center_ang=center))
        rho = observable.unpack(observable(phi))
        assert np.trace(rho[line.initial]).real == pytest.approx(0.94, abs=0.01)
        i, _, _, v = axis_stokes(line.polarization(rho[line.initial]))
        assert v / i == pytest.approx(ell, abs=1e-6)
        yield_per_ang = catalog.yield_per_ang(line, rho[line.initial])
        assert yield_per_ang == pytest.approx(
            catalog.yield_per_ang(line, np.eye(2) * np.trace(rho[line.initial]).real / 2), rel=1e-6
        )


def test_real_dipole_has_no_circular_polarization():
    # Planar limit: a real dipole has |A_+| = |A_-|.
    line = AxialLine(1, 0, 10.0, np.array([[[0.3]], [[0.1]]], dtype=complex))
    polarization = line.polarization(np.eye(1))
    i, q, u, v = axis_stokes(polarization)
    assert v == pytest.approx(0.0, abs=1e-15)
    assert (q, u) == pytest.approx((0.08, 0.06), rel=1e-12)
    assert i == pytest.approx(0.10, rel=1e-12)
    cone = photon_angular_momentum(polarization, 20.0, 0.05)
    assert cone["J_z"] == pytest.approx(0.0, abs=1e-15)
    assert cone["helicity"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("gamma", [1.0001, 1.6, 20.0])
def test_helicity_patterns_sum_to_the_boosted_dipole_and_normalize(gamma):
    theta = np.linspace(0.0, np.pi, 4001)
    for m in (1, -1):
        total = helicity_pattern(theta, gamma, m, 1) + helicity_pattern(theta, gamma, m, -1)
        average = 0.5 * (
            boosted_dipole_pattern(theta, 0.0, gamma)
            + boosted_dipole_pattern(theta, np.pi / 2, gamma)
        )
        np.testing.assert_allclose(total, average, rtol=1e-10)
    full = photon_angular_momentum(np.diag([1.0, 0.0]), gamma)
    assert full["fraction"] == pytest.approx(1.0, abs=1e-12)
    assert full["helicity"] == pytest.approx(0.0, abs=1e-12)  # invariant helicity split 1/2 : 1/2
    assert full["J_z"] == pytest.approx(1.0, abs=1e-12)
    narrow = photon_angular_momentum(np.diag([1.0, 0.0]), gamma, 1e-3 / gamma)
    assert narrow["S_z"] == pytest.approx(1.0, abs=1e-5)  # forward photons: pure spin
    assert narrow["L_z"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
def test_mcwf_line_helicity_tracks_the_surviving_2p_imbalance(tungsten):
    # With TDS jumps the 2p population leaves, but the line's on-axis V/I stays
    # the 2p l-imbalance per 2p electron at every depth (linear in rho).
    from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF

    catalog, _, _ = tungsten
    (line,) = catalog.lines
    profile = catalog.profile
    cell = profile.period_ang[0]
    grid = RealSpaceGrid((4 * cell, 4 * cell), (64, 64))
    center = (2 * cell, 2 * cell)
    dissipator = axial_dissipator(profile, 300e3, (4, 4), grid.shape, mechanisms=("tds",))
    stepper = AxialMCWF(dissipator, 300e3, profile.on_grid((4, 4), grid.shape))
    observable = LevelDensity(catalog, grid, center)
    _, basis = _circular_basis(grid, center, catalog.orbitals(grid, center)[line.initial])
    entry = states.laguerre_gauss(1, 0, waist_ang=0.35, center_ang=center)
    result = stepper.run(
        entry, [0.0, 100.0, 300.0], 64, seed=5, step_ang=1.0, observable=observable
    )
    yields = []
    for values in result.observables:
        rho = observable.unpack(values)[line.initial]
        circular = basis.conj().T @ rho @ basis
        i, _, _, v = axis_stokes(line.polarization(rho))
        population = np.trace(circular).real
        assert v / i == pytest.approx((circular[0, 0] - circular[1, 1]).real / population, abs=2e-2)
        yields.append(catalog.yield_per_ang(line, rho))
    assert yields[2] < 0.5 * yields[0]  # TDS empties 2p within ~10 nm
    assert v / i > 0.5  # survivors keep their helicity


def _winding(field) -> float:
    return float(np.sum(np.diff(np.unwrap(np.angle(field)))) / (2 * np.pi))


@pytest.mark.parametrize("gamma", [1.6, 10.8])
def test_far_field_amplitude_normalizes_and_reduces_to_the_planar_dipole(gamma):
    from pyrite.montecarlo.channeling.dipole import dipole_far_field

    theta = np.linspace(0.0, np.pi, 3001)
    phi = np.linspace(0.0, 2 * np.pi, 361)[:-1]
    t, p = np.meshgrid(theta, phi, indexing="ij")
    for d in ((1.0, 0.0), (1.0, 1j), (0.3, 0.7 - 0.2j)):
        density = np.sum(np.abs(dipole_far_field(t * np.cos(p), t * np.sin(p), gamma, d)) ** 2, 0)
        total = np.trapezoid(density.mean(axis=1) * np.sin(theta), theta) * 2 * np.pi
        assert total == pytest.approx(1.0, abs=1e-4)
    planar = dipole_far_field(t * np.cos(p), t * np.sin(p), gamma, (2.0, 0.0))
    np.testing.assert_allclose(
        np.sum(np.abs(planar) ** 2, 0), boosted_dipole_pattern(t, p, gamma), rtol=1e-9, atol=1e-12
    )
    # A real dipole's field is real up to the fixed basis: E_- = conj(E_+).
    np.testing.assert_allclose(planar[1], planar[0].conj(), atol=1e-12)


def test_far_field_of_a_jz_plus_line_winds_only_the_opposite_circular_component():
    from pyrite.montecarlo.channeling.dipole import dipole_far_field

    gamma = 10.8
    on_axis = dipole_far_field(0.0, 0.0, gamma, (1.0, 1j))
    assert abs(on_axis[1]) < 1e-15 and abs(on_axis[2]) < 1e-15
    phi = np.linspace(0.0, 2 * np.pi, 2001)
    for radius in (0.3, 1.0, 2.5):
        e_plus, e_minus, _ = dipole_far_field(
            radius / gamma * np.cos(phi), radius / gamma * np.sin(phi), gamma, (1.0, 1j)
        )
        assert _winding(e_plus) == pytest.approx(0.0, abs=1e-9)
        assert _winding(e_minus) == pytest.approx(2.0, abs=1e-9)  # l = J_z - sigma
        mirrored = dipole_far_field(
            radius / gamma * np.cos(phi), radius / gamma * np.sin(phi), gamma, (1.0, -1j)
        )
        assert _winding(mirrored[0]) == pytest.approx(-2.0, abs=1e-9)
    # Paraxially the fixed-basis components are the helicity patterns.
    theta = np.array([1e-3, 0.3, 1.0]) / gamma
    field = dipole_far_field(theta, 0.0 * theta, gamma, np.array([1.0, 1j]) / np.sqrt(2))
    np.testing.assert_allclose(
        np.abs(field[0]) ** 2, helicity_pattern(theta, gamma, 1, 1), rtol=2e-2
    )
    np.testing.assert_allclose(
        np.abs(field[1][1:]) ** 2, helicity_pattern(theta[1:], gamma, 1, -1), rtol=2e-2
    )
