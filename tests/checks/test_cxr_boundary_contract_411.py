"""Algebra/contract checks for the unimplemented #411 CXR boundary source."""

import numpy as np
import pytest

from checks.cxr_boundary_contract_411 import BoundaryCells, sample_cells


def _cells():
    # Two patches * two directions * two energies. Already integrated photon
    # counts per incident primary (not rates per sr and not emitted in bulk).
    position = np.repeat([[0.0, 0, 0], [1.0, 0, 0]], 4, axis=0)
    normal = np.tile([0.0, 0, 1.0], (8, 1))
    direction = np.tile([0.0, 0, 1.0], (8, 1))
    direction[1::2] = [0.6, 0, 0.8]
    energy_eV = np.tile([500.0, 1000.0, 500.0, 1000.0], 2)
    weights = 0.001 * np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    return BoundaryCells(position, normal, direction, energy_eV, weights)


def test_weighted_source_conserves_per_primary_and_samples_off_cone():
    source = _cells()
    index, weights, coherency = sample_cells(source, 40_000, seed=411)
    assert coherency is None
    assert weights.sum() == pytest.approx(source.weight_per_primary.sum(), rel=1e-14)
    assert np.any(source.direction[index, 0] > 0.5)  # non-detector support retained
    assert np.any(source.direction[index, 0] == 0.0)
    assert source.position_ang.flags.writeable is False
    np.testing.assert_array_equal(index, sample_cells(source, 40_000, seed=411)[0])


def test_quadrature_marginal_uses_area_energy_and_solid_angle_once():
    # Synthetic uniform per-area crossing density over two surface patches.
    # In a real source it must be derived from the FULL escaping radiation
    # field, not guessed by uniformly distributing mc_cxr_spectrum over area.
    area = np.array([2.0, 3.0])  # arbitrary angstrom**2
    d_energy = 20.0  # eV
    d_omega = 0.1  # sr
    density = 0.03  # photons/(electron angstrom**2 eV sr)
    weights = density * area * d_energy * d_omega
    source = BoundaryCells(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.array([900.0, 900.0]),
        weights,
    )
    # This is the angular marginal matching the units of mc_cxr_spectrum.
    marginal = source.weight_per_primary.sum() / (d_energy * d_omega)
    assert marginal == pytest.approx(density * area.sum())
    assert source.weight_per_primary.sum() == pytest.approx(marginal * d_energy * d_omega)
    assert sample_cells(source, 10, seed=7)[1].sum() == pytest.approx(weights.sum())


def test_combined_pxr_cbs_is_one_nonnegative_population():
    pxr = np.array([1.0 + 0j, 0j])
    cbs = np.array([-0.4 + 0j, 0.2j])
    combined = np.vdot(pxr + cbs, pxr + cbs).real
    diagnostics = np.vdot(pxr, pxr).real + np.vdot(cbs, cbs).real
    assert combined == pytest.approx(0.4)
    assert diagnostics == pytest.approx(1.2)
    assert combined >= 0.0
    assert combined != pytest.approx(diagnostics)
    # The CXR source intensity would be proportional to combined; the
    # separately useful PXR/CBS diagonal arrays must NOT be added as events.


def test_polarization_coherency_is_retained_and_trace_normalized():
    basic = _cells()
    weight = basic.weight_per_primary
    jones = np.zeros((len(weight), 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = weight * 0.8
    jones[:, 1, 1] = weight * 0.2
    jones[:, 0, 1] = weight * 0.1j
    jones[:, 1, 0] = -weight * 0.1j
    basis = np.tile([1.0, 0.0, 0.0], (8, 1))
    basis[1::2] = [0.8, 0.0, -0.6]
    source = BoundaryCells(
        basic.position_ang, basic.normal, basic.direction, basic.energy_eV, weight, jones, basis
    )
    indices, weights, sampled_jones = sample_cells(source, 5000, seed=1, polarized=True)
    np.testing.assert_allclose(np.trace(sampled_jones, axis1=1, axis2=2), weights)
    np.testing.assert_allclose(sampled_jones[:, 0, 0], 0.8 * weights)
    np.testing.assert_array_equal(indices, sample_cells(source, 5000, seed=1)[0])
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        sample_cells(basic, 20, seed=1, polarized=True)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"energy_eV": [0.0] * 8}, "energy must be positive"),
        ({"weight_per_primary": [-1.0] * 8}, "weights nonnegative"),
        ({"direction": [[0.0, 0.0, -1.0]] * 8}, "strictly outwards"),
        ({"direction": [[0.0, 0.0, 2.0]] * 8}, "must be unit vectors"),
        ({"position_ang": [[0.0, 0.0, float("nan")]] * 8}, "must be finite"),
    ],
)
def test_invalid_surface_placement_and_weights_fail_closed(change, match):
    source = _cells()
    data = {
        "position_ang": source.position_ang,
        "normal": source.normal,
        "direction": source.direction,
        "energy_eV": source.energy_eV,
        "weight_per_primary": source.weight_per_primary,
    }
    data.update(change)
    with pytest.raises(ValueError, match=match):
        BoundaryCells(**data)


def test_polarization_nonpositive_or_wrong_trace_is_rejected():
    source = _cells()
    jones = np.zeros((8, 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = source.weight_per_primary
    jones[0, 1, 1] = -0.3 * source.weight_per_primary[0]
    jones[0, 0, 0] *= 1.3  # keep trace correct; eigenvalue now negative
    basis = np.tile([1.0, 0.0, 0.0], (8, 1))
    basis[1::2] = [0.8, 0.0, -0.6]
    with pytest.raises(ValueError, match="positive semidefinite"):
        BoundaryCells(
            source.position_ang,
            source.normal,
            source.direction,
            source.energy_eV,
            source.weight_per_primary,
            jones,
            basis,
        )


def test_zero_source_and_invalid_sample_count():
    source = _cells()
    zeros = BoundaryCells(
        source.position_ang,
        source.normal,
        source.direction,
        source.energy_eV,
        np.zeros(8),
    )
    index, weights, _ = sample_cells(zeros, 10, seed=1)
    assert index.size == weights.size == 0
    with pytest.raises(ValueError, match="nonnegative integer"):
        sample_cells(source, -1, seed=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        sample_cells(source, 1.0, seed=1)


def test_polarization_requires_a_declared_transverse_basis():
    source = _cells()
    coherency = np.zeros((8, 2, 2), dtype=complex)
    coherency[:, 0, 0] = source.weight_per_primary
    with pytest.raises(ValueError, match="supplied together"):
        BoundaryCells(
            source.position_ang, source.normal, source.direction,
            source.energy_eV, source.weight_per_primary, coherency
        )
    with pytest.raises(ValueError, match="transverse"):
        BoundaryCells(
            source.position_ang, source.normal, source.direction,
            source.energy_eV, source.weight_per_primary, coherency,
            np.tile([0.0, 0.0, 1.0], (8, 1))
        )


def test_synthetic_scattered_in_requires_off_acceptance_source_support():
    # Toy linear response, NOT an EPDL scattering kernel: initially in-cone
    # photons survive directly with p=0.7; initially out-of-cone photons enter
    # via an abstract single collision with p=0.2. Never add a ballistic
    # estimate to the tracked direct contribution for these same photons.
    source = _cells()
    idx, history_weight, _ = sample_cells(source, 100_000, seed=411)
    in_cone = source.direction[:, 0] == 0.0
    rng = np.random.Generator(np.random.Philox(412))
    u = rng.random(idx.size)
    direct = history_weight[(in_cone[idx]) & (u < 0.7)].sum()
    redistributed = history_weight[(~in_cone[idx]) & (u < 0.2)].sum()
    expected_direct = 0.7 * source.weight_per_primary[in_cone].sum()
    expected_scatter = 0.2 * source.weight_per_primary[~in_cone].sum()
    assert direct == pytest.approx(expected_direct, rel=0.015)
    assert redistributed == pytest.approx(expected_scatter, rel=0.025)
    assert redistributed > 0  # a source cropped to the cone would lose it
