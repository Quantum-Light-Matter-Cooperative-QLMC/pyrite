"""Analytic momentum instruments; no physical tail inferred from a scalar loss."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling.operators import ScatteringKernel
from pyrite.montecarlo.channeling.overflow import overflow_spectrum


def _kernel(cells, order, blocks, diagonal):
    return ScatteringKernel(
        label="analytic",
        period_ang=2 * np.pi,
        n_cells=cells,
        orders=np.arange(-order, order + 1),
        momentum_kernel=np.asarray(blocks, dtype=complex),
        diagonal_coefficients=np.asarray(diagonal, dtype=complex),
        hbar_v_eV_ang=1.0,
    )


def test_plane_wave_kick_has_no_wrap_and_bloch_carry():
    # Incoming k=1/2 and kick k=3/2 land at k=2 (residue 0, order 2).
    rho = np.zeros((2, 3, 3), complex)
    rho[1, 1, 1] = 0.3
    blocks = np.zeros((2, 5, 5), complex)
    blocks[1, 3, 3] = 2
    extended = _kernel(2, 2, blocks, [2])
    reference = replace(extended, orders=np.arange(-1, 2), momentum_kernel=blocks[:, 1:4, 1:4])
    result = overflow_spectrum(rho, extended, reference_kernel=reference)
    assert result.full_loss_rate == pytest.approx(0.6)
    assert result.reference_overflow_rate == pytest.approx(0.6)
    assert result.outside_rate == pytest.approx(0.6)
    occupied = result.rates_per_ang > 0
    np.testing.assert_array_equal(result.wavevectors_inv_ang[occupied], [2.0])
    assert result.unresolved_rate == pytest.approx(0.0)
    energy = 5e6
    expected = 2 * HBARC_EV_ANG / np.sqrt(energy * (energy + 2 * M_E_EV))
    np.testing.assert_allclose(
        result.sample_angles(energy, 4, np.random.default_rng(522)), expected
    )


def test_coherent_jump_matches_wavefunction_multiplication():
    # L(x)=1+i exp(ix); psi=(exp(-ix)+exp(ix))/sqrt(2).
    # Compare against polynomial multiplication, retaining interference.
    psi = np.array([1, 0, 1], complex) / np.sqrt(2)
    amplitude = np.array([0, 0, 1, 1j, 0], complex)
    blocks = np.outer(amplitude, amplitude.conj())[None]
    extended = _kernel(1, 2, blocks, [2, 1j])
    reference = replace(extended, orders=np.arange(-1, 2), momentum_kernel=blocks[:, 1:4, 1:4])
    result = overflow_spectrum(
        np.outer(psi, psi.conj())[None], extended, reference_kernel=reference
    )
    expected = np.abs(np.polynomial.polynomial.polymul(psi, amplitude)) ** 2
    np.testing.assert_allclose(result.rates_per_ang[0, : expected.size], expected, atol=1e-14)
    assert result.rates_per_ang[0, expected.size :].sum() == 0
    assert result.unresolved_rate == pytest.approx(0, abs=1e-14)


def test_incoming_coherence_changes_total_jump_rate():
    # L=1+i exp(ix), psi=(|0>+i|1>)/sqrt(2): rate 3, not incoherent rate 2.
    psi = np.array([0, 1, 1j]) / np.sqrt(2)
    amplitude = np.array([0, 1, 1j])
    blocks = np.outer(amplitude, amplitude.conj())[None]
    kernel = _kernel(1, 1, blocks, [2, 1j])
    result = overflow_spectrum(np.outer(psi, psi.conj())[None], kernel, reference_kernel=kernel)
    assert result.full_loss_rate == pytest.approx(3)
    assert result.rates_per_ang.sum() == pytest.approx(3)


def test_missing_transfer_can_land_inside_and_blocks_sampling():
    # An omitted kick +2 on incoming -1 lands back inside at +1. Scalar
    # reference overflow=1 cannot be interpreted as outside-band population.
    rho = np.zeros((1, 3, 3), complex)
    rho[0, 0, 0] = 1
    blocks = np.zeros((1, 5, 5), complex)
    blocks[0, 4, 4] = 1
    extended = _kernel(1, 2, blocks, [1])
    reference = replace(extended, orders=np.arange(-1, 2), momentum_kernel=blocks[:, 1:4, 1:4])
    result = overflow_spectrum(rho, extended, reference_kernel=reference)
    assert result.reference_overflow_rate == 1
    assert result.expanded_retained_rate == 1
    assert result.outside_rate == 0
    with pytest.raises(ValueError, match="no positive overflow"):
        result.sample_angles(5e6, 1, np.random.default_rng(522))


def test_unresolved_tail_is_not_renormalized_and_sampling_is_seeded():
    rho = np.zeros((1, 3, 3), complex)
    rho[0, 1, 1] = 1
    blocks = np.zeros((1, 5, 5), complex)
    blocks[0, 0, 0] = blocks[0, 4, 4] = 0.5
    extended = _kernel(1, 2, blocks, [2])  # only half of total rate resolved
    reference = replace(extended, orders=np.arange(-1, 2), momentum_kernel=blocks[:, 1:4, 1:4])
    result = overflow_spectrum(rho, extended, reference_kernel=reference)
    assert result.unresolved_rate == 1
    with pytest.raises(ValueError, match="not closed"):
        result.sample_angles(5e6, 100, np.random.default_rng(522))
    closed = overflow_spectrum(
        rho,
        replace(extended, diagonal_coefficients=np.array([1])),
        reference_kernel=replace(reference, diagonal_coefficients=np.array([1])),
    )
    first = closed.sample_angles(5e6, 1000, np.random.default_rng(522))
    np.testing.assert_array_equal(
        first, closed.sample_angles(5e6, 1000, np.random.default_rng(522))
    )
    assert set(np.sign(first)) == {-1, 1}
    assert abs(np.mean(np.sign(first))) < 0.12  # >3 sigma for a fair ± kick


def test_zero_scattering_and_invalid_density():
    kernel = _kernel(1, 1, np.zeros((1, 3, 3)), [0])
    rho = np.eye(3)[None] / 3
    result = overflow_spectrum(rho, kernel, reference_kernel=kernel)
    assert result.outside_rate == result.full_loss_rate == result.unresolved_rate == 0
    with pytest.raises(ValueError, match="no positive overflow"):
        result.sample_angles(5e6, 1, np.random.default_rng(522))
    with pytest.raises(ValueError, match="positive semidefinite"):
        overflow_spectrum(-rho, kernel, reference_kernel=kernel)
