"""Limits of the first-order Dirac correction used by the W <001> spin bound (#585)."""

import numpy as np
import pytest

from checks import channeling_submev_w001_spin as spin
from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV


def _two_waves(total_eV):
    kvec = np.array([[0.0, 0.0], [1.0, 0.0]])
    u = np.array([[0.0, 2.0], [2.0, 0.0]])
    return spin.dirac_terms(u, kvec, total_eV)


def test_forward_matrix_element_is_unchanged():
    # p' = p: u^dag u = 1, so neither correction touches the diagonal.
    kvec = np.array([[0.3, -0.2]])
    u = np.array([[5.0]])
    darwin, terms = spin.dirac_terms(u, kvec, 2.0 * M_E_EV)
    assert darwin[0, 0] == pytest.approx(0.0, abs=1e-12)
    for s in terms:
        assert s[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_longitudinal_spin_orbit_matches_bmt():
    total = 1.5 * M_E_EV
    _, (sx, sy, sz) = _two_waves(total)
    gamma = total / M_E_EV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    # Row p' = (1, 0) Å^-1, column p = 0: q = p' - p along x, so grad V x z is
    # along -y and only sigma_y couples, with weight hbar c beta / (2 (E + m c^2))
    # times i q V_q (here q = 1 Å^-1, V_q = 2 eV).
    expected = HBARC_EV_ANG * beta / (2.0 * (total + M_E_EV)) * 2.0 * 1.0
    assert abs(sy[1, 0]) == pytest.approx(expected, rel=1e-12)
    assert sx[1, 0] == pytest.approx(0.0, abs=1e-15)
    assert sz[1, 0] == pytest.approx(0.0, abs=1e-15)


def test_spin_hamiltonian_is_hermitian_and_kramers_degenerate():
    kvec = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    rng = np.random.default_rng(3)
    n = len(kvec)
    u = rng.normal(size=(n, n))
    u = u + u.T
    darwin, terms = spin.dirac_terms(u, kvec, 1.2 * M_E_EV)
    h = np.kron(np.diag(np.sum(kvec**2, 1)) + u + darwin, np.eye(2))
    for s, pauli in zip(terms, spin.SIGMA, strict=True):
        h = h + np.kron(s, pauli)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    e = np.linalg.eigvalsh(h)
    # Real potential and time reversal: every level is a Kramers doublet.
    np.testing.assert_allclose(e[0::2], e[1::2], atol=1e-9)
