"""Regressions for the layered frozen-phonon oracle's geometry and observable."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.channeling import longitudinal_kinematics, planar_profile
from pyrite.materials.channeling.transverse import _hamiltonian


@pytest.fixture(scope="module")
def oracle():
    path = Path(__file__).parents[2] / "checks" / "abtem_channeling_oracle.py"
    spec = importlib.util.spec_from_file_location("channeling_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mos2_high_index_beam_has_right_handed_orthogonal_cell(oracle):
    profile = planar_profile("mos2", (0, 0, 1), 295.0)
    beam = np.array([1, 13, 0])
    rows, lattice = oracle.orthogonal_rows(profile, beam)
    # The old fixed +/-24 search missed this shortest y row.
    np.testing.assert_array_equal(rows, [[0, 0, -1], [-25, -11, 0], [1, 13, 0]])
    cart = rows @ lattice
    np.testing.assert_allclose(cart @ cart.T, np.diag(np.sum(cart**2, axis=1)), atol=1e-10)
    assert np.linalg.det(cart) > 0


def test_mos2_grouped_entry_survives_basis_change_and_band_cut(oracle):
    profile = planar_profile("mos2", (0, 0, 1), 295.0)
    gamma, _ = longitudinal_kinematics(16.92e6)
    grouped = []
    for order in (48, 64):
        energies, vectors = np.linalg.eigh(
            _hamiltonian(profile, gamma, 0.0, np.arange(-order, order + 1))
        )
        # Requesting four states must include the partner of state 3.
        groups = oracle.degenerate_groups(energies, 4, 1e-3)
        assert [g.tolist() for g in groups] == [[0], [1], [2], [3, 4]]
        entry = np.abs(vectors[order]) ** 2
        grouped.append([entry[g].sum() for g in groups])
    # Individual mirror-well eigenvectors can swap as H changes; summed
    # occupation of the nearly degenerate pair is the measurable quantity.
    np.testing.assert_allclose(grouped[0], grouped[1], atol=1e-7, rtol=0)
