"""Reflection-batched line coupling tables (#1167).

``_reflection_tables`` tabulates the chi/U rows in reflection batches so the
float64 transient stays bounded for thousands of reflections. Rows are
independent and the REAL cast is elementwise, so batching must be
bit-identical to one call, both for the tables and for the scored spectrum.

Validation: line-reflection-coupling-tables
"""

import numpy as np
import pytest

from pyrite._backend import BackendResourceError
from pyrite.montecarlo import mc_cxr_spectrum
from pyrite.montecarlo.spectrum.cxr import _kernels, _line_tables, _setup

CRYSTAL = "silicon"
HKLS = [(0, 0, 4), (1, 1, 1), (2, 2, 0), (1, 1, 3), (0, 0, 8), (3, 3, 1)]
THICKNESS = 2000.0


def _segments():
    v_hat = np.array([0.0, 0.0, 1.0])
    return {
        "r_mid": np.array([[0.0, 0.0, 0.5 * THICKNESS]]),
        "v_hat": v_hat[None, :],
        "L_ang": np.array([200.0]),
        "E_keV": np.array([100.0]),
        "t_ang": np.zeros(1),
        "t0_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "layer": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": THICKNESS,
        "crystal_width_ang": None,
        "crystal_height_ang": None,
        "n_backscattered": 0,
        "n_missed": 0,
    }


def _one_reflection_per_batch(monkeypatch, n_tab):
    monkeypatch.setattr(_line_tables, "TABLE_BATCH_BYTES", 6 * 8 * n_tab)


def test_batched_tables_equal_single_call(monkeypatch):
    grid = np.linspace(1_000.0, 12_000.0, 3001)
    want = _kernels._reflection_tables(CRYSTAL, HKLS, grid, 0.48, True)
    _one_reflection_per_batch(monkeypatch, grid.size)
    got = _kernels._reflection_tables(CRYSTAL, HKLS, grid, 0.48, True)
    for a, b in zip(got, want, strict=True):
        assert a.dtype == b.dtype and a.shape == (len(HKLS), grid.size)
        np.testing.assert_array_equal(a, b)


def test_batched_spectrum_equals_unbatched(monkeypatch):
    grid = np.linspace(1_000.0, 12_000.0, 2001)
    kwargs = {
        "crystal": CRYSTAL,
        "hkl_list": HKLS,
        "n_hat": np.array([0.8, 0.0, 0.6]),  # exits the back face
        "B_ang2": 0.48,
        "use_henke": True,
    }
    unbatched = mc_cxr_spectrum(_segments(), grid, **kwargs)
    monkeypatch.setattr(_line_tables, "TABLE_BATCH_BYTES", 1)  # one reflection per batch
    batched = mc_cxr_spectrum(_segments(), grid, **kwargs)
    assert unbatched.max() > 0.0
    np.testing.assert_array_equal(batched, unbatched)


def test_line_table_bytes_counts_outputs_and_one_batch():
    n_tab = 69_331
    batch = _line_tables.TABLE_BATCH_BYTES // (6 * 8 * n_tab)
    assert _line_tables.line_table_bytes(5408, n_tab, 4) == (
        4 * 5408 * n_tab * 4 + 6 * 8 * batch * n_tab
    )
    # A list smaller than one batch charges only its own rows.
    assert _line_tables.line_table_bytes(2, 100, 8) == 4 * 2 * 100 * 8 + 6 * 8 * 2 * 100


def test_admission_refuses_tables_over_device_budget(monkeypatch):
    from pyrite.montecarlo.runner import _RESOURCE_POLICY

    monkeypatch.setattr(_RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(_RESOURCE_POLICY, "device_budget_bytes", 1 << 30)
    _setup._admit_line_tables(50, 12_000)  # production-sized: admitted
    with pytest.raises(BackendResourceError, match="5408 reflections"):
        _setup._admit_line_tables(5408, 69_331)
    monkeypatch.setattr(_RESOURCE_POLICY, "gpu", False)
    _setup._admit_line_tables(5408, 69_331)  # host runs are not device-admitted


@pytest.mark.parametrize(("crystal", "henke"), [("silicon", True), ("mos2", True), ("gaas", False)])
def test_coupling_table_rows_match_single_reflection_calls(crystal, henke):
    """Whole-table fill (#1200): every row is bit-identical to its own one-row call.

    Includes a noncentrosymmetric crystal (Bijvoet pairs) and the in-place
    negation of ``emission_coupling_tables``.
    """
    from pyrite.materials.crystal import emission_coupling_tables

    hkls = [(1, 1, 1), (-1, -1, -1), (2, 2, 0), (0, 0, 4), (3, 1, 1), (4, 4, 4), (1, 0, 3)]
    energy = np.geomspace(800.0, 30000.0, 97)
    batch = emission_coupling_tables(crystal, hkls, energy, 0.45, henke)
    for i, hkl in enumerate(hkls):
        single = emission_coupling_tables(crystal, [hkl], energy, 0.45, henke)
        for whole, row in zip(batch, single, strict=True):
            np.testing.assert_array_equal(whole[i], row[0], strict=True)
            assert np.array_equal(np.signbit(whole[i]), np.signbit(row[0]))


def test_polarization_pairs_match_scalar_rows():
    """Vectorized polarization bases (#1200) equal the scalar ones bit for bit."""
    rng = np.random.default_rng(1200)
    k_hat = np.array([0.6, -0.1, 0.79])
    k_hat /= np.linalg.norm(k_hat)
    rows = np.vstack([rng.normal(size=(64, 3)) * 7.0, 3.0 * k_hat])  # last row degenerate
    from pyrite.montecarlo.spectrum.cxr._polarization import _polarization_pairs

    e_s, e_p = _polarization_pairs(k_hat, rows)
    for i, g in enumerate(rows):
        s, p = _kernels._polarization_pair(k_hat, g)
        np.testing.assert_array_equal(e_s[i], s, strict=True)
        np.testing.assert_array_equal(e_p[i], p, strict=True)
