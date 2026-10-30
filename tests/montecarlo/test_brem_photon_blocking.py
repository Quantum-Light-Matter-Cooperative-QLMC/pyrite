"""Photon-grid blocking of the continuum scorers (#819).

Every photon node is scored independently of the others, so contiguous grid
blocks must reproduce one pass. Each node's segment sum is a BLAS ``gemv``
column, whose reduction order depends on the matrix width, so agreement is to
a few ulps of a chunk-length sum, not bit for bit.

Validation: brem-spectrum
"""

import tracemalloc
import warnings

import numpy as np
import pytest

from pyrite._backend import xp
from pyrite.montecarlo import runner
from pyrite.montecarlo.groove import blazed_groove_spec
from pyrite.montecarlo.runner import chunking
from pyrite.montecarlo.spectrum import brem, brem_blocks
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum
from tests.helpers import scaled_rtol
from tests.helpers.bremslib import synthetic_bremslib_arrays

_HOST = getattr(xp, "__name__", "") == "numpy"
_GRID = np.linspace(800.0, 150_000.0, 97)
_BLOCKS = (1, 7, 96)
_LAYERS = [(0.0, 400.0, [("C", 0.1)]), (400.0, 2000.0, [("Si", 0.05)])]
_TILT = np.deg2rad(45.0)
_GROOVE = blazed_groove_spec(20_000.0, np.pi / 2, _TILT, np.pi)
_GROOVE_N_HAT = np.array([np.cos(_TILT), 0.0, -np.sin(_TILT)])


def _assert_same(blocked, whole):
    # A gemv over <= 41 rows per chunk: a few ulps, never a model difference.
    np.testing.assert_allclose(blocked, whole, rtol=scaled_rtol(1e-14, eps_multiple=64.0))


def _segments(n=41, *, thickness=2000.0, T_keV=(12.0, 190.0), seed=819, n_electrons=3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z = rng.uniform(0.05, 0.95, n) * thickness
    return {
        "r_mid": np.column_stack((rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), z)),
        "v_hat": v,
        "L_ang": rng.uniform(5.0, 40.0, n),
        "E_keV": rng.uniform(*T_keV, n),
        "elec_id": rng.integers(0, n_electrons, n),
        "layer": (z >= 400.0).astype(int),
        "Ne": n_electrons,
        "thickness_ang": thickness,
    }


@pytest.fixture
def carbon_table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6, key="k", digest="d")


@pytest.mark.parametrize("model", ["eedl", "bethe-heitler"])
@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("block", _BLOCKS)
def test_blocked_isotropic_continuum_reproduces_one_pass(model, layered, block):
    kwargs = dict(
        composition=[("Si", 0.05)],
        cross_section_model=model,
        chunk=9,
        layers=_LAYERS if layered else None,
        E_cut_keV=15.0,
    )
    segments = _segments()
    _assert_same(
        brem.mc_brem_spectrum(segments, _GRID, photon_block=block, **kwargs),
        brem.mc_brem_spectrum(segments, _GRID, **kwargs),
    )


@pytest.mark.parametrize("block", _BLOCKS)
def test_blocked_grooved_continuum_reproduces_one_pass(block):
    segments = _segments(thickness=20_000.0)
    segments.pop("layer")
    kwargs = dict(
        composition=[("C", 0.176)],
        cross_section_model="eedl",
        n_hat=_GROOVE_N_HAT,
        groove=_GROOVE,
        chunk=11,
    )
    _assert_same(
        brem.mc_brem_spectrum(segments, _GRID, photon_block=block, **kwargs),
        brem.mc_brem_spectrum(segments, _GRID, **kwargs),
    )


@pytest.mark.parametrize("plasma_eV", [0.0, 30.0])
@pytest.mark.parametrize("block", _BLOCKS)
def test_blocked_bremslib_continuum_reproduces_one_pass(carbon_table, plasma_eV, block):
    # A 400 keV row lies outside the synthetic table and takes the EEDL fallback,
    # which dielectric scoring refuses; give that case covered rows only.
    segments = _segments(T_keV=(12.0, 190.0))
    if not plasma_eV:
        segments["E_keV"][0] = 400.0
    kwargs = dict(
        composition=[("C", 0.1)],
        cross_section_model="bremslib",
        bremslib_tables={"C": carbon_table},
        dielectric_plasma_eV=plasma_eV,
        chunk=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        whole = brem.mc_brem_spectrum(segments, _GRID, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blocked = brem.mc_brem_spectrum(segments, _GRID, photon_block=block, **kwargs)
    _assert_same(blocked, whole)
    # Segment-only fallback warnings come once, not once per block.
    assert len([w for w in caught if "outside the BremsLib range" in str(w.message)]) == (
        0 if plasma_eV else 1
    )


@pytest.mark.parametrize("block", _BLOCKS)
def test_blocked_auto_model_resolves_once_and_reproduces_one_pass(carbon_table, block):
    kwargs = dict(composition=[("C", 0.1)], bremslib_tables={"C": carbon_table}, chunk=7)
    segments = _segments()
    _assert_same(
        brem.mc_brem_spectrum(segments, _GRID, photon_block=block, **kwargs),
        brem.mc_brem_spectrum(segments, _GRID, **kwargs),
    )


@pytest.mark.parametrize("block", _BLOCKS)
def test_blocked_coupled_continuum_reproduces_one_pass(carbon_table, block):
    segments = _segments()
    segments["radiative"] = {
        "model": "bremslib-soft-hard",
        "cutoff_eV": 20_000.0,
        "bremslib_tables": (("C", 6, "k", "d"),),
    }
    kwargs = dict(
        cutoff_eV=20_000.0,
        bremslib_tables={"C": carbon_table},
        composition=[("C", 0.1)],
        chunk=6,
    )
    _assert_same(
        mc_coupled_brem_spectrum(segments, _GRID, photon_block=block, **kwargs),
        mc_coupled_brem_spectrum(segments, _GRID, **kwargs),
    )


@pytest.mark.parametrize("model", ["eedl", "bethe-heitler"])
@pytest.mark.parametrize("block", _BLOCKS)
def test_blocked_electron_band_weights_add_block_shares(model, block):
    weights = np.gradient(_GRID) * (_GRID < 60_000.0)
    kwargs = dict(
        composition=[("Si", 0.05)],
        cross_section_model=model,
        chunk=8,
        electron_band_weights=weights,
    )
    segments = _segments()
    whole = brem.mc_brem_spectrum(segments, _GRID, **kwargs)
    blocked = brem.mc_brem_spectrum(segments, _GRID, photon_block=block, **kwargs)
    assert blocked.shape == (segments["Ne"],)
    # Block partial sums reorder the band sum: rounding-level, not bitwise.
    np.testing.assert_allclose(blocked, whole, rtol=scaled_rtol(1e-13, eps_multiple=256.0))


def test_photon_block_must_be_positive_and_band_weights_must_match():
    with pytest.raises(ValueError, match="photon_block must be positive"):
        brem.mc_brem_spectrum(
            _segments(), _GRID, composition=[("C", 0.1)], cross_section_model="eedl", photon_block=0
        )
    with pytest.raises(ValueError, match="electron_band_weights must match"):
        brem.mc_brem_spectrum(
            _segments(),
            _GRID,
            composition=[("C", 0.1)],
            cross_section_model="eedl",
            electron_band_weights=np.ones(3),
            photon_block=5,
        )


def test_photon_block_covers_a_grid_that_fits_and_shrinks_with_the_budget(monkeypatch):
    composition = [("Si", 0.05)]
    assert chunking._brem_photon_block(_GRID.size, composition, "eedl") == _GRID.size
    panels = brem_blocks.eedl_staged_panel_count("Si")
    assert panels > 0
    monkeypatch.setattr(chunking._RESOURCE_POLICY, "spec_budget_mb", 1)
    per_node = panels * (8 + (chunking._real_itemsize() if chunking._RESOURCE_POLICY.gpu else 0))
    per_node += 5 * 8
    assert chunking._brem_photon_block(10**6, composition, "eedl") == 1_000_000 // per_node
    # Bethe--Heitler stages no panel rows, only per-node vectors.
    assert chunking._brem_photon_block(10**6, composition, "bethe-heitler") == 1_000_000 // 40
    assert chunking._brem_photon_block(10**6, [("Xx", 1.0)], "eedl") == 1_000_000 // 40


def test_runner_passes_the_photon_block_to_every_emitting_layer(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_brem_photon_block", lambda n, comp, model, **kw: (n, model, kw))
    monkeypatch.setattr(
        runner, "mc_brem_spectrum", lambda segs, E, **kw: seen.append(kw) or np.zeros(E.shape)
    )
    monkeypatch.setattr(runner, "_segments_in_layer", lambda segs, index: {"L_ang": np.ones(1)})
    from pyrite.montecarlo.runner.emission import _uncoupled_brem_from_segments

    _uncoupled_brem_from_segments(
        {"n_layers": 2, "Ne": 1},
        _GRID,
        {"composition": [("C", 0.1)]},
        None,
        _LAYERS,
        groove=None,
        Ne=1,
        brem_chunk=4,
        model_kwargs=dict(cross_section_model="eedl"),
    )
    assert [kw["photon_block"] for kw in seen] == [(_GRID.size, "eedl", {"n_layers": 2})] * 2


@pytest.mark.slow
def test_mev_wide_grid_staging_stays_inside_the_budget(monkeypatch):
    """A 3 MeV, 1 keV--E0, 10 eV EEDL grid scored under a 256 MB staging budget.

    One pass stages 300k nodes x 327 Si panels (0.8 GB); the estimator's
    blocks keep the traced host peak under twice the budget.
    """
    if not _HOST:
        pytest.skip("host tracemalloc bound is a CPU check")
    budget_mb = 256
    monkeypatch.setattr(chunking._RESOURCE_POLICY, "spec_budget_mb", budget_mb)
    grid = np.arange(1_000.0, 3.0e6 + 1.0, 10.0)
    composition = [("Si", 0.05)]
    block = chunking._brem_photon_block(grid.size, composition, "eedl")
    assert 1 < block < grid.size
    segments = _segments(8, T_keV=(500.0, 2990.0))
    tracemalloc.start()
    try:
        spectrum = brem.mc_brem_spectrum(
            segments,
            grid,
            composition=composition,
            cross_section_model="eedl",
            chunk=4,
            photon_block=block,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert spectrum.shape == grid.shape and np.all(np.isfinite(spectrum))
    assert peak < 2 * budget_mb * 1_000_000
