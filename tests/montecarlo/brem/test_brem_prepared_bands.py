"""Prepared cone scoring preserves primary identities, escape and band units."""

import numpy as np
import pytest

from pyrite._backend import BACKEND, REAL
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum
from pyrite.montecarlo.spectrum.brem_prepared import _PreparedBremsLibBands
from tests.helpers.bremslib import faint_tip_bremslib_arrays, synthetic_bremslib_arrays
from tests.montecarlo.brem.test_brem_photon_blocking import _LAYERS, _segments

# Same FP32 cells and positive quadrature as the scalar CUDA band scorer;
# changed reduction order has a 128-eps score budget, twice that for moments.
_RTOL = 128 * np.finfo(REAL).eps


@pytest.fixture
def inputs():
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    segments = _segments(n=37, n_electrons=7, T_keV=(12, 190))
    segments["elec_id"] = np.resize([5, 0, 2, 8, 3], 37)
    segments["E_repr_keV"] = segments["E_keV"] * 0.99
    segments["radiative"] = {
        "model": "bremslib-soft-hard",
        "cutoff_eV": 1000.0,
        "bremslib_tables": (("C", 6, table.key, table.digest),),
    }
    grids = [np.geomspace(1000, 30000, 17), np.geomspace(40000, 300000, 23)]
    weights = [np.gradient(g) for g in grids]
    weights[0][::3] = 0
    return segments, table, grids, weights


@pytest.mark.parametrize("geometry", ["slab", "finite", "layers"])
@pytest.mark.parametrize("dielectric", [False, True])
def test_prepared_cone_bands_match_scalar_primary_scores(inputs, geometry, dielectric):
    segments, table, grids, weights = inputs
    layers = _LAYERS if geometry == "layers" else None
    if dielectric:
        from pyrite.materials import MediumSpec
        from pyrite.materials.attenuation import plasma_energy_eV

        plasma = plasma_energy_eV(MediumSpec("prepared-test", (("C", 0.1),)))
        segments["radiative"].update(
            dielectric_model="ter-mikaelian-v1", plasma_energy_eV_by_layer=(plasma,)
        )
    if geometry == "finite":
        segments.update(crystal_width_ang=100.0, crystal_height_ang=100.0)
    directions = np.array([[0.6, 0, 0.8], [-0.6, 0, -0.8]])
    angular_weights = np.array([0.0003, 0.0007])
    recipe = _PreparedBremsLibBands(
        grids,
        weights,
        composition=[("C", 0.1)],
        bremslib_tables={"C": table},
        cutoff_eV=1000,
        layers=layers,
        row_chunk=7,
    )
    reference = np.zeros((7, 2))
    for direction, omega in zip(directions, angular_weights, strict=True):
        for band, (grid, q) in enumerate(zip(grids, weights, strict=True)):
            reference[:, band] += omega * mc_coupled_brem_spectrum(
                segments,
                grid,
                cutoff_eV=1000,
                composition=[("C", 0.1)],
                bremslib_tables={"C": table},
                n_hat=direction,
                layers=layers,
                electron_band_weights=q,
                chunk=7,
            )
    candidate = recipe.bind(segments).integrate(directions, angular_weights)
    assert candidate.dtype == np.float64
    np.testing.assert_array_equal(candidate[[1, 4, 6]], 0)
    np.testing.assert_allclose(candidate, reference, rtol=_RTOL)
    np.testing.assert_allclose((candidate**2).sum(0), (reference**2).sum(0), rtol=2 * _RTOL)
    # Recipe reused across independent blocks, with zero-score primaries retained.
    empty = {k: v[:0] if isinstance(v, np.ndarray) else v for k, v in segments.items()}
    np.testing.assert_array_equal(
        recipe.bind(empty).integrate(directions, angular_weights), np.zeros((7, 2))
    )


def test_prepared_rejects_unsupported_partition_and_table_range(inputs):
    segments, table, grids, weights = inputs
    recipe = _PreparedBremsLibBands(
        grids, weights, composition=[("C", 0.1)], bremslib_tables={"C": table}, cutoff_eV=2000
    )
    with pytest.raises(ValueError, match="cutoff"):
        recipe.bind(segments)
    segments.pop("radiative")
    segments["E_repr_keV"][0] = table.maximum_incident_energy_keV * 2
    with pytest.raises(ValueError, match="coverage"):
        recipe.bind(segments)


@pytest.mark.parametrize("bad_weights", [[np.ones(1)], [np.array([-1.0, 2.0])]])
def test_prepared_rejects_invalid_quadrature(bad_weights, inputs):
    _, table, _, _ = inputs
    with pytest.raises(ValueError, match="weights"):
        _PreparedBremsLibBands(
            [np.array([1000, 2000])],
            bad_weights,
            composition=[("C", 0.1)],
            bremslib_tables={"C": table},
            cutoff_eV=1000,
        )


@pytest.mark.skipif(BACKEND.name != "cpu", reason="CPU FP32 emulation of the ROCm path")
def test_fp32_portable_bands_keep_faint_tip_cells_normal(monkeypatch):
    """#1019: ROCm FP32 ``_portable_rows`` scores backward tip cells in millibarn."""
    from pyrite.montecarlo.spectrum import brem_bremslib, brem_prepared

    table = prepare_bremslib_table(
        faint_tip_bremslib_arrays(), atomic_number=6, key="k", digest="d"
    )
    segments = {
        "r_mid": np.array([[0.0, 0.0, 20.0], [0.0, 0.0, 50.0], [0.0, 0.0, 80.0]]),
        "v_hat": np.array([[0.0, 0.0, -1.0], [0.6, 0.0, -0.8], [0.0, 0.8, -0.6]]),
        "L_ang": np.array([40.0, 110.0, 260.0]),
        "E_keV": np.array([29999.0, 29990.0, 29980.0]),
        "elec_id": np.arange(3),
        "Ne": 3,
        "thickness_ang": 100.0,
    }
    # Tip nodes of the HOPG 29,999 keV automatic grid band, as in #979.
    grids = [np.linspace(29.15e6, 29.55e6, 6), np.linspace(29.6e6, 29.96e6, 6)]
    weights = [np.gradient(g) for g in grids]
    monkeypatch.setattr(brem_prepared, "_mu_total_inv_ang", lambda _c, e: np.zeros_like(e))

    def score():
        recipe = _PreparedBremsLibBands(
            grids, weights, composition=[("C", 0.1)], bremslib_tables={"C": table}, cutoff_eV=1000
        )
        return recipe.bind(segments).integrate([[0.0, 0.0, 1.0]], [1.0])

    reference = score()
    assert np.all(reference > 0.0)
    # Accelerator FP32 precision without accelerator hardware (CPU is binary64).
    monkeypatch.setattr(brem_prepared, "REAL", np.float32)
    monkeypatch.setattr(brem_bremslib, "REAL", np.float32)
    np.testing.assert_allclose(score(), reference, rtol=1e-4)
    # Negative control: the same FP32 cells formed in cm² lose the tip.
    monkeypatch.setattr(brem_prepared, "_cell_unit_cm2", lambda _staged: 1.0)
    assert np.max(abs(score() / reference - 1.0)) > 1e-3
