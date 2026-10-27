"""Spectral vocabulary migration: grids, modes, aliases, and legacy identity."""

import os
from dataclasses import asdict, replace

import numpy as np
import pytest

from pyrite import CxrSpectra, Numerics, Result, SpectralDensity
from pyrite.campaign.beam import PointBunch, Spot
from pyrite.campaign.geometry import Footprint
from pyrite.config import Precision
from pyrite.detectors import EnergyBins


def _result():
    return Result(
        cxr=CxrSpectra(SpectralDensity([1, 2], [3, 4]), SpectralDensity([1, 2], [7, 8])),
        brem=SpectralDensity([1, 2, 3], [5, 6, 7]),
        characteristic=SpectralDensity([1, 2], [0.25, 0.5]),
        case={},
        provenance={},
    )


def test_components_keep_their_own_grids_and_separate_densities():
    result = _result()
    assert len(result.cxr.incoherent.photon_energy_eV) == 2
    assert len(result.brem.photon_energy_eV) == 3
    np.testing.assert_array_equal(result.combined_cxr().density, [3.25, 4.5])
    np.testing.assert_array_equal(result.combined_cxr(mode="coherent").density, [7.25, 8.5])
    np.testing.assert_array_equal(result.cxr.incoherent.density, [3, 4])
    assert result.brem.normalization == "source"


@pytest.mark.parametrize(
    "name",
    [
        "energy_eV",
        "spectrum",
        "background_energy_eV",
        "background",
        "coherent_spectrum",
        "characteristic_spectrum",
        "line_total",
    ],
)
def test_removed_flat_result_attributes_are_gone(name):
    assert not hasattr(_result(), name)


def test_result_constructor_is_grouped_keyword_only():
    with pytest.raises(TypeError):
        Result([1, 2], [3, 4], [1, 2, 3], [5, 6, 7], {}, {})  # type: ignore[misc]
    with pytest.raises(TypeError):
        Result(cxr=_result().cxr, energy_eV=[1, 2])  # type: ignore[call-arg]


def test_component_shape_and_normalization_errors_are_explicit():
    with pytest.raises(ValueError, match="grid"):
        SpectralDensity([1, 2], [3])
    with pytest.raises(ValueError, match="normalization"):
        SpectralDensity([1], [2], "counts")
    with pytest.raises(ValueError, match="share"):
        CxrSpectra(SpectralDensity([1], [2]), SpectralDensity([2], [3]))


def test_numerics_uses_canonical_trial_names_and_rejects_old_keywords():
    new = Numerics(cxr_trials=7, brem_trials=3, cxr_chunk=2)
    assert asdict(new)["cxr_trials"] == 7
    assert "n_electrons" not in asdict(new)
    assert replace(new, cxr_trials=9).cxr_trials == 9
    for old in ("n_electrons", "n_electrons_brem", "spec_chunk"):
        with pytest.raises(TypeError, match="unexpected keyword"):
            Numerics(**{old: 7})
        assert not hasattr(new, old)


def test_canonical_energy_bins_and_precision_keep_legacy_serialization():
    with pytest.raises(TypeError, match="unexpected keyword"):
        EnergyBins(line=np.array([1, 2]))
    assert not hasattr(EnergyBins(cxr=np.array([1, 2])), "line")
    canonical = Precision(0.1, 20, 40, 10, observables=("cxr", "brem"))
    with pytest.raises(ValueError, match="observables"):
        Precision(0.1, 20, 40, 10, observables=("line", "brem"))
    assert canonical.to_dict()["observables"] == ("line", "brem")
    assert Precision.from_dict(canonical.to_dict()) == canonical


def test_dataframe_preserves_modes_unequal_grids_and_normalization():
    import polars as pl

    frame = _result().to_dataframe()
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == [
        "component",
        "mode",
        "photon_energy_eV",
        "density",
        "normalization",
    ]
    assert frame.height == 9
    brem = frame.filter(pl.col("component") == "brem")
    np.testing.assert_array_equal(brem["photon_energy_eV"], [1, 2, 3])
    coherent = frame.filter((pl.col("component") == "cxr") & (pl.col("mode") == "coherent"))
    np.testing.assert_array_equal(coherent["density"], [7, 8])
    assert frame.filter(pl.col("component") == "characteristic")["mode"].is_null().all()
    assert set(frame["normalization"]) == {"source"}


def test_dataframe_omits_absent_families():
    import polars as pl

    result = Result(
        cxr=CxrSpectra(SpectralDensity([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])),
        brem=SpectralDensity([100.0, 400.0], [0.5, 0.25]),
        characteristic=None,
        case={"crystal": "hopg"},
        provenance={"identity_digest": "abc"},
    )
    frame = result.to_dataframe()
    counts = dict(frame.group_by("component").len().iter_rows())
    assert counts == {"cxr": 3, "brem": 2}
    assert set(frame.filter(pl.col("component") == "cxr")["mode"]) == {"incoherent"}
    assert frame["photon_energy_eV"].dtype == pl.Float64


def test_catalog_key_translation_preserves_literals_order_and_profile_names():
    from pyrite._cxr_vocabulary import translate_catalog_text

    original = """# n_electrons = 99
[profiles.n_electrons]
label = "n_electrons = 99"
cxr_trials = {values = [7]} # n_electrons = 99
brem_trials = {values = [3]}
[profiles."my.profile".cxr_grid_policy]
quadrature = "bin-mean"
"""
    legacy = translate_catalog_text(original)
    assert "[profiles.n_electrons]" in legacy
    assert 'label = "n_electrons = 99"' in legacy
    assert "n_electrons = {values = [7]} # n_electrons = 99" in legacy
    assert '[profiles."my.profile".line_grid_policy]' in legacy
    assert translate_catalog_text(legacy, canonical=True) == original


def test_catalog_aliases_reject_duplicate_spellings():
    from pyrite._cxr_vocabulary import translate_catalog_fields

    with pytest.raises(ValueError, match="both n_electrons and cxr_trials"):
        translate_catalog_fields({"n_electrons": [7], "cxr_trials": [7]})


def test_old_sweep_axis_path_is_removed():
    from pyrite import Beam, Detector, Scene, Sweep
    from pyrite.geometry import Slab

    scene = Scene(
        Beam(energy_keV=30, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", thickness_ang=1000, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    with pytest.raises(ValueError, match="energy_bins"):
        Sweep(scene, {"detector.energy_bins.line": ([1, 2],)}).expand()
    with pytest.raises(ValueError, match="energy_bins"):
        Sweep(scene, {"detector.energy_bins.cxr": ([1, 2],)})
    expanded = Sweep(scene, {"numerics.energy_bins.cxr": ([1, 2],)}).expand()[0][1]
    np.testing.assert_array_equal(expanded.numerics.energy_bins.cxr, [1, 2])


def test_removed_api_alias_helpers_are_unavailable():
    from pyrite import _api_deprecations

    assert not hasattr(_api_deprecations, "renamed_fields")
    assert not hasattr(_api_deprecations, "warn_renamed")


@pytest.mark.parametrize("suffix", ["CHUNK", "BUDGET_MB"])
@pytest.mark.parametrize("value", ["", "0", "128"])
def test_cxr_environment_uses_canonical_names(monkeypatch, suffix, value):
    from pyrite._env import env_value, set_canonical_env

    canonical = f"PYRITE_MC_CXR_{suffix}"
    legacy = f"PYRITE_MC_SPEC_{suffix}"
    monkeypatch.delenv(canonical, raising=False)
    monkeypatch.setenv(legacy, value)
    assert env_value(canonical) is None
    set_canonical_env(canonical, value)
    assert env_value(canonical) == value
    assert os.environ[canonical] == value


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("pyrite.results", "line_metrics"),
        ("pyrite.results", "LINE_RECORD_KEYS"),
        ("pyrite.montecarlo", "mc_spectrum"),
        ("pyrite.montecarlo", "_lines_for_case"),
        ("pyrite.plots", "_line_brem"),
        ("pyrite._cxr_grid_policy", "LineGridPolicy"),
        ("pyrite.energy_grid.apply", "set_line_artifact"),
    ],
)
def test_removed_cxr_function_exports_are_unavailable(module, name):
    from importlib import import_module

    assert not hasattr(import_module(module), name)
