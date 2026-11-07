"""Spectral vocabulary migration: grids, modes, aliases, and legacy identity."""

import os
from dataclasses import asdict, replace

import numpy as np
import pytest

from pyrite import CxrSpectra, Numerics, Result, SpectralDensity
from pyrite.campaign.beam import PointBunch, Spot
from pyrite.campaign.geometry import Footprint
from pyrite.config import AdaptiveSampling, FixedSampling
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


@pytest.mark.parametrize(
    ("cxr_norm", "brem_norm"),
    [("source", "observation-average"), ("observation-average", "source")],
)
def test_result_rejects_brem_normalization_mismatch(cxr_norm, brem_norm):
    with pytest.raises(ValueError, match="brem and CXR normalization must match"):
        Result(
            cxr=CxrSpectra(SpectralDensity([1, 2], [3, 4], cxr_norm)),
            brem=SpectralDensity([1, 2, 3], [5, 6, 7], brem_norm),
            case={},
            provenance={},
        )


@pytest.mark.parametrize("norm", ["source", "observation-average"])
def test_result_accepts_matching_brem_normalization_on_independent_grid(norm):
    provenance = {"identity_digest": "abc123", "observation_identity_digest": "def456"}
    result = Result(
        cxr=CxrSpectra(SpectralDensity([1, 2], [3, 4], norm)),
        brem=SpectralDensity([0.5, 1.5, 2.5, 3.5], [5, 6, 7, 8], norm),
        case={},
        provenance=provenance,
    )
    assert result.brem.normalization == result.cxr.incoherent.normalization == norm
    np.testing.assert_array_equal(result.brem.photon_energy_eV, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(result.brem.density, [5, 6, 7, 8])
    np.testing.assert_array_equal(result.cxr.incoherent.photon_energy_eV, [1, 2])
    rebuilt = replace(result)
    np.testing.assert_array_equal(rebuilt.brem.density, result.brem.density)
    assert rebuilt.provenance["identity_digest"] == "abc123"
    assert rebuilt.provenance["observation_identity_digest"] == "def456"


def test_numerics_uses_sampling_budget_and_rejects_old_keywords():
    new = Numerics(sampling=FixedSampling(cxr_histories=7, brem_histories=3), cxr_chunk=2)
    assert asdict(new)["sampling"] == {"cxr_histories": 7, "brem_histories": 3}
    assert "n_electrons" not in asdict(new)
    assert replace(new, sampling=FixedSampling(9, 3)).sampling.cxr_histories == 9
    for old in ("n_electrons", "n_electrons_brem", "spec_chunk"):
        with pytest.raises(TypeError, match="unexpected keyword"):
            Numerics(**{old: 7})
        assert not hasattr(new, old)


def test_canonical_energy_bins_and_precision_keep_legacy_serialization():
    with pytest.raises(TypeError, match="unexpected keyword"):
        EnergyBins(line=np.array([1, 2]))
    assert not hasattr(EnergyBins(cxr=np.array([1, 2])), "line")
    canonical = AdaptiveSampling(0.1, 20, 40, 10, observables=("cxr", "brem"))
    with pytest.raises(ValueError, match="observables"):
        AdaptiveSampling(0.1, 20, 40, 10, observables=("line", "brem"))
    assert canonical.to_dict()["observables"] == ("line", "brem")
    assert AdaptiveSampling.from_dict(canonical.to_dict()) == canonical


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
cxr_histories = {values = [7]} # n_electrons = 99
brem_histories = {values = [3]}
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

    with pytest.raises(ValueError, match="both E_grid_line and E_grid_cxr"):
        translate_catalog_fields({"E_grid_line": [7.0], "E_grid_cxr": [7.0]})


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("cxr_trials", "cxr_histories"),
        ("brem_trials", "brem_histories"),
        ("n_electrons", "cxr_histories"),
        ("n_electrons_brem", "brem_histories"),
    ],
)
def test_removed_budget_keys_error_naming_the_new_key(old, new):
    from pyrite._cxr_vocabulary import translate_catalog_fields, translate_catalog_text

    with pytest.raises(ValueError, match=f"{old}.*removed.*{new}"):
        translate_catalog_fields({"profiles": {"p": {old: {"values": [7]}}}})
    with pytest.raises(ValueError, match=f"{old}.*removed.*{new}"):
        translate_catalog_text(f"[profiles.p]\n{old} = {{values = [7]}}\n")
    # Rendering the editable view back is not a read, and a profile may use the name.
    text = f'[profiles.{old}]\nlabel = "{old} = 1" # {old} = 1\n'
    assert translate_catalog_text(text) == text


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


def test_bundled_catalog_has_no_removed_budget_keys():
    import tomllib

    from pyrite._catalog_layout import bundled_catalog, read_text
    from pyrite._cxr_vocabulary import REMOVED_CATALOG_KEYS, translate_catalog_text

    text = read_text(bundled_catalog())
    translate_catalog_text(text)  # raises on a removed key in any key position
    profiles = tomllib.loads(text)["profiles"]
    assert not {key for row in profiles.values() for key in row} & set(REMOVED_CATALOG_KEYS)
