"""PENELOPE-2024 conduction-band and bound-shell oscillator construction."""

import numpy as np
import pytest

from pyrite.materials import load_material_catalog
from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import shell_oscillators as so
from pyrite.montecarlo.transport.shell_transport import validate_shell_cutoff
from pyrite.xsgen.sbethe.catalog import catalog_material


def _shell(z, designator, label, orbital, occupation, energy):
    return config.AtomicShell(z, designator, label, orbital, occupation, energy, 0.1, 0.0, 0.0)


def _closure(result):
    return sum(o.strength * np.log(o.resonance_energy_eV) for o in result.oscillators)


def test_single_bound_shell_resonance_equals_mean_excitation():
    shells = {1: (_shell(1, 1, "K", "1s1/2", 1, 13.6),)}
    result = so.build_shell_oscillators({1: 1.0}, 19.2, 0.3, shells, default_threshold_eV=1.0)
    (shell,) = result.oscillators
    assert shell.resonance_energy_eV == pytest.approx(19.2, rel=1e-12)
    assert result.conduction_shells == ()


def test_measured_band_consumes_whole_outer_shells_and_closes_i():
    shells = {
        14: (
            _shell(14, 1, "K", "1s1/2", 2, 1844.0),
            _shell(14, 2, "L1", "2s1/2", 2, 154.0),
            _shell(14, 3, "L2", "2p1/2", 2, 104.0),
            _shell(14, 4, "L3", "2p3/2", 4, 104.0),
            _shell(14, 5, "M1", "3s1/2", 2, 13.46),
            _shell(14, 6, "M2", "3p1/2", 2, 8.151),
        )
    }
    band = so.ConductionBand(4.0, 16.7, "fixture", {14: 1.0})
    result = so.build_shell_oscillators({14: 1.0}, 173.0, 31.05, shells, band)
    assert result.conduction_shells == ((14, "M2"), (14, "M1"))
    assert result.oscillators[0].resonance_energy_eV == 16.7
    assert sum(o.strength for o in result.oscillators) == pytest.approx(14.0)
    assert _closure(result) == pytest.approx(14.0 * np.log(173.0), rel=1e-12)
    assert 1.0 < result.sternheimer_factor < 4.0


@pytest.mark.parametrize(
    "electrons, error",
    [(3.0, "whole-shell"), (8.0, "equal ionization")],
)
def test_measured_band_rejects_split_shell_boundaries(electrons, error):
    shells = {
        8: (
            _shell(8, 1, "K", "1s1/2", 2, 538.0),
            _shell(8, 2, "L1", "2s1/2", 2, 28.48),
            _shell(8, 3, "L2", "2p1/2", 2, 13.62),
            _shell(8, 4, "L3", "2p3/2", 2, 13.62),
        ),
        14: (_shell(14, 1, "K", "1s1/2", 10, 1844.0), _shell(14, 6, "M2", "3p1/2", 4, 8.151)),
    }
    band = so.ConductionBand(electrons, 20.0, "fixture")
    with pytest.raises(ValueError, match=error):
        so.build_shell_oscillators({14: 1.0, 8: 2.0}, 125.7, 30.2, shells, band)


def test_rejects_formula_mismatch_and_unreachable_mean_excitation():
    shells = {1: (_shell(1, 1, "K", "1s1/2", 1, 13.6),)}
    band = so.ConductionBand(1.0, 10.0, "fixture", {1: 2.0})
    with pytest.raises(ValueError, match="formula"):
        so.build_shell_oscillators({1: 1.0}, 19.2, 0.3, shells, band)
    with pytest.raises(ValueError, match="zero-binding"):
        so.build_shell_oscillators({1: 1.0}, 0.1, 0.3, shells, default_threshold_eV=1.0)


def test_packaged_conduction_bands_are_sourced():
    bands = so.load_conduction_bands()
    assert set(bands) == {
        "silicon",
        "sio2",
        "mos2",
        "hopg",
        "hbn",
        "wse2",
        "mose2",
        "ws2",
        "mote2",
        "nbs2",
        "nbse2",
        "2h_tas2",
        "2h_tase2",
        "zrse2",
    }
    assert all("doi:10." in band.source for band in bands.values())


def test_packaged_conduction_band_keys_resolve_to_bundled_crystals_or_media():
    catalog = load_material_catalog()
    for key in so.load_conduction_bands():
        assert key in catalog.crystals or key in catalog.media, key


def test_conduction_band_cache_keeps_caller_mutations_and_file_edits_isolated(tmp_path):
    path = tmp_path / "bands.toml"
    contents = """[fixture]
formula = { C = 1 }
electrons_per_formula = 2
resonance_eV = 20
source = "fixture"
doi = "10.fixture"
"""
    path.write_text(contents)
    bands = so.load_conduction_bands(path)
    bands["fixture"].formula.clear()
    bands.clear()
    assert so.load_conduction_bands(path)["fixture"].formula == {6: 1.0}
    path.write_text(contents.replace("resonance_eV = 20", "resonance_eV = 25"))
    assert so.load_conduction_bands(path)["fixture"].resonance_eV == 25.0
    path.write_text(contents.replace("resonance_eV = 20", "resonance_eV = -1"))
    with pytest.raises(ValueError, match="finite positive"):
        so.load_conduction_bands(path)


def test_fetched_packaged_conduction_bands_pass_default_shell_cutoff():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    validate_shell_cutoff(tuple(so.load_conduction_bands()), 50.0)


@pytest.mark.parametrize(
    "key, conduction_shells, sternheimer",
    [
        ("silicon", {(14, "M1"), (14, "M2")}, 2.2024),
        ("sio2", {(14, "M1"), (14, "M2"), (8, "L1"), (8, "L2"), (8, "L3")}, 3.4701),
        ("mos2", {(42, "N4"), (42, "O1"), (16, "M1"), (16, "M2"), (16, "M3")}, 1.7797),
        ("hopg", {(6, "L1"), (6, "L2")}, 2.6359),
        ("hbn", {(5, "L1"), (5, "L2"), (7, "L1"), (7, "L2"), (7, "L3")}, 2.5860),
        (
            "wse2",
            {(74, "O4"), (74, "P1"), (34, "N1"), (34, "N2"), (34, "N3")},
            2.0383,
        ),
        ("mose2", {(42, "N4"), (42, "O1"), (34, "N1"), (34, "N2"), (34, "N3")}, 1.9008),
        ("ws2", {(74, "O4"), (74, "P1"), (16, "M1"), (16, "M2"), (16, "M3")}, 2.0033),
        ("mote2", {(42, "N4"), (42, "O1"), (52, "O1"), (52, "O2"), (52, "O3")}, 1.5922),
        ("nbs2", {(41, "N4"), (41, "O1"), (16, "M1"), (16, "M2"), (16, "M3")}, 1.7792),
        ("nbse2", {(41, "N4"), (41, "O1"), (34, "N1"), (34, "N2"), (34, "N3")}, 1.9101),
        ("2h_tas2", {(73, "O4"), (73, "P1"), (16, "M1"), (16, "M2"), (16, "M3")}, 2.0668),
        ("2h_tase2", {(73, "O4"), (73, "P1"), (34, "N1"), (34, "N2"), (34, "N3")}, 2.0800),
        ("zrse2", {(40, "N4"), (40, "O1"), (34, "N1"), (34, "N2"), (34, "N3")}, 1.9056),
    ],
)
def test_fetched_catalog_materials_close_mean_excitation(key, conduction_shells, sternheimer):
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    material = catalog_material(key)
    band = so.load_conduction_bands()[key]
    omega = plasma_energy_eV(key)
    result = so.build_shell_oscillators(
        material.composition,
        material.mean_excitation_eV,
        omega,
        config.load_atomic_shells(),
        band,
    )
    z = result.electrons_per_formula
    assert set(result.conduction_shells) == conduction_shells
    assert sum(o.strength for o in result.oscillators) == pytest.approx(z)
    assert _closure(result) == pytest.approx(z * np.log(material.mean_excitation_eV), rel=1e-12)
    assert result.sternheimer_factor == pytest.approx(sternheimer, abs=1e-4)
    free_electron = np.sqrt(band.electrons_per_formula / z) * omega
    assert band.resonance_eV == pytest.approx(free_electron, rel=0.12)


@pytest.mark.parametrize("key", ["hfs2", "hfse2"])
def test_fetched_hafnium_dichalcogenide_valence_splits_at_hf_4f(key):
    # Bell & Liang's n = 16 plasmon excludes Hf 4f, but pdatconf places 4f
    # (20.0 eV) between the chalcogen p and s shells, so no measured band fits.
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    material = catalog_material(key)
    band = so.ConductionBand(16.0, 20.0, "fixture", dict(material.composition))
    with pytest.raises(ValueError, match="whole-shell"):
        so.build_shell_oscillators(
            material.composition,
            material.mean_excitation_eV,
            plasma_energy_eV(key),
            config.load_atomic_shells(),
            band,
        )


def test_fetched_silicon_default_matches_free_electron_plasmon():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    material = catalog_material("silicon")
    result = so.build_shell_oscillators(
        material.composition,
        material.mean_excitation_eV,
        plasma_energy_eV("silicon"),
        config.load_atomic_shells(),
    )
    band = result.oscillators[0]
    assert (band.label, band.strength) == ("cb", 4.0)
    assert band.resonance_energy_eV == pytest.approx(16.60, abs=0.01)
    assert result.sternheimer_factor == pytest.approx(2.208, abs=1e-3)


@pytest.mark.parametrize(
    "band, mean_excitation, error",
    [
        (so.ConductionBand(4.0, float("inf"), "fixture"), 173.0, "finite positive"),
        (so.ConductionBand(0.0, 16.7, "fixture"), 173.0, "finite positive"),
        (so.ConductionBand(4.0, 200.0, "fixture"), 173.0, "below the mean excitation"),
        (so.ConductionBand(4.0, 16.7, "fixture"), 30.0, "at or below its ionization"),
    ],
)
def test_rejects_unphysical_conduction_inputs(band, mean_excitation, error):
    shells = {
        14: (
            _shell(14, 1, "K", "1s1/2", 2, 1844.0),
            _shell(14, 2, "L1", "2s1/2", 2, 154.0),
            _shell(14, 3, "L2", "2p1/2", 2, 104.0),
            _shell(14, 4, "L3", "2p3/2", 4, 104.0),
            _shell(14, 5, "M1", "3s1/2", 2, 13.46),
            _shell(14, 6, "M2", "3p1/2", 2, 8.151),
        )
    }
    with pytest.raises(ValueError, match=error):
        so.build_shell_oscillators({14: 1.0}, mean_excitation, 31.05, shells, band)


def test_rejects_missing_element_shells():
    with pytest.raises(ValueError, match="no atomic shells"):
        so.build_shell_oscillators({2: 1.0}, 41.8, 1.0, {})
