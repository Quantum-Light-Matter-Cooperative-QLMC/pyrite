"""Vacuum line-centre API anchors and plotting-array contracts."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.materials import load_material_catalog
from pyrite.materials.crystal import HC_EV_ANG
from pyrite.materials.emission import photon_energies


@pytest.fixture(scope="module")
def catalog():
    return load_material_catalog()


def test_root_export_and_hopg_vacuum_anchor(catalog):
    assert pr.photon_energies is photon_energies
    energies = pr.photon_energies("hopg", [30_000, 100_000], theta_obs_deg=60, catalog=catalog)
    # Independent scalar relation E=hc/d * beta/(1-beta*cos(theta)); d002=c/2.
    beta = np.sqrt(1 - (510_998.95 / (510_998.95 + np.array([30_000, 100_000]))) ** 2)
    expected = HC_EV_ANG / (catalog.crystals["hopg"].lattice["c"] / 2) * beta / (1 - beta / 2)
    assert energies.shape == (1, 2)
    np.testing.assert_allclose(energies[0], expected, rtol=1e-8)


@pytest.mark.parametrize(
    ("material", "hkl"),
    [
        ("hopg", (0, 0, 2)),
        ("mos2", (0, 0, 2)),
        ("black_phosphorus", (0, 2, 0)),
        ("ges", (2, 0, 0)),
        ("gep", (2, 0, -2)),
        ("4h_sic", (0, 0, 4)),
        ("6h_sic", (0, 0, 6)),
        ("tis2", (0, 0, 3)),
    ],
)
def test_primary_reflection_matches_explicit(material, hkl, catalog):
    default = photon_energies(material, 30_000, catalog=catalog)
    explicit = photon_energies(material, 30_000, hkl=hkl, catalog=catalog)
    np.testing.assert_allclose(default, explicit)
    assert np.isfinite(default).all()


def test_unpinned_cut_uses_first_nonextinct_harmonic(catalog):
    # Diamond's catalogue (400) cut reduces to (100), but 100,200,300 cancel.
    expected = photon_energies("diamond", 100_000, hkl=(4, 0, 0), catalog=catalog)
    np.testing.assert_allclose(photon_energies("diamond", 100_000, catalog=catalog), expected)
    # Removing HOPG's authoritative pins still finds 002 from the basis.
    crystals = dict(catalog.crystals)
    crystals["hopg"] = replace(crystals["hopg"], hkl_families=())
    custom = replace(catalog, crystals=crystals)
    np.testing.assert_allclose(
        photon_energies("hopg", 30_000, catalog=custom),
        photon_energies("hopg", 30_000, hkl=(0, 0, 2), catalog=custom),
    )


def test_full_catalogue_comparison_defaults_resolve(catalog):
    values = photon_energies(catalog.material_keys, 100_000, catalog=catalog)
    assert values.shape == (len(catalog.material_keys),)
    assert np.isfinite(values).all()
    assert (values > 0).all()


def test_grid_axis_order_and_scalar_agreement(catalog):
    axes = ([30_000, 50_000], [0, 25, 50], [0, 90], [30, 60], [0, 45])
    keys = ["mos2", "hopg"]
    grid = photon_energies(
        keys,
        axes[0],
        tilt_deg=axes[1],
        tilt_azim_deg=axes[2],
        theta_obs_deg=axes[3],
        phi_obs_deg=axes[4],
        mode="grid",
        catalog=catalog,
    )
    assert grid.shape == (2, 2, 3, 2, 2, 2)
    for index in [(0, 0, 0, 0, 0, 0), (1, 1, 2, 1, 1, 1), (0, 1, 1, 0, 1, 0)]:
        m, e, t, a, o, p = index
        scalar = photon_energies(
            keys[m],
            axes[0][e],
            tilt_deg=axes[1][t],
            tilt_azim_deg=axes[2][a],
            theta_obs_deg=axes[3][o],
            phi_obs_deg=axes[4][p],
            catalog=catalog,
        )
        assert grid[index] == pytest.approx(scalar[0])


def test_broadcast_paired_and_shaped_heatmap(catalog):
    paired = photon_energies(
        ["hopg", "mos2"],
        [30_000, 100_000],
        tilt_deg=[0, 45],
        catalog=catalog,
    )
    assert paired.shape == (2, 2)
    for column, (energy, tilt) in enumerate([(30_000, 0), (100_000, 45)]):
        np.testing.assert_allclose(
            paired[:, column],
            photon_energies(["hopg", "mos2"], energy, tilt_deg=tilt, catalog=catalog),
        )
    heatmap = photon_energies(
        "hopg",
        30_000,
        tilt_deg=np.array([0, 30, 60])[:, None],
        theta_obs_deg=np.array([30, 60])[None, :],
        catalog=catalog,
    )
    grid = photon_energies(
        "hopg",
        30_000,
        tilt_deg=[0, 30, 60],
        theta_obs_deg=[30, 60],
        mode="grid",
        catalog=catalog,
    )
    assert heatmap.shape == (1, 3, 2)
    np.testing.assert_allclose(heatmap, grid[:, 0, :, 0, :, 0])


def test_scalar_singleton_empty_axes_and_material_order(catalog):
    assert photon_energies("hopg", 30_000, catalog=catalog).shape == (1,)
    assert photon_energies("hopg", 30_000, mode="grid", catalog=catalog).shape == (1,) * 6
    assert photon_energies([], [30_000, 50_000], catalog=catalog).shape == (0, 2)
    assert photon_energies("hopg", [], catalog=catalog).shape == (1, 0)
    assert photon_energies("hopg", [], mode="grid", catalog=catalog).shape == (1, 0, 1, 1, 1, 1)
    assert photon_energies("hopg", 30_000, phi_obs_deg=[], catalog=catalog).shape == (1, 0)
    ordered = photon_energies(np.array(["mos2", "hopg", "mos2"]), 30_000, catalog=catalog)
    np.testing.assert_array_equal(ordered[[0]], photon_energies("mos2", 30_000, catalog=catalog))
    assert ordered[0] == ordered[2]
    assert ordered[0] != ordered[1]


def test_basal_azimuth_invariance_and_tilt_cosine(catalog):
    result = photon_energies(
        "hopg",
        30_000,
        tilt_deg=30,
        tilt_azim_deg=[0, 60, 180, 360],
        phi_obs_deg=[0, 90, 180, 270],
        catalog=catalog,
    )
    untilted = photon_energies("hopg", 30_000, catalog=catalog)[0]
    np.testing.assert_allclose(result, untilted * np.sqrt(3) / 2)


def test_oblique_reflection_full_orientation_and_friedel_pair(catalog):
    # Custom cubic cut (001), g111=(2pi/a)*(1,1,1); minimal sample tilt about y.
    crystal = catalog.crystals["diamond"]
    custom_crystal = replace(crystal, surface_hkl=(0, 0, 1))
    custom = replace(catalog, crystals={**catalog.crystals, "diamond": custom_crystal})
    azimuths = np.array([0, 90, 180])
    result = photon_energies(
        "diamond",
        100_000,
        tilt_deg=30,
        tilt_azim_deg=azimuths,
        hkl=(1, 1, 1),
        catalog=custom,
    )
    beta = np.sqrt(1 - (510_998.95 / 610_998.95) ** 2)
    projection = np.abs(
        np.sqrt(3) / 2 - 0.5 * (np.cos(np.deg2rad(azimuths)) + np.sin(np.deg2rad(azimuths)))
    )
    expected = HC_EV_ANG / crystal.lattice["a"] * beta * projection
    np.testing.assert_allclose(result[0], expected, rtol=1e-8)
    assert result[0, 2] > result[0, 0]
    np.testing.assert_allclose(
        result,
        photon_energies(
            "diamond",
            100_000,
            tilt_deg=30,
            tilt_azim_deg=azimuths,
            hkl=(-1, -1, -1),
            catalog=custom,
        ),
    )


def test_nonorthogonal_default_is_reciprocal_normal(catalog):
    gep = catalog.crystals["gep"]
    assert gep.lattice["beta"] != 90
    untilted = photon_energies("gep", 30_000, catalog=catalog)[0]
    tilted = photon_energies("gep", 30_000, tilt_deg=60, tilt_azim_deg=[0, 90], catalog=catalog)
    np.testing.assert_allclose(tilted, untilted / 2)


def test_matches_existing_validation_anchor(catalog):
    from pyrite.validation._anchor_conditions import ZhaiAnchor, line_energy_eV

    anchor = ZhaiAnchor()
    energies = photon_energies(
        "hopg",
        [30_000, 100_000],
        theta_obs_deg=np.rad2deg(anchor.theta_obs_rad),
        catalog=catalog,
    )
    np.testing.assert_allclose(
        energies[0],
        [line_energy_eV(anchor, energy) for energy in [30, 100]],
        rtol=1e-12,
    )


def test_no_line_is_shape_preserving_nan(catalog):
    values = photon_energies("hopg", 30_000, tilt_deg=[0, 90, 180, 270], catalog=catalog)
    assert values.shape == (1, 4)
    assert np.isnan(values[0, [1, 3]]).all()
    assert values[0, 0] == pytest.approx(values[0, 2])
    assert np.isnan(photon_energies("hopg", 1, catalog=catalog)).all()
    assert np.isnan(photon_energies("hopg", 30_000, hkl=(0, 0, 1000), catalog=catalog)).all()


@pytest.mark.parametrize(
    "name", ["beam_energy_eV", "tilt_deg", "tilt_azim_deg", "theta_obs_deg", "phi_obs_deg"]
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(name, bad, catalog):
    arguments = {"beam_energy_eV": 30_000, name: bad}
    with pytest.raises(ValueError, match=name):
        photon_energies("hopg", **arguments, catalog=catalog)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_energy_rejected(bad, catalog):
    with pytest.raises(ValueError, match="strictly positive"):
        photon_energies("hopg", bad, catalog=catalog)


@pytest.mark.parametrize("bad", [[30_000 + 1j], "not-a-number", [[30_000], [30_000, 40_000]]])
def test_nonreal_or_ragged_inputs_rejected(bad, catalog):
    with pytest.raises(ValueError, match="real numerical values"):
        photon_energies("hopg", bad, catalog=catalog)


@pytest.mark.parametrize("bad", [(0, 0, 0), (1, 2), (0, 0, 1.5), (0, 0, np.inf)])
def test_invalid_miller_indices_rejected(bad, catalog):
    with pytest.raises(ValueError, match="Miller indices"):
        photon_energies("hopg", 30_000, hkl=bad, catalog=catalog)


def test_shape_mode_material_and_override_errors(catalog):
    with pytest.raises(ValueError, match="incompatible broadcast shapes"):
        photon_energies("hopg", [30_000, 40_000], tilt_deg=[0, 30, 60], catalog=catalog)
    with pytest.raises(ValueError, match="one-dimensional"):
        photon_energies("hopg", [[30_000]], mode="grid", catalog=catalog)
    with pytest.raises(ValueError, match="mode"):
        photon_energies("hopg", 30_000, mode="invalid", catalog=catalog)
    with pytest.raises(ValueError, match="unknown catalogue material"):
        photon_energies("not-a-material", 30_000, catalog=catalog)
    with pytest.raises(ValueError, match="sequence of keys"):
        photon_energies([["hopg"]], 30_000, catalog=catalog)
    with pytest.raises(ValueError, match="not requested"):
        photon_energies("hopg", 30_000, hkl={"mos2": (0, 0, 2)}, catalog=catalog)


def test_mapping_overrides_and_ambiguous_default(catalog):
    overridden = photon_energies(
        ["hopg", "mos2"],
        30_000,
        hkl={"hopg": (0, 0, 4)},
        catalog=catalog,
    )
    default = photon_energies(["hopg", "mos2"], 30_000, catalog=catalog)
    np.testing.assert_allclose(overridden, default * [2, 1])
    custom = replace(
        catalog,
        crystals={
            **catalog.crystals,
            "hopg": replace(
                catalog.crystals["hopg"],
                surface_hkl=None,
                beam_uvw=(0, 0, 1),
            ),
        },
    )
    with pytest.raises(ValueError, match="provide hkl explicitly"):
        photon_energies("hopg", 30_000, catalog=custom)
    np.testing.assert_allclose(
        photon_energies("hopg", 30_000, hkl=(0, 0, 2), catalog=custom),
        default[:1],
    )


def test_external_catalog_path_and_environment(tmp_path, monkeypatch, catalog):
    from tests.helpers.user_catalog import copy_full_catalog

    path = copy_full_catalog(tmp_path / "catalog")
    # Reflection pins change only in this catalogue; catch stale CRYSTALS use.
    crystal_file = path / "crystals" / "hopg.toml"
    crystal_file.write_text(
        crystal_file.read_text().replace(
            "[[0, 0, 2], [0, 0, 4]]",
            "[[0, 0, 4]]",
        )
    )
    expected = photon_energies("hopg", 30_000, catalog=catalog) * 2
    np.testing.assert_allclose(photon_energies("hopg", 30_000, catalog=path), expected)
    monkeypatch.setenv("PYRITE_CATALOG", str(path))
    np.testing.assert_allclose(photon_energies("hopg", 30_000), expected)


def test_call_does_not_initialize_execution_or_plotting():
    script = """
import sys
import pyrite as pr
assert 'pyrite.materials.emission' not in sys.modules
assert pr.photon_energies('hopg', 30_000).shape == (1,)
assert {'numba', 'pyrite._backend', 'pyrite.api', 'pyrite.montecarlo',
        'matplotlib', 'plotly', 'cupy', 'pyrite.checkpoints'}.isdisjoint(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
