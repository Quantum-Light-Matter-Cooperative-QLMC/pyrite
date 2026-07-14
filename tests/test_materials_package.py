"""Contract tests for the material-physics package boundary."""

import numpy as np


def test_materials_package_exports_only_registry_conveniences():
    from cxr_mc import materials

    assert set(materials.__all__) == {
        "CRYSTALS",
        "CATALOG",
        "MaterialCatalog",
        "CrystalInfo",
        "CrystalSpec",
        "MediumSpec",
        "MaterialSpec",
        "ScanSpec",
        "LayerSpec",
        "MaterialConfigError",
        "load_material_catalog",
        "MATERIAL_CONFIGS",
        "MATERIAL_GRIDS",
        "CRYSTAL_PARAMS",
        "MATERIAL_LABELS",
        "MATERIALS",
        "Layer",
        "ScalarOrSeq",
        "MaterialConfig",
        "MaterialGrid",
        "CrystalParamsGrid",
        "material_crystal_key",
        "material_scan_grid",
        "crystal_config",
    }


def test_materials_package_preserves_crystal_registry_and_attenuation_behavior():
    from cxr_mc.materials import CRYSTALS
    from cxr_mc.materials.attenuation import _mu_total_inv_ang, _stack_tau

    assert {"silicon", "hbn", "mose2", "sapphire"} <= set(CRYSTALS)

    z_mid = np.array([50.0, 200.0, 480.0])
    energy_eV = np.full(3, 1438.0)
    film = [("Mo", 0.019), ("Se", 0.038)]
    substrate = [("Si", 0.02205), ("O", 0.04410)]
    film_thickness, substrate_thickness, n_z = 500.0, 1.5e4, 0.7
    tau_film = _stack_tau([(0.0, film_thickness, film)], z_mid, n_z, energy_eV)
    tau_stack = _stack_tau(
        [
            (0.0, film_thickness, film),
            (film_thickness, film_thickness + substrate_thickness, substrate),
        ],
        z_mid,
        n_z,
        energy_eV,
    )

    assert np.allclose(
        tau_stack - tau_film,
        _mu_total_inv_ang(substrate, energy_eV) * substrate_thickness / abs(n_z),
    )
