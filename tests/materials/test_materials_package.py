"""Contract tests for the material-physics package boundary."""

import subprocess
import sys

import numpy as np


def test_materials_package_exports_only_registry_conveniences():
    from pyrite import materials

    assert set(materials.__all__) == {
        "CRYSTALS",
        "CATALOG",
        "MaterialCatalog",
        "CrystalInfo",
        "CrystalSpec",
        "MediumSpec",
        "MaterialIdentity",
        "MaterialSpec",
        "MaterialValidationSpec",
        "ScanSpec",
        "LayerSpec",
        "MaterialConfigError",
        "load_material_catalog",
        "MATERIAL_LABELS",
        "MATERIALS",
    }

    for legacy in (
        "MATERIAL_CONFIGS",
        "MATERIAL_GRIDS",
        "CRYSTAL_PARAMS",
        "MaterialGrid",
        "CrystalParamsGrid",
        "Layer",
    ):
        assert not hasattr(materials, legacy)


def test_catalog_package_export_is_a_singleton():
    import pyrite.materials as materials
    from pyrite.materials import CATALOG

    assert materials.CATALOG is CATALOG


def test_importing_catalog_module_does_not_load_bundled_manifest():
    script = r"""
from pathlib import Path

real_open = Path.open

def guarded_open(path, *args, **kwargs):
    if "catalog" in Path(path).parts:
        raise AssertionError("bundled catalog was loaded eagerly")
    return real_open(path, *args, **kwargs)

real_iterdir = Path.iterdir

def guarded_iterdir(path):
    if "catalog" in path.parts:
        raise AssertionError("bundled catalog was listed eagerly")
    return real_iterdir(path)

Path.open = guarded_open
Path.iterdir = guarded_iterdir
import pyrite.materials.catalog
print("catalog module imported lazily")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "catalog module imported lazily" in result.stdout


def test_legacy_registry_and_crystal_toml_are_removed():
    from pathlib import Path

    import pyrite

    package = Path(pyrite.__file__).parent
    assert not (package / "materials" / "registry.py").exists()
    assert not (package / "data" / "crystal_structures.toml").exists()


def test_materials_package_preserves_crystal_registry_and_attenuation_behavior():
    from pyrite.materials import CRYSTALS
    from pyrite.materials.attenuation import _mu_total_inv_ang, _stack_tau

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
