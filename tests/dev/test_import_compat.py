import importlib
import importlib.resources
import pickle
import subprocess
import sys


def test_legacy_root_and_deep_imports_are_canonical_module_objects():
    legacy_root = importlib.import_module("cxr_mc")
    canonical_root = importlib.import_module("pyrite")
    legacy_leaf = importlib.import_module("cxr_mc.campaign.sweep")
    canonical_leaf = importlib.import_module("pyrite.campaign.sweep")

    assert legacy_root is canonical_root
    assert legacy_leaf is canonical_leaf
    assert sys.modules["cxr_mc.campaign.sweep"] is canonical_leaf


def test_legacy_import_order_does_not_duplicate_classes():
    legacy = importlib.import_module("cxr_mc.materials.catalog")
    canonical = importlib.import_module("pyrite.materials.catalog")

    assert legacy is canonical
    assert legacy.MaterialCatalog is canonical.MaterialCatalog


def test_existing_legacy_pickle_global_resolves_to_canonical_class():
    from pyrite.campaign.sweep import Sweep

    legacy_global = b"ccxr_mc.campaign.sweep\nSweep\n."

    assert pickle.loads(legacy_global) is Sweep


def test_new_pickle_uses_canonical_module_path():
    from pyrite.campaign.sweep import Sweep

    payload = pickle.dumps(Sweep(material="hopg"))

    assert b"pyrite.campaign.sweep" in payload
    assert b"cxr_mc.campaign.sweep" not in payload


def test_legacy_resource_lookup_uses_canonical_package_data():
    material_data = importlib.resources.files("cxr_mc").joinpath("data/materials.toml")

    assert material_data.is_file()


def test_legacy_python_m_entry_executes_canonical_module():
    completed = subprocess.run(
        [sys.executable, "-m", "cxr_mc._entry.scan", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Usage:" in completed.stdout
