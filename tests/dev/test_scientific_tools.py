"""Calculator facade contracts: supported identity and dependency-free discovery."""

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = tomllib.loads((ROOT / "api-surface.toml").read_text())["package_exports"][
    "pyrite.tools"
]


@pytest.mark.parametrize("name", sorted(INVENTORY))
def test_tools_preserve_supported_implementation_identity(name):
    from pyrite import tools

    module, attribute = INVENTORY[name]["target"].split(":")
    original = getattr(importlib.import_module(module), attribute)
    assert getattr(tools, name) is original
    assert name in tools.__all__
    assert name in dir(tools)


def test_tools_discovery_without_scientific_dependencies():
    script = """
import sys
from importlib.abc import MetaPathFinder

blocked = {'numpy', 'numba', 'cupy', 'matplotlib', 'pandas', 'h5py', 'pint', 'scipy'}

class DenyScientific(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or fullname.startswith('pyrite.materials'):
            raise AssertionError(f'eager scientific import: {fullname}')
        return None

sys.meta_path.insert(0, DenyScientific())
from pyrite import tools
assert 'structure_factor' in dir(tools)
assert 'photon_energies_q' in dir(tools)
assert len(tools.__all__) == len(set(tools.__all__))
try:
    tools.not_a_calculator
except AttributeError:
    pass
else:
    raise AssertionError('unknown tool accepted')
assert not (blocked & sys.modules.keys())
assert 'pyrite.montecarlo' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tools_exclude_simulation_storage_and_capture_readers():
    from pyrite import tools

    assert set(tools.__all__) == set(INVENTORY)
    assert not set(tools.__all__) & {
        "simulate",
        "fetch_tables",
        "mc_cxr_spectrum",
        "directional_source",
        "summarize_trajectories",
        "capture_secondary_energy_balance",
        "capture_beam_metrics",
    }
