"""Advanced scientific aliases preserve engine contracts without eager imports."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrite.devtools.api_packages import read_package_inventory

ROOT = Path(__file__).resolve().parents[2]
PACKAGES, _, _ = read_package_inventory(ROOT / "api-surface.toml")


@pytest.mark.parametrize(
    "row",
    [row for domain in ("transport", "radiation") for row in PACKAGES[f"pyrite.{domain}"].values()],
    ids=lambda row: f"{row.package}.{row.name}",
)
def test_scientific_alias_identity_and_discovery(row):
    facade = importlib.import_module(row.package)
    assert getattr(facade, row.name) is getattr(importlib.import_module(row.module), row.attribute)
    assert row.name in facade.__all__
    assert row.name in dir(facade)
    with pytest.raises(AttributeError, match="no attribute"):
        _ = facade.unsupported_kernel


def test_namespace_import_does_not_attempt_engines_or_dataset_access():
    script = """
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

blocked = {'numpy', 'numba', 'cupy', 'matplotlib', 'dask', 'distributed', 'optuna'}
class DenyEngines(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or fullname.startswith((
            'pyrite.montecarlo', 'pyrite.materials', 'pyrite.directional', 'pyrite.results'
        )):
            raise AssertionError(f'eager scientific import: {fullname}')
        return None
sys.meta_path.insert(0, DenyEngines())
def no_data(*args, **kwargs):
    raise AssertionError('namespace import attempted dataset I/O')
Path.open = no_data
import pyrite.transport as transport
import pyrite.radiation as radiation
assert 'simulate_trajectories' in dir(transport)
assert 'directional_sources' in dir(radiation)
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_transport_alias_matches_seeded_segments_and_metadata():
    from pyrite.montecarlo.transport import simulate_trajectories as original
    from pyrite.transport import simulate_trajectories

    options = dict(
        composition=[("C", 0.1136)],
        E_cut_keV=5.0,
        seed=4,
        elastic_model="sr",
        transport_core="lockstep",
        energy_model="frozen",
    )
    canonical = simulate_trajectories(5.01, 1, 1e8, **options)
    legacy = original(5.01, 1, 1e8, **options)
    assert canonical.keys() == legacy.keys()
    for key in canonical:
        if isinstance(canonical[key], np.ndarray):
            np.testing.assert_array_equal(canonical[key], legacy[key], err_msg=key)
        else:
            assert canonical[key] == legacy[key], key
    assert canonical["Ne"] == canonical["n_cutoff_stopped"] == 1


def test_photon_energy_alias_preserves_eV_broadcast_contract():
    from pyrite.materials.emission import photon_energies as original
    from pyrite.radiation import photon_energies

    options = dict(theta_obs_deg=[90.0, 119.0], hkl=(0, 0, 2))
    actual = photon_energies("hopg", [30e3, 50e3], **options)
    np.testing.assert_array_equal(actual, original("hopg", [30e3, 50e3], **options))
    assert actual.shape == (1, 2)
    assert np.all((actual > 0) & (actual < [30e3, 50e3]))


@pytest.mark.parametrize(
    "name", ["mc_cxr_spectrum", "mc_cxr_spectrum_solid_angle", "mc_brem_spectrum"]
)
def test_spectrum_aliases_preserve_small_segment_densities(name):
    from pyrite import radiation
    from pyrite.montecarlo import spectrum
    from tests.helpers import to_host

    segments = dict(
        r_mid=np.array([[0.0, 0.0, 5.0]]),
        v_hat=np.array([[0.0, 0.0, 1.0]]),
        L_ang=np.array([10.0]),
        E_keV=np.array([30.0]),
        t_ang=np.zeros(1),
        t0_ang=np.zeros(1),
        elec_id=np.zeros(1, dtype=int),
        layer=np.zeros(1, dtype=int),
        Ne=1,
        thickness_ang=10.0,
        n_backscattered=0,
        n_missed=0,
    )
    grid = np.linspace(700.0, 1500.0, 17)
    direction = np.array([1.0, 0.0, 0.1]) / np.linalg.norm([1.0, 0.0, 0.1])
    if name == "mc_brem_spectrum":
        kwargs = dict(
            composition=[("C", 0.1136)], n_hat=direction, cross_section_model="bethe-heitler"
        )
    else:
        kwargs = dict(crystal="hopg", hkl_list=[(0, 0, 2)], B_ang2=0.8)
        kwargs.update(
            dict(n_hats=[direction], weights=[0.02])
            if name.endswith("solid_angle")
            else dict(n_hat=direction)
        )
    canonical = getattr(radiation, name)(segments, grid, **kwargs)
    legacy = getattr(spectrum, name)(segments, grid, **kwargs)
    actual = to_host(canonical)
    np.testing.assert_array_equal(actual, to_host(legacy))
    assert actual.shape == grid.shape
    assert np.all(np.isfinite(actual)) and np.max(actual) > 0
