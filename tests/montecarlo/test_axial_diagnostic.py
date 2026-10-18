"""The diagnostic assembles existing solvers without changing their conventions."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig, axial_diagnostic
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.realspace import SplitStepPropagator
from pyrite.montecarlo.channeling.states import gaussian, laguerre_gauss, mixture
from pyrite.montecarlo.channeling.vortex import CylindricalProjection


@pytest.fixture
def config():
    return AxialDiagnosticConfig(
        shape=(12, 12),
        depths_ang=(0.0, 10.0, 20.0),
        tilt_rad=(1e-4, -2e-4),
        n_electrons=4,
        seed=547,
    )


def test_coherent_matches_direct_propagator_and_projection(config):
    config = replace(config, oam_string=0)
    entry = laguerre_gauss(1, waist_ang=0.5, center_ang=(0.2, 0.3))
    result = axial_diagnostic(config, entry=entry)
    solver = SplitStepPropagator(
        result.grid,
        config.kinetic_energy_eV,
        potential_eV=result.potential_eV,
        tilt_rad=config.tilt_rad,
    )
    phi = result.grid.sample(entry)
    direct = np.stack(
        [
            state
            for _, state in solver.trajectory(
                phi / np.sqrt(result.grid.norm(phi)), config.depths_ang, step_ang=config.step_ang
            )
        ]
    )
    np.testing.assert_allclose(result.wavefunctions_per_ang, direct)
    np.testing.assert_allclose(result.density_per_ang2, np.abs(direct) ** 2)
    np.testing.assert_allclose(result.surviving, 1, atol=1e-13)
    projection = CylindricalProjection(result.grid, result.oam_origin_ang, 0.5, 8)
    np.testing.assert_allclose(
        result.oam_populations[0], projection(direct[0], solver.tilt_inv_ang)
    )
    assert result.sampled_norms == (result.grid.norm(phi),)
    assert result.config.seed == 547
    assert "experimental" in result.experimental_model


def test_coherent_mixture_adds_densities_not_amplitudes(config):
    a = gaussian(waist_ang=0.5)
    b = laguerre_gauss(-1, waist_ang=0.7)
    result = axial_diagnostic(config, entry=mixture([(1, a), (3, b)]))
    assert result.wavefunctions_per_ang is None
    expected = 0.25 * axial_diagnostic(config, entry=a).density_per_ang2
    expected += 0.75 * axial_diagnostic(config, entry=b).density_per_ang2
    np.testing.assert_allclose(result.density_per_ang2, expected)


def test_mcwf_matches_direct_seeded_solver(config):
    config = replace(
        config, mode="mcwf", depths_ang=(0.0, 100.0, 500.0), mechanisms=("tds",), oam_string=0
    )
    entry = mixture([(1, gaussian(waist_ang=0.5)), (1, laguerre_gauss(1, waist_ang=0.7))])
    result = axial_diagnostic(config, entry=entry)
    dissipator = axial_dissipator(
        result.profile,
        config.kinetic_energy_eV,
        config.cells,
        config.shape,
        mechanisms=config.mechanisms,
    )
    direct = AxialMCWF(dissipator, config.kinetic_energy_eV, result.potential_eV).run(
        entry,
        config.depths_ang,
        config.n_electrons,
        seed=config.seed,
        step_ang=config.step_ang,
        tilt_rad=config.tilt_rad,
    )
    np.testing.assert_array_equal(result.density_per_ang2, direct.observables)
    np.testing.assert_array_equal(result.surviving, direct.surviving)
    assert result.jumps == direct.jumps
    assert result.wavefunctions_per_ang is None
    np.testing.assert_allclose(
        result.density_per_ang2.sum((1, 2)) * result.grid.cell_area_ang2, result.surviving
    )
    again = axial_diagnostic(config, entry=entry)
    np.testing.assert_array_equal(result.oam_populations, again.oam_populations)
    np.testing.assert_allclose(
        result.density(conditional=True).sum((1, 2)) * result.grid.cell_area_ang2,
        (result.surviving > 0).astype(float),
    )
    extinct = replace(
        result,
        surviving=np.zeros_like(result.surviving),
        density_per_ang2=np.zeros_like(result.density_per_ang2),
    )
    assert not extinct.density(conditional=True).any()


@pytest.mark.parametrize(
    "updates",
    [
        {"kinetic_energy_eV": np.nan},
        {"temperature_K": 0},
        {"step_ang": np.inf},
        {"depths_ang": ()},
        {"depths_ang": (0, np.nan)},
        {"depths_ang": (10, 0)},
        {"depths_ang": (0, 1)},
        {"shape": (12.5, 12)},
        {"cells": (0, 1)},
        {"axis": (0, 0, 0)},
        {"axis": (0, 0, 1.2)},
        {"tilt_rad": (np.inf, 0)},
        {"mode": "other"},
        {"n_electrons": 1.5},
        {"seed": -1},
        {"mechanisms": ("other",)},
        {"oam_string": -1},
        {"oam_radius_ang": 0},
    ],
)
def test_invalid_inputs(config, updates):
    with pytest.raises(ValueError):
        axial_diagnostic(replace(config, **updates))


def test_refuses_planar_entry(config):
    with pytest.raises(ValueError, match="2D"):
        axial_diagnostic(config, entry=gaussian(waist_ang=0.5, dimension=1))


def test_core_import_needs_no_app_or_plotting_extras():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pyrite.montecarlo.channeling.axial_diagnostic import axial_diagnostic
assert 'marimo' not in sys.modules
assert 'matplotlib' not in sys.modules
""",
        ],
        check=True,
    )


def test_diamond110_and_empty_sample(config, monkeypatch):
    from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

    diamond = axial_diagnostic(replace(config, crystal="diamond", axis=(1, 1, 0)))
    assert diamond.profile.crystal == "diamond"
    np.testing.assert_allclose(diamond.surviving, 1)
    monkeypatch.setattr(RealSpaceGrid, "sample", lambda self, state: np.zeros(self.shape))
    with pytest.raises(ValueError, match="sampled norm"):
        axial_diagnostic(config)


@pytest.mark.parametrize("mode", ["coherent", "mcwf"])
def test_caller_observable_is_weighted_like_the_density(mode):
    config = AxialDiagnosticConfig(
        shape=(16, 16), depths_ang=(0.0, 50.0), mode=mode, n_electrons=4, mechanisms=("tds",)
    )
    entry = mixture(
        [(0.25, laguerre_gauss(1, waist_ang=0.5)), (0.75, laguerre_gauss(-1, waist_ang=0.5))]
    )

    def observable(phi, tilt):
        return np.abs(phi).ravel() ** 2 + 0j  # complex, as LevelDensity returns

    result = axial_diagnostic(replace(config, oam_string=0), entry=entry, observable=observable)
    np.testing.assert_allclose(
        result.observables.reshape(result.density_per_ang2.shape), result.density_per_ang2
    )
    assert np.isrealobj(result.density_per_ang2) and np.isrealobj(result.oam_populations)
    plain = axial_diagnostic(replace(config, oam_string=0), entry=entry)
    assert plain.observables is None
    np.testing.assert_allclose(result.oam_populations, plain.oam_populations)
