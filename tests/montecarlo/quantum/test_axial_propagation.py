"""The diagnostic assembles existing solvers without changing their conventions."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.axial_propagation import AxialPropagationConfig, propagate_axial
from pyrite.montecarlo.channeling.realspace import SplitStepPropagator
from pyrite.montecarlo.channeling.states import gaussian, laguerre_gauss, mixture
from pyrite.montecarlo.channeling.vortex import CylindricalProjection


@pytest.fixture
def config():
    return AxialPropagationConfig(
        shape=(12, 12),
        depths_ang=(0.0, 10.0, 20.0),
        tilt_rad=(1e-4, -2e-4),
        histories=4,
        seed=547,
    )


def test_unitary_matches_direct_propagator_and_projection(config):
    config = replace(config, oam_string=0)
    entry = laguerre_gauss(1, waist_ang=0.5, center_ang=(0.2, 0.3))
    result = propagate_axial(config, entry=entry)
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


def test_unitary_mixture_adds_densities_not_amplitudes(config):
    a = gaussian(waist_ang=0.5)
    b = laguerre_gauss(-1, waist_ang=0.7)
    result = propagate_axial(config, entry=mixture([(1, a), (3, b)]))
    assert result.wavefunctions_per_ang is None
    expected = 0.25 * propagate_axial(config, entry=a).density_per_ang2
    expected += 0.75 * propagate_axial(config, entry=b).density_per_ang2
    np.testing.assert_allclose(result.density_per_ang2, expected)


def test_mcwf_matches_direct_seeded_solver(config):
    config = replace(
        config, mode="mcwf", depths_ang=(0.0, 100.0, 500.0), mechanisms=("tds",), oam_string=0
    )
    entry = mixture([(1, gaussian(waist_ang=0.5)), (1, laguerre_gauss(1, waist_ang=0.7))])
    result = propagate_axial(config, entry=entry)
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
        config.histories,
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
    again = propagate_axial(config, entry=entry)
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
        {"histories": 1.5},
        {"seed": -1},
        {"mechanisms": ("other",)},
        {"oam_string": -1},
        {"oam_radius_ang": 0},
    ],
)
def test_invalid_inputs(config, updates):
    with pytest.raises(ValueError):
        propagate_axial(replace(config, **updates))


def test_retired_coherent_mode_names_replacement(config):
    with pytest.raises(ValueError, match="'coherent' is now 'unitary'"):
        propagate_axial(replace(config, mode="coherent"))


def test_refuses_planar_entry(config):
    with pytest.raises(ValueError, match="2D"):
        propagate_axial(config, entry=gaussian(waist_ang=0.5, dimension=1))


def test_core_import_needs_no_app_or_plotting_extras():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pyrite.montecarlo.channeling.axial_propagation import propagate_axial
assert 'marimo' not in sys.modules
assert 'matplotlib' not in sys.modules
""",
        ],
        check=True,
    )


def test_diamond110_and_empty_sample(config, monkeypatch):
    from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

    diamond = propagate_axial(replace(config, crystal="diamond", axis=(1, 1, 0)))
    assert diamond.profile.crystal == "diamond"
    np.testing.assert_allclose(diamond.surviving, 1)
    monkeypatch.setattr(RealSpaceGrid, "sample", lambda self, state: np.zeros(self.shape))
    with pytest.raises(ValueError, match="sampled norm"):
        propagate_axial(config)


@pytest.mark.parametrize("mode", ["unitary", "mcwf"])
def test_caller_observable_is_weighted_like_the_density(mode):
    config = AxialPropagationConfig(
        shape=(16, 16), depths_ang=(0.0, 50.0), mode=mode, histories=4, mechanisms=("tds",)
    )
    entry = mixture(
        [(0.25, laguerre_gauss(1, waist_ang=0.5)), (0.75, laguerre_gauss(-1, waist_ang=0.5))]
    )

    def observable(phi, tilt):
        return np.abs(phi).ravel() ** 2 + 0j  # complex, as LevelDensity returns

    result = propagate_axial(replace(config, oam_string=0), entry=entry, observable=observable)
    np.testing.assert_allclose(
        result.observables.reshape(result.density_per_ang2.shape), result.density_per_ang2
    )
    assert np.isrealobj(result.density_per_ang2) and np.isrealobj(result.oam_populations)
    plain = propagate_axial(replace(config, oam_string=0), entry=entry)
    assert plain.observables is None
    np.testing.assert_allclose(result.oam_populations, plain.oam_populations)


def test_renamed_mcwf_run_reproduces_pre_rename_numbers(monkeypatch):
    """#1147 rename is name-only: values pinned from ``axial_diagnostic`` at bbc0b1023.

    Pinned before the measured bonding charge (#1177), so the run uses the
    independent-atom silicon potential the pins were taken with.
    """
    from pyrite.materials.crystal import CRYSTALS

    monkeypatch.setitem(CRYSTALS["silicon"], "bonding_dF", {})
    config = AxialPropagationConfig(
        shape=(12, 12),
        depths_ang=(0.0, 100.0, 500.0),
        mode="mcwf",
        histories=4,
        seed=547,
        oam_string=0,
    )
    with pytest.warns(RuntimeWarning, match="fewer than 16 samples"):
        result = propagate_axial(config, entry=gaussian(waist_ang=0.5), record_metrics=True)
    np.testing.assert_array_equal(result.surviving, [1.0, 1.0, 0.75])
    assert result.jumps == {"tds": 1, "sink": 1, "electronic": 1}
    np.testing.assert_allclose(
        result.density_per_ang2.sum((1, 2)),
        [39.0578979907728, 39.05789799077281, 29.293423493079608],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.density_per_ang2[:, 3, 5],
        [0.0028334262453719237, 0.004104938290016683, 0.08202343184537898],
        rtol=1e-10,
    )
    np.testing.assert_allclose(result.oam_populations[-1, 8], 0.43365615704023575, rtol=1e-10)
    metrics = result.metrics
    assert metrics.histories == 4
    np.testing.assert_allclose(
        metrics.purity, [1.3108679465419728, 1.4031347466141946, 0.7929287917230612], rtol=1e-10
    )
    np.testing.assert_allclose(
        metrics.energy_mean_eV,
        [-18.756631959857245, -17.65256573783696, -1.029748312353929],
        rtol=1e-10,
    )


def test_pre_rename_count_keywords_and_fields_are_gone(config):
    from dataclasses import fields

    from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWFResult
    from pyrite.montecarlo.channeling.metrics import EnsembleMetrics

    with pytest.raises(TypeError, match="n_electrons"):
        AxialPropagationConfig(n_electrons=4)
    assert "n_electrons" not in {f.name for f in fields(AxialMCWFResult)}
    assert "histories" in {f.name for f in fields(AxialMCWFResult)}
    assert "electrons" not in {f.name for f in fields(EnsembleMetrics)}
    result = propagate_axial(replace(config, mode="mcwf", mechanisms=()))
    dissipator = axial_dissipator(
        result.profile, config.kinetic_energy_eV, config.cells, config.shape
    )
    solver = AxialMCWF(dissipator, config.kinetic_energy_eV, result.potential_eV)
    with pytest.raises(TypeError, match="n_electrons"):
        solver.run(gaussian(waist_ang=0.5), (0.0, 10.0), n_electrons=1)
