"""B1: experimental real-space/axial inputs refuse Pint quantities (#755)."""

from __future__ import annotations

import pytest

from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid, SplitStepPropagator
from pyrite.units import ureg


def test_realspace_grid_extent_refuses_quantities() -> None:
    for extent in ((1 * ureg.nm, 1 * ureg.nm), (1.0, 10 * ureg.angstrom), (1, 1) * ureg.nm):
        with pytest.raises(TypeError, match=r"RealSpaceGrid\.extent_ang is numeric-only"):
            RealSpaceGrid(extent, (8, 8))
    assert RealSpaceGrid((10.0, 10.0), (8, 8)).x[1] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "tilt",
    [
        (1e-3 * ureg.dimensionless, 0.0),  # bare dimensionless is not an angle
        (1 * ureg.mrad, 0.0),  # numeric-only: refused, not converted
        (0.0, 0.0) * ureg.rad,
    ],
)
def test_split_step_tilt_refuses_dimensionless_and_angle_quantities(tilt) -> None:
    grid = RealSpaceGrid((10.0, 10.0), (8, 8))
    with pytest.raises(TypeError, match=r"SplitStepPropagator\.tilt_rad is numeric-only"):
        SplitStepPropagator(grid, 1e6, tilt_rad=tilt)
    plain = SplitStepPropagator(grid, 1e6, tilt_rad=(1e-3, 0.0))
    assert plain.tilt_inv_ang[1] == 0.0


def test_split_step_energy_refuses_quantity() -> None:
    grid = RealSpaceGrid((10.0, 10.0), (8, 8))
    with pytest.raises(TypeError, match="kinetic_energy_eV is numeric-only"):
        SplitStepPropagator(grid, 1 * ureg.MeV)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("kinetic_energy_eV", 5 * ureg.MeV),
        ("temperature_K", 293.15 * ureg.kelvin),
        ("depths_ang", (0.0, 5 * ureg.nm)),
        ("step_ang", 0.5 * ureg.nm),
        ("tilt_rad", (1e-3 * ureg.dimensionless, 0.0)),
        ("oam_radius_ang", 0.05 * ureg.nm),
        ("spectral_cutoff_eV", 1 * ureg.eV),
    ],
)
def test_axial_diagnostic_config_refuses_quantities_at_construction(name, value) -> None:
    with pytest.raises(TypeError, match=rf"AxialDiagnosticConfig\.{name} is numeric-only"):
        AxialDiagnosticConfig(**{name: value})
    assert AxialDiagnosticConfig().kinetic_energy_eV == 5e6
