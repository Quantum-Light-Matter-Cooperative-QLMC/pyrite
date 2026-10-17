"""Scientific and reactive contracts of the planar channeling demo."""

import ast
from pathlib import Path

import numpy as np
import pytest

from pyrite.apps.channeling import critical_angle_rad
from pyrite.materials.channeling import longitudinal_kinematics, planar_profile, transverse_states
from pyrite.materials.crystal import M_E_EV

APP = Path(__file__).parents[2] / "src/pyrite/apps/channeling_app.py"


def test_si110_critical_angle_uses_the_well_depth():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    gamma, beta = longitudinal_kinematics(5e6)
    minimum, maximum = profile.extrema()
    expected = np.sqrt(2 * (maximum - minimum) / (gamma * M_E_EV * beta**2))
    assert critical_angle_rad(profile, 5e6) == pytest.approx(expected)
    assert expected > 0


def test_display_selectors_do_not_feed_expensive_cells():
    tree = ast.parse(APP.read_text())
    expensive = {"propagate", "channeling_spectrum", "entry_populations", "transverse_states"}
    found = set()
    for cell in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        if called & expensive:
            found |= called & expensive
            assert not {"state_ui", "sample_ui", "conditional_ui"} & {
                arg.arg for arg in cell.args.args
            }
    assert found == expensive


def test_every_plotted_eigenfunction_has_unit_probability_per_period():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    gamma, _ = longitudinal_kinematics(5e6)
    states = transverse_states(profile, gamma, n_plane_waves=16)
    x = np.linspace(0, profile.period_ang, 257)
    for state in range(states.energies_eV.size):
        assert np.trapezoid(np.abs(states.wavefunction(x, state)) ** 2, x) == pytest.approx(
            1,
            abs=1e-13,
        )
