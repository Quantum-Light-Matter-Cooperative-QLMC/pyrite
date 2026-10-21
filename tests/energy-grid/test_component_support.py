"""Optical source-only grids never relax another component's data support."""

import numpy as np
import pytest

from pyrite._grid_semantics import node_bin_edges_and_widths
from pyrite.energy_grid.component_support import ComponentSupport, component_grid
from pyrite.energy_grid.floor import DATA_SUPPORT_LIMITS_EV, photon_continuum_floor_eV
from pyrite.materials.optical import gold_rakic_response


def _optical():
    return ComponentSupport("Au", "transition_radiation", gold_rakic_response().bands_eV)


def test_optical_grid_is_explicit_and_retains_nodes_and_midpoint_bins():
    nodes = np.array([0.3, 0.5, 1.0, 2.0, 4.0])
    result = component_grid(
        nodes, required=(("Au", "transition_radiation"),), support=(_optical(),)
    )
    np.testing.assert_array_equal(result, nodes)
    for expected, actual in zip(
        node_bin_edges_and_widths(nodes), node_bin_edges_and_widths(result), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert result is not nodes
    assert photon_continuum_floor_eV("silicon") > 30.0
    assert DATA_SUPPORT_LIMITS_EV["eaglexo_quantum_efficiency"] == 12.0


@pytest.mark.parametrize(
    "component",
    ["epdl_attenuation", "eedl_bremsstrahlung", "chantler", "eaglexo_qe", "tensor_optics"],
)
def test_missing_component_is_named_and_never_inherits_tr_support(component):
    with pytest.raises(ValueError, match=f"Au/{component}: no supported-band"):
        component_grid(
            [1.0, 2.0],
            required=(("Au", "transition_radiation"), ("Au", component)),
            support=(_optical(),),
        )


def test_narrower_component_band_is_named():
    qe = ComponentSupport("sensor", "measured_qe", ((1.5, 3.0),))
    with pytest.raises(ValueError, match="sensor/measured_qe: unsupported"):
        component_grid(
            [1.0, 2.0],
            required=(("Au", "transition_radiation"), ("sensor", "measured_qe")),
            support=(_optical(), qe),
        )


def test_nodes_and_bins_have_distinct_coverage_contracts():
    support = ComponentSupport("sample", "source", ((1.0, 2.0), (3.0, 4.0)))
    nodes = [1.0, 2.0, 3.0, 4.0]
    component_grid(nodes, required=(("sample", "source"),), support=(support,))
    with pytest.raises(ValueError, match="sample/source: unsupported"):
        component_grid(
            nodes, required=(("sample", "source"),), support=(support,), representation="bins"
        )
    with pytest.raises(ValueError, match="gap"):
        component_grid([1.5, 2.5], required=(("sample", "source"),), support=(support,))
    np.testing.assert_array_equal(
        component_grid(
            [1.25, 1.75],
            required=(("sample", "source"),),
            support=(support,),
            representation="bins",
        ),
        [1.25, 1.75],
    )


@pytest.mark.parametrize(
    "nodes", [[0.0, 1.0], [2.0, 1.0], [1.0, 1.0], [1.0, np.nan], [1.0], [[1.0, 2.0]]]
)
def test_invalid_grid_is_refused(nodes):
    with pytest.raises(ValueError, match="component grid"):
        component_grid(nodes, required=(("Au", "transition_radiation"),), support=(_optical(),))


def test_required_components_and_unique_support_are_mandatory():
    with pytest.raises(ValueError, match="required"):
        component_grid([1.0, 2.0], required=(), support=(_optical(),))
    with pytest.raises(ValueError, match="duplicate"):
        component_grid(
            [1.0, 2.0], required=(("Au", "transition_radiation"),), support=(_optical(), _optical())
        )


@pytest.mark.parametrize(
    "bands", [(), ((0.0, 1.0),), ((1.0, np.inf),), ((1.0, 3.0), (2.0, 4.0)), ((2.0, 1.0),)]
)
def test_invalid_capability_is_refused(bands):
    with pytest.raises(ValueError, match="support"):
        ComponentSupport("sample", "source", bands)
