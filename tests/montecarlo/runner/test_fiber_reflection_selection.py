"""Frozen #1301 transport yields anchor the texture-filtered candidate domain.

Validation: reflection-yield-tolerance-selector
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo.runner.lattice_share import continuum_case, scored_reflections
from pyrite.montecarlo.runner.reflection_cutoff import (
    axis_yield_weights,
    band_bin_grid,
    reflection_selection_record,
    selected_line_spectrum,
)


@pytest.mark.parametrize("index", [0, 1, 2], ids=["bells-30", "cleo-1000", "cleo-30"])
def test_frozen_fiber_shells_select_and_reconstruct_the_same_basal_set(index):
    evidence = json.loads(
        (Path(__file__).parents[2] / "data/reflection_selection_fiber_1301.json").read_text()
    )
    sample = evidence["samples"][index]
    rows = sample["shells"]
    by_hkl = {tuple(h): row for row in rows for h in row["hkls"]}
    grid = band_bin_grid(*sample["band_eV"])
    width = axis_yield_weights(grid, "bin-mean").sum()

    def yield_of(hkls, key):
        return sum(by_hkl[tuple(h)][key] / len(by_hkl[tuple(h)]["hkls"]) for h in hkls)

    def evaluate(case):
        return np.full(grid.shape, yield_of(case["hkl_list"], "yield_") / width)

    class MeasuredContinuum:
        def applies(self, layer):
            return True

        def share(self, layer, hkls):
            return yield_of(hkls, "share")

        def background(self, layer):
            return sample["background_yield"]

    case = dict(
        crystal="hopg",
        B_ang2=CATALOG.crystal("hopg").B_ang2,
        composition=list(CATALOG.crystal("hopg").composition),
        hkl_list=sample["pinned_hkls"],
        line_quadrature="bin-mean",
        reflection_selection=reflection_selection_record(1e-3),
    )
    spectrum, record = selected_line_spectrum(
        case, evaluate, grid, continuum=lambda _: MeasuredContinuum()
    )
    (radiator,) = record["radiators"]
    expected = sample["selector_stop"]
    assert radiator["converged"] == expected["converged"]
    assert radiator["cutoff_invang"] == expected["cutoff_invang"]
    assert radiator["tail_fraction"] == pytest.approx(expected["tail_fraction"], rel=1e-12)
    assert radiator["texture"] == "fiber" and radiator["texture_axis_hkl"] == (0, 0, 1)
    assert len(radiator["selected_set_sha256"]) == 64
    rebuilt = scored_reflections(continuum_case(case, record), None)
    assert list(map(list, rebuilt)) == expected["hkl_list"]
    np.testing.assert_allclose(spectrum, evaluate(dict(case, hkl_list=rebuilt)), rtol=1e-14)
