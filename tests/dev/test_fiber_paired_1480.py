"""Regression tests for the research-only paired diagnostic (#1480)."""

import numpy as np
import pytest

from checks.fiber_paired_1480 import (
    azimuth_average,
    merge_shell_weights,
    observables,
    paired_differences,
    photon_axis,
    reflection_domain,
    tensor_groups,
    validate_case,
)


def test_common_axis_and_observables():
    grid, edges = photon_axis(10, 15, 2)
    np.testing.assert_allclose(edges[[0, -1]], [10, 15])
    widths = np.diff(edges)
    cxr = np.array([0., 3., 0.])
    brem = np.ones_like(cxr)
    result = observables(grid, widths, cxr, brem)
    assert result["cxr_yield"] == pytest.approx(5)
    assert result["total_yield"] == pytest.approx(10)
    assert result["cxr_centroid_eV"] == pytest.approx(12.5)
    assert result["cxr_fwhm_eV"] == pytest.approx(widths[0])
    zero = observables(grid, widths, np.zeros_like(cxr), brem)
    assert zero["cxr_fwhm_eV"] is None
    metrics = dict(A=result, B=result, C=result, D=zero)
    delta = paired_differences(metrics)
    assert delta["B-A"]["cxr_yield"] == dict(absolute=0., relative=0.)
    assert delta["A-D"]["cxr_yield"]["absolute"] == pytest.approx(5.)
    assert delta["A-D"]["cxr_yield"]["relative"] is None
    assert delta["A-D"]["cxr_centroid_eV"]["absolute"] is None


def test_merge_shells_before_clamp():
    g, weights = merge_shell_weights([
        (np.array([1., 3.]), {"C": np.array([2., 4.])}),
        (np.array([2., 3.]), {"C": np.array([5., 7.])}),
    ])
    np.testing.assert_array_equal(g, [1, 2, 3])
    np.testing.assert_array_equal(weights["C"], [2, 5, 11])


def test_azimuth_average_sums_intensities_and_preserves_nodes():
    grid = np.array([10., 11.])
    angles = np.array([0., np.pi])
    groups = {1.: [(1, 0, 0)], 2.: [(0, 1, 0)]}
    calls = []

    def score(b, hkls, phi):
        calls.append((b, hkls, phi))
        return np.full_like(grid, b * (2 + np.cos(phi)))

    average, samples = azimuth_average(score, groups, angles, grid)
    np.testing.assert_allclose(samples, [[9, 9], [3, 3]])
    np.testing.assert_allclose(average, [6, 6])
    assert [call[2] for call in calls] == [0, 0, np.pi, np.pi]


def test_refuses_mismatched_populations():
    case = {"crystal": "hopg"}
    tp = {"segs": {"Ne": 3}, "Ne_lines": 2, "Ne_brem": 3}
    with pytest.raises(ValueError, match="same complete electron population"):
        validate_case(case, tp)


def test_production_domain_and_tensor_projection():
    from pyrite.materials.channeling.thermal import thermal_site
    from pyrite.materials.crystal import CRYSTALS

    basal, full = reflection_domain("hopg", 8.)
    assert set(basal) < set(full)
    assert (0, 0, 2) in basal
    assert (1, 0, 2) in full
    assert all(tuple(-i for i in h) in full for h in full)
    assert CRYSTALS["hopg"]["texture"] == "fiber"
    site = thermal_site("hopg", "C")
    groups = tensor_groups("hopg", full, site, 300.)
    b33 = site.b_factor_along(1., 300.)
    assert set(groups[b33]) == set(basal)
    assert min(groups) == pytest.approx(site.b_factor_along(0., 300.))
    assert {h for hkls in groups.values() for h in hkls} == set(full)
    assert sum(map(len, groups.values())) == len(full)
