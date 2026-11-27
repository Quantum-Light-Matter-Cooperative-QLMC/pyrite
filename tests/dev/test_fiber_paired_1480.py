"""Regression tests for the research-only paired diagnostic (#1480)."""

import numpy as np
import pytest

from checks.fiber_paired_1480 import (
    azimuth_average,
    integrity_passes,
    merge_shell_weights,
    observables,
    paired_differences,
    photon_axis,
    reflection_domain,
    roll_angles,
    tensor_groups,
    validate_case,
)


def test_common_axis_and_observables():
    grid, edges = photon_axis(10, 15, 2)
    np.testing.assert_allclose(edges[[0, -1]], [10, 15])
    widths = np.diff(edges)
    cxr = np.array([0.0, 3.0, 0.0])
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
    assert delta["B-A"]["cxr_yield"] == dict(absolute=0.0, relative=0.0)
    assert delta["A-D"]["cxr_yield"]["absolute"] == pytest.approx(5.0)
    assert delta["A-D"]["cxr_yield"]["relative"] is None
    assert delta["A-D"]["cxr_centroid_eV"]["absolute"] is None


def test_merge_shells_before_clamp():
    g, weights = merge_shell_weights(
        [
            (np.array([1.0, 3.0]), {"C": np.array([2.0, 4.0])}),
            (np.array([2.0, 3.0]), {"C": np.array([5.0, 7.0])}),
        ]
    )
    np.testing.assert_array_equal(g, [1, 2, 3])
    np.testing.assert_array_equal(weights["C"], [2, 5, 11])


def test_azimuth_average_sums_intensities_and_preserves_nodes():
    grid = np.array([10.0, 11.0])
    angles = np.array([0.0, np.pi])
    groups = {1.0: [(1, 0, 0)], 2.0: [(0, 1, 0)]}
    calls = []

    def score(b, hkls, phi):
        calls.append((b, hkls, phi))
        return np.full_like(grid, b * (2 + np.cos(phi)))

    average, samples = azimuth_average(score, groups, angles, grid)
    np.testing.assert_allclose(samples, [[9, 9], [3, 3]])
    np.testing.assert_allclose(average, [6, 6])
    assert [call[2] for call in calls] == [0, 0, np.pi, np.pi]


@pytest.mark.parametrize("count", [6, 8])
def test_roll_angles_are_distinct_modulo_the_sixfold_period(count):
    angles = roll_angles(0.2, count)
    assert len(set(np.round((angles - 0.2) % (np.pi / 3), 12))) == count
    np.testing.assert_allclose(angles, 0.2 + (np.pi / 3) * (np.arange(count) + 0.5) / count)


def test_ladder_integrity_tolerance_follows_precision():
    record = dict(cxr_max_rel=1e-6, brem_max_rel=0.0)
    assert integrity_passes(record, fp64=False)
    assert not integrity_passes(record, fp64=True)
    assert not integrity_passes(dict(record, brem_max_rel=float("nan")), fp64=False)


def test_refuses_mismatched_populations():
    case = {"crystal": "hopg"}
    tp = {"segs": {"Ne": 3}, "Ne_lines": 2, "Ne_brem": 3}
    with pytest.raises(ValueError, match="same complete electron population"):
        validate_case(case, tp)


def test_production_domain_and_tensor_projection():
    from pyrite.materials.channeling.thermal import thermal_site
    from pyrite.materials.crystal import CRYSTALS

    basal, full = reflection_domain("hopg", 8.0)
    assert set(basal) < set(full)
    assert (0, 0, 2) in basal
    assert (1, 0, 2) in full
    assert all(tuple(-i for i in h) in full for h in full)
    assert CRYSTALS["hopg"]["texture"] == "fiber"
    site = thermal_site("hopg", "C")
    groups = tensor_groups("hopg", full, site, 300.0)
    b33 = site.b_factor_along(1.0, 300.0)
    assert set(groups[b33]) == set(basal)
    assert min(groups) == pytest.approx(site.b_factor_along(0.0, 300.0))
    assert {h for hkls in groups.values() for h in hkls} == set(full)
    assert sum(map(len, groups.values())) == len(full)


def test_ladder_bands_disjoint_and_cumulative():
    from checks.fiber_paired_1480 import ladder_bands
    from pyrite.config import FiberAverage
    from pyrite.montecarlo.spectrum.cxr.fiber import FiberCXRModel

    domains = [
        FiberCXRModel(FiberAverage(cutoff_invang=cut), 300.0).prepare() for cut in (4.0, 6.0)
    ]
    bands = ladder_bands(domains)
    seen = set()
    for domain, band in zip(domains, bands, strict=True):
        rows = band["basal"] + band["nonbasal"]
        assert all(h[:2] == (0, 0) for h, _ in band["basal"])
        assert not any(h[:2] == (0, 0) for h, _ in band["nonbasal"])
        assert seen.isdisjoint(h for h, _ in rows)
        seen |= {h for h, _ in rows}
        assert seen == set(domain.hkls)
    with pytest.raises(ValueError, match="nested"):
        ladder_bands(domains[::-1])
