"""#954: support-aware yields and explicit conservative resampling.

Every expected value below is an analytic integral, written independently of
the implementation's cumulative-sum formulation.

Validation: spectral-support-yield-conservation
"""

import numpy as np
import pytest

from pyrite.api import _line_sampling
from pyrite.instrument import PixelGrid, PlanarDetector, PlanarPose
from pyrite.observations.plan import PixelSampling
from pyrite.results import (
    BinnedSpectrum,
    CxrSpectra,
    QuantifiedDirectionalSpectrum,
    QuantifiedSpatialSpectrum,
    Result,
    SpectralDensity,
    SpectralYield,
)
from pyrite.results.directional import DirectionalSpectrum
from pyrite.results.model import PixelRayMap, SpatialResult, SpectralFactors
from pyrite.results.spectral_support import (
    bin_mean_yield,
    combine_yields,
    interpolate_nodes,
    node_yield,
    nodes_to_bins,
    rebin_bin_means,
)
from pyrite.units import ureg

CASE = {"domega_sr": 0.01, "bunch_charge_pc": 2.0, "rep_rate_hz": 1.0e4}


def _fixture() -> Result:
    return Result(
        cxr=CxrSpectra(SpectralDensity([100.0, 300.0], [2.0, 4.0])),
        brem=SpectralDensity([100.0, 200.0, 400.0], [1.0, 1.0, 1.0]),
        characteristic=SpectralDensity([100.0, 300.0], [0.5, 0.5]),
        case=CASE,
        provenance={},
    )


def test_native_component_sum_is_1000_and_legacy_union_table_stays_1225():
    result = _fixture()
    yields = result.integrated_yield()
    # Trapezoids: 200 * (2 + 4) / 2, 200 * 0.5, 300 * 1.
    assert yields["cxr"].total == pytest.approx(600.0, rel=1e-15)
    assert yields["characteristic"].total == pytest.approx(100.0, rel=1e-15)
    assert yields["brem"].total == pytest.approx(300.0, rel=1e-15)
    assert yields["total"].inside == pytest.approx(1000.0, rel=1e-15)
    assert yields["total"].support_eV == ((100.0, 400.0),)
    assert yields["total"].complete
    # The legacy union table zero-fills beyond each family's support; the extra
    # 225 are the exterior triangles 100 * 4 / 2 (cxr) and 100 * 0.5 / 2.
    table = result.spectral_flux(units="electron")
    legacy = np.trapezoid(table["total"].to_numpy(), table["photon_energy_eV"].to_numpy())
    assert legacy == pytest.approx(1225.0, rel=1e-15)
    np.testing.assert_allclose(table["total"], [3.5, 4.5, 5.5, 1.0])


def test_nonzero_endpoints_and_clipped_window_split_mass_exactly():
    # f(E) = 3 + 2E on [1, 4]: F(a, b) = 3(b - a) + (b^2 - a^2).
    energy = np.array([1.0, 2.5, 4.0])
    density = 3.0 + 2.0 * energy

    def exact(a, b):
        return 3.0 * (b - a) + (b * b - a * a)

    whole = node_yield(energy, density)
    assert whole.total == pytest.approx(exact(1.0, 4.0))
    assert whole.window_eV is None and whole.complete
    clipped = node_yield(energy, density, (1.7, 3.2))
    assert clipped.inside == pytest.approx(exact(1.7, 3.2))
    assert clipped.below == pytest.approx(exact(1.0, 1.7))
    assert clipped.above == pytest.approx(exact(3.2, 4.0))
    beyond = node_yield(energy, density, (0.0, 5.0))
    assert beyond.inside == pytest.approx(exact(1.0, 4.0))
    assert beyond.below == beyond.above == 0.0
    # The support does not establish zero emission beyond it.
    assert beyond.uncovered_eV == ((0.0, 1.0), (4.0, 5.0))
    assert not beyond.complete
    outside = node_yield(energy, density, (6.0, 7.0))
    assert outside.inside == 0.0
    assert outside.below == pytest.approx(exact(1.0, 4.0))
    assert outside.uncovered_eV == ((6.0, 7.0),)


def test_disjoint_supports_sum_native_integrals_and_report_the_gap():
    low = node_yield([0.0, 1.0], [2.0, 2.0], (0.0, 4.0))
    high = node_yield([3.0, 4.0], [1.0, 3.0], (0.0, 4.0))
    total = combine_yields([low, high])
    assert total.inside == pytest.approx(2.0 + 2.0)
    assert total.support_eV == ((0.0, 1.0), (3.0, 4.0))
    assert total.uncovered_eV == ((1.0, 3.0),)
    with pytest.raises(ValueError, match="share one energy window"):
        combine_yields([low, node_yield([3.0, 4.0], [1.0, 3.0])])


def test_nonuniform_node_to_bin_integration_is_exact_and_conservative():
    # Piecewise-linear source on a nonuniform grid; target bins straddle nodes.
    energy = np.array([10.0, 11.0, 15.0, 30.0, 31.0])
    density = np.array([0.0, 5.0, 1.0, 1.0, 4.0])
    edges = np.array([12.0, 14.0, 20.0, 30.5])
    binned = nodes_to_bins(energy, density, edges, boundary="raise")

    def f(x):
        return np.interp(x, energy, density)

    def exact(a, b):
        # Composite Simpson is exact for a linear piece; split at every node.
        points = np.unique(np.concatenate(([a, b], energy[(energy > a) & (energy < b)])))
        lo, hi = points[:-1], points[1:]
        return float(np.sum((hi - lo) / 6.0 * (f(lo) + 4.0 * f(0.5 * (lo + hi)) + f(hi))))

    np.testing.assert_allclose(
        binned.mass, [exact(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    )
    assert binned.below == pytest.approx(exact(10.0, 12.0))
    assert binned.above == pytest.approx(exact(30.5, 31.0))
    assert binned.total == pytest.approx(node_yield(energy, density).total)
    np.testing.assert_allclose(binned.mean_density, binned.mass / np.diff(edges))


def test_clip_boundary_reports_partial_coverage_as_nan_not_zero():
    binned = nodes_to_bins([1.0, 3.0], [2.0, 2.0], [0.0, 2.0, 3.0, 4.0], boundary="clip")
    np.testing.assert_allclose(binned.mass, [2.0, 2.0, 0.0])
    np.testing.assert_allclose(binned.supported_width_eV, [1.0, 1.0, 0.0])
    mean = binned.mean_density
    assert np.isnan(mean[0]) and np.isnan(mean[2]) and mean[1] == 2.0
    with pytest.raises(ValueError, match="beyond the declared support"):
        nodes_to_bins([1.0, 3.0], [2.0, 2.0], [0.0, 2.0], boundary="raise")
    with pytest.raises(ValueError, match="boundary must be"):
        nodes_to_bins([1.0, 3.0], [2.0, 2.0], [1.5, 2.0], boundary="zero")  # type: ignore[arg-type]


def test_bin_mean_overlap_rebinning_conserves_mass_with_cropped_tails():
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    means = np.array([4.0, 1.0, 2.0])  # masses 4, 2, 6
    binned = rebin_bin_means(edges, means, [0.5, 2.0, 5.0], boundary="raise")
    np.testing.assert_allclose(binned.mass, [0.5 * 4 + 1 * 1, 1 * 1 + 2 * 2])
    assert binned.below == pytest.approx(2.0)
    assert binned.above == pytest.approx(2.0)
    assert binned.total == pytest.approx(12.0)
    window = bin_mean_yield(edges, means, (0.5, 5.0))
    assert window.inside == pytest.approx(float(binned.mass.sum()))
    # A window inside one bin takes its overlap fraction under the bin-mean assumption.
    assert bin_mean_yield(edges, means, (3.5, 4.0)).inside == pytest.approx(1.0)
    # Finer target bins inherit their parent mean; no structure is invented.
    fine = rebin_bin_means(edges, means, np.linspace(0.0, 6.0, 13), boundary="raise")
    np.testing.assert_allclose(fine.mean_density, np.repeat(means, [2, 4, 6]))
    disjoint = rebin_bin_means(edges, means, [7.0, 8.0], boundary="clip")
    assert disjoint.mass.tolist() == [0.0] and disjoint.below == pytest.approx(12.0)


def test_spectral_density_dispatches_on_declared_sampling_and_keeps_native_values():
    edges = np.array([0.0, 150.0, 250.0, 400.0])
    values = np.array([1.0, 10.0, 1.0])
    line = SpectralDensity([100.0, 200.0, 300.0], values, sampling="bin-mean", bin_edges_eV=edges)
    # Bin-mean mass 150 + 1000 + 150; the node trapezoid would give 1100.
    assert line.integrate().total == pytest.approx(1300.0)
    assert line.integrate().support_eV == ((0.0, 400.0),)
    rebinned = line.rebin([0.0, 200.0, 400.0], boundary="raise")
    assert isinstance(rebinned, BinnedSpectrum)
    np.testing.assert_allclose(rebinned.mass, [650.0, 650.0])
    np.testing.assert_array_equal(line.density, values)
    np.testing.assert_array_equal(line.bin_edges_eV, edges)
    with pytest.raises(ValueError, match="not node-sampled"):
        line.interpolate([150.0], outside="raise")
    with pytest.raises(ValueError, match="requires its native bin_edges_eV"):
        SpectralDensity([1.0, 2.0], [1.0, 1.0], sampling="bin-mean")
    with pytest.raises(ValueError, match="only meaningful for bin-mean"):
        SpectralDensity([1.0, 2.0], [1.0, 1.0], bin_edges_eV=[0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match="one more entry"):
        SpectralDensity([1.0, 2.0], [1.0, 1.0], sampling="bin-mean", bin_edges_eV=[0.5, 2.5])


def test_mixed_combined_cxr_refuses_one_quadrature_but_result_yield_sums_parts():
    energy = np.array([100.0, 200.0, 300.0])
    edges = np.array([50.0, 150.0, 250.0, 350.0])
    result = Result(
        cxr=CxrSpectra(SpectralDensity(energy, [1.0, 1.0, 1.0])),
        brem=SpectralDensity(energy, [0.0, 0.0, 0.0]),
        characteristic=SpectralDensity(
            energy, [0.0, 2.0, 0.0], sampling="bin-mean", bin_edges_eV=edges
        ),
        case=CASE,
        provenance={},
    )
    combined = result.combined_cxr()
    assert combined.sampling == "mixed"
    np.testing.assert_allclose(combined.density, [1.0, 3.0, 1.0])
    with pytest.raises(ValueError, match="mixed"):
        combined.integrate()
    yields = result.integrated_yield()
    assert yields["cxr"].total == pytest.approx(200.0)
    assert yields["characteristic"].total == pytest.approx(200.0)
    assert yields["total"].inside == pytest.approx(400.0)
    assert yields["total"].support_eV == ((50.0, 350.0),)
    only_lines = result.combined_cxr(include_characteristic=False)
    assert only_lines.sampling == "nodes"


@pytest.mark.parametrize(
    ("energy", "density", "message"),
    [
        ([], [], "at least 2"),
        ([1.0], [1.0], "at least 2"),
        ([1.0, 1.0, 2.0], [1.0, 1.0, 1.0], "strictly increasing"),
        ([2.0, 1.0], [1.0, 1.0], "strictly increasing"),
        ([1.0, np.inf], [1.0, 1.0], "NaN or infinite"),
        ([1.0, 2.0], [1.0, np.nan], "never zeroed"),
        ([1.0, 2.0], np.ma.masked_array([1.0, 2.0], mask=[False, True]), "masked"),
    ],
)
def test_undeclared_or_invalid_values_are_refused_never_zeroed(energy, density, message):
    with pytest.raises(ValueError, match=message):
        node_yield(energy, density)


def test_masked_density_is_refused_at_construction():
    with pytest.raises(ValueError, match="masked"):
        SpectralDensity([1.0, 2.0], np.ma.masked_array([1.0, 2.0], mask=[True, False]))


def test_interpolation_is_pointwise_and_marks_out_of_support():
    energy, density = [0.0, 1.0, 2.0], [0.0, 10.0, 0.0]
    np.testing.assert_allclose(interpolate_nodes(energy, density, [0.5], outside="raise"), [5.0])
    marked = interpolate_nodes(energy, density, [-1.0, 1.0, 3.0], outside="nan")
    assert np.isnan(marked[0]) and marked[1] == 10.0 and np.isnan(marked[2])
    with pytest.raises(ValueError, match="beyond the declared support"):
        interpolate_nodes(energy, density, [3.0], outside="raise")
    # Coarse targets do not conserve: the line's mass is 10, the coarse trapezoid 0.
    coarse = interpolate_nodes(energy, density, [0.0, 2.0], outside="raise")
    assert np.trapezoid(coarse, [0.0, 2.0]) == 0.0
    assert node_yield(energy, density).total == pytest.approx(10.0)


def test_energy_conversion_preserves_node_and_bin_mean_integrals():
    edges = np.array([0.0, 150.0, 250.0, 400.0])
    line = SpectralDensity([100.0, 200.0, 300.0], [1.0, 10.0, 1.0], "source", "bin-mean", edges)
    ev = line.quantify()
    kev = ev.to_energy("keV")
    assert kev.sampling == "bin-mean"
    mass_kev = float(np.sum(kev.density.magnitude * np.diff(kev.photon_energy_edges.magnitude)))
    assert mass_kev == pytest.approx(line.integrate().total)
    np.testing.assert_allclose(kev.photon_energy_edges.to("eV").magnitude, edges)
    nodes = SpectralDensity([200.0, 400.0, 1000.0], [1.0, 3.0, 2.0]).quantify().to_energy("keV")
    assert nodes.sampling == "nodes"
    with pytest.raises(ValueError, match="only bin-mean"):
        _ = nodes.photon_energy_edges
    assert np.trapezoid(nodes.density.magnitude, nodes.photon_energy.magnitude) == pytest.approx(
        node_yield([200.0, 400.0, 1000.0], [1.0, 3.0, 2.0]).total
    )


def test_producers_declare_bin_mean_lines_on_clamped_midpoint_edges():
    energy = np.array([10.0, 40.0, 50.0])
    cxr, characteristic = _line_sampling({}, energy)
    assert cxr == {}
    assert characteristic["sampling"] == "bin-mean"
    # Midpoints with the low edge clamped at 0 eV (10 - 15 < 0).
    np.testing.assert_allclose(characteristic["bin_edges_eV"], [0.0, 25.0, 45.0, 55.0])
    cxr, _ = _line_sampling({"line_quadrature": "bin-mean"}, energy)
    assert cxr["sampling"] == "bin-mean"


def test_yield_types_are_frozen_values():
    result = node_yield([0.0, 1.0], [1.0, 1.0])
    assert isinstance(result, SpectralYield)
    with pytest.raises(AttributeError):
        result.inside = 2.0  # type: ignore[misc]
    binned = nodes_to_bins([0.0, 1.0], [1.0, 1.0], [0.0, 1.0], boundary="raise")
    with pytest.raises(ValueError):
        binned.mass[0] = 3.0


# -- planar and directional quantified views (#959) ----------------------------


ENERGY = np.array([10.0, 40.0, 50.0])
EDGES = np.array([0.0, 25.0, 45.0, 55.0])


def _factor(**sampling) -> SpectralFactors:
    return SpectralFactors(ENERGY, np.ones((1, 3)), np.zeros((0, 3)), **sampling)


def _spatial(**factors) -> SpatialResult:
    ray_map = PixelRayMap(
        tile_index=np.array([[0]]),
        solid_angle_sr=np.array([[0.5]]),
        path_length_mm=np.zeros((1, 1, 0)),
    )
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 10.0)),
        pixels=PixelGrid((1, 1), (1.0, 1.0)),
        response=_ScaleResponse(),
    )
    return SpatialResult(ray_map, factors.pop("cxr", _factor()), _factor(), detector, **factors)


class _ScaleResponse:
    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        return np.asarray(intrinsic_density) * (0.5 * scale)


BIN_MEAN = {"sampling": "bin-mean", "bin_edges_eV": EDGES}


def test_pixel_views_report_declared_sampling_and_convert_edges():
    spatial = _spatial(characteristic=_factor(**BIN_MEAN), channeling=_factor())
    characteristic = spatial.quantify_spectra(pixels=[(0, 0)], component="characteristic")
    assert characteristic.sampling == "bin-mean"
    assert characteristic.photon_energy_edges.units == ureg.eV
    np.testing.assert_array_equal(characteristic.photon_energy_edges.magnitude, EDGES)
    kev = characteristic.to_energy("keV")
    assert kev.sampling == "bin-mean"
    np.testing.assert_allclose(kev.photon_energy_edges.magnitude, EDGES / 1e3)
    np.testing.assert_allclose(
        np.sum(kev.density.magnitude * np.diff(kev.photon_energy_edges.magnitude), axis=1),
        np.sum(characteristic.density.magnitude * np.diff(EDGES), axis=1),
    )
    for component in ("cxr", "brem", "channeling"):
        view = spatial.quantify_spectra(pixels=[(0, 0)], component=component)
        assert view.sampling == "nodes"
        with pytest.raises(ValueError, match="only bin-mean"):
            _ = view.photon_energy_edges


def test_combined_pixel_view_matches_result_combination_rule():
    mixed = _spatial(characteristic=_factor(**BIN_MEAN))
    view = mixed.quantify_spectra(pixels=[(0, 0)], component="cxr_total")
    assert view.sampling == "mixed"
    with pytest.raises(ValueError, match="only bin-mean"):
        _ = view.photon_energy_edges
    # Equal native edges still integrate as one bin-mean family, as in combined_cxr.
    uniform = _spatial(cxr=_factor(**BIN_MEAN), characteristic=_factor(**BIN_MEAN))
    assert uniform.quantify_spectra(pixels=[(0, 0)], component="cxr").sampling == "bin-mean"
    total = uniform.quantify_spectra(pixels=[(0, 0)], component="cxr_total")
    assert total.sampling == "bin-mean"
    np.testing.assert_array_equal(total.photon_energy_edges.magnitude, EDGES)
    other = _factor(sampling="bin-mean", bin_edges_eV=EDGES + np.array([0.0, 1.0, 1.0, 0.0]))
    disagree = _spatial(cxr=_factor(**BIN_MEAN), characteristic=other)
    assert disagree.quantify_spectra(pixels=[(0, 0)], component="cxr_total").sampling == "mixed"
    # No extras: a total is its base component.
    assert _spatial().quantify_spectra(pixels=[(0, 0)], component="cxr_total").sampling == "nodes"


def test_sampling_labels_do_not_change_pixel_values():
    plain = _spatial(characteristic=_factor())
    labeled = _spatial(characteristic=_factor(**BIN_MEAN))
    for component in ("characteristic", "cxr_total"):
        for measured in (False, True):
            for left, right in zip(
                plain.spectra(pixels=[(0, 0)], component=component, measured=measured),
                labeled.spectra(pixels=[(0, 0)], component=component, measured=measured),
                strict=True,
            ):
                np.testing.assert_array_equal(left, right)


def test_measured_pixel_view_is_node_sampled_without_native_edges():
    spatial = _spatial(characteristic=_factor(**BIN_MEAN))
    view = spatial.quantify_spectra(pixels=[(0, 0)], component="characteristic", measured=True)
    assert view.measured is True
    assert view.sampling == "nodes"
    with pytest.raises(ValueError, match="only bin-mean"):
        _ = view.photon_energy_edges
    assert view.to_energy("keV").sampling == "nodes"
    with pytest.raises(ValueError, match="node-sampled"):
        QuantifiedSpatialSpectrum(
            ENERGY,
            np.ones((1, 3)),
            np.zeros((1, 2), dtype=int),
            "characteristic",
            True,
            sampling="bin-mean",
            _bin_edges_eV=EDGES,
        )


@pytest.mark.parametrize(
    ("sampling", "edges", "message"),
    [
        ("bin-mean", None, "requires"),
        ("nodes", EDGES, "requires"),
        ("mixed", EDGES, "requires"),
        ("bin-mean", EDGES[:-1], "one more entry"),
        ("bin_means", None, "sampling must be"),
    ],
)
def test_sampling_carriers_refuse_inconsistent_declarations(sampling, edges, message):
    with pytest.raises(ValueError, match=message):
        _factor(sampling=sampling, bin_edges_eV=edges)
    with pytest.raises(ValueError, match=message):
        _directional(sampling=sampling, bin_edges_eV=edges)
    with pytest.raises(ValueError, match=message):
        QuantifiedSpatialSpectrum(
            ENERGY,
            np.ones((1, 3)),
            np.zeros((1, 2), dtype=int),
            "cxr",
            False,
            sampling=sampling,
            _bin_edges_eV=edges,
        )
    with pytest.raises(ValueError, match=message):
        QuantifiedDirectionalSpectrum(
            np.zeros(1),
            ENERGY,
            np.ones((1, 3)),
            np.ones(1, dtype=bool),
            "characteristic",
            sampling=sampling,
            _bin_edges_eV=edges,
        )


def _directional(**sampling) -> DirectionalSpectrum:
    return DirectionalSpectrum(
        np.zeros(1),
        ENERGY,
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.ones((1, 3)),
        np.ones(1, dtype=bool),
        {},
        {},
        **sampling,
    )


def test_directional_view_keeps_declared_sampling_and_converts_edges():
    view = _directional(component="characteristic", **BIN_MEAN).quantify()
    assert view.sampling == "bin-mean"
    kev = view.to_energy("keV")
    assert kev.sampling == "bin-mean"
    np.testing.assert_allclose(kev.photon_energy_edges.magnitude, EDGES / 1e3)
    np.testing.assert_allclose(
        np.sum(kev.density.magnitude * np.diff(kev.photon_energy_edges.magnitude), axis=1),
        np.sum(view.density.magnitude * np.diff(EDGES), axis=1),
    )
    node = _directional(component="bremsstrahlung").quantify()
    assert node.sampling == "nodes"
    with pytest.raises(ValueError, match="only bin-mean"):
        _ = node.photon_energy_edges


@pytest.mark.parametrize(
    ("line_quadrature", "cxr_sampling"), [(None, "nodes"), ("bin-mean", "bin-mean")]
)
def test_planar_producer_declares_factor_sampling(line_quadrature, cxr_sampling):
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 10.0)), size_mm=(1.0, 1.0), pixels=PixelGrid((1, 1), (1.0, 1.0))
    )
    sampling = PixelSampling.of(detector, (1, 1))
    output = {
        "E_grid": ENERGY,
        "E_grid_brem": ENERGY,
        "spec_by_direction": np.ones((1, 3)),
        "brem_wide_by_direction": np.ones((1, 3)),
        "spec_characteristic_by_direction": np.ones((1, 3)),
        "spec_channeling_by_direction": np.ones((1, 3)),
        "spec_transition_by_direction": np.ones((1, 3)),
        "spec_coherent_by_direction": np.ones((1, 3)),
    }
    spatial = sampling.spatial(
        output, detector=detector, filters=(), line_quadrature=line_quadrature
    )
    assert spatial.cxr.sampling == cxr_sampling
    assert spatial.characteristic.sampling == "bin-mean"
    np.testing.assert_array_equal(spatial.characteristic.bin_edges_eV, EDGES)
    for name in ("brem", "coherent_cxr", "channeling", "transition"):
        assert getattr(spatial, name).sampling == "nodes"
    if line_quadrature:
        np.testing.assert_array_equal(spatial.cxr.bin_edges_eV, EDGES)
    assert spatial.quantify_spectra(pixels=[(0, 0)], component="cxr").sampling == cxr_sampling
    # Channeling and transition are node-sampled, so this total is always mixed.
    total = spatial.quantify_spectra(pixels=[(0, 0)], component="cxr_total")
    assert total.sampling == "mixed"
