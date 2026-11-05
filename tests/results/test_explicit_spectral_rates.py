"""Analytic mean-current/acceptance normalization and native export contracts."""

from dataclasses import asdict, replace

import numpy as np
import pint
import pytest
from polars.testing import assert_frame_equal

from pyrite.results import CxrSpectra, QuantifiedSpatialSpectrum, Result, SpectralDensity
from pyrite.results.aperture import aperture_dataframe
from pyrite.results.directional import DirectionalSpectrum
from pyrite.units import Q_

# Exact SI elementary charge; independent of the implementation's constant import.
E_CHARGE = 1.602176634e-19


def scalar_result(normalization="source", spatial=None):
    return Result(
        cxr=CxrSpectra(SpectralDensity([100, 300], [2.0, 2.0], normalization)),
        brem=SpectralDensity([200, 400, 800], [3.0, 3.0, 3.0], normalization),
        case={"domega_sr": 0.1, "bunch_charge_pc": 0.0, "rep_rate_hz": 1000.0},
        provenance={},
        spatial=spatial,
    )


def directional():
    return DirectionalSpectrum(
        np.array([0.0, 90.0]),
        np.array([100.0, 300.0]),
        np.eye(3)[:2],
        np.eye(3)[:2],
        np.array([[2.0, 4.0], [np.nan, np.nan]]),
        np.array([True, False]),
        {"bunch_charge_pc": 2.0, "rep_rate_hz": 3000.0},
        {},
    )


def aperture():
    from pyrite.photon_observation import PhotonContinuumTally

    return PhotonContinuumTally(
        edges_eV=np.array([0.0, 100.0, 300.0]),
        total_per_primary_eV=np.array([0.5, 1.5]),
        direct_per_primary_eV=np.array([0.3, 1.0]),
        redistributed_per_primary_eV=np.array([0.2, 0.5]),
        detected_per_primary=355.0,
        unbinned_detected_per_primary=5.0,
        n_primaries=10,
        n_proposals=100,
        charged_feedback=False,
        source_identity_digest="source",
        observation_identity_digest="observation",
        provenance={},
    )


@pytest.fixture(params=["scalar", "directional", "spatial", "aperture"])
def view(request):
    if request.param == "scalar":
        return scalar_result().cxr.incoherent.quantify()
    if request.param == "directional":
        return directional().quantify()
    if request.param == "spatial":
        return QuantifiedSpatialSpectrum(
            np.array([100.0, 300.0]),
            np.array([[2.0, 4.0]]),
            np.array([[0, 1]]),
            "cxr",
            True,
            12.0,
        )
    return aperture().quantify()


def test_equivalent_current_energy_and_semantic_states(view):
    original = view.density.magnitude.copy()
    for current in (Q_(20, "nA"), Q_(0.02, "uA"), pint.UnitRegistry().Quantity(2e-8, "A")):
        rate = view.rate(current=current)
        np.testing.assert_allclose(rate.density.magnitude, original * (2e-8 / E_CHARGE))
        assert rate.normalization_basis == "per_second"
        assert rate.count_basis == "photons_per_second"
        assert rate.measure == view.measure
        assert rate.normalization == view.normalization
        assert rate.acceptance_applied == view.acceptance_applied
        converted = rate.to_energy("keV")
        np.testing.assert_allclose(converted.density.magnitude, rate.density.magnitude * 1000)
        np.testing.assert_allclose(
            view.to_energy("keV").rate(current=current).density.magnitude,
            converted.density.magnitude,
        )
        assert converted.normalization_basis == "per_second"
    np.testing.assert_array_equal(view.density.magnitude, original)


def test_zero_and_per_current_without_division(view):
    zero = view.rate(current=Q_(0, "A"))
    finite = np.isfinite(view.density.magnitude)
    np.testing.assert_array_equal(zero.density.magnitude[finite], 0)
    np.testing.assert_array_equal(np.isnan(zero.density.magnitude), ~finite)
    coefficient = view.rate_per_current()
    np.testing.assert_allclose(coefficient.density.magnitude, view.density.magnitude / E_CHARGE)
    assert coefficient.normalization_basis == "per_current"
    assert coefficient.count_basis == "photons_per_second_per_ampere"
    target = "1/keV/s/nA" + ("/sr" if view.measure.endswith("steradian") else "")
    np.testing.assert_allclose(
        coefficient.to_energy("keV").density.to(target).magnitude,
        view.density.magnitude * (1e-6 / E_CHARGE),
    )
    for transformed in (zero, coefficient):
        with pytest.raises(ValueError, match="already"):
            transformed.rate(current=Q_(1, "A"))
        with pytest.raises(ValueError, match="already"):
            transformed.rate_per_current()


@pytest.mark.parametrize(
    "current,error",
    [
        (20, TypeError),
        (None, TypeError),
        (Q_(np.array(True), "A"), TypeError),
        (Q_([1], "A"), TypeError),
        (Q_(1j, "A"), TypeError),
        (Q_(-1, "nA"), ValueError),
        (Q_(np.nan, "A"), ValueError),
        (Q_(np.inf, "A"), ValueError),
        (Q_(1, "sr"), ValueError),
        (Q_(1, "C"), ValueError),
    ],
)
def test_current_input_refusals(current, error):
    with pytest.raises(error):
        scalar_result().brem.quantify().rate(current=current)


def test_native_tables_units_and_snapshot_ownership(view):
    transformed = view.to_energy("keV").rate(current=Q_(1, "nA"))
    table = transformed.to_dataframe()
    metadata = transformed.dataframe_units()
    assert table.height == view.density.magnitude.size
    assert "total" not in table.columns
    assert table["normalization_basis"].unique().to_list() == ["per_second"]
    assert metadata["density"]["basis"] == "photons_per_second"
    assert Q_(1, metadata["density"]["unit"]).units == transformed.density.units
    metadata["density"]["basis"] = "changed"
    assert transformed.dataframe_units()["density"]["basis"] == "photons_per_second"
    expected = transformed.density.magnitude.copy()
    transformed.density.magnitude[:] = -1
    np.testing.assert_array_equal(transformed.density.magnitude, expected)
    if hasattr(view, "supported"):
        assert table["supported"].to_list() == [True, True, False, False]
    if hasattr(view, "measured"):
        assert transformed.measured and transformed.fwhm_eV == 12
        assert table["pixel_column"].to_list() == [1, 1]


def test_source_acceptance_native_grids_modes_and_integrated_total():
    result = scalar_result()
    with pytest.raises(ValueError, match="assume_uniform"):
        result.accepted_spectra()
    with pytest.raises(TypeError, match="boolean"):
        result.accepted_spectra(assume_uniform=1)
    with pytest.raises(ValueError, match="coherent"):
        result.accepted_spectra(mode="coherent", assume_uniform=True)
    with pytest.raises(ValueError, match="mode"):
        result.accepted_spectra(mode="both", assume_uniform=True)
    accepted = result.accepted_spectra(assume_uniform=True)
    assert set(accepted) == {"cxr", "brem"}
    assert accepted["cxr"].photon_energy.size == 2
    assert accepted["brem"].photon_energy.size == 3
    for name, view in accepted.items():
        source = result.cxr.incoherent if name == "cxr" else result.brem
        np.testing.assert_allclose(view.density.magnitude, source.density * 0.1)
        assert view.measure == "per_energy" and view.acceptance_applied
        assert view.accepted_solid_angle_sr == 0.1
        assert view.normalization == "source"
        assert not hasattr(view, "accepted_spectra")
    # Native component integrals are 400 and 1800 per primary/sr, not a union-grid total.
    yields = result.integrated_yield()
    assert yields["total"].inside == pytest.approx(2200)
    integrated_rate = yields["total"].inside * 0.1 * (20e-9 / E_CHARGE)
    table_integrals = sum(
        np.trapezoid(
            s.rate(current=Q_(20, "nA")).to_energy("keV").density.magnitude,
            s.to_energy("keV").photon_energy.magnitude,
        )
        for s in accepted.values()
    )
    assert table_integrals == pytest.approx(integrated_rate)


def test_bin_mean_edges_and_mixed_sampling_are_not_reinterpreted():
    component = SpectralDensity(
        [50, 200], [0.5, 1.5], sampling="bin-mean", bin_edges_eV=[0, 100, 300]
    )
    result = Result(
        cxr=CxrSpectra(component), brem=scalar_result().brem, case={"domega_sr": 0.1}, provenance={}
    )
    view = result.accepted_spectra(assume_uniform=True)["cxr"].to_energy("keV")
    table = view.to_dataframe()
    assert table["photon_energy_left"].to_list() == [0, 0.1]
    assert table["photon_energy_right"].to_list() == [0.1, 0.3]
    assert table["sampling"].to_list() == ["bin-mean"] * 2
    assert np.sum(
        view.density.magnitude * np.diff(view.photon_energy_edges.magnitude)
    ) == pytest.approx(35)
    mixed = SpectralDensity([50, 200], [1, 1], sampling="mixed")
    assert mixed.quantify().rate_per_current().sampling == "mixed"
    with pytest.raises(ValueError, match="mixed"):
        mixed.integrate()


def test_legacy_quantity_current_paths_preserve_return_types_and_schemas():
    source = directional()
    np.testing.assert_allclose(source.rate(current=Q_(6, "nA")), source.rate(current_na=6))
    np.testing.assert_allclose(source.rate(), source.rate(current=Q_(6, "nA")))
    with pytest.raises(TypeError, match="only one"):
        source.rate(6, current=Q_(6, "nA"))
    tally = aperture()
    before = asdict(tally)
    frame = aperture_dataframe(tally, current=Q_(20, "nA"))
    assert_frame_equal(
        frame, aperture_dataframe(tally, current_nA=20), check_exact=False, rel_tol=1e-14
    )
    with pytest.raises(TypeError, match="only one"):
        aperture_dataframe(tally, current_nA=20, current=Q_(20, "nA"))
    scaled = tally.quantify().rate(current=Q_(20, "nA")).to_energy("keV")
    np.testing.assert_allclose(scaled.counts.magnitude, np.array([50, 300]) * 20e-9 / E_CHARGE)
    assert scaled.unbinned_counts.magnitude == pytest.approx(5 * 20e-9 / E_CHARGE)
    assert scaled.detected_counts.magnitude == pytest.approx(355 * 20e-9 / E_CHARGE)
    np.testing.assert_array_equal(tally.total_per_primary_eV, before["total_per_primary_eV"])
    assert tally.source_identity_digest == before["source_identity_digest"]


def test_observation_acceptance_and_measured_spectra_never_reapply_response(monkeypatch):
    from pyrite.instrument import PixelGrid, PlanarDetector, PlanarPose
    from pyrite.results import PixelRayMap, SpatialResult, SpectralFactors

    class HalfResponse:
        def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
            return intrinsic_density * 0.5 * scale

    rays = PixelRayMap(
        tile_index=np.array([[0, 1]]),
        solid_angle_sr=np.array([[0.1, 0.2]]),
        path_length_mm=np.array([[[0.0], [1.0]]]),
    )
    factors = SpectralFactors(
        np.array([100.0, 300.0]), np.array([[1.0, 1.0], [2.0, 2.0]]), np.full((1, 2), np.log(2.0))
    )
    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 10.0)),
        pixels=PixelGrid((1, 2), (1.0, 1.0)),
        response=HalfResponse(),
    )
    spatial = SpatialResult(rays, factors, factors, detector)
    result = scalar_result("observation-average", spatial)
    average = SpectralDensity([100, 300], [1.0, 1.0], "observation-average")
    result = replace(result, cxr=CxrSpectra(average))
    measured = spatial.quantify_spectra(pixels=[(0, 0), (0, 1)], measured=True)
    # Pixel 1 has twice the source density, but half transmission; response halves both.
    np.testing.assert_allclose(measured.density.magnitude, [[0.05, 0.05], [0.1, 0.1]])

    def refuse(*args, **kwargs):
        raise AssertionError("current/acceptance transformation must not score")

    monkeypatch.setattr(SpatialResult, "spectra", refuse)
    monkeypatch.setattr(HalfResponse, "score", refuse)
    accepted = result.accepted_spectra()["cxr"]
    np.testing.assert_allclose(accepted.density.magnitude, [0.3, 0.3])
    assert accepted.normalization == "observation-average"
    assert accepted.accepted_solid_angle_sr == pytest.approx(0.3)  # case carries .1
    rate = measured.rate(current=Q_(1, "nA")).to_energy("keV")
    np.testing.assert_allclose(rate.density.magnitude, measured.density.magnitude * 1e-6 / E_CHARGE)
    assert rate.measured and rate.normalization == "pixel-accepted"
    with pytest.raises(ValueError, match="planar detector"):
        scalar_result("observation-average").accepted_spectra()


def test_acquisition_counts_cannot_be_current_normalized():
    from pyrite.results import QuantifiedAcquisitionBatch

    assert not hasattr(QuantifiedAcquisitionBatch, "rate")
    assert not hasattr(QuantifiedAcquisitionBatch, "rate_per_current")


def test_legacy_selector_warning_window_preserves_numeric_tables():
    from pyrite._api_deprecations import API_DEPRECATIONS

    result = scalar_result()
    for name in ("spectral_flux", "spectral_flux_units"):
        entry = API_DEPRECATIONS[f"Result.{name}"]
        assert (entry.deprecated_in, entry.remove_in) == ("0.26.1", "0.28.0")
    with pytest.warns(DeprecationWarning, match="0.28.0"):
        frame = result.spectral_flux(units="electron")
    with pytest.warns(DeprecationWarning, match="0.28.0"):
        metadata = result.spectral_flux_units(units="electron")
    assert frame.columns == ["photon_energy_eV", "photon_energy_keV", "cxr", "brem", "total"]
    assert list(metadata) == frame.columns
    np.testing.assert_array_equal(frame["photon_energy_eV"], [100, 200, 300, 400, 800])
    np.testing.assert_array_equal(frame["total"], [2, 5, 5, 3, 3])
    assert result.to_dataframe().columns == [
        "component",
        "mode",
        "photon_energy_eV",
        "density",
        "normalization",
    ]


def test_scalar_snapshot_refuses_hidden_device_conversion():
    from pyrite.results import QuantifiedSpectrum

    class DeviceArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError("must refuse device conversion before materializing")

    with pytest.raises(TypeError, match="CPU"):
        QuantifiedSpectrum(np.array([100.0, 300.0]), DeviceArray(), "source")


def test_native_result_serialization_and_identity_are_unchanged():
    import pickle

    result = scalar_result()
    before = pickle.dumps(result)
    reopened = pickle.loads(before)
    reopened.accepted_spectra(assume_uniform=True)["cxr"].rate(current=Q_(20, "nA")).to_dataframe()
    reopened.brem.quantify().rate_per_current().to_energy("keV").dataframe_units()
    assert pickle.dumps(reopened) == before


@pytest.mark.parametrize("mode", [None, "both", "other"])
def test_accepted_spectra_requires_one_explicit_cxr_mode(mode):
    with pytest.raises(ValueError, match="mode"):
        scalar_result().accepted_spectra(mode=mode, assume_uniform=True)


def test_quantity_directional_current_refuses_nonprimary_units():
    with pytest.raises(ValueError, match="per-primary"):
        replace(directional(), units="ph/s/eV/sr").rate(current=Q_(1, "nA"))


def test_current_rate_factor_must_be_representable():
    with pytest.raises(ValueError, match="representable"):
        scalar_result().brem.quantify().rate(current=Q_(1e300, "A"))


def test_steradian_dimensionlessness_cannot_change_semantic_measure():
    source = scalar_result().brem.quantify()
    # Pint permits dropping sr; the spectrum must still declare a directional measure.
    source.density.to("1/eV")
    assert source.measure == "per_energy_per_steradian" and not source.acceptance_applied
    with pytest.raises(ValueError, match="acceptance"):
        replace(source, measure="per_energy")
    with pytest.raises(ValueError, match="acceptance"):
        replace(source, acceptance_applied=True)


@pytest.mark.parametrize("solid_angle", [0, -1, np.inf, np.nan])
def test_scalar_acceptance_refuses_nonpositive_or_nonfinite_solid_angle(solid_angle):
    result = replace(scalar_result(), case={"domega_sr": solid_angle})
    with pytest.raises(ValueError, match="solid angle"):
        result.accepted_spectra(assume_uniform=True)


def test_accepted_tables_retain_selected_cxr_mode_without_summing_alternatives():
    result = scalar_result()
    result = replace(
        result, cxr=CxrSpectra(result.cxr.incoherent, SpectralDensity([100, 300], [7.0, 7.0]))
    )
    view = result.accepted_spectra(mode="coherent", assume_uniform=True)["cxr"]
    table = view.rate(current=Q_(1, "nA")).to_energy("keV").to_dataframe()
    assert table["component"].unique().to_list() == ["cxr"]
    assert table["mode"].unique().to_list() == ["coherent"]
    np.testing.assert_allclose(view.density.magnitude, [0.7, 0.7])
    assert view.normalization == "source"
