"""results.line_metrics: the coherent_brem_ratio metric (formula + edge case).



Uses a synthetic record (a Gaussian line on a flat brem) so it stays fast and

needs no Monte-Carlo run."""

import numpy as np
import pytest

from pyrite.campaign.config import default_settings
from pyrite.plots import _common
from pyrite.results import (
    Settings,
    beam_current_na,
    line_metrics,
    selection_score,
    store_result,
    summary_table,
)


def _record(E, spec, brem):
    # the only fields line_metrics reads

    return {"E_grid": E, "spec": spec, "brem": brem, "scale": 1.0}


def test_pulse_charge_and_rep_rate_convert_to_average_current_na():
    """1 pC at 5 kHz is 5 nA, not 5,000 nA (pC x Hz -> pA)."""
    assert beam_current_na({"bunch_charge_pc": 1.0, "rep_rate_hz": 5000.0}) == 5.0
    assert beam_current_na({"bunch_charge_pc": 2.0, "rep_rate_hz": 10_000.0}) == 20.0


def test_new_default_record_stamps_beam_current_outside_case_identity():
    E = np.arange(50.0, 151.0, 1.0)
    case = {
        "name": "default-pulse",
        "E0_keV": 30.0,
        "theta_obs_rad": np.pi / 2,
        "dtheta_obs_rad": 0.0,
        "domega_sr": 1.0,
    }
    out = {
        "E_grid": E,
        "spec": np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2),
        "brem": np.full_like(E, 0.1),
        "eta": 1.0,
    }
    results = {}
    store_result(results, case, out)
    record = results["default-pulse"][30.0]

    assert "bunch_charge_pc" not in record["case"]
    assert "rep_rate_hz" not in record["case"]
    assert record["source_current_na"] == 5.0
    assert beam_current_na(record, Settings(beam_current_na=123.0)) == 5.0


def test_line_metrics_cache_keys_on_record_pulse_current():
    E = np.arange(50.0, 151.0, 1.0)
    records = []
    for current_na in (5.0, 20.0):
        record = _record(E, np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2), np.full_like(E, 0.1))
        record["case"] = {"name": "same", "E0_keV": 30.0}
        record["source_current_na"] = current_na
        records.append(record)
    _common._LINE_METRICS_CACHE.clear()

    first = _common._cached_line_metrics(records[0], Settings(), 0.03, "sharpness")
    second = _common._cached_line_metrics(records[1], Settings(), 0.03, "sharpness")

    assert second["coherent_flux"] == pytest.approx(4.0 * first["coherent_flux"])
    _common._LINE_METRICS_CACHE.clear()


def test_default_pulse_rate_preserves_existing_count_rates_and_reports_per_na():
    E = np.arange(50.0, 151.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)
    brem = np.full_like(E, 0.1)
    legacy = line_metrics(_record(E, spec, brem), Settings())
    pulsed_record = _record(E, spec, brem)
    pulsed_record["case"] = {"bunch_charge_pc": 1.0, "rep_rate_hz": 5000.0}
    pulsed = line_metrics(pulsed_record, Settings(beam_current_na=123.0))

    for name in ("peak_flux", "coherent_flux", "line_flux", "total_flux"):
        assert pulsed[name] == legacy[name]
        assert pulsed[f"{name}_per_na"] * 5.0 == pulsed[name]


def test_nondefault_pulse_rate_controls_count_rate_not_settings_override():
    E = np.arange(50.0, 151.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)
    brem = np.full_like(E, 0.1)
    default = _record(E, spec, brem)
    default["case"] = {"bunch_charge_pc": 1.0, "rep_rate_hz": 5000.0}
    nondefault = _record(E, spec, brem)
    nondefault["case"] = {"bunch_charge_pc": 2.0, "rep_rate_hz": 10_000.0}

    base = line_metrics(default, Settings())
    rate = line_metrics(nondefault, Settings(beam_current_na=999.0))
    assert rate["coherent_flux"] == pytest.approx(4.0 * base["coherent_flux"])
    assert rate["coherent_flux_per_na"] == base["coherent_flux_per_na"]


def test_summary_table_reports_intrinsic_and_rep_rate_count_rates():
    E = np.arange(50.0, 151.0, 1.0)
    record = _record(E, np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2), np.full_like(E, 0.1))
    record.update(
        fwhm=1.0,
        case={
            "crystal": "mose2",
            "thickness_ang": 100.0,
            "tilt_deg": 5.0,
            "tilt_azim_deg": 0.0,
            "E0_keV": 30.0,
            "bunch_charge_pc": 2.0,
            "rep_rate_hz": 10_000.0,
        },
    )
    row = summary_table([record], Settings(beam_current_na=999.0)).iloc[0]

    # Presentation rounds each column independently, so allow its displayed
    # 0.1-count precision rather than comparing the unrounded ratio exactly.
    assert row[("", "line [cts/s]")] == pytest.approx(20.0 * row[("", "line [cts/s/nA]")], abs=0.5)
    assert row[("", "total [cts/s]")] == pytest.approx(
        20.0 * row[("", "total [cts/s/nA]")], abs=0.5
    )


def test_coherent_brem_ratio_matches_integral_ratio():
    E = np.arange(50.0, 150.0, 1.0)

    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)  # a clean line

    brem = np.full_like(E, 0.1)  # flat background

    m = line_metrics(_record(E, spec, brem), default_settings())

    expected = float(np.trapezoid(spec, E)) / float(np.trapezoid(brem, E))

    assert m["coherent_brem_ratio"] == pytest.approx(expected, rel=1e-9)

    # ratio is scale/current independent -> equals coherent_flux/brem_flux too

    assert m["coherent_brem_ratio"] > 0


def test_coherent_brem_ratio_nan_without_brem():
    E = np.arange(50.0, 150.0, 1.0)

    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)

    m = line_metrics(_record(E, spec, np.zeros_like(E)), default_settings())

    assert np.isnan(m["coherent_brem_ratio"])


def test_line_brem_ratio_uses_the_dominant_line_window():
    E = np.arange(50.0, 151.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)
    brem = 0.1 + 0.01 * E
    m = line_metrics(_record(E, spec, brem), default_settings())
    idx = int(np.argmax(spec))
    half_window_eV = 3.0 * m["fwhm_eV"] / 2.0
    bounds = (E[idx] - half_window_eV, E[idx] + half_window_eV)
    inside = (E > bounds[0]) & (E < bounds[1])
    E_window = np.concatenate(([bounds[0]], E[inside], [bounds[1]]))
    spec_window = np.interp(E_window, E, spec)
    brem_window = np.interp(E_window, E, brem)
    expected = np.trapezoid(spec_window, E_window) / np.trapezoid(brem_window, E_window)
    assert m["line_brem_ratio"] == pytest.approx(expected)


@pytest.mark.parametrize("brem_level", [0.0, -0.1])
def test_line_brem_ratio_is_nan_for_nonpositive_local_brem_integral(brem_level):
    E = np.arange(50.0, 151.0, 1.0)
    spec = np.exp(-0.5 * ((E - 100.0) / 3.0) ** 2)

    m = line_metrics(_record(E, spec, np.full_like(E, brem_level)), default_settings())

    assert np.isnan(m["line_brem_ratio"])


def test_line_brem_ratio_is_a_selection_mode():
    assert selection_score({"line_brem_ratio": 2.0}, "line_brem_ratio") == 2.0
    assert selection_score({"line_brem_ratio": np.nan}, "line_brem_ratio") == -np.inf


def test_default_selection_uses_grid_independent_line_flux():
    metrics = {
        "line_flux": 4.0,
        "peak_flux": 100.0,
        "coherent_flux": 7.0,
        "line_quality": 0.5,
    }
    assert selection_score(metrics) == 2.0


def test_peak_density_is_explicit_and_records_local_sample_spacing():
    E = np.array([80.0, 90.0, 97.0, 100.0, 102.0, 110.0, 125.0])
    spec = np.exp(-0.5 * ((E - 100.0) / 4.0) ** 2)

    metrics = line_metrics(_record(E, spec, np.zeros_like(E)), default_settings())

    assert metrics["peak_spectral_flux_density"] == metrics["peak_flux"]
    assert metrics["peak_spectral_flux_density_per_na"] == metrics["peak_flux_per_na"]
    assert metrics["peak_sample_spacing_eV"] == 2.5


def test_fwhm_uses_physical_energies_on_nonuniform_grid():
    E = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0])
    sigma_eV = 4.0
    spec = np.exp(-0.5 * ((E - 11.0) / sigma_eV) ** 2)

    metrics = line_metrics(_record(E, spec, np.zeros_like(E)), default_settings())

    expected_fwhm_eV = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_eV
    assert metrics["fwhm_eV"] == pytest.approx(expected_fwhm_eV, rel=0.08)


def test_fwhm_and_line_flux_are_stable_across_resolution_ladder():
    sigma_eV = 20.0
    fwhm_values = []
    line_flux_values = []
    for spacing_eV in (12.0, 6.0, 3.0, 1.5, 0.75):
        E = np.arange(4.0, 196.0 + spacing_eV, spacing_eV)
        spec = np.exp(-0.5 * ((E - 100.0) / sigma_eV) ** 2)
        metrics = line_metrics(_record(E, spec, np.zeros_like(E)), default_settings())
        fwhm_values.append(metrics["fwhm_eV"])
        line_flux_values.append(metrics["line_flux_per_na"])

    expected_fwhm_eV = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_eV
    assert fwhm_values == pytest.approx([expected_fwhm_eV] * 5, rel=0.02)
    assert max(line_flux_values) / min(line_flux_values) < 1.02
