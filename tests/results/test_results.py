"""results.line_metrics: the coherent_brem_ratio metric (formula + edge case).



Uses a synthetic record (a Gaussian line on a flat brem) so it stays fast and

needs no Monte-Carlo run."""

import numpy as np
import pytest
from scipy.signal import peak_widths

from cxr_mc.campaign.config import default_settings
from cxr_mc.plots import _common
from cxr_mc.results import (
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
    width = float(peak_widths(spec, [idx], rel_height=0.5)[0][0])
    half = max(round(3.0 * width / 2.0), 1)
    lo, hi = max(idx - half, 0), min(idx + half + 1, spec.size)
    expected = np.trapezoid(spec[lo:hi], E[lo:hi]) / np.trapezoid(brem[lo:hi], E[lo:hi])
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
