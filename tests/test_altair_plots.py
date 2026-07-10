"""Guard tests for the Altair spectrum renderer (cxr_mc.plots.altair_spectra).

The renderer shares its physics/data prep with the matplotlib path; these tests
exercise only the NEW rendering layer, on synthetic records (no GPU, no
checkpoint), keeping the data small enough to stay under Vega-Lite's default
5000-row cap so ``chart.to_dict()`` serializes without touching global config.
"""

from types import SimpleNamespace

import altair as alt
import numpy as np

from cxr_mc.plots.altair_spectra import spectrum_chart, spectrum_frame


def _settings():
    # apply_detector_qe=False -> no QE table; convolve_with_det=False -> no
    # convolution; brem_source="mc" -> brem reads straight from r["brem"]. This
    # keeps _line_brem/detected_background on the pure-numpy path.
    return SimpleNamespace(
        apply_detector_qe=False,
        convolve_with_det=False,
        brem_source="mc",
    )


def _record(E0_keV, tilt_deg, azim_deg, n=200, wide_brem=False):
    E = np.linspace(1000.0, 5000.0, n)  # eV
    peak = np.exp(-(((E - 2500.0) / 50.0) ** 2))  # a single sharp line
    brem = np.linspace(1.0, 0.2, n)  # a smooth falling continuum
    rec = {
        "E_grid": E,
        "spec": peak,
        "brem": brem,
        "scale": 2.0,
        "fwhm": 30.0,
        "case": {
            "name": "HOPG bulk",
            "E0_keV": E0_keV,
            "tilt_deg": tilt_deg,
            "tilt_azim_deg": azim_deg,
            "thickness_ang": 5.0e4,
        },
    }
    if wide_brem:
        Eb = np.linspace(0.0, E0_keV * 1000.0, 120)
        rec["E_grid_brem"] = Eb
        rec["brem_wide"] = np.linspace(1.2, 0.02, Eb.size)
    return rec


def _store():
    # {name: {E0: record}} -- one polar tilt, two beam energies.
    return {
        "HOPG bulk": {
            30.0: _record(30.0, -20.0, 0.0),
            60.0: _record(60.0, -20.0, 0.0),
        }
    }


def _dataset(spec):
    return spec["datasets"][spec["data"]["name"]]


def test_spectrum_frame_shapes_and_components():
    df = spectrum_frame([_record(30.0, -20.0, 0.0), _record(60.0, -20.0, 0.0)], _settings())
    assert list(df.columns) == ["energy_eV", "intensity", "E0_keV", "azimuth_deg", "component"]
    # 2 energies x 2 components x 200 grid points
    assert len(df) == 2 * 2 * 200
    assert set(df["component"]) == {"total", "brem"}
    assert sorted(df["E0_keV"].unique()) == [30.0, 60.0]
    # total = line + brem >= brem everywhere (line is non-negative)
    assert (
        df[df.component == "total"]["intensity"].to_numpy()
        >= df[df.component == "brem"]["intensity"].to_numpy() - 1e-9
    ).all()


def test_spectrum_frame_broadband_uses_wide_brem_tail():
    df = spectrum_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings(), band="broad")

    assert df["energy_eV"].max() == 30000.0
    tail = df[(df.component == "total") & (df.energy_eV > 5000.0)]
    brem_tail = df[(df.component == "brem") & (df.energy_eV > 5000.0)]
    assert not tail.empty
    assert np.allclose(tail["intensity"].to_numpy(), brem_tail["intensity"].to_numpy())


def test_spectrum_frame_broadband_clips_tail_to_beam_energy():
    rec = _record(30.0, -20.0, 0.0, wide_brem=True)
    rec["E_grid_brem"] = np.array([0.0, 5000.0, 30000.0, 30500.0])
    rec["brem_wide"] = np.array([1.0, 0.5, 0.2, 99.0])

    df = spectrum_frame([rec], _settings(), band="broad")

    assert df["energy_eV"].max() == 30000.0
    assert 30500.0 not in set(df["energy_eV"])


def test_spectrum_frame_narrow_ignores_wide_brem_tail():
    df = spectrum_frame([_record(30.0, -20.0, 0.0, wide_brem=True)], _settings())

    assert df["energy_eV"].max() == 5000.0


def test_spectrum_frame_peak_preserving_decimation_keeps_line_peak():
    rec = _record(30.0, -20.0, 0.0, n=1000)
    df = spectrum_frame([rec], _settings(), include_brem=False, max_points=40)

    total = df[df.component == "total"]
    assert len(total) <= 40
    assert np.isclose(total["intensity"].max(), rec["spec"].max() * rec["scale"])


def test_spectrum_frame_excludes_brem_when_disabled():
    df = spectrum_frame([_record(30.0, -20.0, 0.0)], _settings(), include_brem=False)
    assert set(df["component"]) == {"total"}


def test_spectrum_chart_builds_valid_spec():
    chart = spectrum_chart(_store(), _settings())
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()  # raises if the spec is malformed / over the row cap
    enc = spec["layer"][0]["encoding"]
    assert enc["x"]["field"] == "energy_eV"
    assert enc["y"]["field"] == "intensity"
    assert enc["color"]["field"] == "E0_keV"
    # total + brem layers
    assert len(spec["layer"]) == 2


def test_spectrum_chart_accepts_log_x_scale_and_broadband():
    store = {"HOPG bulk": {30.0: _record(30.0, -20.0, 0.0, wide_brem=True)}}
    chart = spectrum_chart(
        store,
        _settings(),
        band="broad",
        x_domain=(50.0, 30000.0),
        x_type="log",
        y_type="log",
    )

    spec = chart.to_dict()
    enc = spec["layer"][0]["encoding"]
    assert enc["x"]["scale"] == {"domain": [50.0, 30000.0], "type": "log"}
    assert enc["y"]["scale"] == {"type": "log"}
    assert max(row["energy_eV"] for row in _dataset(spec)) == 30000.0


def test_spectrum_chart_single_layer_without_brem():
    chart = spectrum_chart(_store(), _settings(), include_brem=False)
    assert len(chart.to_dict()["layer"]) == 1


def test_spectrum_chart_none_on_empty():
    assert spectrum_chart({}, _settings()) is None
