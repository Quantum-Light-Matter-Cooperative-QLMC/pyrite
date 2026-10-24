"""Read-time component selection, signed presentation and diagnostic adapters."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.apps.optical_diagnostics import (
    collection_convergence,
    finite_source_demo,
    inference_demo,
    load_saved_optical,
)
from pyrite.apps.radiation_evidence import radiation_evidence
from pyrite.apps.radiation_figures import (
    dtr_reference_figure,
    optical_scan_figure,
    optical_source_figure,
    saved_optical_figure,
)
from pyrite.apps.transition_radiation import cached_dtr_reference
from pyrite.plots.plotly.spectra import compare_spectrum_chart, multi_case_spectrum_chart
from pyrite.results.radiation import radiation_band_metrics, two_beam_record


def record():
    return dict(
        E_grid=np.array([100.0, 200.0, 300.0]),
        spec=np.array([1.0, 2.0, 1.0]),
        spec_transition=np.array([-4.0, -3.0, 2.0]),
        brem=np.zeros(3),
        scale=1.0,
        fwhm=0.0,
        case=dict(
            name="test",
            E0_keV=30.0,
            tilt_deg=0.0,
            tilt_azim_deg=0.0,
            thickness_ang=100.0,
            bunch_charge_pc=1.0,
            Ne_brem=4,
        ),
    )


def test_band_endpoints_signed_yields_and_population():
    r = record()
    report = radiation_band_metrics(r, (150.0, 250.0))
    assert report["yields"]["transition"] == pytest.approx(-250.0)
    assert report["physical_electrons"] == pytest.approx(6.241509074e6)
    assert report["sampled_histories"] == 4
    assert report["negative_transition_bins"] == 2
    with pytest.raises(ValueError, match="inside"):
        radiation_band_metrics(r, (50.0, 250.0))
    r["case"]["transition_radiation"] = {"model": "two-beam"}
    assert two_beam_record(r)
    assert "joint-field correction" in radiation_band_metrics(r)["basis"]


@pytest.mark.parametrize("basket", [False, True])
def test_plotly_tr_isolation_exclusion_and_signed_axis(basket):
    r = record()
    settings = SimpleNamespace(apply_detector_qe=False, convolve_with_det=False, brem_source="mc")

    def chart(**flags):
        kwargs = dict(
            include_brem=False,
            include_cxr=False,
            include_characteristic=False,
            include_channeling=False,
            sum_components=True,
            y_type="log",
            **flags,
        )
        if basket:
            return multi_case_spectrum_chart([(r, "case")], settings, **kwargs)
        return compare_spectrum_chart({"test": {30.0: r}}, settings, hue="E0_keV", **kwargs)

    isolated = chart(include_transition=True)
    assert isolated.layout.yaxis.type == "linear"
    np.testing.assert_allclose(isolated.data[0].y, r["spec_transition"] * 1000)
    excluded = chart(include_transition=False)
    np.testing.assert_array_equal(excluded.data[0].y, [np.nan] * 3)
    del r["spec_transition"]
    legacy = chart(include_transition=True)
    assert legacy is not None


def test_finite_optical_adapter_and_figures():
    band, field, reading = finite_source_demo()
    assert finite_source_demo()[0] is band
    assert np.all(np.isfinite(band.collected_photons_per_eV_per_pulse))
    assert reading.incident_joule_per_pulse > 0
    assert reading.average_incident_power_w == pytest.approx(
        reading.incident_joule_per_pulse * 1000
    )
    assert asdict(reading)["response_sha256"]
    fig = optical_source_figure(band, field)
    assert np.asarray(fig.data[1].z).shape == (128, 128)


def test_inference_saved_roundtrip_and_missing_band_refusal(tmp_path):
    scan, fit, interferogram = inference_demo()
    assert fit.rms_duration_fs == pytest.approx(200.0, rel=0.005)
    assert fit.is_identified_95
    assert fit.confidence_interval_95_fs[0] < 200 < fit.confidence_interval_95_fs[1]
    assert len(optical_scan_figure(scan, fit).data) == 7
    for obj, kind in ((fit, "Fit"), (interferogram, "Interferogram")):
        path = obj.save(tmp_path / f"{kind}.npz")
        loaded = load_saved_optical(path, kind)
        assert loaded.metadata == obj.metadata
        assert saved_optical_figure(loaded, kind).data
    _, high_fit, _ = inference_demo(200.0, 0.02)
    assert not high_fit.is_identified_95
    assert None in high_fit.confidence_interval_95_fs


@pytest.mark.slow
def test_optical_convergence_reports_independent_errors():
    report = collection_convergence()
    assert (
        set(
            (
                "pitch_relative_change",
                "window_relative_change",
                "hemisphere_fft_bias",
                "green_relative_difference",
                "converged",
            )
        )
        <= report.keys()
    )
    assert np.isfinite(report["green_relative_difference"])


@pytest.mark.slow
def test_dtr_view_reconstructs_reference_without_relabelling():
    columns, metadata = cached_dtr_reference(True)
    assert cached_dtr_reference(True)[0] is columns
    np.testing.assert_allclose(
        columns["two_beam_total"],
        columns["two_beam_transition"] + columns["two_beam_line"] + columns["two_beam_cross"],
        atol=1e-14,
    )
    fig = dtr_reference_figure(columns)
    trace = next(trace for trace in fig.data if trace.name == "two_beam_cross")
    np.testing.assert_array_equal(trace.y, columns["two_beam_cross"])
    assert "unverified" in fig.layout.title.text
    assert metadata


def test_evidence_missing_malformed_and_provenance(tmp_path):
    rows = radiation_evidence(tmp_path)
    assert all(row["status"] == "missing" for row in rows)
    (tmp_path / "reference.json").write_text('{"model": "experimental", "hash": "abc"}')
    assert radiation_evidence(tmp_path)[-1]["record"]["hash"] == "abc"
    (tmp_path / "reference.json").write_text("invalid")
    assert "unreadable" in radiation_evidence(tmp_path)[-1]["status"]


@pytest.mark.parametrize(
    "view_name",
    [
        pytest.param("Source and collection", marks=pytest.mark.slow),
        "Instrument and inference",
        "Cherenkov cone",
        "Saved outputs",
    ],
)
def test_live_optical_app_views_execute_without_implicit_transport(view_name, tmp_path):
    from pyrite.apps.optical_diagnostics_app import app

    overrides = {"view": SimpleNamespace(value=view_name)}
    if view_name == "Source and collection":
        from pyrite.apps.optical_diagnostics import ScreenConfig

        d = ScreenConfig()
        overrides["screen_form"] = SimpleNamespace(
            value=dict(
                outer=d.outer_um,
                hole=d.hole_um,
                beta=d.beta,
                duration=d.duration_fs,
                charge=d.charge_pc,
                distance=d.distance_um,
                aperture=d.aperture_um,
                low=d.low_eV,
                high=d.high_eV,
                points=d.energy_points,
                grid=d.grid,
                pitch=d.pitch_um,
                device="cpu",
            )
        )
    elif view_name == "Instrument and inference":
        overrides["infer_form"] = SimpleNamespace(
            value=dict(sigma=200.0, low=0.0005, high=0.025, mirror=1.2, nodes=61, noise=15.0)
        )
    elif view_name == "Cherenkov cone":
        pass
    else:
        path = inference_demo()[1].save(tmp_path / "fit.npz")
        overrides.update(
            load_run=SimpleNamespace(value=True),
            saved_path=SimpleNamespace(value=str(path)),
            saved_kind=SimpleNamespace(value="Fit"),
        )
    outputs, definitions = app.run(defs=overrides)
    assert outputs
    assert definitions["view"].value == view_name


def test_initial_optical_app_defers_numerical_work(monkeypatch):
    from pyrite.apps import optical_diagnostics
    from pyrite.apps.optical_diagnostics_app import app

    def forbidden(*args, **kwargs):
        raise AssertionError("initial rendering must not compute")

    for name in (
        "finite_source_demo",
        "collection_convergence",
        "inference_demo",
        "load_saved_optical",
    ):
        monkeypatch.setattr(optical_diagnostics, name, forbidden)
    _, definitions = app.run()
    assert definitions["screen_form"].value is None


def test_radiation_panel_keeps_single_energy_checkpoints_usable():
    import marimo as mo

    from pyrite.apps.analysis_ui.radiation import radiation_panel

    r = record()
    for name in ("E_grid", "spec", "spec_transition", "brem"):
        r[name] = r[name][:1]
    assert radiation_panel(mo, [r]) is not None


def test_cherenkov_cone_map_threshold_and_cone():
    """Validation: frank-tamm-homogeneous-dielectric (constant-n app view)."""
    from pyrite.apps.optical_diagnostics import cherenkov_cone_map

    panel, threshold, cone = cherenkov_cone_map(1.46, 2.5, (100.0, 1e5), 64, 90)
    assert threshold == pytest.approx(510.99895 * (1 / np.sqrt(1 - 1 / 1.46**2) - 1), rel=1e-6)
    assert cone == pytest.approx(np.rad2deg(np.arccos(1 / 1.46)))
    below = panel.energy_eV < threshold
    assert below.any() and not panel.values[:, below].any()
    # Above threshold each kinetic energy fills exactly one bin, at theta_C.
    hit = panel.values[:, ~below] > 0
    assert np.all(hit.sum(axis=0) == 1)
    assert panel.angle_deg[hit[:, -1]][0] == pytest.approx(cone, abs=1.0)
    _, none, _ = cherenkov_cone_map(1.0, 2.5, (100.0, 1e5), 8, 10)
    assert none == np.inf


def test_screen_config_validation_and_defaults():
    from pyrite.apps.optical_diagnostics import ScreenConfig

    assert ScreenConfig().window_um == 64.0
    for bad in (
        {"hole_um": 30.0},
        {"beta": 1.0},
        {"low_eV": 2.0},
        {"grid": 8},
        {"device": "tpu"},
    ):
        with pytest.raises(ValueError):
            ScreenConfig(**bad)
