"""Fast unit tests for checks/anchor_figures.py (the P1 #2 validation figures).

The heavy MC figure run lives in checks/; here we test only the cheap, pure
pieces -- theory anchors, the reference-CSV loader, series matching, the
single-segment lineshape-normalization anchor, and that the figure builders
return Figures on synthetic data -- so the suite stays CPU-only and fast.
"""

import sys
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless; no display in CI

# anchor_figures lives in checks/, not on the package path -- add it.
_CHECKS = Path(__file__).resolve().parent.parent / "checks"
if str(_CHECKS) not in sys.path:
    sys.path.insert(0, str(_CHECKS))

import anchor_figures as af  # noqa: E402


@pytest.fixture(scope="module")
def anchor():
    return af.ZhaiAnchor()


def test_line_energy_matches_dispersion(anchor):
    """line_energy_eV reproduces E = hbar c beta g / (1 - beta cos theta)."""
    from cxr_mc.materials.crystal import (
        CRYSTALS,
        HBARC_EV_ANG,
        reciprocal_g_vector,
    )
    from cxr_mc.montecarlo import beta_from_keV

    info = CRYSTALS[anchor.crystal]
    _, g = reciprocal_g_vector(anchor.hkl, info["lattice"])
    for E0 in anchor.energies_keV:
        beta = beta_from_keV(E0)
        expect = HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(anchor.theta_obs_rad))
        assert af.line_energy_eV(anchor, E0) == pytest.approx(expect, rel=1e-12)


def test_line_energies_monotonic_and_in_window(anchor):
    lines = af.theory_line_energies(anchor)
    assert set(lines) == set(anchor.energies_keV)
    vals = [lines[E0] for E0 in anchor.energies_keV]
    assert vals == sorted(vals)  # energy rises with beam energy
    assert all(anchor.e_min_eV < v < anchor.e_max_eV for v in vals)


def test_reference_curve_absent_returns_none(anchor, tmp_path):
    assert af.reference_curve(anchor, path=tmp_path / "nope.csv") is None
    # the default location ships no real zhai_fig1c.csv, only the .example.csv
    assert af.reference_curve(anchor) is None


def test_reference_curve_loads_and_groups(tmp_path):
    csv = tmp_path / "ref.csv"
    csv.write_text(
        "# a comment\nseries,energy_eV,intensity\n25keV,900,0.1\n25keV,940,1.0\n17.5keV,690,0.9\n"
    )
    ref = af.reference_curve(path=csv)
    assert set(ref) == {"25keV", "17.5keV"}
    e, inten = ref["25keV"]
    assert np.allclose(e, [900, 940])
    assert np.allclose(inten, [0.1, 1.0])


def test_example_csv_is_valid_schema():
    """The shipped template parses under the documented schema."""
    example = _CHECKS / "reference_data" / "zhai_fig1c.example.csv"
    ref = af.reference_curve(path=example)
    assert ref is not None and len(ref) == 4


def test_match_series_tolerant():
    ref = {"17.5keV": None, "25.0 keV": None, "20": None}
    assert af._match_series(ref, 17.5) == "17.5keV"
    assert af._match_series(ref, 25.0) == "25.0 keV"
    assert af._match_series(ref, 20.0) == "20"
    assert af._match_series(ref, 99.0) is None


# NB: single_segment_anchor() and model_spectra() call mc_spectrum, which runs on
# the GPU array module (xp=cupy on a CUDA box). Per repo convention the fast suite
# stays CPU-only, so the lineshape-normalization anchor (the single-segment MC vs
# Eq.(12) ratio) is exercised by the CPU-forced check run -- it is a column in
# validation_table() printed by anchor_figures.main() -- not here.


def _synthetic_model(anchor):
    """A cheap stand-in for model_spectra() output, for figure smoke tests."""
    E = anchor.E_grid
    lines = af.theory_line_energies(anchor)
    model = {}
    for E0 in anchor.energies_keV:
        peak = lines[E0]
        spec = np.exp(-0.5 * ((E - peak) / 8.0) ** 2)
        brem = 0.02 * np.ones_like(E)
        model[E0] = {
            "spec": spec,
            "spec_det": spec,
            "brem": brem,
            "brem_det": brem,
            "E_peak": float(peak),
            "fwhm": 30.0,
            "line_flux_per_e": float(np.trapezoid(spec, E) * anchor.domega_sr),
            "backscatter": 0.1,
        }
    E0 = anchor.energies_keV[-1]
    film = 0.3 * np.exp(-0.5 * ((E - lines[E0]) / 8.0) ** 2)
    model["film"] = {
        "E0_keV": E0,
        "spec": film,
        "spec_det": film,
        "line_flux_per_e": float(np.trapezoid(film, E) * anchor.domega_sr),
        "n_transmitted": 5,
    }
    return model


def test_cached_model_spectra_round_trip(anchor, tmp_path, monkeypatch):
    expected = _synthetic_model(anchor)
    calls = []

    def fake_model_spectra(received_anchor, ne, ne_brem):
        calls.append((received_anchor, ne, ne_brem))
        return expected

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)

    first, first_hit, path = af.cached_model_spectra(anchor, ne=12, ne_brem=7, cache_dir=tmp_path)
    second, second_hit, second_path = af.cached_model_spectra(
        anchor, ne=12, ne_brem=7, cache_dir=tmp_path
    )

    assert not first_hit
    assert second_hit
    assert path == second_path
    assert path.exists()
    assert calls == [(anchor, 12, 7)]
    assert first.keys() == second.keys()


def test_cached_model_spectra_refreshes(anchor, tmp_path, monkeypatch):
    calls = []

    def fake_model_spectra(received_anchor, ne, ne_brem):
        calls.append((received_anchor, ne, ne_brem))
        return {"generation": len(calls)}

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)
    af.cached_model_spectra(anchor, ne=12, ne_brem=7, cache_dir=tmp_path)
    refreshed, cache_hit, _ = af.cached_model_spectra(
        anchor, ne=12, ne_brem=7, cache_dir=tmp_path, refresh=True
    )

    assert not cache_hit
    assert refreshed == {"generation": 2}
    assert len(calls) == 2


def test_figure_spectra_smoke(anchor):
    from matplotlib.figure import Figure

    model = _synthetic_model(anchor)
    fig = af.figure_spectra(anchor, model)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    fig.canvas.draw()
    assert all(any(label.get_visible() for label in ax.get_xticklabels()) for ax in fig.axes)
    assert all(ax.get_xlabel() == "Photon energy (eV)" for ax in fig.axes)
    scale = anchor.domega_sr * anchor.per_nA
    first_energy = anchor.energies_keV[0]
    assert np.allclose(fig.axes[1].lines[0].get_ydata(), model[first_energy]["spec_det"] * scale)
    assert np.allclose(
        fig.axes[2].lines[0].get_ydata(),
        (model[first_energy]["spec_det"] + model[first_energy]["brem_det"]) * scale,
    )


def test_figure_spectra_with_reference_overlay(anchor):
    model = _synthetic_model(anchor)
    ref = af.reference_curve(path=_CHECKS / "reference_data" / "zhai_fig1c.example.csv")
    fig = af.figure_spectra(anchor, model, reference=ref)
    # detector panel gains scatter collections from the overlay
    assert len(fig.axes[2].collections) >= 1


def test_figure_enhancement_smoke(anchor):
    from matplotlib.figure import Figure

    fig = af.figure_enhancement(anchor, _synthetic_model(anchor))
    assert isinstance(fig, Figure)


def _synthetic_supplementary_spectra(study):
    """One narrow coherent line per requested condition."""
    return {
        condition: np.exp(
            -0.5
            * (
                (study.E_grid - (860.0 + 2.0 * condition.polar_tilt_deg + condition.energy_keV))
                / 8.0
            )
            ** 2
        )
        for condition in study.conditions
    }


def test_supplementary_detected_spectrum_matches_fig1c_detector_scaling():
    study = af.supplementary_study("wse2")
    condition = study.conditions[0]
    spectrum = _synthetic_supplementary_spectra(study)[condition]
    peak_eV = float(study.E_grid[np.argmax(spectrum)])
    anchor = af.ZhaiAnchor()
    fwhm_eV = float(
        np.hypot(
            af.eds_fwhm_eV(peak_eV),
            af.aperture_fwhm_eV(
                peak_eV,
                af.beta_from_keV(condition.energy_keV),
                study.theta_obs_rad,
                anchor.dtheta_obs_rad,
            ),
        )
    )
    spectrum_eff = spectrum * af.detector_efficiency(study.E_grid)
    expected = af.convolve_detector(study.E_grid, spectrum_eff, fwhm_eV)
    expected *= anchor.domega_sr * anchor.per_nA

    assert np.allclose(af._supplementary_detected_spectrum(study, condition, spectrum), expected)


@pytest.mark.parametrize("crystal", ["wse2", "hbn"])
def test_supplementary_figures_use_detected_units_and_hard_bounds(crystal):
    study = af.supplementary_study(crystal)
    spectra = _synthetic_supplementary_spectra(study)
    fig = (
        af.figure_supplementary_hbn(study, study.thicknesses_nm[0], spectra)
        if crystal == "hbn"
        else af.figure_supplementary_tmd(study, study.thicknesses_nm[0], spectra)
    )

    expected = af._supplementary_detected_spectrum(
        study, study.conditions[0], spectra[study.conditions[0]]
    )
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), expected)
    assert all(ax.get_ylabel() == "Intensity (Phs/eV/s/nA)" for ax in fig.axes)
    assert all(ax.get_ylim()[0] == 0.0 for ax in fig.axes)
    assert all(ax.get_xlim() == (study.e_min_eV, study.e_max_eV) for ax in fig.axes)
    if crystal == "wse2":
        assert tuple(fig.get_size_inches()) == (8.0, 7.0)


def test_supplementary_studies_match_requested_windows_and_thicknesses():
    wse2 = af.supplementary_study("wse2")
    mose2 = af.supplementary_study("mose2")
    hbn = af.supplementary_study("hbn")
    hopg = af.supplementary_study("hopg")

    assert wse2.thicknesses_nm == (42.0, 55.0, 75.0)
    assert mose2.thicknesses_nm == (47.0, 112.0, 147.0)
    assert hbn.thicknesses_nm == (42.0, 109.0, 219.0, 659.0, 921.0, 170_000.0)
    assert hopg.thicknesses_nm == (29.0, 76.0, 150.0, 17_000.0, 500_000.0, 1_000_000.0)
    assert (wse2.E_grid[0], wse2.E_grid[-1]) == (800.0, 1199.0)
    assert (hbn.E_grid[0], hbn.E_grid[-1]) == (600.0, 1199.0)
    assert tuple(
        (condition.energy_keV, condition.polar_tilt_deg, condition.azimuth_deg)
        for condition in wse2.conditions
    ) == (
        (200.0, 10.0, None),
        (200.0, 15.0, None),
        (200.0, 17.5, None),
        (200.0, 20.0, None),
    )
    assert mose2.conditions == wse2.conditions
    assert tuple(
        (condition.energy_keV, condition.polar_tilt_deg, condition.azimuth_deg)
        for condition in hbn.conditions_for(921.0)
    ) == (
        (17.5, 17.0, 130.0),
        (20.0, 17.0, 130.0),
        (22.5, 17.0, 130.0),
        (25.0, 17.0, 130.0),
    )
    with pytest.raises(ValueError, match="unknown Zhai supplementary crystal"):
        af.supplementary_study("graphite")


def test_supplementary_table4_conditions_are_bound_to_each_thickness():
    """Table 4 angles must not leak between distinct HOPG/h-BN samples."""
    hopg = af.supplementary_study("hopg")
    hbn = af.supplementary_study("hbn")

    assert hopg.thicknesses_nm == (29.0, 76.0, 150.0, 17_000.0, 500_000.0, 1_000_000.0)
    assert {
        (condition.polar_tilt_deg, condition.azimuth_deg) for condition in hopg.conditions_for(76.0)
    } == {(13.0, 120.0)}
    assert {
        (condition.polar_tilt_deg, condition.azimuth_deg)
        for condition in hopg.conditions_for(17_000.0)
    } == {(11.0, 60.0)}
    assert hbn.thicknesses_nm == (42.0, 109.0, 219.0, 659.0, 921.0, 170_000.0)
    assert {
        (condition.polar_tilt_deg, condition.azimuth_deg) for condition in hbn.conditions_for(219.0)
    } == {(11.5, 65.0), (17.0, 130.0)}
    assert len(hbn.conditions_for(219.0)) == 8


def test_zhai_thermal_beam_uses_the_reported_gaussian_99_9_percent_diameter():
    beam = af.ZhaiThermalBeam()

    assert beam.energy_keV == 300.0
    assert beam.diameter_mm == 1.0
    assert beam.enclosed_fraction == 0.999
    assert beam.fwhm_mm == pytest.approx(np.sqrt(np.log(2.0) / np.log(1000.0)))


def test_supplementary_model_uses_reported_hbn_azimuth(monkeypatch):
    geometry_calls = []

    def fake_geometry(theta_obs_rad, polar_rad, azimuth_rad):
        geometry_calls.append((theta_obs_rad, polar_rad, azimuth_rad))
        return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(af, "tilted_geometry", fake_geometry)
    monkeypatch.setattr(af, "simulate_trajectories", lambda *args, **kwargs: np.zeros((1, 2)))
    monkeypatch.setattr(af, "mc_spectrum", lambda *args, **kwargs: np.zeros(600))

    study = af.supplementary_study("hbn")
    spectra = af.model_coherent_spectra(study, 921.0, ne=1)

    assert set(spectra) == set(study.conditions_for(921.0))
    assert len(geometry_calls) == 4
    assert all(call[1] == pytest.approx(np.deg2rad(17.0)) for call in geometry_calls)
    assert all(call[2] == pytest.approx(np.deg2rad(130.0)) for call in geometry_calls)


def test_supplementary_model_requires_explicit_unreported_azimuth(monkeypatch):
    study = af.supplementary_study("wse2")
    with pytest.raises(ValueError, match="azimuth is unreported"):
        af.model_coherent_spectra(study, 42.0, ne=1)

    geometry_azimuths = []

    def fake_geometry(theta_obs_rad, polar_rad, azimuth_rad):
        geometry_azimuths.append(azimuth_rad)
        return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(af, "tilted_geometry", fake_geometry)
    monkeypatch.setattr(af, "simulate_trajectories", lambda *args, **kwargs: np.zeros((1, 2)))
    monkeypatch.setattr(af, "mc_spectrum", lambda *args, **kwargs: np.zeros(400))

    af.model_coherent_spectra(study, 42.0, ne=1, exploratory_azimuth_deg=35.0)

    assert geometry_azimuths == pytest.approx([np.deg2rad(35.0)] * 4)
    assert all(condition.azimuth_deg is None for condition in study.conditions)


def test_supplementary_tmd_figure_smoke():
    from matplotlib.figure import Figure

    study = af.supplementary_study("wse2")
    fig = af.figure_supplementary_tmd(
        study, study.thicknesses_nm[0], _synthetic_supplementary_spectra(study)
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    assert all(ax.get_xlabel() == "Photon energy (eV)" for ax in fig.axes)
    fig.canvas.draw()


def test_supplementary_tmd_panels_autoscale_and_show_upper_x_tick_labels():
    study = af.supplementary_study("wse2")
    unit_spectra = _synthetic_supplementary_spectra(study)
    spectra = {
        condition: scale * unit_spectra[condition]
        for condition, scale in zip(study.conditions, (1.0, 2.0, 4.0, 8.0), strict=True)
    }

    fig = af.figure_supplementary_tmd(study, study.thicknesses_nm[0], spectra)
    fig.canvas.draw()

    upper_limits = [ax.get_ylim()[1] for ax in fig.axes]
    assert len(set(upper_limits)) == len(fig.axes)
    assert all(ax.get_ylim()[1] > np.max(ax.lines[0].get_ydata()) for ax in fig.axes)
    assert all(any(label.get_visible() for label in ax.get_xticklabels()) for ax in fig.axes[:2])
    assert fig.subplotpars.left == pytest.approx(1.0 - fig.subplotpars.right)


def test_supplementary_tmd_figure_renders_at_physical_intensity_scale():
    study = af.supplementary_study("wse2")
    spectra = {
        condition: 1e-9 * spectrum
        for condition, spectrum in _synthetic_supplementary_spectra(study).items()
    }

    fig = af.figure_supplementary_tmd(study, study.thicknesses_nm[0], spectra)

    fig.savefig(BytesIO(), format="png", bbox_inches="tight")


def test_supplementary_hbn_figure_smoke():
    from matplotlib.figure import Figure

    study = af.supplementary_study("hbn")
    fig = af.figure_supplementary_hbn(
        study, study.thicknesses_nm[0], _synthetic_supplementary_spectra(study)
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == 4
    assert fig.axes[0].get_xlabel() == "Photon energy (eV)"
    fig.canvas.draw()


def test_supplementary_sem_figure_limits_to_selected_orientation():
    study = af.supplementary_study("hbn")
    thickness_nm = 219.0
    spectra = {
        condition: np.exp(-0.5 * ((study.E_grid - 900.0) / 8.0) ** 2)
        for condition in study.conditions_for(thickness_nm)
    }
    fig = af.figure_supplementary_sem(
        study,
        thickness_nm,
        spectra,
        orientation=(11.5, 65.0),
    )

    assert len(fig.axes[0].lines) == 4
    assert [line.get_label() for line in fig.axes[0].lines] == [
        "17.5 keV",
        "20 keV",
        "22.5 keV",
        "25 keV",
    ]
    assert "polar 11.5°" in fig.axes[0].get_title()
    assert "azimuth 65°" in fig.axes[0].get_title()


def test_supplementary_sem_figure_defaults_to_first_reported_orientation():
    study = af.supplementary_study("hbn")
    thickness_nm = 219.0
    spectra = {
        condition: np.exp(-0.5 * ((study.E_grid - 900.0) / 8.0) ** 2)
        for condition in study.conditions_for(thickness_nm)
    }

    fig = af.figure_supplementary_sem(study, thickness_nm, spectra)

    assert len(fig.axes[0].lines) == 4
    assert "polar 11.5°" in fig.axes[0].get_title()
    assert "azimuth 65°" in fig.axes[0].get_title()


def test_supplementary_overview_figure_smoke():
    from matplotlib.figure import Figure

    spectra = {}
    for crystal in ("wse2", "mose2", "hbn", "hopg"):
        study = af.supplementary_study(crystal)
        thickness_nm = study.thicknesses_nm[0]
        synthetic = {
            condition: np.exp(-0.5 * ((study.E_grid - (860.0 + condition.energy_keV)) / 8.0) ** 2)
            for condition in study.conditions_for(thickness_nm)
        }
        spectra[crystal] = synthetic[study.conditions_for(thickness_nm)[-1]]

    fig = af.figure_supplementary_overview(spectra)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "WSe_2" in titles and "MoSe_2" in titles and "h-BN" in titles and "HOPG" in titles
    fig.canvas.draw()


def test_supplementary_overview_rejects_missing_material():
    with pytest.raises(ValueError, match="wse2"):
        af.figure_supplementary_overview({"wse2": np.zeros(4), "mose2": np.zeros(4)})


def test_supplementary_overview_rejects_unknown_key():
    spectra = {crystal: np.zeros(4) for crystal in ("wse2", "mose2", "hbn", "hopg")}
    spectra["graphite"] = spectra.pop("hopg")
    with pytest.raises(ValueError, match="hbn"):
        af.figure_supplementary_overview(spectra)


def test_reproduce_all_populates_every_cache_and_reuses_it(tmp_path, monkeypatch):
    calls = []

    def fake_model_spectra(received_anchor, ne, ne_brem):
        calls.append(("fig1c", ne, ne_brem))
        return {"peak": True}

    def fake_coherent(study, thickness_nm, ne, exploratory_azimuth_deg=None):
        calls.append((study.crystal, thickness_nm, ne, exploratory_azimuth_deg))
        return {condition: np.zeros(4) for condition in study.conditions_for(thickness_nm)}

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)
    monkeypatch.setattr(af, "model_coherent_spectra", fake_coherent)

    results = af.reproduce_all(ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path)

    assert len(results) == 19
    labels = [label for label, _path, _hit in results]
    assert labels[0] == "zhai-fig1c"
    assert (
        "wse2-42nm" in labels
        and "mose2-147nm" in labels
        and "hbn-170000nm" in labels
        and "hopg-1000000nm" in labels
    )
    assert all(path.exists() for _label, path, _hit in results)
    assert all(hit is False for _label, _path, hit in results)  # first run: no cache hits
    assert len(calls) == 19

    calls.clear()
    results2 = af.reproduce_all(ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path)

    assert calls == []  # fully cache-hit; no MC re-run
    assert [label for label, _p, _h in results2] == labels
    assert all(hit for _label, _path, hit in results2)


def test_export_all_figures_writes_expected_files(tmp_path, monkeypatch):
    anchor = af.ZhaiAnchor()
    backend_calls = []

    def fake_model_spectra(received_anchor, ne, ne_brem):
        return _synthetic_model(anchor)

    def fake_coherent(study, thickness_nm, ne, exploratory_azimuth_deg=None):
        return {
            condition: np.exp(-0.5 * ((study.E_grid - 900.0) / 8.0) ** 2)
            for condition in study.conditions_for(thickness_nm)
        }

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)
    monkeypatch.setattr(af, "model_coherent_spectra", fake_coherent)
    monkeypatch.setattr(matplotlib, "use", lambda backend: backend_calls.append(backend))

    outdir = tmp_path / "figures"
    written = af.export_all_figures(
        outdir=outdir, ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path / "cache"
    )

    names = {p.name for p in written}
    assert "zhai_fig1c_spectra_vs_theory.png" in names
    assert "zhai_flux_anchor.png" in names
    assert "zhai_bulk_vs_film_enhancement.png" in names
    assert "zhai_supplementary_wse2_42nm.png" in names
    assert "zhai_supplementary_mose2_147nm.png" in names
    assert "zhai_supplementary_hbn_921nm.png" in names
    assert "zhai_supplementary_hopg_1000000nm.png" in names
    assert "zhai_supplementary_overview.png" in names
    for path in written:
        assert path.exists()
        assert path.with_suffix(".pdf").exists()
    assert backend_calls == ["Agg"]


def test_supplementary_tmd_figure_rejects_hbn_crystal():
    study = af.supplementary_study("hbn")
    with pytest.raises(ValueError, match="only for the WSe2 and MoSe2"):
        af.figure_supplementary_tmd(
            study, study.thicknesses_nm[0], _synthetic_supplementary_spectra(study)
        )


def test_supplementary_tmd_figure_rejects_incomplete_condition_set():
    study = af.supplementary_study("wse2")
    spectra = _synthetic_supplementary_spectra(study)
    del spectra[study.conditions[0]]
    with pytest.raises(ValueError, match="exactly the study's four conditions"):
        af.figure_supplementary_tmd(study, study.thicknesses_nm[0], spectra)


def test_supplementary_hbn_figure_rejects_non_hbn_crystal():
    study = af.supplementary_study("wse2")
    with pytest.raises(ValueError, match="only for the h-BN study"):
        af.figure_supplementary_hbn(
            study, study.thicknesses_nm[0], _synthetic_supplementary_spectra(study)
        )


def test_supplementary_hbn_figure_rejects_incomplete_condition_set():
    study = af.supplementary_study("hbn")
    spectra = _synthetic_supplementary_spectra(study)
    del spectra[study.conditions[0]]
    with pytest.raises(ValueError, match="exactly the requested conditions"):
        af.figure_supplementary_hbn(study, study.thicknesses_nm[0], spectra)


def test_physics_source_tree_is_lf_only():
    """The Zhai cache key hashes raw bytes of every src/cxr_mc/**/*.py file
    (_zhai_cache_key / _supplementary_cache_key); cxr remote check relies on
    the box and the laptop hashing identical bytes for a pulled cache to be a
    hit. The repo's .gitattributes pins `* text=auto eol=lf`, so a CRLF file
    slipping into src/ would silently produce a different hash per platform
    and break that invariant with no visible error -- catch it here instead."""
    repo_root = _CHECKS.parent
    offenders = [
        str(path.relative_to(repo_root))
        for path in sorted((repo_root / "src" / "cxr_mc").glob("**/*.py"))
        if b"\r\n" in path.read_bytes()
    ]
    assert offenders == [], f"CRLF line endings found (breaks box<->laptop cache hash): {offenders}"
