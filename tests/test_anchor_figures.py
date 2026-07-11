"""Fast unit tests for checks/anchor_figures.py (the P1 #2 validation figures).

The heavy MC figure run lives in checks/; here we test only the cheap, pure
pieces -- theory anchors, the reference-CSV loader, series matching, the
single-segment lineshape-normalization anchor, and that the figure builders
return Figures on synthetic data -- so the suite stays CPU-only and fast.
"""

import sys
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
    from cxr_mc.crystallography import (
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
    """One distinct narrow coherent line per requested polar tilt."""
    return {
        tilt: np.exp(-0.5 * ((study.E_grid - (900.0 + 2.0 * tilt)) / 8.0) ** 2)
        for tilt in study.polar_tilts_deg
    }


def test_supplementary_studies_match_requested_windows_and_thicknesses():
    wse2 = af.supplementary_study("wse2")
    mose2 = af.supplementary_study("mose2")
    hbn = af.supplementary_study("hbn")

    assert wse2.energy_keV == mose2.energy_keV == hbn.energy_keV == 200.0
    assert wse2.thicknesses_nm == (42.0, 55.0, 75.0)
    assert mose2.thicknesses_nm == (47.0, 112.0, 147.0)
    assert hbn.thicknesses_nm == (921.0,)
    assert (wse2.E_grid[0], wse2.E_grid[-1]) == (800.0, 1199.0)
    assert (hbn.E_grid[0], hbn.E_grid[-1]) == (600.0, 1199.0)
    assert wse2.polar_tilts_deg == (-10.0, -15.0, -17.5, -20.0)
    with pytest.raises(ValueError, match="unknown Zhai supplementary crystal"):
        af.supplementary_study("hopg")


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


def test_supplementary_overview_figure_smoke():
    from matplotlib.figure import Figure

    spectra = {
        crystal: _synthetic_supplementary_spectra(af.supplementary_study(crystal))[-20.0]
        for crystal in ("wse2", "mose2", "hbn")
    }

    fig = af.figure_supplementary_overview(spectra)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "WSe_2" in titles and "MoSe_2" in titles and "h-BN" in titles
    fig.canvas.draw()


def test_supplementary_overview_rejects_missing_material():
    with pytest.raises(ValueError, match="wse2"):
        af.figure_supplementary_overview({"wse2": np.zeros(4), "mose2": np.zeros(4)})


def test_supplementary_overview_rejects_unknown_key():
    spectra = {
        crystal: _synthetic_supplementary_spectra(af.supplementary_study(crystal))[-20.0]
        for crystal in ("wse2", "mose2", "hbn")
    }
    spectra["hopg"] = spectra.pop("hbn")
    with pytest.raises(ValueError, match="hbn"):
        af.figure_supplementary_overview(spectra)


def test_reproduce_all_populates_every_cache_and_reuses_it(tmp_path, monkeypatch):
    calls = []

    def fake_model_spectra(received_anchor, ne, ne_brem):
        calls.append(("fig1c", ne, ne_brem))
        return {"peak": True}

    def fake_coherent(study, thickness_nm, ne):
        calls.append((study.crystal, thickness_nm, ne))
        return {tilt: np.zeros(4) for tilt in study.polar_tilts_deg}

    monkeypatch.setattr(af, "model_spectra", fake_model_spectra)
    monkeypatch.setattr(af, "model_coherent_spectra", fake_coherent)

    results = af.reproduce_all(ne=11, ne_brem=3, ne_supp=5, cache_dir=tmp_path)

    assert len(results) == 8
    labels = [label for label, _path, _hit in results]
    assert labels[0] == "zhai-fig1c"
    assert "wse2-42nm" in labels and "mose2-147nm" in labels and "hbn-921nm" in labels
    assert all(path.exists() for _label, path, _hit in results)
    assert all(hit is False for _label, _path, hit in results)  # first run: no cache hits
    assert len(calls) == 8

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

    def fake_coherent(study, thickness_nm, ne):
        return _synthetic_supplementary_spectra(study)

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
    assert "zhai_supplementary_overview.png" in names
    for path in written:
        assert path.exists()
        assert path.with_suffix(".pdf").exists()
    assert backend_calls == ["Agg"]
