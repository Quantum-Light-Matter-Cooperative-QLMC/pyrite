"""Precomputed tutorials must stay in sync and render without transport."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tutorials():
    path = Path(__file__).parents[2] / "docs" / "_tutorials.py"
    spec = importlib.util.spec_from_file_location("docs_tutorials", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_build_rejects_missing_and_stale_outputs(tutorials, monkeypatch, tmp_path):
    monkeypatch.setattr(tutorials, "EXAMPLES", tmp_path)
    monkeypatch.setattr(tutorials, "NAMES", ("python_api_workflow",))
    source = tmp_path / "python_api_workflow.py"
    source.write_text("original")
    with pytest.raises(FileNotFoundError):
        tutorials._check_outputs(None)
    (tmp_path / "data").mkdir()
    output = tmp_path / "data" / "python_api_workflow.json"
    output.write_text(
        json.dumps({"source_sha256": tutorials.source_digest(source.stem), "output": {}})
    )
    tutorials._check_outputs(None)
    source.write_text("changed")
    with pytest.raises(ValueError, match="stale tutorial output"):
        tutorials._check_outputs(None)


def test_cli_transcript_matches_recorded_success(tutorials, monkeypatch, tmp_path):
    example = tutorials.load_example("getting_started")

    def forbid_transport():
        pytest.fail("documentation rendering must never execute the CLI run")

    monkeypatch.setattr(example, "compute", forbid_transport)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    data = tutorials.checked_output("getting_started")
    assert data["run"]["ok"]
    assert data["run"]["payload"]["completed_materials"] == ["hopg"]
    assert not data["run"]["payload"]["resumable"]
    assert "1 records" in data["display_text"]
    assert "quickstart" in data["display_text"]

    monkeypatch.setattr(tutorials, "EXAMPLES", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "getting_started.py").write_text("source")
    (tmp_path / "data" / "getting_started.json").write_text(
        json.dumps(
            {
                "source_sha256": tutorials.source_digest("getting_started"),
                "output": data,
            }
        )
    )
    transcript = tmp_path / "data" / "getting_started.txt"
    transcript.write_text("changed")
    with pytest.raises(ValueError, match="stale tutorial transcript"):
        tutorials.checked_output("getting_started")
    transcript.write_text(data["display_text"])
    assert tutorials.checked_output("getting_started") == data


def test_precomputed_spectrum_renders_without_simulating(tutorials, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("python_api_workflow")

    def forbid_transport():
        pytest.fail("documentation rendering must never execute transport")

    monkeypatch.setattr(example, "compute", forbid_transport)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    data = tutorials.checked_output("python_api_workflow")
    assert data["seed"] == 1
    assert data["provenance"]["backend"] == "cpu"
    for spectrum in data["spectra"].values():
        energy, density = map(np.asarray, (spectrum["energy_eV"], spectrum["density"]))
        assert energy.ndim == 1 and energy.shape == density.shape
        assert np.isfinite(energy).all() and np.all(np.diff(energy) > 0)
        assert np.isfinite(density).all() and np.all(density >= 0) and np.any(density > 0)
    try:
        tutorials.python_api_workflow()
        figure = plt.gcf()
        assert len(figure.axes[0].lines) == 2
        figure.savefig(tmp_path / "spectrum.svg")
        assert (tmp_path / "spectrum.svg").stat().st_size > 1_000
    finally:
        plt.close("all")


def test_configuration_exercise_isolated_and_reproducible(tutorials, monkeypatch, tmp_path):
    # Real CLI writes must stay out of an existing user's config and catalog.
    for key in ("XDG_CONFIG_HOME", "PYRITE_USER_CATALOG", "PYRITE_HOME", "PYRITE_CATALOG"):
        location = tmp_path / key
        location.mkdir()
        (location / "sentinel").write_text("unchanged")
        monkeypatch.setenv(key, str(location))
    example = tutorials.load_example("configuration_cookbook")
    data = example.compute()
    assert data == tutorials.checked_output("configuration_cookbook")
    assert data["exit_code"] == 0
    assert "thickness: [50000] (inherited)" in data["display_text"]
    assert "energy: [30, 60] (inherited)" in data["display_text"]
    assert "profile.current  lab-scan  config store" in data["display_text"]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == sorted(
        item
        for key in ("XDG_CONFIG_HOME", "PYRITE_USER_CATALOG", "PYRITE_HOME", "PYRITE_CATALOG")
        for item in (key, f"{key}/sentinel")
    )


def test_configuration_build_rejects_changed_shell_source(tutorials, monkeypatch, tmp_path):
    data = tutorials.checked_output("configuration_cookbook")
    monkeypatch.setattr(tutorials, "EXAMPLES", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "configuration_cookbook.py").write_text("source")
    (tmp_path / "configuration_cookbook.sh").write_text("changed shell source")
    (tmp_path / "data" / "configuration_cookbook.json").write_text(
        json.dumps(
            {
                "source_sha256": tutorials.source_digest("configuration_cookbook"),
                "output": data,
            }
        )
    )
    with pytest.raises(ValueError, match="stale tutorial script"):
        tutorials.checked_output("configuration_cookbook")


def test_saved_results_render_without_checkpoint_or_transport(tutorials, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrite.results.radiation import radiation_band_metrics

    example = tutorials.load_example("working_with_results")

    def forbid_compute(*args, **kwargs):
        pytest.fail("documentation rendering must use only the committed reduced output")

    monkeypatch.setattr(example, "compute", forbid_compute)
    monkeypatch.setattr(example, "load_checkpoint", forbid_compute)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    data = tutorials.checked_output("working_with_results")
    provenance = data["provenance"]
    assert provenance["dataset_identity"]["catalog_profile"] == "quickstart"
    assert data["seed"] == provenance["case"]["seed"] == 1
    assert provenance["adaptive_sampling"]["realized_electrons"] > 0
    assert set(provenance["components_sha256"]) == {"line.h5", "brem.h5", "characteristic.h5"}
    assert all(len(digest) == 64 for digest in provenance["components_sha256"].values())
    line = data["spectra"]["PXR + CBS"]
    characteristic = data["spectra"]["Characteristic"]
    continuum = data["spectra"]["Bremsstrahlung"]
    for spectrum in data["spectra"].values():
        energy, density = map(np.asarray, (spectrum["energy_eV"], spectrum["density"]))
        assert energy.shape == density.shape and energy.ndim == 1
        assert np.isfinite(energy).all() and np.all(np.diff(energy) > 0)
        assert np.isfinite(density).all() and np.all(density >= 0)
    assert line["energy_eV"] == characteristic["energy_eV"]
    assert continuum["energy_eV"] != line["energy_eV"]
    # Recheck displayed band yields from the committed native density arrays.
    metrics = radiation_band_metrics(
        {
            "E_grid": line["energy_eV"],
            "spec": line["density"],
            "spec_characteristic": characteristic["density"],
        },
        tuple(data["band_eV"]),
    )
    assert set(data["band_yields"]) == {"line", "characteristic"}
    for name, value in data["band_yields"].items():
        assert value == pytest.approx(metrics["yields"][name])
    try:
        tutorials.working_with_results()
        figure = plt.gcf()
        assert len(figure.axes) == 2
        assert len(figure.axes[0].lines) == 3
        assert len(figure.axes[1].patches) == 2
        figure.savefig(tmp_path / "saved-results.svg")
        assert (tmp_path / "saved-results.svg").stat().st_size > 1_000
    finally:
        plt.close("all")


def test_directional_map_renders_native_grids_without_transport(tutorials, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("directional_source_heatmap")

    def forbid_transport(*args, **kwargs):
        pytest.fail("documentation rendering must never execute directional transport")

    monkeypatch.setattr(example, "compute", forbid_transport)
    monkeypatch.setattr(example, "calculate", forbid_transport)
    monkeypatch.setattr(example, "directional_source", forbid_transport)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    data = tutorials.checked_output("directional_source_heatmap")
    angles, energy, density = (np.asarray(data[k]) for k in ("angle_deg", "energy_eV", "density"))
    assert density.shape == (len(angles), len(energy)) == (48, 513)
    assert angles[0] == -180 and angles[-1] < 180 and np.all(np.diff(angles) == 7.5)
    assert np.all(np.diff(energy) > 0) and energy[0] >= 10
    assert energy[-1] <= data["case"]["E0_keV"] * 1_000
    assert energy[-1] >= data["provenance"]["line_window_ceiling_eV"]
    assert np.isfinite(density).all() and np.all(density >= 0) and np.any(density > 0)
    assert np.all(data["supported"])  # Finite footprint has no unsupported grazing directions.
    assert data["units"] == "ph/incident-electron/eV/sr"
    assert data["seed"] == data["case"]["seed"] == 1
    assert data["case"]["Ne"] == data["case"]["Ne_brem"] == 64
    assert data["provenance"]["backend"] == "cpu"
    assert any("LineOpticalDataWarning" in warning for warning in data["warnings"])
    try:
        tutorials.directional_source_heatmap()
        figure = plt.gcf()
        # Three spectral cuts retain pointwise per-sr densities, without bin factors.
        for line, angle in zip(figure.axes[1].lines, (0, 90, 135), strict=True):
            np.testing.assert_array_equal(line.get_xdata(), energy)
            np.testing.assert_array_equal(
                line.get_ydata(), density[np.flatnonzero(angles == angle)[0]]
            )
        mesh = figure.axes[0].collections[0]
        assert mesh.get_array().shape == (len(energy), len(angles) + 1)
        figure.savefig(tmp_path / "directional-map.svg")
        assert (tmp_path / "directional-map.svg").stat().st_size > 1_000
    finally:
        plt.close("all")


def test_detector_counts_render_without_transport(tutorials, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("detector_observations")

    def forbid_transport(*args, **kwargs):
        pytest.fail("documentation rendering must never simulate or acquire counts")

    monkeypatch.setattr(example, "compute", forbid_transport)
    monkeypatch.setattr(example, "calculate", forbid_transport)
    monkeypatch.setattr(example, "simulate", forbid_transport)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    data = tutorials.checked_output("detector_observations")
    expected, realized = (np.asarray(data[k]) for k in ("expected", "realized"))
    assert expected.shape == realized.shape == (16, 16)
    assert np.isfinite(expected).all() and np.all(expected > 0)
    assert np.issubdtype(realized.dtype, np.integer) and np.all(realized >= 0)
    assert not np.array_equal(expected, realized)
    assert data["measured_edges_eV"] == [500.0, 4_000.0]
    assert set(data["components"]) == {"cxr", "brem", "characteristic"}
    assert data["case"]["Ne"] == data["case"]["Ne_brem"] == 64
    assert data["seed"] == data["case"]["seed"] == 1
    provenance = data["provenance"]
    assert provenance["backend"] == "cpu"
    observation = provenance["observation"]
    assert observation["response"]["event_semantics"] == "ideal-energy-preserving-photon-event"
    assert observation["acquisition"]["config"]["mode"] == "poisson"
    assert observation["acquisition"]["config"]["seed"] == 23
    assert observation["acquisition"]["normalization"] == {
        "rep_rate_hz": 1_000_000.0,
        "bunch_charge_pc": 0.01,
    }
    try:
        tutorials.detector_observations()
        figure = plt.gcf()
        for ax, array in zip(figure.axes[:2], (expected, realized), strict=True):
            np.testing.assert_array_equal(ax.images[0].get_array(), array)
        assert figure.axes[0].images[0].get_clim() == figure.axes[1].images[0].get_clim()
        figure.savefig(tmp_path / "detector-counts.svg")
        assert (tmp_path / "detector-counts.svg").stat().st_size > 1_000
    finally:
        plt.close("all")


def test_imported_beam_fixture_and_cached_figure(tutorials, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrite.montecarlo.gdf import load_gdf_beam

    example = tutorials.load_example("gpt_gdf_beams")
    data = tutorials.checked_output("gpt_gdf_beams")
    beam = load_gdf_beam(example.fixture_path(), time_s=1e-9, normalization="gdf_charge")
    assert len(beam.energy_keV) == 96
    assert beam.absolute_charge_c == pytest.approx(1e-12)
    np.testing.assert_allclose(beam.energy_keV, data["energy_keV"])
    np.testing.assert_allclose(beam.probabilities, data["probabilities"])
    np.testing.assert_allclose(beam.position_m[:, 0] * 1e6, data["x_um"])
    np.testing.assert_allclose(beam.beta[:, 0] / beam.beta[:, 2] * 1000, data["slope_mrad"])
    assert data["case"]["gdf_source"]["sha256"] == beam.sha256
    assert data["case"]["gdf_source"]["normalization"] == "gdf_charge"
    assert data["case"]["bunch_charge_pc"] == pytest.approx(1.0)
    assert data["case"]["rep_rate_hz"] == 1_000_000.0
    assert data["case"]["E0_keV"] == pytest.approx(max(data["energy_keV"]))
    assert data["case"]["Ne"] == data["case"]["Ne_brem"] == 64
    assert data["seed"] == data["case"]["seed"] == 1
    assert data["provenance"]["backend"] == "cpu"
    assert data["input_kind"].startswith("SYNTHETIC DEMONSTRATION")

    def forbid_compute(*args, **kwargs):
        pytest.fail("documentation rendering must never import or simulate the beam")

    monkeypatch.setattr(example, "compute", forbid_compute)
    monkeypatch.setattr(example, "calculate", forbid_compute)
    monkeypatch.setattr(example, "simulate", forbid_compute)
    monkeypatch.setattr(example.easygdf, "load", forbid_compute)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    try:
        tutorials.gpt_gdf_beams()
        figure = plt.gcf()
        np.testing.assert_allclose(figure.axes[0].collections[0].get_offsets()[:, 0], data["x_um"])
        assert sum(p.get_height() for p in figure.axes[1].patches) == pytest.approx(1.0)
        for line, spectrum in zip(figure.axes[2].lines, data["spectra"].values(), strict=True):
            assert np.all(np.asarray(spectrum["density"]) >= 0)
            np.testing.assert_array_equal(line.get_xdata(), spectrum["energy_eV"])
            np.testing.assert_array_equal(line.get_ydata(), spectrum["density"])
        figure.savefig(tmp_path / "imported-beam.svg")
        assert (tmp_path / "imported-beam.svg").stat().st_size > 1_000
    finally:
        plt.close("all")


def test_tutorial_build_rejects_changed_input_fixture(tutorials, monkeypatch, tmp_path):
    data = tutorials.checked_output("gpt_gdf_beams")
    monkeypatch.setattr(tutorials, "EXAMPLES", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "gpt_gdf_beams.py").write_text("source")
    (tmp_path / "data" / "gpt_gdf_beams.json").write_text(
        json.dumps({"source_sha256": tutorials.source_digest("gpt_gdf_beams"), "output": data})
    )
    fixture = tmp_path / "data" / "synthetic_beam.gdf"
    with pytest.raises(FileNotFoundError):
        tutorials.checked_output("gpt_gdf_beams")
    fixture.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="stale tutorial input"):
        tutorials.checked_output("gpt_gdf_beams")
