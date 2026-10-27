"""Precomputed tutorials must stay in sync and render without transport."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest


def test_optimization_plots_preserve_recorded_selection_uncertainty_and_units(
    tutorials, monkeypatch, tmp_path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("optimization_studies")
    data = tutorials.checked_output("optimization_studies")
    assert example.compute() == data
    assert data["simulation_source"]["code_revision"].startswith("c941146c")
    assert data["units"] == "photons/(sr incident electron)"
    rows = data["comparisons"]
    assert len(rows) == 9 and all(row["precision_pass"] for row in rows)

    def forbid_compute():
        pytest.fail("documentation rendering must use checked optimization evidence")

    monkeypatch.setattr(example, "compute", forbid_compute)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    try:
        tutorials.optimization_studies()
        ax = plt.gcf().axes[0]
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), [row["selected_search_yield"] * 1e8 for row in rows]
        )
        container = ax.containers[0]
        np.testing.assert_allclose(
            container.lines[0].get_ydata(orig=False), [row["validation_mean"] * 1e8 for row in rows]
        )
        expected = [
            [
                [index, (row["validation_mean"] - row["validation_sem"]) * 1e8],
                [index, (row["validation_mean"] + row["validation_sem"]) * 1e8],
            ]
            for index, row in enumerate(rows)
        ]
        np.testing.assert_allclose(container.lines[2][0].get_segments(), expected)
        tutorials.optimization_studies(view="precision")
        ax = plt.gcf().axes[0]
        assert ax.get_yscale() == "log"
        np.testing.assert_allclose(
            ax.lines[1].get_ydata(),
            [row["max_relative_paired_grid_difference"] * 100 for row in rows],
        )
        np.testing.assert_allclose(ax.lines[2].get_ydata(), 5)
        np.testing.assert_allclose(ax.lines[3].get_ydata(), 2)
        plt.gcf().savefig(tmp_path / "optimization.svg")
    finally:
        plt.close("all")

    evidence = tmp_path / "evidence.json"
    evidence.write_text("changed")
    monkeypatch.setattr(tutorials, "DOCS", tmp_path / "docs")
    data["repository_input_sha256"] = {evidence.name: "0" * 64}
    cache = tmp_path / "data"
    cache.mkdir()
    (tmp_path / "optimization_studies.py").write_text("source")
    monkeypatch.setattr(tutorials, "EXAMPLES", tmp_path)
    (cache / "optimization_studies.json").write_text(
        json.dumps(
            {"source_sha256": tutorials.source_digest("optimization_studies"), "output": data}
        )
    )
    with pytest.raises(ValueError, match="stale tutorial repository input"):
        tutorials.checked_output("optimization_studies")


def test_multiple_inline_guide_plots_have_distinct_image_names():
    guides = Path(__file__).parents[2] / "docs" / "guides"
    for page in guides.glob("*.md"):
        blocks = re.findall(
            r"^\.\. plot::[^\S\n]*\n((?:[ \t]+[^\n]*\n|\n)*)", page.read_text(), re.MULTILINE
        )
        if len(blocks) < 2:
            continue
        prefixes = [
            re.search(r"^[ \t]+:filename-prefix:[ \t]+(\S+)[ \t]*$", block, re.MULTILINE)
            for block in blocks
        ]
        assert all(prefixes), f"{page.name}: each inline plot needs a filename-prefix"
        names = [prefix.group(1) for prefix in prefixes]
        assert len(set(names)) == len(names), f"{page.name}: duplicate plot filename-prefix"


def test_planar_diagnostic_renders_cached_native_arrays_without_solving(tutorials, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("experimental_planar_channeling")

    def forbid_solve(*args, **kwargs):
        pytest.fail("documentation rendering must never run planar quantum solvers")

    for name in ("compute", "calculate", "propagate_planar_source", "score_planar_source"):
        monkeypatch.setattr(example, name, forbid_solve)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    data = tutorials.checked_output("experimental_planar_channeling")
    depth = np.asarray(data["depth_ang"])
    energy = np.asarray(data["energy_eV"])
    assert len(depth) == 101 and depth[0] == 0 and depth[-1] == 2_000
    assert len(energy) == 1_025 and energy[0] == 1_000 and energy[-1] == 50_000
    assert np.all(np.diff(energy) > 0)
    assert data["units"]["spectrum"] == "ph/incident-electron/eV/sr"
    assert sum(data["entry"].values()) == pytest.approx(1.0, abs=2e-3)
    retained = np.asarray(data["retained_population"])
    bound = np.asarray(data["bound_population"])
    assert np.all(bound <= retained + 1e-12)
    # The default history returns the bound-band prefix, not all retained bands.
    assert np.all(np.sum(data["band_populations"], axis=1) <= retained + 1e-12)
    for name in ("spectrum", "capture_spectrum"):
        values = np.asarray(data[name])
        assert values.shape == energy.shape and np.isfinite(values).all()
        assert np.any(values > 0)
    assert data["settings"]["backend"] == "cpu" and not data["settings"]["absorption"]
    try:
        tutorials.experimental_planar_channeling()
        figure = plt.gcf()
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), depth / 10_000)
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), retained)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_xdata(), energy / 1_000)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), data["spectrum"])
        np.testing.assert_array_equal(figure.axes[1].lines[1].get_ydata(), data["capture_spectrum"])
    finally:
        plt.close("all")


def test_axial_diagnostic_renders_cached_density_and_overlap_without_solving(
    tutorials, monkeypatch
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("experimental_axial_channeling")

    def forbid_solve(*args, **kwargs):
        pytest.fail("documentation rendering must never run axial quantum solvers")

    for name in ("compute", "calculate", "axial_diagnostic"):
        monkeypatch.setattr(example, name, forbid_solve)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    data = tutorials.checked_output("experimental_axial_channeling")
    density = np.asarray(data["density_per_ang2"])
    depth = np.asarray(data["depth_ang"])
    norm = np.asarray(data["retained_norm"])
    overlap = np.asarray(data["initial_state_overlap"])
    assert density.shape == (3, 64, 64)
    assert data["snapshot_indices"] == [0, 25, 50]
    assert depth[0] == 0 and depth[-1] == 500 and len(depth) == 51
    assert np.isfinite(density).all() and np.all(density >= 0)
    np.testing.assert_allclose(density.sum(axis=(1, 2)) * data["cell_area_ang2"], norm[[0, 25, 50]])
    np.testing.assert_allclose(norm, 1, atol=1e-12)
    assert overlap[0] == pytest.approx(1)
    assert np.all(overlap >= 0) and np.all(overlap <= norm + 1e-12)
    assert data["config"]["mode"] == "coherent" and data["config"]["backend"] == "cpu"
    assert data["units"]["density"] == "per incident electron / angstrom^2"
    try:
        tutorials.experimental_axial_channeling(view="density")
        figure = plt.gcf()
        for ax, values in zip(figure.axes[:3], density, strict=True):
            np.testing.assert_array_equal(ax.images[0].get_array(), values.T)
        assert len({ax.images[0].get_clim() for ax in figure.axes[:3]}) == 1
        tutorials.experimental_axial_channeling(view="evolution")
        ax = plt.gcf().axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), norm)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), overlap)
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), depth / 10)
    finally:
        plt.close("all")


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
        assert [line.get_label() for line in figure.axes[0].lines] == [
            "PXR + CBS",
            "Bremsstrahlung",
            "Total Spectrum",
        ]
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
    assert data["case"]["Ne"] == data["case"]["Ne_brem"] == 4_096
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


def test_tr_ctr_app_output_renders_signed_native_arrays_without_transport(
    tutorials, monkeypatch, tmp_path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("transition_radiation_demo")

    def forbid_compute():
        pytest.fail("documentation rendering must never execute the app calculations")

    monkeypatch.setattr(example, "compute", forbid_compute)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    data = tutorials.checked_output("transition_radiation_demo")
    assert data["units"] == "ph/incident-electron/eV/sr"
    assert data["physical_electrons"] == pytest.approx(624.1509074460763)
    analytic = data["analytic"]
    np.testing.assert_array_equal(analytic["energy_eV"], np.arange(200, 1010, 10))
    np.testing.assert_array_equal(analytic["angle_deg"], np.arange(100, 176))
    np.testing.assert_allclose(
        analytic["ctr"],
        np.asarray(analytic["film"])
        * (1 + (data["physical_electrons"] - 1) * np.asarray(analytic["form_factor"])),
    )
    for record in data["transport"].values():
        assert record["case"]["Ne"] == 32 and record["case"]["Ne_brem"] == 64
        assert record["seed"] == record["case"]["seed"] == 1
        assert record["provenance"]["backend"] == "cpu"
        np.testing.assert_array_equal(record["energy_eV"], analytic["energy_eV"])
        np.testing.assert_allclose(
            record["combined_cxr"],
            np.asarray(record["transition"]) + np.asarray(record["other_cxr"]),
        )
    negative = np.count_nonzero(np.asarray(data["transport"]["bunch"]["transition"]) < 0)
    assert negative > 0
    assert any(f"negative in {negative} of 81 bins" in warning for warning in data["warnings"])
    try:
        for view in ("analytic", "transport"):
            tutorials.transition_radiation_demo(view)
            figure = plt.gcf()
            assert len(figure.axes) == 2
            if view == "analytic":
                np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), analytic["film"])
                np.testing.assert_array_equal(
                    figure.axes[1].lines[1].get_ydata(), analytic["ctr_cut"]
                )
            else:
                for ax, mode in zip(figure.axes, ("history", "bunch"), strict=True):
                    assert ax.get_yscale() == "symlog"
                    np.testing.assert_array_equal(
                        ax.lines[0].get_ydata(), data["transport"][mode]["transition"]
                    )
            figure.savefig(tmp_path / f"tr-ctr-{view}.svg")
            assert (tmp_path / f"tr-ctr-{view}.svg").stat().st_size > 1_000
    finally:
        plt.close("all")


def test_optical_diagnostic_profiles_render_without_inference(tutorials, monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = tutorials.load_example("radiation_diagnostics")
    data = tutorials.checked_output("radiation_diagnostics")
    wide = data["bands"]["turnover_in_band"]
    high = data["bands"]["turnover_below_band"]
    assert data["settings"]["nodes"] == 61
    assert wide["fit"]["is_identified_95"]
    assert wide["fit"]["rms_duration_fs"] == pytest.approx(200, rel=0.005)
    assert wide["fit"]["confidence_interval_95_fs"] == pytest.approx((198.62745, 201.38390))
    assert not high["fit"]["is_identified_95"]
    assert high["fit"]["is_multimodal"]
    assert high["fit"]["confidence_interval_95_fs"] == [None, None]
    for record in (wide, high):
        scan, fit = record["scan"], record["fit"]
        np.testing.assert_allclose(scan["uncertainty_v_ports"], 15e-6)
        np.testing.assert_allclose(
            np.asarray(scan["voltage_v_ports"]) - np.asarray(fit["predicted_voltage_v_ports"]),
            fit["residual_v_ports"],
        )
        assert "SYNTHETIC noise-free" in scan["provenance"]
        provenance = record["instrument_provenance"]
        assert "unanchored; synthetic reference only" in provenance["validation"]
        assert all(len(digest) == 64 for digest in provenance["detector_response_sha256"])

    def forbid_compute():
        pytest.fail("documentation rendering must never run the optical forward model or fit")

    monkeypatch.setattr(example, "compute", forbid_compute)
    monkeypatch.setattr(tutorials, "load_example", lambda _: example)
    tutorials._check_outputs(None)
    try:
        for view in ("signal", "profile"):
            tutorials.radiation_diagnostics(view)
            fig = plt.gcf()
            if view == "signal":
                assert len(fig.axes) == 2
                np.testing.assert_array_equal(
                    fig.axes[0].lines[0].get_xdata(),
                    np.asarray(wide["scan"]["mirror_displacement_m"]) * 1e3,
                )
                np.testing.assert_array_equal(
                    fig.axes[0].lines[0].get_ydata(),
                    np.asarray(wide["scan"]["voltage_v_ports"])[:, 0] * 1e3,
                )
                np.testing.assert_array_equal(
                    fig.axes[1].lines[0].get_ydata(),
                    np.asarray(wide["fit"]["residual_v_ports"])[:, 0] * 1e6,
                )
            else:
                assert len(fig.axes) == 1
                assert fig.axes[0].get_xscale() == "log"
                assert fig.axes[0].get_yscale() == "symlog"
                for line, record in zip(fig.axes[0].lines, (wide, high), strict=False):
                    profile = record["fit"]["metadata"]["diagnostic_profile"]
                    np.testing.assert_array_equal(line.get_xdata(), profile["sigma_fs"])
                    np.testing.assert_array_equal(line.get_ydata(), profile["delta_chi2"])
            target = tmp_path / f"optical-{view}.svg"
            fig.savefig(target)
            assert target.stat().st_size > 1_000
    finally:
        plt.close("all")
