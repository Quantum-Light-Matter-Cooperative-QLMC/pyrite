"""Native GPT fixtures exercise parsing, correlated sampling, and transport."""

from dataclasses import replace

import easygdf
import numpy as np
import pytest
from scipy.constants import c, electron_mass, elementary_charge

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.materials import MaterialConfigError
from pyrite.montecarlo.gdf import list_gdf_times, load_gdf_beam
from pyrite.montecarlo.transport import simulate_trajectories


def fields():
    return {
        "x": np.array([1e-5, -2e-5]),
        "y": np.array([3e-5, 4e-5]),
        "z": np.array([0.1, 0.1]),
        "Bx": np.array([0.01, -0.02]),
        "By": np.array([-0.01, 0.02]),
        "Bz": np.array([0.328, 0.328]),
        "m": np.full(2, electron_mass),
        "q": np.full(2, -elementary_charge),
        "nmacro": np.array([1.0, 9.0]),
    }


def write_gdf(path, arrays=None, times=(1.0000000000000003e-9,), creator="GPT", kind="time"):
    arrays = fields() if arrays is None else arrays
    blocks = [
        {
            "name": kind,
            "value": time,
            "children": [{"name": key, "value": value} for key, value in arrays.items()],
        }
        for time in times
    ]
    easygdf.save(str(path), blocks=blocks, creator=creator)
    return path


def test_times_and_selection(tmp_path):
    path = write_gdf(tmp_path / "beam.gdf", times=tuple(i * 5e-11 for i in range(21)))
    assert len(list_gdf_times(path)) == 21
    assert all(count == 2 for _, count in list_gdf_times(path))
    assert load_gdf_beam(path, 1e-9).time_s == pytest.approx(1e-9)
    for time, tolerance, match in [
        (None, 1e-15, "required"),
        (1.01e-9, 1e-15, "unavailable"),
        (1e-10, 1e-10, "ambiguous"),
    ]:
        with pytest.raises(MaterialConfigError, match=match) as error:
            load_gdf_beam(path, time, tolerance)
        assert "particles" in str(error.value)
    path = write_gdf(path)
    assert load_gdf_beam(path, 1e-9).time_s == 1.0000000000000003e-9
    assert load_gdf_beam(path).time_s > 0


@pytest.mark.parametrize(
    "time,tolerance", [(-1, 1e-15), (np.nan, 1e-15), (np.inf, 0), (0, -1), (0, np.nan), (0, np.inf)]
)
def test_bad_time_parameters(tmp_path, time, tolerance):
    with pytest.raises(MaterialConfigError, match="finite and non-negative"):
        load_gdf_beam(tmp_path / "unused", time, tolerance)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("x", None, "missing required"),
        ("q", None, "missing required"),
        ("m", np.ones(2), "electrons"),
        ("q", np.array([-elementary_charge, -2 * elementary_charge]), "electrons"),
        ("q", np.ones(2), "electrons"),
        ("x", np.zeros(3), "lengths"),
        ("x", np.array([np.nan, 0]), "finite"),
        ("Bx", np.ones(2), "invalid beta"),
        ("Bz", np.full(2, np.inf), "finite"),
        ("G", np.ones(2) * 2, "maximum discrepancy"),
        ("nmacro", np.array([0, 1]), "strictly positive"),
        ("nmacro", np.array([np.nan, 1]), "finite"),
    ],
)
def test_invalid_arrays(tmp_path, key, value, message):
    arrays = fields()
    if value is None:
        arrays.pop(key)
    else:
        arrays[key] = value
    with pytest.raises(MaterialConfigError, match=message):
        load_gdf_beam(write_gdf(tmp_path / "bad.gdf", arrays))


def test_empty_zero_speed_and_missing_weight(tmp_path):
    path = tmp_path / "bad.gdf"
    with pytest.raises(MaterialConfigError, match="empty"):
        load_gdf_beam(write_gdf(path, {k: v[:0] for k, v in fields().items()}))
    arrays = fields()
    for key in ("Bx", "By", "Bz"):
        arrays[key][:] = 0
    with pytest.raises(MaterialConfigError, match="nonzero"):
        load_gdf_beam(write_gdf(path, arrays))
    arrays = fields()
    arrays.pop("nmacro")
    write_gdf(path, arrays)
    np.testing.assert_array_equal(load_gdf_beam(path).probabilities, [0.5, 0.5])
    with pytest.raises(MaterialConfigError, match="requires nmacro"):
        load_gdf_beam(path, normalization="gdf_charge")


def test_bad_files(tmp_path):
    path = tmp_path / "bad.gdf"
    with pytest.raises(MaterialConfigError, match="cannot read"):
        load_gdf_beam(path)
    path.write_bytes(b"not a gdf")
    with pytest.raises(MaterialConfigError, match="cannot read"):
        load_gdf_beam(path)
    write_gdf(path, kind="position")
    with pytest.raises(MaterialConfigError, match="screen-only"):
        load_gdf_beam(path)
    write_gdf(path, creator="other")
    with pytest.raises(MaterialConfigError, match="GPT-produced"):
        load_gdf_beam(path)


def test_sample_values(tmp_path):
    arrays = fields()
    arrays.update(
        Bx=np.zeros(2),
        By=np.zeros(2),
        Bz=np.full(2, 0.328376176361),
        G=np.full(2, 1.05870853551),
        nmacro=np.full(2, 62415.0907446 * 50),
    )
    beam = load_gdf_beam(write_gdf(tmp_path / "sample.gdf", arrays), normalization="gdf_charge")
    np.testing.assert_allclose(beam.energy_keV, 30, rtol=1e-8)
    assert beam.signed_charge_c == pytest.approx(-1e-12, rel=1e-10)
    assert beam.absolute_charge_c == pytest.approx(1e-12, rel=1e-10)


def test_weighted_correlated_seeded_sampling(tmp_path):
    beam = load_gdf_beam(write_gdf(tmp_path / "weighted.gdf"))
    first = beam.sample(10000, 17, 0.1)
    second = beam.sample(10000, 17, 0.1)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.dtype.kind == "f"
    position, direction, energy, time = first
    row_one = position[:, 0] > 0
    assert row_one.mean() == pytest.approx(0.1, abs=0.012)
    assert np.all(direction[row_one, 0] > 0)
    assert np.all(direction[~row_one, 0] < 0)
    np.testing.assert_allclose(energy[row_one], beam.energy_keV[0])
    np.testing.assert_allclose(energy[~row_one], beam.energy_keV[1])
    np.testing.assert_array_equal(time, 0)
    # Shift origin by 1 mm: individual slopes determine the transverse drift.
    moved, _, _, _ = beam.sample(10000, 17, 0.101)
    np.testing.assert_allclose(
        moved[:, :2] - position[:, :2], 1e7 * direction[:, :2] / direction[:, 2, None]
    )
    np.testing.assert_array_equal(moved[:, 2], 0)


def gdf_sweep(path, **kwargs):
    return material_sweep(
        "hopg",
        source="gpt_gdf",
        gdf_path=str(path),
        gdf_z_origin_m=0.1,
        tilt_deg=[10.0],
        tilt_azim_deg=[0.0],
        thickness_ang=[10.0],
        **kwargs,
    )


@pytest.mark.parametrize("screen", [False, True])
def test_normalization_and_cpu_transport(tmp_path, screen):
    arrays = fields()
    if screen:
        arrays["t"] = np.array([1e-9, 2e-9])
    path = write_gdf(
        tmp_path / "beam.gdf",
        arrays,
        times=(0.1,) if screen else (1e-9,),
        kind="position" if screen else "time",
    )
    selection = {"gdf_screen_position_m": 0.1} if screen else {}
    beam = load_gdf_beam(path, screen_position_m=0.1 if screen else None)
    current = build_cases(
        gdf_sweep(path, bunch_charge_pc=2.0, rep_rate_hz=1000, **selection),
        n_electrons=8,
        n_electrons_brem=8,
    )[0]
    charge = build_cases(
        gdf_sweep(path, gdf_normalization="gdf_charge", gdf_repetition_rate_hz=1e6, **selection),
        n_electrons=8,
        n_electrons_brem=8,
    )[0]
    assert current["bunch_charge_pc"] * current["rep_rate_hz"] == 2000
    assert charge["bunch_charge_pc"] == pytest.approx(10 * elementary_charge * 1e12)
    assert charge["rep_rate_hz"] == 1e6
    assert charge["E0_keV"] == pytest.approx(beam.energy_keV.max())
    result = simulate_trajectories(
        charge["E0_keV"],
        8,
        10,
        element="C",
        n_atoms_per_ang3=0.1,
        elastic_model="sr",
        seed=17,
        gdf_source=charge["gdf_source"],
        transport_core="lockstep",
    )
    expected = beam.sample(8, 17, 0.1)
    for key, values in zip(
        ("initial_r_ang", "initial_v_hat", "initial_E_keV", "initial_t0_ang"), expected, strict=True
    ):
        np.testing.assert_array_equal(result[key], values)
    assert result["Ne"] == 8
    assert result["n_step_limited"] == 0
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|cannot read"):
        simulate_trajectories(
            30,
            2,
            10,
            element="C",
            n_atoms_per_ang3=0.1,
            elastic_model="sr",
            gdf_source=charge["gdf_source"],
        )


def test_analytic_compatibility_and_conflicts(tmp_path):
    sweep = material_sweep("hopg", tilt_deg=[10], tilt_azim_deg=[0], thickness_ang=[10])
    cases = build_cases(sweep)
    assert all("gdf_source" not in case for case in cases)
    with pytest.raises(ValueError, match="require source"):
        build_cases(replace(sweep, beam=replace(sweep.beam, gdf_path="wrong.gdf")))
    path = write_gdf(tmp_path / "beam.gdf")
    with pytest.raises(ValueError, match="repetition"):
        build_cases(gdf_sweep(path, gdf_normalization="gdf_charge"))
    with pytest.raises(ValueError, match="analytic"):
        build_cases(gdf_sweep(path, energy_spread_frac=0.1))
    with pytest.raises(ValueError, match="incoherent"):
        build_cases(gdf_sweep(path), coherent_emission=True)


def test_nonrelativistic_limit(tmp_path):
    arrays = fields()
    arrays.update(Bx=np.zeros(2), By=np.zeros(2), Bz=np.full(2, 1e-5))
    beam = load_gdf_beam(write_gdf(tmp_path / "slow.gdf", arrays))
    expected = 0.5 * electron_mass * (1e-5 * c) ** 2 / (elementary_charge * 1000)
    np.testing.assert_allclose(beam.energy_keV, expected, rtol=1e-9)


def test_device_upload_boundary_accepts_only_numeric_arrays(tmp_path, monkeypatch):
    """Exercise the CUDA-shared uploader with NumPy storage, without a GPU."""
    from pyrite.montecarlo.transport import api, batching
    from pyrite.montecarlo.transport.lut import TransportLUTConfig

    uploaded = []

    class NumericBackend:
        def asarray(self, value, *args, **kwargs):
            array = np.asarray(value, *args, **kwargs)
            assert array.dtype.kind in "biuf"
            uploaded.append(array.copy())
            return array

        def asnumpy(self, value):
            return np.asarray(value)

        def __getattr__(self, name):
            return getattr(np, name)

    original = batching._run_per_electron_transport

    def with_numeric_backend(core, xp, *args, **kwargs):
        return original(core, NumericBackend(), *args, **kwargs)

    monkeypatch.setattr(api, "_run_per_electron_transport", with_numeric_backend)
    path = write_gdf(tmp_path / "beam.gdf")
    case = build_cases(gdf_sweep(path), n_electrons=2, n_electrons_brem=2)[0]
    result = simulate_trajectories(
        case["E0_keV"],
        2,
        1,
        element="C",
        n_atoms_per_ang3=0.1,
        elastic_model="sr",
        gdf_source=case["gdf_source"],
        seed=7,
        transport_core="per-electron",
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    assert result["Ne"] == 2
    assert any(a.shape == (2, 3) for a in uploaded)
    assert all(a.dtype.kind in "biuf" for a in uploaded)


def test_absolute_weight_scale_and_identity(tmp_path):
    from pyrite.campaign.config import default_settings
    from pyrite.campaign.profiles import case_content_key, dataset_identity
    from pyrite.results.store import beam_current_na

    path = write_gdf(tmp_path / "beam.gdf")
    beam = load_gdf_beam(path)
    sweep = gdf_sweep(path, gdf_normalization="gdf_charge", gdf_repetition_rate_hz=1e6)
    before = build_cases(sweep)[0]
    identity_before = dataset_identity("hopg", "full", default_settings(), sweep)
    shape = replace(sweep, beam=replace(sweep.beam, gdf_shape_only=True))
    identity_shape = dataset_identity("hopg", "full", default_settings(), shape)
    assert identity_before["parameter_sha256"] != identity_shape["parameter_sha256"]
    assert case_content_key(before) != case_content_key(build_cases(shape)[0])
    arrays = fields()
    arrays["nmacro"] *= 100
    write_gdf(path, arrays)
    scaled = load_gdf_beam(path)
    np.testing.assert_array_equal(beam.probabilities, scaled.probabilities)
    assert scaled.absolute_charge_c == pytest.approx(100 * beam.absolute_charge_c)
    after = build_cases(sweep)[0]
    identity_after = dataset_identity("hopg", "full", default_settings(), sweep)
    assert beam_current_na(after) == pytest.approx(100 * beam_current_na(before))
    assert case_content_key(before) != case_content_key(after)
    assert identity_before["parameter_sha256"] != identity_after["parameter_sha256"]
    configured = gdf_sweep(path, bunch_charge_pc=2, rep_rate_hz=3000)
    assert beam_current_na(build_cases(configured)[0]) == 6.0


def test_tilt_projection_matches_analytic_geometry(tmp_path):
    from pyrite.montecarlo.geometry import project_beam_entry, tilted_geometry

    arrays = fields()
    arrays["Bx"][:] = arrays["By"][:] = 0
    beam = load_gdf_beam(write_gdf(tmp_path / "beam.gdf", arrays))
    flat, _, _, _ = beam.sample(32, 12, 0.1)
    tilted, directions, _, _ = beam.sample(32, 12, 0.1, 0.3, 0.7)
    expected = project_beam_entry(flat[:, :2], 0.3, 0.7)
    np.testing.assert_allclose(tilted[:, :2], expected, rtol=1e-12, atol=1e-8)
    direction, _ = tilted_geometry(np.pi / 2, 0.3, 0.7)
    np.testing.assert_allclose(directions, np.tile(direction, (32, 1)), atol=1e-14)


def test_screen_selection_timing_and_coordinates(tmp_path):
    from pyrite.montecarlo.gdf import inspect_gdf

    arrays = fields()
    arrays["t"] = np.array([2e-9, 5e-9])
    path = write_gdf(tmp_path / "screen.gdf", arrays, times=(0.0, 0.2), kind="position")
    assert inspect_gdf(path)["selected"] is None
    beam = load_gdf_beam(path, screen_position_m=0.0)
    assert beam.time_s is None
    assert beam.screen_position_m == 0.0
    position, _, _, times = beam.sample(100, 42, 0.1)
    first = position[:, 0] > 0
    assert first.any() and (~first).any()
    np.testing.assert_allclose(times[first], 0)
    np.testing.assert_allclose(times[~first], 3e-9 * c * 1e10)
    report = inspect_gdf(path, screen_position_m=0.0)["selected"]
    assert report["centroid_z_origin_m"] == pytest.approx(0.1)
    assert report["coordinates"]["x"]["weighted_mean_m"] == pytest.approx(-1.7e-5)
    with pytest.raises(MaterialConfigError, match="mutually exclusive"):
        load_gdf_beam(path, 1e-9, screen_position_m=0)
    with pytest.raises(MaterialConfigError, match="ambiguous"):
        load_gdf_beam(path, screen_position_m=0.1, screen_tolerance_m=0.11)
    with pytest.raises(MaterialConfigError, match="unavailable"):
        load_gdf_beam(path, screen_position_m=0.3)
    assert load_gdf_beam(path, screen_position_m=1e-10).screen_position_m == 0


@pytest.mark.parametrize("value", [None, np.array([0.0]), np.array([0.0, np.nan])])
def test_screen_requires_valid_crossing_times(tmp_path, value):
    arrays = fields()
    if value is not None:
        arrays["t"] = value
    path = write_gdf(tmp_path / "screen.gdf", arrays, times=(0.1,), kind="position")
    with pytest.raises(MaterialConfigError, match="missing required|lengths|finite"):
        load_gdf_beam(path, screen_position_m=0.1)


def test_mixed_outputs_keep_selections_separate(tmp_path):
    from pyrite.montecarlo.gdf import inspect_gdf

    path = write_gdf(tmp_path / "mixed.gdf")
    with path.open("rb") as stream:
        blocks = easygdf.load(stream)["blocks"]
    arrays = fields()
    arrays["t"] = np.array([1e-9, 2e-9])
    blocks.append(
        {
            "name": "position",
            "value": -0.1,
            "children": [{"name": key, "value": value} for key, value in arrays.items()],
        }
    )
    easygdf.save(str(path), blocks=blocks, creator="GPT")
    assert len(list_gdf_times(path)) == 1
    assert inspect_gdf(path)["selected"]["time_s"] is not None
    assert load_gdf_beam(path).arrival_time_s is None
    assert load_gdf_beam(path, screen_position_m=-0.1).arrival_time_s is not None


def test_shape_only_sweep_and_cpu_clock(tmp_path):
    arrays = fields()
    arrays["Bz"] = np.full(2, 1e-5)
    arrays["Bx"] = np.array([1e-6, -2e-6])
    arrays["By"] = np.zeros(2)
    arrays["t"] = np.array([1e-9, 2e-9])
    path = write_gdf(tmp_path / "shape.gdf", arrays, times=(0.1,), kind="position")
    beam = load_gdf_beam(path, screen_position_m=0.1)
    sweep = gdf_sweep(path, gdf_screen_position_m=0.1, gdf_shape_only=True, energy_keV=[20, 30])
    cases = build_cases(sweep, n_electrons=8, n_electrons_brem=8)
    assert {case["E0_keV"] for case in cases} == {20, 30}
    original = beam.sample(100, 42, 0.101)
    for energy in (20, 30):
        pos, dirs, energies, clock = beam.sample(100, 42, 0.101, energy_keV=energy)
        np.testing.assert_array_equal(pos, original[0])
        np.testing.assert_array_equal(dirs, original[1])
        np.testing.assert_array_equal(energies, np.full(100, energy))
        gamma = 1 + energy * 1000 * elementary_charge / (electron_mass * c**2)
        beta = np.sqrt(1 - gamma**-2)
        np.testing.assert_allclose(clock, 1e7 / dirs[:, 2] / beta, rtol=1e-12)
        at_plane = beam.sample(100, 42, 0.1, energy_keV=energy)
        np.testing.assert_array_equal(at_plane[3], 0)
    for case in cases:
        assert case["gdf_source"]["shape_only"] is True
        result = simulate_trajectories(
            case["E0_keV"],
            8,
            10,
            element="C",
            n_atoms_per_ang3=0.1,
            elastic_model="sr",
            seed=42,
            gdf_source=case["gdf_source"],
            transport_core="lockstep",
        )
        np.testing.assert_array_equal(result["initial_E_keV"], np.full(8, case["E0_keV"]))
        np.testing.assert_array_equal(result["initial_t0_ang"], 0)
    for invalid in (0, -1, np.nan, np.inf):
        with pytest.raises(MaterialConfigError, match="finite and positive"):
            beam.sample(2, 42, 0.1, energy_keV=invalid)
