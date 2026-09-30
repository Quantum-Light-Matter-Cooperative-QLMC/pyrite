"""montecarlo.py: the "no silent default material" guards + the transport table.



Importing montecarlo prints a GPU/CPU banner and is otherwise CPU-only here; no

full sweep is run (that lives in checks/)."""

import pathlib
import warnings

import numpy as np
import pytest

import pyrite.montecarlo.transport as transport
from pyrite.campaign.sweep import BeamSpec, crystal_params
from pyrite.montecarlo import (
    TRANSPORT_ELEMENTS,
    _normalize_composition,
    mc_spectrum,
    simulate_trajectories,
)
from pyrite.montecarlo.runner import scheduling


def test_normalize_requires_material():
    with pytest.raises(ValueError):
        _normalize_composition(None, None, None)


def test_normalize_composition_passthrough():
    assert _normalize_composition(None, None, [("Mo", 0.01)]) == [("Mo", 0.01)]


def test_simulate_trajectories_refuses_silent_default():
    with pytest.raises(ValueError):
        simulate_trajectories(30.0, 4, 1e4)  # no element / composition given


def test_mc_spectrum_requires_crystal_and_hkl():
    with pytest.raises(TypeError):
        mc_spectrum({"E_keV": np.array([30.0])}, np.arange(50.0, 100.0, 1.0))


def test_mc_spectrum_requires_B_ang2():
    # crystal + hkl supplied, but B_ang2 left at its sentinel -> explicit ValueError

    with pytest.raises(ValueError):
        mc_spectrum({}, np.arange(50.0, 100.0, 1.0), crystal="silicon", hkl_list=[(1, 1, 1)])


def test_tellurium_in_transport_table():
    assert TRANSPORT_ELEMENTS["Te"]["Z"] == 52

    assert TRANSPORT_ELEMENTS["Te"]["A"] == pytest.approx(127.6, abs=0.1)


def test_vanadium_transport_parameters():
    assert TRANSPORT_ELEMENTS["V"] == {
        "Z": 23,
        "A": pytest.approx(50.9415, abs=0.0001),
        "J_keV": 0.245,
    }


def test_titanium_transport_parameters():
    assert TRANSPORT_ELEMENTS["Ti"] == {
        "Z": 22,
        "A": pytest.approx(47.867, abs=0.001),
        "J_keV": 0.233,
    }


def test_niobium_transport_parameters():
    params = TRANSPORT_ELEMENTS["Nb"]
    assert params == {"Z": 41, "A": pytest.approx(92.906, abs=0.001), "J_keV": 0.417}


def test_mott_missing_table_raises_naming_config_key(monkeypatch, tmp_path):
    """No SRD 64 table: ``"mott"`` fails naming the fix, never SR angles (#263)."""
    fallback_calls = []
    monkeypatch.setattr(
        transport.scattering,
        "_sample_cos_theta_sr_numba",
        lambda *args: fallback_calls.append(args),
    )
    monkeypatch.setenv("PYRITE_MOTT_TABLES_DIR", str(tmp_path))
    energies = np.array([30.0, 25.0])
    with pytest.raises(transport.MottTableUnavailableError, match=r"mott\.tables_dir") as err:
        transport._sample_cos_theta(41, energies, np.random.default_rng(1), "mott", "Nb")
    assert "DisplayCalcTCSTableForNb.csv" in str(err.value)
    assert "srdata.nist.gov/srd64" in str(err.value)
    assert fallback_calls == []

    monkeypatch.delenv("PYRITE_MOTT_TABLES_DIR")
    with pytest.raises(transport.MottTableUnavailableError, match="is not set"):
        transport._mott_alpha_table("Nb", 41)


def test_mott_missing_table_fails_layer_tables_before_transport(monkeypatch, tmp_path):
    from pyrite.montecarlo.transport.layer_tables import build_layer_tables

    monkeypatch.setenv("PYRITE_MOTT_TABLES_DIR", str(tmp_path))
    layers = [(0.0, 1.0e4, [("Si", 0.05)])]
    with pytest.raises(transport.MottTableUnavailableError, match="Si"):
        build_layer_tables(layers, "mott", None)
    # Other models never read the tables.
    build_layer_tables(layers, "sr", None)


def test_synthetic_mott_fixture_reproduces_analytic_screening():
    """The test fixture's tables calibrate back to Joy/Bishop alpha (see helper)."""
    logE, logA = transport._mott_alpha_table("Si", 14)
    energy_keV = np.logspace(-1.5, 4.5, 200)
    alpha = 10.0 ** np.interp(np.log10(energy_keV * 1e3), logE, logA)
    np.testing.assert_allclose(alpha, transport._alpha_sr_joy(14, energy_keV), rtol=1e-12)


@pytest.mark.parametrize(
    ("element", "expected"),
    [
        ("Fe", {"Z": 26, "A": 55.845, "J_keV": 0.286}),
        ("Bi", {"Z": 83, "A": 208.98040, "J_keV": 0.823}),
        ("Re", {"Z": 75, "A": 186.207, "J_keV": 0.736}),
        ("Ta", {"Z": 73, "A": 180.94788, "J_keV": 0.718}),
    ],
)
def test_new_element_transport_parameters(element, expected):
    assert TRANSPORT_ELEMENTS[element] == expected


def test_hbn_composition_runs_transport():
    cp = crystal_params("hbn")

    segs = simulate_trajectories(
        30.0,
        4,
        100.0,
        composition=cp["composition"],
        seed=123,
        max_steps=5,
    )

    assert segs["Ne"] == 4
    assert len(segs["E_keV"]) > 0
    np.testing.assert_array_equal(segs["initial_E_keV"], np.full(4, 30.0))
    np.testing.assert_array_equal(
        segs["initial_v_hat"],
        np.tile([0.0, 0.0, 1.0], (4, 1)),
    )
    np.testing.assert_array_equal(segs["initial_t0_ang"], np.zeros(4))


@pytest.mark.parametrize("shape", ["gaussian", "uniform"])
def test_longitudinal_bunch_sampling_matches_requested_rms(shape):
    sigma_fs = 17.0
    offsets_ang = transport._sample_bunch_offsets(
        200_000,
        sigma_fs,
        shape,
        None,
        seed=842,
    )
    offsets_fs = offsets_ang / transport.C_ANG_PER_FS

    assert offsets_fs.mean() == pytest.approx(0.0, abs=1e-13)
    assert offsets_fs.std() == pytest.approx(sigma_fs, rel=5e-3)


def test_longitudinal_bunch_stream_does_not_perturb_transport():
    cp = crystal_params("hbn")
    common = dict(composition=cp["composition"], seed=17, max_steps=10)
    point = simulate_trajectories(30.0, 40, 100.0, **common)
    bunched = simulate_trajectories(30.0, 40, 100.0, bunch_length_fs=10.0, **common)

    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer"):
        np.testing.assert_array_equal(point[key], bunched[key])
    assert np.any(bunched["initial_t0_ang"] != 0.0)
    np.testing.assert_array_equal(
        bunched["t0_ang"],
        bunched["initial_t0_ang"][bunched["elec_id"]],
    )


def test_explicit_longitudinal_offsets_override_shape_and_center():
    cp = crystal_params("hbn")
    supplied_fs = np.array([1.0, 2.0, 4.0, 9.0])
    segs = simulate_trajectories(
        30.0,
        4,
        100.0,
        composition=cp["composition"],
        seed=123,
        max_steps=5,
        bunch_length_fs=99.0,
        long_shape="not-used",
        long_offsets_fs=tuple(supplied_fs),
    )
    expected = (supplied_fs - supplied_fs.mean()) * transport.C_ANG_PER_FS

    np.testing.assert_allclose(segs["initial_t0_ang"], expected)
    np.testing.assert_array_equal(segs["t0_ang"], expected[segs["elec_id"]])


def test_longitudinal_bunch_zero_length_is_point_bunch():
    point = transport._sample_bunch_offsets(8, None, "gaussian", None, seed=7)
    zero = transport._sample_bunch_offsets(8, 0.0, "gaussian", None, seed=7)
    np.testing.assert_array_equal(point, zero)


@pytest.mark.parametrize(
    ("offsets", "match"),
    [
        (np.zeros((2, 2)), "one-dimensional"),
        ([0.0, np.nan], "finite"),
    ],
)
def test_explicit_longitudinal_offsets_reject_invalid_arrays(offsets, match):
    with pytest.raises(ValueError, match=match):
        transport._sample_bunch_offsets(
            4 if np.asarray(offsets).ndim > 1 else 2, None, "gaussian", offsets, 1
        )


@pytest.mark.parametrize("sigma", [-1.0, np.inf, np.nan])
def test_longitudinal_bunch_rejects_invalid_rms(sigma):
    with pytest.raises(ValueError, match="finite and non-negative"):
        transport._sample_bunch_offsets(2, sigma, "gaussian", None, 1)


def test_beam_fwhm_mm_zero_and_none_are_equivalent():
    # 0 is falsy, same branch as None -> point source, no RNG draw for the offset
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=123, max_steps=5)
    none_segs = simulate_trajectories(30.0, 20, 100.0, beam_fwhm_mm=None, **kw)
    zero_segs = simulate_trajectories(30.0, 20, 100.0, beam_fwhm_mm=0.0, **kw)
    assert np.array_equal(none_segs["r_mid"], zero_segs["r_mid"])


def test_beam_fwhm_mm_offsets_transverse_position_only():
    # A finite beam spot must be a pure (x, y) translation of each electron's
    # WHOLE trajectory -- every other physics output (energies, directions,
    # segment lengths, ages, backscatter/transmit counts) is untouched, because
    # the offset is drawn from an RNG stream independent of the transport rng.
    cp = crystal_params("hbn")
    args = (30.0, 200, 100.0)
    common = dict(composition=cp["composition"], seed=7, max_steps=20)
    point = simulate_trajectories(*args, beam_fwhm_mm=None, **common)
    finite = simulate_trajectories(*args, beam_fwhm_mm=1.0, **common)

    for key in ("E_keV", "v_hat", "L_ang", "t_ang", "elec_id", "layer"):
        assert np.array_equal(point[key], finite[key]), key
    for key in (
        "n_backscattered",
        "n_transmitted",
        "n_stopped",
        "n_side_exited",
        "n_missed",
        "Ne",
    ):
        assert point[key] == finite[key]

    # depth (z) is identical; transverse (x, y) differs by a per-electron
    # constant offset shared across every segment that electron emits
    assert np.array_equal(point["r_mid"][:, 2], finite["r_mid"][:, 2])
    d_xy = finite["r_mid"][:, :2] - point["r_mid"][:, :2]
    per_electron_offset = {}
    for eid, dxy in zip(finite["elec_id"], d_xy, strict=False):
        if eid in per_electron_offset:
            assert np.allclose(per_electron_offset[eid], dxy)
        else:
            per_electron_offset[eid] = dxy
    assert any(np.linalg.norm(v) > 0 for v in per_electron_offset.values())


def test_beam_fwhm_mm_matches_gaussian_sigma():
    # statistical check: the sampled transverse spread matches sigma = FWHM /
    # (2 sqrt(2 ln 2)), converted mm -> Angstrom (1 mm = 1e7 Ang)
    cp = crystal_params("hbn")
    fwhm_mm = 1.0
    segs = simulate_trajectories(
        30.0,
        20000,
        100.0,
        composition=cp["composition"],
        seed=7,
        max_steps=200,
        beam_fwhm_mm=fwhm_mm,
    )
    # first segment per electron carries the raw entry offset (before any
    # elastic-scattering-driven lateral drift accumulates)
    _, first_idx = np.unique(segs["elec_id"], return_index=True)
    x0y0 = (
        segs["r_mid"][first_idx, :2]
        - 0.5 * segs["L_ang"][first_idx, None] * segs["v_hat"][first_idx, :2]
    )
    expected_sigma_ang = fwhm_mm * 1.0e7 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.std(x0y0) == pytest.approx(expected_sigma_ang, rel=0.05)


def test_beam_fwhm_mm_seed_reproducible():
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=42, max_steps=5, beam_fwhm_mm=1.0)
    a = simulate_trajectories(30.0, 50, 100.0, **kw)
    b = simulate_trajectories(30.0, 50, 100.0, **kw)
    assert np.array_equal(a["r_mid"], b["r_mid"])


def test_finite_footprint_truncates_lateral_transport_and_counts_side_exit():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30.0,
        64,
        1e8,
        composition=cp["composition"],
        seed=4,
        elastic_model="sr",
        beam_dir=[0.8, 0.0, 0.6],
        max_steps=1,
        crystal_width_mm=1e-6,
        crystal_height_mm=1e-6,
    )
    assert segs["n_side_exited"] == 64
    assert segs["n_transmitted"] == segs["n_backscattered"] == 0
    assert np.all(np.abs(segs["r_mid"][:, 0]) < 5.0)


def test_finite_footprint_counts_missed_gaussian_entries_without_changing_ne():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30.0,
        2_000,
        100.0,
        composition=cp["composition"],
        seed=7,
        elastic_model="sr",
        max_steps=1,
        beam_fwhm_mm=1.0,
        crystal_width_mm=1e-6,
        crystal_height_mm=1e-6,
    )
    assert segs["Ne"] == 2_000
    assert segs["n_missed"] > 1_900
    assert np.all(np.abs(segs["r_mid"][:, :2]) <= 5.0)


def test_all_missed_entries_return_typed_empty_segment_arrays():
    cp = crystal_params("hbn")
    segs = simulate_trajectories(
        30.0,
        4,
        100.0,
        composition=cp["composition"],
        seed=2,
        elastic_model="sr",
        beam_fwhm_mm=1e8,
        crystal_width_mm=1e-6,
        crystal_height_mm=1e-6,
    )
    assert segs["n_missed"] == segs["Ne"] == 4
    assert segs["r_mid"].shape == segs["v_hat"].shape == (0, 3)
    assert segs["L_ang"].shape == segs["E_keV"].shape == segs["t_ang"].shape == (0,)
    assert segs["elec_id"].dtype == np.dtype("int64")
    assert segs["layer"].dtype == np.dtype("int16")


def test_grazing_incidence_projects_beam_off_finite_sample():
    # A 1 mm beam sits well inside a 5 mm sample at normal incidence (no
    # misses). At 89 deg tilt the entry spot stretches by 1/cos(89 deg) ~ 57x
    # along the tilt azimuth, overfilling the 5 mm sample, so most of the beam
    # now lands off the tilted face and is counted as missed -- the geometric
    # overlap loss that competes with the 1/cos path-length yield enhancement.
    from pyrite.montecarlo.geometry import tilted_geometry

    cp = crystal_params("hbn")
    kw = dict(
        composition=cp["composition"],
        seed=11,
        elastic_model="sr",
        max_steps=200,
        beam_fwhm_mm=1.0,
        crystal_width_mm=5.0,
        crystal_height_mm=5.0,
    )
    beam0, _ = tilted_geometry(np.deg2rad(20.0), 0.0, 0.0)
    beam89, _ = tilted_geometry(np.deg2rad(20.0), np.deg2rad(89.0), 0.0)
    normal = simulate_trajectories(
        30.0, 4000, 100.0, beam_dir=beam0, tilt_polar_rad=0.0, tilt_azim_rad=0.0, **kw
    )
    grazing = simulate_trajectories(
        30.0, 4000, 100.0, beam_dir=beam89, tilt_polar_rad=np.deg2rad(89.0), tilt_azim_rad=0.0, **kw
    )
    assert normal["n_missed"] == 0
    assert grazing["n_missed"] > 3000
    assert grazing["Ne"] == 4000  # misses stay in Ne (per-incident normalization)


def test_grazing_projection_identity_at_normal_incidence_is_bitwise():
    # tilt_polar_rad=0 makes project_beam_entry the identity, so passing the new
    # tilt kwargs leaves every beam_fwhm_mm result bit-for-bit.
    cp = crystal_params("hbn")
    kw = dict(
        composition=cp["composition"],
        seed=5,
        elastic_model="sr",
        max_steps=5,
        beam_fwhm_mm=1.0,
        crystal_width_mm=5.0,
        crystal_height_mm=5.0,
    )
    a = simulate_trajectories(30.0, 60, 100.0, **kw)
    b = simulate_trajectories(30.0, 60, 100.0, tilt_polar_rad=0.0, tilt_azim_rad=0.0, **kw)
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "elec_id"):
        assert np.array_equal(a[key], b[key]), key


def test_omitted_footprint_is_bitwise_legacy_transport():
    cp = crystal_params("hbn")
    kw = dict(composition=cp["composition"], seed=19, elastic_model="sr", max_steps=5)
    old = simulate_trajectories(30.0, 40, 100.0, **kw)
    new = simulate_trajectories(
        30.0,
        40,
        100.0,
        crystal_width_mm=None,
        crystal_height_mm=None,
        **kw,
    )
    for key in ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "elec_id", "layer"):
        assert np.array_equal(old[key], new[key]), key


@pytest.mark.parametrize(
    ("width_mm", "height_mm"), [(None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0)]
)
def test_finite_footprint_rejects_invalid_dimension_pairs(width_mm, height_mm):
    cp = crystal_params("hbn")
    with pytest.raises(ValueError):
        simulate_trajectories(
            30.0,
            4,
            100.0,
            composition=cp["composition"],
            crystal_width_mm=width_mm,
            crystal_height_mm=height_mm,
        )


def test_run_cases_should_stop_halts_new_dispatch(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(scheduling, "run_case", lambda case: {"name": case["name"]})
    calls = {"n": 0}

    def stop_after_two():
        return calls["n"] >= 2

    seen = []

    def cb(i, case, out):
        calls["n"] += 1
        seen.append(case["name"])

    cases = [{"name": f"c{i}"} for i in range(5)]
    results = runner.run_cases(
        cases, max_workers=0, progress=False, callback=cb, should_stop=stop_after_two
    )
    assert seen == ["c0", "c1"]
    assert results[0] is not None and results[1] is not None
    assert results[2] is None and results[3] is None and results[4] is None


def test_run_cases_should_stop_none_runs_everything(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(scheduling, "run_case", lambda case: {"name": case["name"]})
    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=0, progress=False, should_stop=None)
    assert all(r is not None for r in results)


# ---- engine= regime-split dispatch (2026-07-18 design) ----------------------
# CI has no GPU (_GPU is False), so the dispatch tests below monkeypatch
# runner._RESOURCE_POLICY.gpu to exercise all four branch combinations, and stand in a
# same-process "pool" (real, already-finished concurrent.futures.Future
# objects, so as_completed's normal machinery still works) for
# ProcessPoolExecutor so no real subprocess/CUDA context is ever requested.
# Only test_run_cases_engine_cpu_end_to_end_returns_finite_spectrum spins up a
# real (1-worker) pool, to prove the NumPy spectrum path actually works when
# a worker is forced onto it (design doc Sec. 2 verification item).


class _SyncProcessPoolExecutor:
    """Stand-in for ProcessPoolExecutor that runs every submission synchronously
    in THIS process and returns a real (already-finished) Future, so the CPU-
    pool branch's as_completed(...) drain works unmodified. Records its
    constructor args on the class so a test can assert which initializer/
    initargs run_cases chose without touching a real worker process."""

    captured: dict = {}

    def __init__(self, max_workers=None, initializer=None, initargs=(), mp_context=None):
        type(self).captured = dict(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
            mp_context=mp_context,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future

        fut = Future()
        fut.set_result(fn(*args))
        return fut


def test_run_cases_invalid_engine_raises():
    from pyrite.montecarlo import runner

    with pytest.raises(ValueError, match="engine"):
        runner.run_cases([{"name": "c0"}], engine="bogus")


def test_run_cases_engine_auto_uses_cpu_pool_when_no_gpu(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(scheduling, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=2, progress=False, engine="auto")

    assert [r["name"] for r in results] == ["c0", "c1", "c2"]
    assert _SyncProcessPoolExecutor.captured["initializer"] is runner._worker_init
    assert _SyncProcessPoolExecutor.captured["initargs"] == (False,)


def test_run_cases_nsys_uses_spawn_processes(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "nsys", True)
    monkeypatch.setattr(scheduling, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(3)]
    runner.run_cases(cases, max_workers=2, progress=False, engine="auto")

    context = _SyncProcessPoolExecutor.captured["mp_context"]
    assert context is not None
    assert context.get_start_method() == "spawn"


def test_run_cases_engine_cpu_forces_cpu_pool_when_gpu_present(monkeypatch):
    from pyrite.montecarlo import runner

    # _GPU True simulates a GPU box; engine="cpu" must still take the
    # full-case CPU pool (not the GPU-pipeline branch, which submits
    # _transport_case/_spectrum_case instead and would blow up on these
    # name-only stub cases).
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=2, progress=False, engine="cpu")

    assert [r["name"] for r in results] == ["c0", "c1", "c2"]
    assert _SyncProcessPoolExecutor.captured["initializer"] is runner._worker_init
    assert _SyncProcessPoolExecutor.captured["initargs"] == (True,)


def test_run_cases_engine_gpu_errors_when_accelerator_unavailable(monkeypatch):
    from pyrite._backend import BackendUnavailableError
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)

    cases = [{"name": f"c{i}"} for i in range(3)]
    with pytest.raises(BackendUnavailableError, match="engine='gpu'"):
        runner.run_cases(cases, max_workers=2, progress=False, engine="gpu")


def test_run_cases_engine_cpu_end_to_end_returns_finite_spectrum():
    """A real (1-worker) engine="cpu" run: proves _worker_init(force_cpu=True)
    doesn't break the worker, and that mc_spectrum/mc_brem_spectrum's xp/REAL
    (rebound by force_cpu) still produce a finite spectrum -- the design doc's
    Sec. 2 verification item, covered end-to-end rather than assumed."""
    from pyrite.campaign.sweep import Sweep, build_cases
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.montecarlo import runner

    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30),
        tilt_deg=30.0,
        detector=Detector(
            energy_bins=EnergyBins(
                line=np.arange(50.0, 300.0, 5.0),
                brem=np.arange(0.0, 1000.0, 100.0),
            )
        ),
    )
    cases = build_cases(sweep, n_electrons=40, n_electrons_brem=20)
    results = runner.run_cases(cases, max_workers=1, progress=False, engine="cpu")

    assert len(results) == 1
    out = results[0]
    assert out["spec"].size > 0
    assert np.all(np.isfinite(out["spec"]))
    assert np.all(np.isfinite(out["brem"]))


# ---- _cpu_pool_workers memory-aware sizing (2026-07-18 OOM fix) -------------
# The uncapped ncpu*3//4 = 24-worker pool OOM'd remote-host (45 GiB box, ~5.5 GB
# anon-rss per full-case worker at 200 keV): the kernel killed one worker and
# BrokenProcessPool lost the whole run. These tests pin the host probes so the
# sizing is deterministic on any machine.


def test_case_progress_label_names_single_material():
    from pyrite.montecarlo import runner

    assert runner._case_progress_label([{"crystal": "hopg"}, {"crystal": "hopg"}]) == ("hopg cases")
    assert runner._case_progress_label([{"crystal": "hopg"}, {"crystal": "hbn"}]) == ("mixed cases")


def _patch_host(
    monkeypatch,
    *,
    ncpus=32,
    avail_mb=44_900,
    total_mb=48_000,
    budget_mb=6_144,
    pipeline_budget_mb=1_536,
):
    """Fake a host for _cpu_pool_workers; defaults reproduce remote-host's shape."""
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner._RESOURCE_POLICY, "n_cpus", ncpus)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "available_mem_mb", lambda: avail_mb)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "total_mem_mb", total_mb)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "worker_mem_mb", budget_mb)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "pipeline_worker_mem_mb", pipeline_budget_mb)
    return runner


def test_cpu_pool_workers_autosize_is_memory_bound_on_remote_host_shape(monkeypatch):
    """remote-host regression: 32 cores asks for 24 workers, RAM only carries 7."""
    runner = _patch_host(monkeypatch)
    # 44_900 // 6_144 = 7
    assert runner._cpu_pool_workers(None, 980) == 7


def test_cpu_pool_workers_autosize_is_cpu_bound_with_ample_ram(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=1_000_000, total_mb=1_000_000)
    assert runner._cpu_pool_workers(None, 980) == 24  # ncpu * 3 // 4


def test_cpu_pool_workers_pin_is_clamped_by_memory(monkeypatch):
    """An explicit pin cannot re-create the OOM; raise PYRITE_MC_WORKER_MEM_MB to
    deliberately run tighter than the measured per-worker budget."""
    runner = _patch_host(monkeypatch)
    assert runner._cpu_pool_workers(24, 980) == 7


def test_cpu_pool_workers_pin_under_cap_is_honored(monkeypatch):
    runner = _patch_host(monkeypatch)
    assert runner._cpu_pool_workers(3, 980) == 3


def test_cpu_pool_workers_bounded_by_cases_and_floored_at_one(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=1_000_000, total_mb=1_000_000)
    assert runner._cpu_pool_workers(None, 2) == 2  # never more workers than cases
    _patch_host(monkeypatch, avail_mb=1_000, total_mb=1_000)  # cap rounds to 0
    assert runner._cpu_pool_workers(None, 980) == 1  # degrade, don't refuse


def test_cpu_pool_workers_unknown_cpu_count_falls_back(monkeypatch):
    runner = _patch_host(monkeypatch, ncpus=None, avail_mb=1_000_000, total_mb=1_000_000)
    assert runner._cpu_pool_workers(None, 980) == 6


def test_run_cases_cpu_pool_receives_the_capped_worker_count(monkeypatch):
    """Wiring check: the executor must see the clamped count. Guards the
    fall-through-returns-None failure, which ProcessPoolExecutor would
    silently accept as 'use all cores' -- worse than the bug being fixed."""
    runner = _patch_host(monkeypatch)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(scheduling, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(10)]
    runner.run_cases(cases, progress=False, engine="cpu")
    assert _SyncProcessPoolExecutor.captured["max_workers"] == 7


# ---- _gpu_pipeline_workers memory-aware sizing -------------------------------
# The GPU-pipeline transport pool spawns full worker processes just like the CPU
# pool, but originally sized them as ncpu//2 with no RAM cap. On remote-host that OOM'd
# a worker at pool startup and the first submit raised BrokenProcessPool, losing
# the whole hopg scan (exit 1). The cap now binds this path too.


def test_gpu_pipeline_workers_autosize_is_memory_bound(monkeypatch):
    """32 cores would ask for ncpu // 2 = 16 transport workers; RAM carries 2."""
    runner = _patch_host(monkeypatch, avail_mb=11_000, total_mb=16_000)
    # 11_000 // 1_536 = 7 slots, minus the 2 prefetched ahead, halved between
    # each worker and the payload it hands the driver.
    assert runner._gpu_pipeline_workers(None, 980) == 2


def test_gpu_pipeline_workers_autosize_is_cpu_bound_with_ample_ram(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=1_000_000, total_mb=1_000_000)
    assert runner._gpu_pipeline_workers(None, 980) == 16  # ncpu // 2


def test_gpu_pipeline_workers_uses_the_transport_only_budget(monkeypatch):
    """Transport-only workers must not be charged the full-case footprint.

    24-core / 23.4 GB box: the shared 6144 MiB budget capped the pipeline at 2
    workers while measured child RSS was 552-1033 MB."""
    runner = _patch_host(monkeypatch, ncpus=24, avail_mb=16_687, total_mb=24_600)
    assert runner._mem_worker_cap() == 2  # full-case budget, unchanged
    assert runner._pipeline_slot_cap() == 10  # 16_687 // 1_536
    assert runner._gpu_pipeline_workers(None, 980) == 4  # (10 - 2) // 2


def test_gpu_pipeline_workers_pin_is_clamped_by_memory_and_warns(monkeypatch):
    """An explicit --max-workers pin cannot re-create the OOM either -- but the
    clamp is announced, not silent."""
    runner = _patch_host(monkeypatch, avail_mb=11_000, total_mb=16_000)
    with pytest.warns(RuntimeWarning, match="host RAM admits 2"):
        assert runner._gpu_pipeline_workers(16, 980) == 2


def test_gpu_pipeline_workers_pin_under_cap_is_silent(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=20_000, total_mb=24_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert runner._gpu_pipeline_workers(4, 980) == 4


def test_gpu_pipeline_workers_degrades_to_serial_under_memory_pressure(monkeypatch):
    """Cap rounds to 0/1 -> caller sees < 2 and drops to serial, never OOMs."""
    runner = _patch_host(monkeypatch, avail_mb=1_000, total_mb=1_000)
    assert runner._gpu_pipeline_workers(None, 980) < 2


# ---- in-flight payload budgeting (2026-08-08 remote-host swap incident) -------------
# Budgeting only the workers let a MoSe2 campaign run 16 workers with 18 cases in
# flight on a 45 GB box -- 34 host-resident segment payloads, peak tree RSS
# 50.3 GB, 12.9 GB of swap, ssh unreachable for ~15 min. The driver holds one
# payload per in-flight case, so they are charged like workers.


def test_gpu_pipeline_holds_worker_and_prefetch_payloads_inside_the_budget(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=11_000, total_mb=16_000)
    nw = runner._gpu_pipeline_workers(None, 980)
    prefetch = runner._gpu_pipeline_prefetch(nw, 980)
    assert prefetch == nw + 2
    assert nw + prefetch <= runner._pipeline_slot_cap()


def test_gpu_pipeline_prefetch_is_reclamped_when_memory_moved(monkeypatch):
    """Sizing and the driver loop read the budget at different moments."""
    runner = _patch_host(monkeypatch, avail_mb=11_000, total_mb=16_000)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "available_mem_mb", lambda: 6_000)  # 3 slots left
    assert runner._gpu_pipeline_prefetch(2, 980) == 1


def test_gpu_pipeline_prefetch_never_exceeds_the_case_count(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=1_000_000, total_mb=1_000_000)
    assert runner._gpu_pipeline_prefetch(16, 3) == 3


def test_run_cases_pipeline_holds_no_more_than_the_budgeted_prefetch(monkeypatch):
    """Wiring check: the driver loop must use the budgeted depth, not nw + 2.

    Pinned to 6 workers against a 7-slot budget, so the two differ: the old
    hard-coded nw + 2 would put 8 payloads in the driver, the budget allows 1."""
    runner = _patch_host(monkeypatch, avail_mb=11_000, total_mb=16_000)
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 6)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr(scheduling, "_transport_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr(
        scheduling,
        "_spectrum_case_retry",
        lambda case, tp, **_kw: {"name": tp["name"]},
    )
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    activity = []
    cases = [{"name": f"c{i}", "Ne": 10} for i in range(12)]
    runner.run_cases(cases, progress=False, on_activity=activity.append)

    assert _SyncProcessPoolExecutor.captured["max_workers"] == 6
    assert {event["transport_prefetch_count"] for event in activity} == {1}
    assert max(event["in_flight_case_count"] for event in activity) == 1


def test_gpu_pipeline_deadline_completes_at_most_one_case_after_expiry(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(scheduling, "_gpu_pipeline_workers", lambda *_args: 2)
    monkeypatch.setattr(scheduling, "_gpu_pipeline_prefetch", lambda *_args: 4)
    monkeypatch.setattr(scheduling, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(scheduling, "_process_pool_kwargs", lambda: {})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    clock = {"seconds": 0}
    started = []

    def transport(case):
        started.append(case["name"])
        return {"name": case["name"]}

    def spectrum(_case, payload, **_kwargs):
        clock["seconds"] += 10
        return payload

    monkeypatch.setattr(scheduling, "_transport_case", transport)
    monkeypatch.setattr(scheduling, "_spectrum_case_retry", spectrum)
    cases = [{"name": f"c{i}", "Ne": 10} for i in range(6)]
    activity = []
    results = runner.run_cases(
        cases,
        progress=False,
        should_stop=lambda: clock["seconds"] >= 15,
        on_activity=activity.append,
    )

    assert started == ["c0", "c1"]
    assert [r["name"] if r else None for r in results] == ["c0", "c1", None, None, None, None]
    assert max(event["in_flight_case_count"] for event in activity) == 1
    assert clock["seconds"] == 20

    started.clear()
    results = runner.run_cases(cases, progress=False, should_stop=lambda: clock["seconds"] >= 15)
    assert started == []
    assert all(result is None for result in results)


# ---- _usable_cpus: the allocation, not the machine ---------------------------
# os.cpu_count() reports the box. Under the lab's --cpus-per-task=8 SLURM
# allocation on a 32-core node it still returned 32, so the pipeline sized a
# 16-worker pool into 8 CPUs -- the other half of the swap incident above.


def test_usable_cpus_honors_the_affinity_mask(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(runner, "_cgroup_cpu_quota", lambda: None)
    assert runner._usable_cpus() == 8


def test_usable_cpus_honors_slurm_and_the_cgroup_quota(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _pid: set(range(32)))
    monkeypatch.setattr(runner, "_cgroup_cpu_quota", lambda: 6)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert runner._usable_cpus() == 6  # tightest limit wins
    monkeypatch.setattr(runner, "_cgroup_cpu_quota", lambda: None)
    assert runner._usable_cpus() == 8


def test_usable_cpus_ignores_an_unparseable_slurm_value(monkeypatch):
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _pid: set(range(4)))
    monkeypatch.setattr(runner, "_cgroup_cpu_quota", lambda: None)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8(x2)")  # heterogeneous job syntax
    assert runner._usable_cpus() == 4


def test_cgroup_cpu_quota_reads_v2_then_v1(monkeypatch, tmp_path):
    from pyrite.montecarlo import runner

    (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / "self").mkdir()
    (tmp_path / "proc" / "self" / "cgroup").write_text("0::/slurm/job_1\n")
    leaf = tmp_path / "sys" / "slurm" / "job_1"
    leaf.mkdir(parents=True)
    (leaf / "cpu.max").write_text("800000 100000\n")
    (tmp_path / "sys" / "cpu.max").write_text("max 100000\n")

    def _path(*parts):
        joined = "/".join(str(p) for p in parts)
        joined = joined.replace("/proc/", f"{tmp_path}/proc/", 1)
        joined = joined.replace("/sys/fs/cgroup", f"{tmp_path}/sys", 1)
        return pathlib.Path(joined)

    monkeypatch.setattr(runner, "Path", _path)
    assert runner._cgroup_cpu_quota() == 8

    (leaf / "cpu.max").write_text("max 100000\n")
    assert runner._cgroup_cpu_quota() is None  # unquotaed -> no limit


# ---- _adaptive_chunk grid-aware sizing (2026-07-18 OOM fix, part 2) ---------
# The fixed spec/brem chunk defaults were tuned on the old ~2000-bin line grid;
# widening the grid to 30000 eV (~6000 bins) silently tripled the (chunk, nbins)
# matmul transients per worker. _adaptive_chunk holds the byte product constant:
# chunk = budget_bytes // (3 arrays * nbins * 8 B), clamped to [1, 100_000].


def _patch_cpu_chunk_policy(monkeypatch, *, budget_mb=1920, device_budget_bytes=None):
    """Pin chunk tests to the CPU/fp64 path, independent of test hardware."""
    from pyrite.montecarlo import runner

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)
    monkeypatch.setattr(runner._spectrum_mod, "REAL", np.float64)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "spec_budget_mb", budget_mb)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "device_budget_bytes", device_budget_bytes)
    return runner


def test_adaptive_chunk_reproduces_old_default_on_narrow_grid(monkeypatch):
    runner = _patch_cpu_chunk_policy(monkeypatch)
    assert runner._adaptive_chunk(2000) == 40_000


def test_adaptive_chunk_shrinks_on_the_widened_grid(monkeypatch):
    runner = _patch_cpu_chunk_policy(monkeypatch)
    assert runner._adaptive_chunk(6000) == 13_333


def test_adaptive_chunk_preferred_size_follows_budget_and_ceiling(monkeypatch):
    runner = _patch_cpu_chunk_policy(monkeypatch)
    # 1.92e9 B // (3 * 8 B * 200_000 bins) = 400: no 1000-segment floor.
    assert runner._adaptive_chunk(200_000) == 400
    assert runner._adaptive_chunk(1) == 100_000


def test_adaptive_chunk_cpu_path_does_not_apply_device_admission(monkeypatch):
    runner = _patch_cpu_chunk_policy(
        monkeypatch,
        device_budget_bytes=1,
    )
    # CPU mode deliberately ignores accelerator device-memory admission.
    assert runner._adaptive_chunk(10**9) == 1


def test_adaptive_chunk_honors_budget_override(monkeypatch):
    runner = _patch_cpu_chunk_policy(monkeypatch, budget_mb=480)
    assert runner._adaptive_chunk(2000) == 10_000


def test_adaptive_chunk_accounts_for_eedl_brem_intermediates(monkeypatch):
    runner = _patch_cpu_chunk_policy(monkeypatch)

    assert runner._adaptive_chunk(2000, intermediates=8) == 15_000
    assert runner._admit_chunk(40_000, 2000, intermediates=8) == 40_000
