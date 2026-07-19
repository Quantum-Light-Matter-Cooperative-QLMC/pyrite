"""montecarlo.py: the "no silent default material" guards + the transport table.



Importing montecarlo prints a GPU/CPU banner and is otherwise CPU-only here; no

full sweep is run (that lives in checks/)."""

import numpy as np
import pytest

import cxr_mc.montecarlo.transport as transport
from cxr_mc.montecarlo import (
    TRANSPORT_ELEMENTS,
    _normalize_composition,
    mc_spectrum,
    simulate_trajectories,
)
from cxr_mc.sweep import crystal_params


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


def test_niobium_transport_parameters_and_fallback(monkeypatch):
    params = TRANSPORT_ELEMENTS["Nb"]
    assert params == {"Z": 41, "A": pytest.approx(92.906, abs=0.001), "J_keV": 0.417}

    mott_calls = []
    alpha_calls = []
    original_alpha_sr_joy = transport._alpha_sr_joy

    def missing_mott_table(element, Z):
        mott_calls.append((element, Z))
        raise FileNotFoundError

    def spy_alpha_sr_joy(Z, E_keV):
        alpha_calls.append((Z, E_keV.copy()))
        return original_alpha_sr_joy(Z, E_keV)

    monkeypatch.setattr(transport, "_mott_alpha_table", missing_mott_table)
    monkeypatch.setattr(transport, "_alpha_sr_joy", spy_alpha_sr_joy)

    previous_no_mott = set(transport._NO_MOTT)
    transport._NO_MOTT.discard("Nb")
    energies = np.array([30.0, 25.0])
    try:
        transport._sample_cos_theta(41, energies, np.random.default_rng(1), "mott", "Nb")
        transport._sample_cos_theta(41, energies, np.random.default_rng(2), "mott", "Nb")

        assert mott_calls == [("Nb", 41)]
        assert len(alpha_calls) == 2
        for Z, called_energies in alpha_calls:
            assert Z == 41
            np.testing.assert_array_equal(called_energies, energies)
        assert "Nb" in transport._NO_MOTT
    finally:
        transport._NO_MOTT.clear()
        transport._NO_MOTT.update(previous_no_mott)


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


@pytest.mark.parametrize(("element", "Z"), [("Fe", 26), ("Bi", 83), ("Re", 75), ("Ta", 73)])
def test_new_elements_use_analytic_fallback_without_mott_table(monkeypatch, element, Z):
    mott_calls = []
    alpha_calls = []
    original_alpha_sr_joy = transport._alpha_sr_joy

    def missing_mott_table(called_element, called_Z):
        mott_calls.append((called_element, called_Z))
        raise FileNotFoundError

    def spy_alpha_sr_joy(called_Z, E_keV):
        alpha_calls.append((called_Z, E_keV.copy()))
        return original_alpha_sr_joy(called_Z, E_keV)

    monkeypatch.setattr(transport, "_mott_alpha_table", missing_mott_table)
    monkeypatch.setattr(transport, "_alpha_sr_joy", spy_alpha_sr_joy)

    previous_no_mott = set(transport._NO_MOTT)
    transport._NO_MOTT.discard(element)
    energies = np.array([30.0, 25.0])
    try:
        cos_theta = transport._sample_cos_theta(
            Z, energies, np.random.default_rng(1), "mott", element
        )
        transport._sample_cos_theta(Z, energies, np.random.default_rng(2), "mott", element)

        assert mott_calls == [(element, Z)]
        assert len(alpha_calls) == 2
        for called_Z, called_energies in alpha_calls:
            assert called_Z == Z
            np.testing.assert_array_equal(called_energies, energies)
        assert np.all((-1.0 <= cos_theta) & (cos_theta <= 1.0))
        assert element in transport._NO_MOTT
    finally:
        transport._NO_MOTT.clear()
        transport._NO_MOTT.update(previous_no_mott)


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
        max_steps=1,
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
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
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
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=0, progress=False, should_stop=None)
    assert all(r is not None for r in results)


# ---- engine= regime-split dispatch (2026-07-18 design) ----------------------
# CI has no GPU (_GPU is False), so the dispatch tests below monkeypatch
# runner._GPU to exercise all four branch combinations, and stand in a
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

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        type(self).captured = dict(
            max_workers=max_workers, initializer=initializer, initargs=initargs
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
    from cxr_mc.montecarlo import runner

    with pytest.raises(ValueError, match="engine"):
        runner.run_cases([{"name": "c0"}], engine="bogus")


def test_run_cases_engine_auto_uses_cpu_pool_when_no_gpu(monkeypatch):
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "_GPU", False)
    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=2, progress=False, engine="auto")

    assert [r["name"] for r in results] == ["c0", "c1", "c2"]
    assert _SyncProcessPoolExecutor.captured["initializer"] is runner._worker_init
    assert _SyncProcessPoolExecutor.captured["initargs"] == (False,)


def test_run_cases_engine_cpu_forces_cpu_pool_when_gpu_present(monkeypatch):
    from cxr_mc.montecarlo import runner

    # _GPU True simulates a GPU box; engine="cpu" must still take the
    # full-case CPU pool (not the GPU-pipeline branch, which submits
    # _transport_case/_spectrum_case instead and would blow up on these
    # name-only stub cases).
    monkeypatch.setattr(runner, "_GPU", True)
    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(3)]
    results = runner.run_cases(cases, max_workers=2, progress=False, engine="cpu")

    assert [r["name"] for r in results] == ["c0", "c1", "c2"]
    assert _SyncProcessPoolExecutor.captured["initializer"] is runner._worker_init
    assert _SyncProcessPoolExecutor.captured["initargs"] == (True,)


def test_run_cases_engine_gpu_falls_back_to_cpu_pool_with_warning(monkeypatch):
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "_GPU", False)
    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(3)]
    with pytest.warns(UserWarning, match="engine='gpu'"):
        results = runner.run_cases(cases, max_workers=2, progress=False, engine="gpu")

    assert [r["name"] for r in results] == ["c0", "c1", "c2"]
    assert _SyncProcessPoolExecutor.captured["initargs"] == (False,)  # engine != "cpu"


def test_run_cases_engine_cpu_end_to_end_returns_finite_spectrum():
    """A real (1-worker) engine="cpu" run: proves _worker_init(force_cpu=True)
    doesn't break the worker, and that mc_spectrum/mc_brem_spectrum's xp/REAL
    (rebound by force_cpu) still produce a finite spectrum -- the design doc's
    Sec. 2 verification item, covered end-to-end rather than assumed."""
    from cxr_mc.montecarlo import runner
    from cxr_mc.sweep import Sweep, build_cases

    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        energy_keV=30,
        tilt_deg=30.0,
        E_grid_line=np.arange(50.0, 300.0, 5.0),
        E_grid_brem=np.arange(0.0, 1000.0, 100.0),
    )
    cases = build_cases(sweep, n_electrons=40, n_electrons_brem=20)
    results = runner.run_cases(cases, max_workers=1, progress=False, engine="cpu")

    assert len(results) == 1
    out = results[0]
    assert out["spec"].size > 0
    assert np.all(np.isfinite(out["spec"]))
    assert np.all(np.isfinite(out["brem"]))


# ---- _cpu_pool_workers memory-aware sizing (2026-07-18 OOM fix) -------------
# The uncapped ncpu*3//4 = 24-worker pool OOM'd qlmc (45 GiB box, ~5.5 GB
# anon-rss per full-case worker at 200 keV): the kernel killed one worker and
# BrokenProcessPool lost the whole run. These tests pin the host probes so the
# sizing is deterministic on any machine.


def test_case_progress_label_names_single_material():
    from cxr_mc.montecarlo import runner

    assert runner._case_progress_label([{"crystal": "hopg"}, {"crystal": "hopg"}]) == ("hopg cases")
    assert runner._case_progress_label([{"crystal": "hopg"}, {"crystal": "hbn"}]) == ("mixed cases")


def _patch_host(monkeypatch, *, ncpus=32, avail_mb=44_900, total_mb=48_000, budget_mb=6_144):
    """Fake a host for _cpu_pool_workers; defaults reproduce qlmc's shape."""
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "_N_CPUS", ncpus)
    monkeypatch.setattr(runner, "_available_mem_mb", lambda: avail_mb)
    monkeypatch.setattr(runner, "_TOTAL_MEM", total_mb)
    monkeypatch.setattr(runner, "_WORKER_MEM_MB", budget_mb)
    return runner


def test_cpu_pool_workers_autosize_is_memory_bound_on_qlmc_shape(monkeypatch):
    """qlmc regression: 32 cores asks for 24 workers, RAM only carries 6."""
    runner = _patch_host(monkeypatch)
    # min(44_900, 0.85 * 48_000 = 40_800) // 6_144 = 6
    assert runner._cpu_pool_workers(None, 980) == 6


def test_cpu_pool_workers_autosize_is_cpu_bound_with_ample_ram(monkeypatch):
    runner = _patch_host(monkeypatch, avail_mb=1_000_000, total_mb=1_000_000)
    assert runner._cpu_pool_workers(None, 980) == 24  # ncpu * 3 // 4


def test_cpu_pool_workers_pin_is_clamped_by_memory(monkeypatch):
    """An explicit pin cannot re-create the OOM; raise CXR_MC_WORKER_MEM_MB to
    deliberately run tighter than the measured per-worker budget."""
    runner = _patch_host(monkeypatch)
    assert runner._cpu_pool_workers(24, 980) == 6


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
    monkeypatch.setattr(runner, "_GPU", False)
    monkeypatch.setattr(runner, "run_case", lambda case: {"name": case["name"]})
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncProcessPoolExecutor)

    cases = [{"name": f"c{i}"} for i in range(10)]
    runner.run_cases(cases, progress=False, engine="cpu")
    assert _SyncProcessPoolExecutor.captured["max_workers"] == 6


# ---- _adaptive_chunk grid-aware sizing (2026-07-18 OOM fix, part 2) ---------
# The fixed spec/brem chunk defaults were tuned on the old ~2000-bin line grid;
# widening the grid to 30000 eV (~6000 bins) silently tripled the (chunk, nbins)
# matmul transients per worker. _adaptive_chunk holds the byte product constant:
# chunk = budget_bytes // (3 arrays * nbins * 8 B), clamped to [1000, 100_000].


def test_adaptive_chunk_reproduces_old_default_on_narrow_grid():
    """Default 1920 MB budget was chosen so the pre-2026-07 ~2000-bin grid gets
    back exactly the old fixed chunk=40000 -- no behavior change on old runs."""
    from cxr_mc.montecarlo import runner

    assert runner._adaptive_chunk(2000) == 40_000


def test_adaptive_chunk_shrinks_on_the_widened_grid():
    from cxr_mc.montecarlo import runner

    # 1920e6 // (3 * 6000 * 8) = 13_333: ~3x fewer segments for ~3x more bins,
    # so the matmul transient stays ~1.9 GB instead of the ~5.7 GB that OOM'd.
    assert runner._adaptive_chunk(6000) == 13_333


def test_adaptive_chunk_clamps_to_floor_and_ceiling():
    from cxr_mc.montecarlo import runner

    assert runner._adaptive_chunk(10**9) == 1000  # degenerate wide grid: slow, not zero
    assert runner._adaptive_chunk(10) == 100_000  # coarse brem grid: bound kernel size


def test_adaptive_chunk_honors_budget_override(monkeypatch):
    from cxr_mc.montecarlo import runner

    monkeypatch.setattr(runner, "_SPEC_BUDGET_MB", 480)  # CXR_MC_SPEC_BUDGET_MB
    assert runner._adaptive_chunk(2000) == 10_000
