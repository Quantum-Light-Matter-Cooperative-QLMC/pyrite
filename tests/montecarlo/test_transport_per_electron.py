"""Per-electron transport core and its CUDA port.

The CPU tests here pin the properties the GPU port depends on: a counter-based
RNG that is pure integer arithmetic, output slots addressed by electron index,
and replay-exactness so a capacity overflow can be retried rather than
resampled. The CUDA tests re-check the same properties on device and are skipped
without a GPU.
"""

import os

import numpy as np
import pytest

from pyrite.montecarlo.transport import (
    CUDA_TRANSPORT_MIN_ELECTRONS,
    PerElectronTransportConfig,
    TransportLUTConfig,
    _splitmix64,
    _stream_key_scalar,
    _stream_uniform_scalar,
    simulate_trajectories,
    spliced_stopping_keV_per_ang,
    stream_keys,
)

SEGMENT_KEYS = ("r_mid", "v_hat", "L_ang", "E_keV", "t_ang", "t0_ang", "elec_id", "layer")
COUNT_KEYS = (
    "n_backscattered",
    "n_transmitted",
    "n_side_exited",
    "n_cutoff_stopped",
    "n_step_limited",
    "n_stopped",
    "n_missed",
)

BASE_CASE = dict(
    E0_keV=30.0,
    Ne=120,
    thickness_ang=5.0e4,
    element="Si",
    n_atoms_per_ang3=0.04996,
    seed=11,
)

U64_MASK = (1 << 64) - 1


def _reference_splitmix64(x):
    """Independent pure-Python SplitMix64, from the published constants."""
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return x ^ (x >> 31)


def _reference_uniform(key, counter):
    z = _reference_splitmix64((key + 0x9E3779B97F4A7C15 * (counter + 1)) & U64_MASK)
    return (z >> 11) * 2.0**-53


def _identical(a, b):
    return all(np.array_equal(a[k], b[k]) for k in SEGMENT_KEYS) and all(
        a[k] == b[k] for k in COUNT_KEYS
    )


def _run(**overrides):
    case = dict(BASE_CASE)
    case.update(overrides)
    return simulate_trajectories(**case)


# ---- counter-based RNG --------------------------------------------------------


@pytest.mark.parametrize("x", [0, 1, 2, 3, 2**32 - 1, 2**63 + 7, 2**64 - 1])
def test_splitmix64_matches_published_reference(x):
    assert int(_splitmix64(np.uint64(x))) == _reference_splitmix64(x)


def test_stream_keys_match_the_scalar_form():
    keys = stream_keys(7, 512)
    assert keys.dtype == np.uint64
    assert [int(k) for k in keys] == [int(_stream_key_scalar(7, e)) for e in range(512)]


def test_stream_uniform_matches_the_pure_python_reference():
    # Integer-only until the final 2**-53 scaling, so host and device agree
    # exactly; this pins the arithmetic the CUDA kernel reimplements.
    for elec in (0, 1, 4095):
        key = int(_stream_key_scalar(3, elec))
        for counter in (0, 1, 2, 997, 10**6):
            got = _stream_uniform_scalar(np.uint64(key), np.uint64(counter))
            assert got == _reference_uniform(key, counter)


def test_stream_draws_are_uniform_and_in_range():
    key = np.uint64(_stream_key_scalar(0, 0))
    draws = np.array([_stream_uniform_scalar(key, np.uint64(i)) for i in range(200_000)])
    assert draws.min() >= 0.0
    assert draws.max() < 1.0
    # A signed-shift bug halves the range while leaving individual draws
    # plausible, so assert the moments rather than the bounds alone.
    assert abs(draws.mean() - 0.5) < 0.01
    assert abs(draws.var() - 1.0 / 12.0) < 0.005


def test_adjacent_electron_streams_are_uncorrelated():
    n = 100_000
    a, b = (
        np.array(
            [
                _stream_uniform_scalar(np.uint64(_stream_key_scalar(0, elec)), np.uint64(i))
                for i in range(n)
            ]
        )
        for elec in (0, 1)
    )
    assert abs(np.corrcoef(a, b)[0, 1]) < 4.0 / np.sqrt(n)
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 4.0 / np.sqrt(n)


# ---- per-electron core --------------------------------------------------------


def test_default_core_is_the_lockstep_core_below_the_cuda_threshold():
    # BASE_CASE is 120 electrons, so "auto" resolves to the historical core on
    # every box, GPU or not. The threshold policy itself lives in
    # test_transport_core_default.py.
    assert BASE_CASE["Ne"] <= CUDA_TRANSPORT_MIN_ELECTRONS
    assert _identical(_run(), _run(transport_core="lockstep"))


def test_per_electron_core_is_deterministic():
    assert _identical(_run(transport_core="per-electron"), _run(transport_core="per-electron"))


@pytest.mark.parametrize("capacity", [4, 37, 256, 4096])
def test_capacity_replay_reproduces_the_discarded_run(capacity):
    # An electron that overflows its slots forces the whole batch to replay.
    # Counter-addressed streams make that replay exact, so every capacity --
    # including ones far below the ~800 segments an electron actually needs --
    # must give the same trajectories.
    base = _run(transport_core="per-electron")
    replayed = _run(
        transport_core="per-electron",
        per_electron_config=PerElectronTransportConfig(seg_capacity=capacity),
    )
    assert _identical(base, replayed)


@pytest.mark.parametrize("budget", [1 << 18, 1 << 22, 1 << 27])
def test_results_do_not_depend_on_batch_size(budget):
    base = _run(transport_core="per-electron")
    batched = _run(
        transport_core="per-electron",
        per_electron_config=PerElectronTransportConfig(scratch_budget_bytes=budget),
    )
    assert _identical(base, batched)


@pytest.mark.parametrize("probe", [1, 8, 64, 100000])
def test_capacity_probe_does_not_change_results(probe):
    # The probe only shortens the first batch, and output slots are addressed by
    # electron index, so probing must be invisible in the segments.
    base = _run(transport_core="per-electron")
    probed = _run(
        transport_core="per-electron",
        per_electron_config=PerElectronTransportConfig(
            seg_capacity=8, scratch_budget_bytes=1 << 14, probe_electrons=probe
        ),
    )
    assert _identical(base, probed)


def _core_calls(monkeypatch, **config_kwargs):
    """``(electrons, capacity)`` of every core call, replays included, and the run."""
    from pyrite.montecarlo import transport as tr

    seen = []
    real = tr._alloc_scratch

    def counting(xp, m, cap, midpoint=False):
        seen.append((m, cap))
        return real(xp, m, cap, midpoint)

    # The per-electron run loops resolve _alloc_scratch in their own module
    # (transport.batching), not the package re-export.
    monkeypatch.setattr(tr.batching, "_alloc_scratch", counting)
    try:
        out = _run(
            transport_core="per-electron",
            per_electron_config=PerElectronTransportConfig(**config_kwargs),
        )
    finally:
        monkeypatch.undo()
    return seen, out


# `seg_capacity=8` is far below what these electrons need, and the budget is wide
# enough that without a probe the first batch is the whole run -- the shape that
# makes an overflow expensive.
_PROBE_BUDGET = dict(seg_capacity=8, scratch_budget_bytes=1 << 20)


def test_capacity_probe_shrinks_the_replayed_batch(monkeypatch):
    Ne = BASE_CASE["Ne"]
    pinned, pinned_out = _core_calls(monkeypatch, probe_electrons=0, **_PROBE_BUDGET)
    probed, probed_out = _core_calls(monkeypatch, probe_electrons=4, **_PROBE_BUDGET)

    assert _identical(pinned_out, probed_out)
    # Pinned, the whole run is gambled on `seg_capacity` and discarded; probed,
    # only the probe is. Discarded electrons are the work the replay repeats.
    assert pinned[0][0] == Ne
    assert probed[0][0] == 4
    assert sum(m for m, _ in pinned) == 2 * Ne
    assert sum(m for m, _ in probed) < Ne + Ne // 4


@pytest.mark.parametrize("probe", [0, 4])
def test_capacity_is_sized_from_the_measurement(monkeypatch, probe):
    # Once a batch has run, `cap` tracks the segments an electron actually needs
    # rather than the guess it started from -- an over-provisioned capacity costs
    # launches the same way an under-provisioned one costs replays.
    calls, out = _core_calls(monkeypatch, probe_electrons=probe, **_PROBE_BUDGET)

    assert calls[0][1] == 8
    settled = calls[-1][1]
    needed = int(np.bincount(out["elec_id"]).max())
    assert needed <= settled <= int(np.ceil(1.25 * needed))


def test_segments_are_electron_major_and_step_minor():
    out = _run(transport_core="per-electron")
    elec_id = out["elec_id"]
    assert np.all(np.diff(elec_id) >= 0)
    # Within one electron the clock only advances, which is the step ordering.
    for e in np.unique(elec_id)[:20]:
        t = out["t_ang"][elec_id == e]
        assert np.all(np.diff(t) > 0)


def test_per_electron_core_matches_lockstep_physics():
    # The two cores realize different samples of the same distribution -- they
    # cannot be compared trajectory by trajectory, only in aggregate.
    # Keep the stronger sample for normal runs. With Numba disabled, every
    # scalar transport step executes under the Python tracer; use the existing
    # minimal case so the coverage measurement remains bounded while still
    # comparing every aggregate observable below across independent seeds.
    tracing_numba = os.environ.get("NUMBA_DISABLE_JIT") == "1"
    case = dict(BASE_CASE)
    case.update(Ne=BASE_CASE["Ne"] if tracing_numba else 3000)
    seeds = range(1, 5) if tracing_numba else range(1, 9)
    metrics = {
        "backscatter fraction": lambda r: r["n_backscattered"] / r["Ne"],
        "transmit fraction": lambda r: r["n_transmitted"] / r["Ne"],
        "segments per electron": lambda r: r["L_ang"].size / r["Ne"],
        "mean segment length": lambda r: r["L_ang"].mean(),
        "mean segment energy": lambda r: r["E_keV"].mean(),
        "mean depth": lambda r: r["r_mid"][:, 2].mean(),
    }

    # Reduce each run to its observables at once: holding all 16 full segment
    # tables alive costs ~3 GB, which caps how many test workers fit in RAM.
    def observables(**core):
        rows = []
        for s in seeds:
            run = simulate_trajectories(**{**case, "seed": s, **core})
            rows.append({name: fn(run) for name, fn in metrics.items()})
        return rows

    lockstep = observables()
    per_electron = observables(transport_core="per-electron")

    def observable(rows, name):
        vals = np.array([row[name] for row in rows])
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

    for name in metrics:
        a, a_err = observable(lockstep, name)
        b, b_err = observable(per_electron, name)
        spread = np.hypot(a_err, b_err)
        assert abs(a - b) < 4.0 * spread, f"{name}: {a} +-{a_err} vs {b} +-{b_err}"


def test_multilayer_transport_reports_valid_layers():
    out = _run(
        transport_core="per-electron",
        layers=[
            (0.0, 1.0e4, [("C", 0.176)]),
            (1.0e4, 5.0e4, [("W", 0.0632)]),
        ],
        elastic_model="sr",
    )
    assert out["n_layers"] == 2
    assert set(np.unique(out["layer"])) <= {0, 1}
    assert (out["layer"] == 1).any()


def test_finite_footprint_counts_side_exits():
    out = _run(
        transport_core="per-electron",
        crystal_width_mm=2.0e-4,
        crystal_height_mm=2.0e-4,
    )
    assert out["n_side_exited"] > 0
    assert out["n_backscattered"] + out["n_transmitted"] + out["n_side_exited"] <= out["Ne"]


def test_compound_target_selects_between_elements():
    out = _run(transport_core="per-electron", composition=[("Mo", 0.0186), ("Se", 0.0372)])
    assert out["L_ang"].size > 0
    assert np.all(np.isfinite(out["E_keV"]))


def test_grooved_transport_rejects_the_per_electron_core():
    from pyrite.montecarlo.groove import GrooveSpec

    with pytest.raises(ValueError, match="grooved transport"):
        _run(
            transport_core="per-electron",
            groove=GrooveSpec(spacing_ang=1.0e4, depth_ang=1.0e3, tilt_polar_rad=0.2),
        )


def test_unknown_transport_core_is_rejected():
    with pytest.raises(ValueError, match="transport_core must be"):
        _run(transport_core="opencl")


# ---- CUDA port ----------------------------------------------------------------

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")


@pytest.mark.hardware
@requires_cuda
def test_launcher_signature_tracks_the_reference_core():
    import inspect

    from pyrite.montecarlo.transport import _transport_core_ungrooved_perelectron
    from pyrite.montecarlo.transport._jit_launch import run_transport_kernel

    reference = list(inspect.signature(_transport_core_ungrooved_perelectron.py_func).parameters)
    launcher = [
        name
        for name, p in inspect.signature(run_transport_kernel).parameters.items()
        if p.kind is not p.KEYWORD_ONLY and name != "config"
    ]
    assert launcher == reference


@pytest.mark.hardware
@requires_cuda
def test_cuda_core_is_deterministic():
    assert _identical(_run(transport_core="cuda"), _run(transport_core="cuda"))


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("nthreads", [32, 128, 512])
def test_cuda_results_do_not_depend_on_launch_geometry(nthreads, monkeypatch):
    from pyrite.montecarlo.transport import _jit_launch as tjl

    base = _run(transport_core="cuda")
    monkeypatch.setattr(
        tjl, "DEFAULT_TRANSPORT_KERNEL_CONFIG", tjl.TransportKernelConfig(nthreads=nthreads)
    )
    assert _identical(base, _run(transport_core="cuda"))


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("capacity", [4, 256, 4096])
def test_cuda_capacity_replay_reproduces_the_discarded_run(capacity):
    base = _run(transport_core="cuda")
    replayed = _run(
        transport_core="cuda",
        per_electron_config=PerElectronTransportConfig(seg_capacity=capacity),
    )
    assert _identical(base, replayed)


@pytest.mark.hardware
@requires_cuda
def test_cuda_first_step_agrees_with_the_cpu_reference():
    # Both cores draw the same numbers and take the same branches, so the first
    # recorded segment differs only by libm rounding (CUDA's log/pow are a few
    # ulp from the host's). Later segments are not compared: transport is
    # chaotic and amplifies that difference.
    cpu = _run(transport_core="per-electron")
    gpu = _run(transport_core="cuda")

    cpu_first = np.flatnonzero(np.diff(cpu["elec_id"], prepend=-1))
    gpu_first = np.flatnonzero(np.diff(gpu["elec_id"], prepend=-1))
    assert np.array_equal(cpu["elec_id"][cpu_first], gpu["elec_id"][gpu_first])
    np.testing.assert_allclose(cpu["L_ang"][cpu_first], gpu["L_ang"][gpu_first], rtol=1e-12)
    np.testing.assert_allclose(cpu["E_keV"][cpu_first], gpu["E_keV"][gpu_first], rtol=1e-12)
    np.testing.assert_allclose(cpu["r_mid"][cpu_first], gpu["r_mid"][gpu_first], rtol=1e-12)


@pytest.mark.hardware
@requires_cuda
def test_cuda_matches_the_cpu_reference_in_aggregate():
    case = dict(BASE_CASE)
    case.update(Ne=3000)
    seeds = range(1, 9)
    cpu = [
        simulate_trajectories(**{**case, "seed": s, "transport_core": "per-electron"})
        for s in seeds
    ]
    gpu = [simulate_trajectories(**{**case, "seed": s, "transport_core": "cuda"}) for s in seeds]

    for name, fn in {
        "backscatter fraction": lambda r: r["n_backscattered"] / r["Ne"],
        "segments per electron": lambda r: r["L_ang"].size / r["Ne"],
        "mean segment energy": lambda r: r["E_keV"].mean(),
    }.items():
        a = np.array([fn(r) for r in cpu])
        b = np.array([fn(r) for r in gpu])
        spread = np.hypot(a.std(ddof=1), b.std(ddof=1)) / np.sqrt(len(a))
        assert abs(a.mean() - b.mean()) < 4.0 * spread, name


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("use_lut", [False, True])
def test_cuda_midpoint_first_flight_agrees_with_the_cpu_reference(use_lut):
    # Same argument as the frozen first-step test: only the opening segment of
    # each electron is comparable across libm implementations. Here it also
    # pins the flight identity and end-state fields the controlled propagator
    # adds, so a mis-ordered kernel draw or a dropped substep shows up.
    controlled = dict(
        energy_model="midpoint",
        max_dE_frac=0.05,
        transport_lut_config=TransportLUTConfig(enabled=use_lut),
    )
    cpu = _run(transport_core="per-electron", **controlled)
    gpu = _run(transport_core="cuda", **controlled)

    assert set(gpu) >= {"E_end_keV", "t_end_ang", "E_repr_keV", "flight_id", "substep_id"}
    first = np.flatnonzero(np.diff(cpu["elec_id"], prepend=-1))
    gpu_first = np.flatnonzero(np.diff(gpu["elec_id"], prepend=-1))
    assert np.array_equal(first, gpu_first)
    for key in ("flight_id", "substep_id", "layer"):
        np.testing.assert_array_equal(cpu[key][first], gpu[key][first])
    for key in ("L_ang", "E_start_keV", "E_end_keV", "E_repr_keV", "t_end_ang"):
        np.testing.assert_allclose(cpu[key][first], gpu[key][first], rtol=1e-12)


@pytest.mark.hardware
@requires_cuda
def test_cuda_cutoff_crossing_truncates_the_terminal_flight():
    composition = [("C", 0.1136)]
    cutoff = 5.0
    out = simulate_trajectories(
        5.01,
        1,
        1.0e8,
        composition=composition,
        E_cut_keV=cutoff,
        elastic_model="sr",
        seed=4,
        max_steps=20,
        transport_core="cuda",
        # The endpoint oracle below is the frozen left-endpoint rule.
        energy_model="frozen",
    )

    terminal_start = out["E_keV"][-1]
    # The oracle has to be the model the kernel evaluates. 5 keV is already above
    # carbon's Joy--Luo/Berger--Seltzer crossover, so restating the Joy--Luo
    # branch here would describe the model this branch retired, not the CUDA
    # core -- the same repointing the CPU cutoff tests took.
    stopping = -spliced_stopping_keV_per_ang(composition, terminal_start)
    endpoint = terminal_start - stopping * out["L_ang"][-1]

    # Device transport stores REAL segments in float32; this is a roughly
    # four-ulp absolute bound at the 5 keV cutoff, not an observed-error fit.
    assert endpoint == pytest.approx(cutoff, abs=2e-6)
    assert out["n_cutoff_stopped"] == out["n_stopped"] == 1
    assert out["n_step_limited"] == 0


@pytest.mark.hardware
@requires_cuda
def test_cuda_step_limited_histories_raise_with_incomplete_count():
    with pytest.raises(RuntimeError, match=r"n_step_limited=3, Ne=3, max_steps=1"):
        simulate_trajectories(
            30.0,
            3,
            1.0e8,
            composition=[("C", 0.1136)],
            E_cut_keV=5.0,
            elastic_model="sr",
            seed=2,
            max_steps=1,
            transport_core="cuda",
        )


# ---- device-resident segments -------------------------------------------------

DIAGNOSTIC_KEYS = ("initial_r_ang", "initial_v_hat", "initial_E_keV", "initial_t0_ang")


@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
def test_device_residency_requires_the_cuda_core(core):
    """Only the CUDA core produces segments anywhere but host memory, so asking
    the others to leave them there is a mistake rather than a no-op."""
    with pytest.raises(ValueError, match="keep_segments_on_device"):
        _run(transport_core=core, keep_segments_on_device=True)


@pytest.mark.hardware
@requires_cuda
def test_device_resident_segments_are_identical_to_the_host_run():
    """Residency moves the payload; it must not touch a value. Same seed, same
    core, same dtypes -- only where the arrays live differs."""
    import cupy

    host = _run(transport_core="cuda")
    device = _run(transport_core="cuda", keep_segments_on_device=True)

    assert host["L_ang"].size > 0
    brought_down = {k: cupy.asnumpy(device[k]) for k in SEGMENT_KEYS}
    for k in SEGMENT_KEYS:
        assert brought_down[k].dtype == host[k].dtype, k
    assert _identical(host, brought_down | {k: device[k] for k in COUNT_KEYS})


@pytest.mark.hardware
@requires_cuda
def test_only_the_segments_stay_on_the_device():
    """Per-electron diagnostics and the counts are not what the spectrum kernels
    read, and keeping them on the device would only make them awkward."""
    import cupy

    device = _run(transport_core="cuda", keep_segments_on_device=True)

    for k in SEGMENT_KEYS:
        assert isinstance(device[k], cupy.ndarray), k
    for k in DIAGNOSTIC_KEYS:
        assert isinstance(device[k], np.ndarray), k
    for k in COUNT_KEYS:
        assert not isinstance(device[k], cupy.ndarray), k


@pytest.mark.hardware
@requires_cuda
@pytest.mark.parametrize("capacity", [4, 4096])
def test_device_residency_survives_batching_and_capacity_replay(capacity):
    """The join replaces a preallocated buffer, so batch count and ordering are
    exactly what it has to get right. A capacity of 4 forces replays and many
    small batches; 4096 runs the whole thing in one."""
    base = _run(transport_core="cuda")
    joined = _run(
        transport_core="cuda",
        keep_segments_on_device=True,
        per_electron_config=PerElectronTransportConfig(seg_capacity=capacity),
    )

    import cupy

    assert _identical(base, dict(joined) | {k: cupy.asnumpy(joined[k]) for k in SEGMENT_KEYS})


@pytest.mark.hardware
@requires_cuda
def test_device_resident_segments_carry_the_layer_index():
    """Multilayer stacks are the one case where a downstream kernel slices the
    segments by a field of their own, so the joined `layer` has to line up."""
    import cupy

    stack = {
        "layers": [(0.0, 1.0e4, [("C", 0.176)]), (1.0e4, 5.0e4, [("W", 0.0632)])],
        "elastic_model": "sr",
    }
    host = _run(transport_core="cuda", **stack)
    device = _run(transport_core="cuda", keep_segments_on_device=True, **stack)

    assert np.unique(host["layer"]).size == 2
    for k in SEGMENT_KEYS:
        assert np.array_equal(cupy.asnumpy(device[k]), host[k]), k
