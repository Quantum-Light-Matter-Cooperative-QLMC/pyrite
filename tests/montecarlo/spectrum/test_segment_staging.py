"""One backend copy of the segments per case, shared by that case's kernels.

Each spectrum kernel used to reach for its own slice of the transport's output
with ``xp.asarray(segments[k], dtype=REAL)``, so a case uploaded overlapping
arrays once per kernel. ``_segments_on_device`` stages that upload; these gates
pin the two properties the sharing rests on -- staged segments produce the same
spectra as host ones, and a case stages exactly once however many kernels run.
"""

import numpy as np
import pytest

from pyrite._backend import REAL, _to_cpu
from pyrite.montecarlo import mc_brem_spectrum, mc_cxr_spectrum
from pyrite.montecarlo.spectrum import (
    _SEG_ARRAYS,
    _segments_in_layer,
    _segments_on_device,
)

E_GRID = np.arange(700.0, 1500.0)

LINE_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "n_hat": np.array([1.0, 0.0, 0.01]),
}
BREM_KWARGS = {
    "element": "C",
    "n_atoms_per_ang3": 0.1136,
    "n_hat": np.array([1.0, 0.0, 0.01]),
}


def _segments(count=4, n_layers=1):
    """A synthetic segment set in the shape `simulate_trajectories` returns.

    `E_end_keV` / `t_end_ang` are deliberately absent: the registry carries them
    but only an ``energy_model="midpoint"`` run produces them, so every per-row
    transform has to tolerate their absence.
    """
    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "E_start_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t_start_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": np.arange(count),
        "layer": np.arange(count) % n_layers,
        "Ne": count,
        "thickness_ang": 10.0,
        "crystal_width_ang": 10.0,
        "crystal_height_ang": 10.0,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": n_layers,
    }


@pytest.mark.parametrize("coherent", [False, True])
def test_staged_segments_give_bit_for_bit_identical_lines(coherent):
    """Staging only hoists the cast to REAL each kernel already did."""
    segments = _segments()

    reference = mc_cxr_spectrum(segments, E_GRID, coherent=coherent, **LINE_KWARGS)
    staged = mc_cxr_spectrum(
        _segments_on_device(segments), E_GRID, coherent=coherent, **LINE_KWARGS
    )

    assert np.max(np.abs(reference)) > 0.0
    np.testing.assert_array_equal(_to_cpu(staged), _to_cpu(reference))


def test_staged_segments_give_bit_for_bit_identical_brem():
    segments = _segments()

    reference = mc_brem_spectrum(segments, E_GRID, **BREM_KWARGS)
    staged = mc_brem_spectrum(_segments_on_device(segments), E_GRID, **BREM_KWARGS)

    assert np.max(np.abs(reference)) > 0.0
    np.testing.assert_array_equal(_to_cpu(staged), _to_cpu(reference))


def test_staging_carries_the_kernels_dtypes_and_values():
    segments = _segments()

    staged = _segments_on_device(segments)

    for k in _SEG_ARRAYS:
        if k not in segments:
            continue
        want = np.asarray(segments[k])
        got = _to_cpu(staged[k])
        if want.dtype.kind == "f":
            assert got.dtype == np.dtype(REAL), k
        else:
            assert got.dtype == want.dtype, k
        np.testing.assert_array_equal(got, want.astype(got.dtype), err_msg=k)


def test_staging_leaves_the_scalar_fields_alone():
    """Ne, thickness, the footprint and the layer count still ride along, so the
    per-electron normalization and geometry a kernel reads are unchanged."""
    segments = _segments()

    staged = _segments_on_device(segments)

    assert set(staged) == set(segments)
    for k in set(segments) - set(_SEG_ARRAYS):
        assert staged[k] is segments[k], k


def test_staging_an_already_staged_set_copies_nothing():
    """Idempotence is what makes it safe to stage at a case boundary: a kernel
    (or a nested helper) that stages again must not re-upload."""
    staged = _segments_on_device(_segments())

    restaged = _segments_on_device(staged)

    for k in _SEG_ARRAYS:
        if k not in staged:
            continue
        assert restaged[k] is staged[k], k


def test_staged_segments_slice_by_layer_identically():
    """The multilayer path masks the staged arrays instead of host ones."""
    segments = _segments(count=6, n_layers=3)
    staged = _segments_on_device(segments)

    for layer in range(3):
        host = _segments_in_layer(segments, layer)
        device = _segments_in_layer(staged, layer)
        assert host["L_ang"].size == 2
        for k in _SEG_ARRAYS:
            if k not in host:
                continue
            want = np.asarray(host[k])
            got = _to_cpu(device[k])
            np.testing.assert_array_equal(got, want.astype(got.dtype), err_msg=k)


def test_a_case_stages_once_and_every_kernel_reads_that_copy(monkeypatch):
    """Four kernels, one upload: two line totals, characteristic, and brem all
    receive the same staged segment dictionary."""
    import pyrite.montecarlo.runner as runner

    stages = []
    real_stage = runner._segments_on_device

    def counting_stage(segments):
        out = real_stage(segments)
        stages.append(out)
        return out

    seen = []
    monkeypatch.setattr(runner.case_driver, "_segments_on_device", counting_stage)
    monkeypatch.setattr(
        runner,
        "_cxr_for_segments",
        lambda segs, *a, **k: (seen.append(segs), np.zeros_like(E_GRID))[1],
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
        lambda segs, *a, **k: (seen.append(segs), np.zeros_like(E_GRID))[1],
    )
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda segs, *a, **k: (seen.append(segs), np.zeros_like(E_GRID))[1],
    )

    segments = _segments()
    runner._spectrum_case_impl(
        {
            "crystal": "hopg",
            "hkl_list": LINE_KWARGS["hkl_list"],
            "B_ang2": LINE_KWARGS["B_ang2"],
            "composition": None,
            "E0_keV": 30.0,
            "coherent_emission": True,
        },
        {
            "E_grid": E_GRID,
            "E_brem": E_GRID,
            "n_hat": LINE_KWARGS["n_hat"],
            "segs": segments,
            "Ne_lines": segments["Ne"],
            "Ne_brem": segments["Ne"],
            "groove": None,
        },
    )

    assert len(stages) == 1
    assert len(seen) == 4
    assert all(got is stages[0] for got in seen)


def test_a_case_reports_its_segment_counts_from_the_host_set(monkeypatch):
    """The staged copy is for the kernels only -- the run summary still counts
    segments and backscatter on `tp["segs"]`, which carries the scalar fields a
    device array cannot."""
    import pyrite.montecarlo.runner as runner

    monkeypatch.setattr(runner, "_cxr_for_segments", lambda *a, **k: np.zeros_like(E_GRID))
    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(E_GRID))
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *a, **k: np.zeros_like(E_GRID),
    )

    segments = _segments(count=6)
    out = runner._spectrum_case_impl(
        {
            "crystal": "hopg",
            "hkl_list": LINE_KWARGS["hkl_list"],
            "B_ang2": LINE_KWARGS["B_ang2"],
            "composition": None,
            "E0_keV": 30.0,
        },
        {
            "E_grid": E_GRID,
            "E_brem": E_GRID,
            "n_hat": LINE_KWARGS["n_hat"],
            "segs": segments,
            "Ne_lines": segments["Ne"],
            "Ne_brem": segments["Ne"],
            "groove": None,
        },
    )

    assert out["n_segments"] == 6
    assert out["eta"] == 0.0
    assert out["hit_frac"] == 1.0


# ---- composition with device-resident transport --------------------------------

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    from pyrite._backend import xp as _xp

    # A device is not enough. This test hands CuPy arrays to `_segments_on_device`,
    # which stages through the SESSION's `xp` -- and `tests/conftest.py` pins the
    # session to NumPy, so on a CPU-pinned run the staging call would raise rather
    # than measure anything. Run it with `PYRITE_TEST_BACKEND=cuda`.
    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0 and _xp.__name__ == "cupy"
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(
    not _HAS_CUDA, reason="no CUDA device, or the session is pinned to the NumPy backend"
)


@pytest.mark.hardware
@requires_cuda
def test_device_resident_transport_feeds_the_kernels_the_same_spectra():
    """The two halves compose: transport that never comes down, staged once, has
    to give the kernels exactly what a downloaded-then-uploaded run gives them.
    Staging is what makes it transparent -- it casts to REAL either way, and on
    segments that are already device arrays it is the only copy in the path."""
    from pyrite.montecarlo import simulate_trajectories

    case = dict(
        E0_keV=30.0,
        Ne=200,
        thickness_ang=5.0e4,
        element="C",
        n_atoms_per_ang3=0.1136,
        seed=7,
        transport_core="cuda",
    )
    host = simulate_trajectories(**case)
    device = simulate_trajectories(**case, keep_segments_on_device=True)
    assert host["L_ang"].size > 0
    assert isinstance(device["L_ang"], cupy.ndarray)

    for kernel, kwargs in (
        (mc_cxr_spectrum, {**LINE_KWARGS, "E_grid": E_GRID}),
        (mc_cxr_spectrum, {**LINE_KWARGS, "E_grid": E_GRID, "coherent": True}),
        (mc_brem_spectrum, {**BREM_KWARGS, "E_grid": E_GRID}),
    ):
        grid = kwargs.pop("E_grid")
        want = kernel(_segments_on_device(host), grid, **kwargs)
        got = kernel(_segments_on_device(device), grid, **kwargs)
        np.testing.assert_array_equal(_to_cpu(got), _to_cpu(want))


def _adversarial_f64():
    rng = np.random.default_rng(3)
    f32_min = float(np.finfo(np.float32).tiny)
    f32_max = float(np.finfo(np.float32).max)
    special = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            f32_min,
            np.nextafter(f32_min, 0),
            f32_min / 2,
            -f32_min / 3,
            1e-45,
            1e-46,
            7e-46,
            f32_max,
            np.nextafter(f32_max, np.inf),
            3.5e38,
            -3.5e38,
            1e300,
            -1e300,
            5e-324,
        ]
    )
    nans = np.array(
        [
            0x7FF8000000000000,
            0xFFF8000000000001,
            0x7FF0000000000001,
            0x7FFFFFFFFFFFFFFF,
            0x7FF123456789ABCD,
        ],
        dtype=np.uint64,
    ).view(np.float64)
    sub = rng.uniform(-1.2e-38, 1.2e-38, 4096)
    normal = rng.standard_normal(4096) * 10.0 ** rng.integers(-37, 38, 4096)
    return np.concatenate([special, nans, sub, normal])


@pytest.mark.parametrize("repeat", [1, 3_000_000 // 4000])
def test_float_row_staging_is_bitwise_the_host_cast(repeat):
    """Staged float rows equal ``astype(float32)`` bit for bit (incl. subnormals,
    overflow to inf, NaN payloads); repeat>1 exercises the multi-chunk path."""
    from pyrite._backend import xp

    if not hasattr(xp, "cuda") or REAL is not xp.float32:
        pytest.skip("CuPy float32 staging path only")
    from pyrite.montecarlo.spectrum.cxr._kernels import _stage_row

    base = _adversarial_f64()
    a = np.tile(base, repeat)
    got = _to_cpu(_stage_row(a))
    want = a.astype(np.float32)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    two_d = a[: (a.size // 3) * 3].reshape(-1, 3)
    got2 = _to_cpu(_stage_row(two_d))
    np.testing.assert_array_equal(got2.view(np.uint32), two_d.astype(np.float32).view(np.uint32))
