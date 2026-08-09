"""One backend copy of the segments per case, shared by that case's kernels.

Each spectrum kernel used to reach for its own slice of the transport's output
with ``xp.asarray(segments[k], dtype=REAL)``, so a case uploaded overlapping
arrays once per kernel. ``_segments_on_device`` stages that upload; these gates
pin the two properties the sharing rests on -- staged segments produce the same
spectra as host ones, and a case stages exactly once however many kernels run.
"""

import numpy as np
import pytest

from cxr_mc.montecarlo import mc_brem_spectrum, mc_spectrum
from cxr_mc.montecarlo._backend import REAL, _to_cpu
from cxr_mc.montecarlo.spectrum import (
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
    """A synthetic segment set in the shape `simulate_trajectories` returns."""
    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
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

    reference = mc_spectrum(segments, E_GRID, coherent=coherent, **LINE_KWARGS)
    staged = mc_spectrum(_segments_on_device(segments), E_GRID, coherent=coherent, **LINE_KWARGS)

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
            want = np.asarray(host[k])
            got = _to_cpu(device[k])
            np.testing.assert_array_equal(got, want.astype(got.dtype), err_msg=k)


def test_a_case_stages_once_and_every_kernel_reads_that_copy(monkeypatch):
    """The point of the change: three kernels, one upload. A case that also
    wants the coherent spectrum runs two line kernels and the brem kernel, and
    all three must be handed the SAME staged dict."""
    import cxr_mc.montecarlo.runner as runner

    stages = []
    real_stage = runner._segments_on_device

    def counting_stage(segments):
        out = real_stage(segments)
        stages.append(out)
        return out

    seen = []
    monkeypatch.setattr(runner, "_segments_on_device", counting_stage)
    monkeypatch.setattr(
        runner,
        "_lines_for_segments",
        lambda segs, *a, **k: (seen.append(segs), np.zeros_like(E_GRID))[1],
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
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
    assert len(seen) == 3
    assert all(got is stages[0] for got in seen)


def test_a_case_reports_its_segment_counts_from_the_host_set(monkeypatch):
    """The staged copy is for the kernels only -- the run summary still counts
    segments and backscatter on `tp["segs"]`, which carries the scalar fields a
    device array cannot."""
    import cxr_mc.montecarlo.runner as runner

    monkeypatch.setattr(runner, "_lines_for_segments", lambda *a, **k: np.zeros_like(E_GRID))
    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(E_GRID))

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
