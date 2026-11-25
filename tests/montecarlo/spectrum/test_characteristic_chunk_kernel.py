"""Fused per-chunk characteristic kernel (#1455, option B) against the grouped chain.

The fused route only regroups FP64 sums (#1313), so it must agree with the
grouped chain to FP64 rounding. Without a device the NumPy twin, which runs
the kernel's per-row terms and block reduction tree, stands in for it; the
CuPy tests run the kernel itself and skip without one.
"""

import numpy as np
import pytest

from pyrite._backend import REAL, xp
from pyrite.montecarlo.spectrum import _characteristic_chunk_kernel as fused
from pyrite.montecarlo.spectrum import characteristic
from pyrite.montecarlo.spectrum.segment_escape import segment_escape_paths

RTOL = 1.0e-12
LAYERS = [(0.0, 100.0, [("W", 0.063)]), (100.0, 1000.0, [("Si", 0.05)])]
ENERGY = np.linspace(1.0e3, 1.2e4, 200)
ON_DEVICE = getattr(xp, "__name__", "") == "cupy" and np.dtype(REAL) == np.float32


def _segments(seed=1455, n=6_000):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return {
        "r_mid": np.column_stack((np.zeros(n), np.zeros(n), rng.uniform(1.0, 999.0, n))),
        "v_hat": v / np.linalg.norm(v, axis=1, keepdims=True),
        "L_ang": rng.uniform(0.5, 50.0, n),
        "E_keV": rng.uniform(3.0, 190.0, n),
        "elec_id": np.repeat(np.arange(n // 4), 4),
        "layer": (rng.uniform(size=n) > 0.5).astype(int),
        "Ne": n // 4,
        "thickness_ang": 1000.0,
    }


def _scoring_inputs(element, layers=LAYERS, weighted=False):
    """The grouped scorer's arguments for one emitting element."""
    segments = _segments()
    index = xp.arange(segments["L_ang"].size)
    n_hat = np.array([0.3, 0.0, -0.95])
    n_hat /= np.linalg.norm(n_hat)
    owner, fraction, path_start, path_end = segment_escape_paths(
        segments, index, n_hat, layers=layers, xp=xp
    )
    seg_L = xp.asarray(segments["L_ang"], dtype=REAL)[owner] * fraction
    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)[owner]
    table = characteristic.load_characteristic_cross_sections(element)
    composition = [(el, n) for _top, _bottom, comp in layers for el, n in comp if el == element]
    edges, widths = characteristic._energy_bin_edges_and_widths(ENERGY)
    prepared = characteristic._prepare_lines(
        table, composition, layers, edges, widths, table.recommended_cutoff_eV
    )
    weight = None
    if weighted:
        rng = np.random.default_rng(7)
        n_shell = len(table.ionization_shell_labels)
        weight = xp.asarray(rng.uniform(0.2, 3.0, (int(seg_E.size), n_shell)), dtype=REAL)
    return dict(
        table=table,
        response=prepared.response,
        mu_by_line=prepared.mu_by_line,
        seg_E=seg_E,
        seg_L=seg_L,
        path_start=path_start,
        path_end=path_end,
        number_density_ang3=composition[0][1],
        shell_tables=prepared.shell_tables,
        packed_shells=prepared.packed_shells,
        shell_weight=weight,
    )


def _grouped(inputs, chunk):
    keys = (
        "table",
        "response",
        "mu_by_line",
        "seg_E",
        "seg_L",
        "path_start",
        "path_end",
    )
    return characteristic._to_cpu(
        characteristic._accumulate_line_yields(
            *(inputs[key] for key in keys),
            chunk=chunk,
            number_density_ang3=inputs["number_density_ang3"],
            shell_tables=inputs["shell_tables"],
            packed_shells=inputs["packed_shells"],
            shell_weight=inputs["shell_weight"],
        )
    )


def _twin(inputs, chunk):
    table, shells = inputs["table"], inputs["shell_tables"]

    def shell_sigma(energy_keV):
        return characteristic._to_cpu(
            characteristic._interpolate_shell_cross_sections(table, xp.asarray(energy_keV), shells)
        )

    host = {
        key: characteristic._to_cpu(inputs[key])
        for key in ("seg_E", "seg_L", "path_start", "path_end", "response", "mu_by_line")
    }
    weight = inputs["shell_weight"]
    return fused.host_chunk_line_yields(
        host["seg_E"],
        host["seg_L"],
        host["path_start"],
        host["path_end"],
        chunk=chunk,
        number_density_ang3=inputs["number_density_ang3"],
        shell_sigma=shell_sigma,
        response=host["response"],
        mu_by_line=host["mu_by_line"],
        shell_weight=None if weight is None else characteristic._to_cpu(weight),
    )


@pytest.mark.parametrize("element", ["W", "Si"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("chunk", [997, 4_096, 10**6])
def test_twin_matches_grouped_chain_to_fp64_rounding(monkeypatch, element, weighted, chunk):
    """Several elements, two absorbing layers, flux weights and split line groups."""
    # Force several line groups per chunk in the grouped chain.
    monkeypatch.setattr(characteristic, "_LINE_BATCH_ELEMENTS", 20_000)
    inputs = _scoring_inputs(element, weighted=weighted)
    reference = _grouped(inputs, chunk)
    twin = _twin(inputs, chunk)
    assert reference.size > 1 and reference.max() > 0.0
    np.testing.assert_allclose(twin, reference, rtol=RTOL, atol=0.0)


def test_twin_single_slab_matches_grouped_chain():
    layers = [(0.0, 1000.0, [("W", 0.063)])]
    inputs = _scoring_inputs("W", layers=layers)
    np.testing.assert_allclose(_twin(inputs, 1_500), _grouped(inputs, 1_500), rtol=RTOL, atol=0.0)


def test_block_tree_sum_is_the_kernel_reduction_order():
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, (fused.BLOCK_ROWS * 3 + 17, 5))
    partial = fused._block_tree_sum(values)
    assert partial.shape == (4, 5)
    np.testing.assert_allclose(partial.sum(axis=0), values.sum(axis=0), rtol=1e-14)
    # Pairwise halving within a block: element 0 meets element 128 first.
    block = np.zeros((fused.BLOCK_ROWS, 1))
    block[0], block[fused.BLOCK_ROWS // 2] = 1.0, 2.0**-60
    assert fused._block_tree_sum(block)[0, 0] == 1.0 + 2.0**-60


@pytest.mark.parametrize(
    ("mode", "chunk", "expected"),
    [
        (None, fused.FUSED_MIN_CHUNK_ROWS, True),
        (None, fused.FUSED_MIN_CHUNK_ROWS - 1, False),
        ("auto", 1 << 20, True),
        ("", 997, False),
        ("fused", 997, True),
        ("FUSED", 1 << 20, True),
        ("grouped", 1 << 20, False),
        ("grouped", 997, False),
        ("unknown", 1 << 20, False),
    ],
)
def test_route_gate_and_env_overrides(monkeypatch, mode, chunk, expected):
    """Job 15970: fused by default from 64k-row chunks; the env var overrides."""
    if mode is None:
        monkeypatch.delenv(fused.KERNEL_ENV, raising=False)
    else:
        monkeypatch.setenv(fused.KERNEL_ENV, mode)
    assert fused.FUSED_MIN_CHUNK_ROWS == 65_536
    assert fused.fused_selected(chunk) is expected


class _FakeKernel:
    """Stands in for the RawKernel: counts launches, writes zero partials."""

    def __init__(self):
        self.launches = 0

    def __call__(self, grid, block, args):
        self.launches += 1
        args[-1][...] = 0.0


def _route_args(inputs, chunk, packed_shells):
    positional = (inputs["seg_E"], inputs["seg_L"], inputs["path_start"], inputs["path_end"])
    keywords = dict(
        chunk=chunk,
        number_density_ang3=inputs["number_density_ang3"],
        packed_shells=packed_shells,
        response=inputs["response"],
        mu_by_line=inputs["mu_by_line"],
    )
    return positional, keywords


@pytest.mark.parametrize(
    ("mode", "chunk", "launches"),
    [
        (None, fused.FUSED_MIN_CHUNK_ROWS, 1),
        (None, fused.FUSED_MIN_CHUNK_ROWS - 1, 0),
        ("fused", 997, 7),
        ("grouped", fused.FUSED_MIN_CHUNK_ROWS, 0),
    ],
)
def test_cupy_float32_path_dispatches_by_gate(monkeypatch, mode, chunk, launches):
    """On the CuPy float32 path the gate picks the kernel (B) or the chain (A)."""
    inputs = _scoring_inputs("Si")
    inputs["seg_E"] = np.asarray(characteristic._to_cpu(inputs["seg_E"]), dtype=np.float32)
    n_shell = int(inputs["response"].shape[0])
    packed = (None, None, np.arange(n_shell + 1))
    kernel = _FakeKernel()
    monkeypatch.setattr(fused, "_is_cupy", lambda *arrays: True)
    monkeypatch.setattr(fused, "_kernel", lambda cp: kernel)
    if mode is None:
        monkeypatch.delenv(fused.KERNEL_ENV, raising=False)
    else:
        monkeypatch.setenv(fused.KERNEL_ENV, mode)
    positional, keywords = _route_args(inputs, chunk, packed)
    result = fused.chunk_line_yields(*positional, **keywords)
    assert kernel.launches == launches  # 6000 rows: one chunk, or 7 of 997
    assert (result is None) is (launches == 0)


@pytest.mark.parametrize("mode", [None, "fused", "auto"])
def test_cpu_and_fp64_runs_always_take_the_grouped_chain(monkeypatch, mode):
    inputs = _scoring_inputs("Si")
    if mode is None:
        monkeypatch.delenv(fused.KERNEL_ENV, raising=False)
    else:
        monkeypatch.setenv(fused.KERNEL_ENV, mode)
    positional, keywords = _route_args(inputs, 1 << 20, inputs["packed_shells"])
    if not ON_DEVICE:
        assert fused.chunk_line_yields(*positional, **keywords) is None
    # Without packed float32 tables (the FP64 and CPU paths) it always declines.
    positional, keywords = _route_args(inputs, 1 << 20, None)
    assert fused.chunk_line_yields(*positional, **keywords) is None


@pytest.mark.skipif(not ON_DEVICE, reason="needs a CuPy float32 device")
@pytest.mark.parametrize("element", ["W", "Si"])
@pytest.mark.parametrize("weighted", [False, True])
def test_device_kernel_matches_grouped_chain_and_twin(monkeypatch, element, weighted):
    inputs = _scoring_inputs(element, weighted=weighted)
    monkeypatch.setenv(fused.KERNEL_ENV, "grouped")
    reference = _grouped(inputs, 997)
    monkeypatch.setenv(fused.KERNEL_ENV, "fused")
    kernel = _grouped(inputs, 997)
    np.testing.assert_allclose(kernel, reference, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(kernel, _twin(inputs, 997), rtol=RTOL, atol=0.0)
    # No atomics: a rerun is bitwise identical.
    np.testing.assert_array_equal(_grouped(inputs, 997), kernel)
