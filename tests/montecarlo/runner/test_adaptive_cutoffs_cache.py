"""Adaptive transport host work independent of ``max_histories`` (#1452).

Piecewise cutoffs must equal the dense per-history array; an adaptive case must
not allocate arrays sized by ``max_histories``; and the shell inelastic tables
are built once per case, not once per transport block. Equality claims are
exact; seeds are arbitrary fixed integers.
"""

import tracemalloc
from dataclasses import replace

import numpy as np
import pytest

from pyrite.montecarlo.runner import block_transport
from pyrite.montecarlo.runner.adaptive import (
    AdaptiveSettings,
    run_case_adaptive,
    transport_case_adaptive,
)
from pyrite.montecarlo.runner.cutoffs import (
    PiecewiseCutoffs,
    case_cutoffs,
    cutoff_floor,
    electron_cutoffs,
)
from pyrite.montecarlo.transport import host_memory, shell_transport
from pyrite.montecarlo.transport.beam_entry import initial_energies_keV

from .test_adaptive_stopping import _SETTINGS, _assert_identical, _case, _fixed

_CASES = [
    {"Ne": 10, "Ne_brem": 10},
    {"Ne": 10, "Ne_brem": 4},
    {"Ne": 4, "Ne_brem": 10},
    {"Ne": 7, "Ne_brem": 0},
    {"Ne": 0, "Ne_brem": 7},
    {"Ne": 9, "Ne_brem": 9, "E_cut_lines_keV": 0.5, "E_cut_brem_keV": 2.0},
    {"Ne": 9, "Ne_brem": 3, "E_cut_lines_keV": 0.5, "E_cut_brem_keV": 2.0},
]


@pytest.mark.parametrize("case", _CASES)
def test_piecewise_cutoffs_equal_the_dense_array(case):
    dense = electron_cutoffs(case)
    cuts = case_cutoffs(case)
    n = dense.size
    assert cuts.size == len(cuts) == n
    np.testing.assert_array_equal(np.asarray(cuts), dense)
    for start in range(n + 1):
        for stop in range(start, n + 1):
            np.testing.assert_array_equal(cuts[start:stop], dense[start:stop])
    for k in range(1, n + 1):
        assert cuts.prefix_min(k) == cutoff_floor(dense, k) == cutoff_floor(cuts, k)


@pytest.mark.parametrize("case", [c for c in _CASES if c["Ne"] == c["Ne_brem"]])
def test_realized_pair_cutoffs_equal_the_dense_rewrite(case):
    """The post-transport rewrite: truncate to ``realized``, line cutoff past ``brem``."""
    dense = electron_cutoffs(case)
    for realized in range(1, dense.size + 1):
        for brem in range(1, realized + 1):
            expected = dense[:realized].copy()
            expected[brem:] = case.get("E_cut_lines_keV", 5.0)
            pair = {**case, "Ne": realized, "Ne_brem": brem}
            got = electron_cutoffs(pair, piecewise=True)
            np.testing.assert_array_equal(np.asarray(got), expected)


def test_piecewise_cutoffs_reject_non_slice_indexing():
    with pytest.raises(TypeError):
        PiecewiseCutoffs(((4, 1.0),))[np.array([0, 1])]


def test_huge_population_slices_without_a_dense_array():
    cuts = case_cutoffs({"Ne": 10**9, "Ne_brem": 10**9})
    assert cuts.size == 10**9
    tracemalloc.start()
    try:
        block = cuts[10**8 : 10**8 + 1000]
        floor = cuts.prefix_min(10**9)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert block.shape == (1000,) and floor == 1.0
    assert peak < 1 << 20


@pytest.mark.parametrize("n", [1, 5, 7, 64])
def test_chunked_energy_maximum_equals_the_dense_maximum(monkeypatch, n):
    monkeypatch.setattr(block_transport, "_ENERGY_SCAN_CHUNK", 8)
    spread = {"energy_spread_frac": 1e-2}
    dense = initial_energies_keV(30.0, 64, 5, 1e-2)
    max_energy = block_transport._prefix_max_energy(30.0, 5, spread)
    assert max_energy(n) == float(np.max(dense[:n]))
    assert max_energy(n) == float(np.max(dense[:n]))  # cached chunks
    mono = block_transport._prefix_max_energy(30.0, 5, {})
    assert mono(n) == 30.0


def test_adaptive_transport_allocates_nothing_sized_by_max_histories(monkeypatch):
    """Peak host memory is flat in ``max_histories`` (it was ~48 B per history)."""
    # The admission projection to max_histories would switch routes; transport only.
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 10**13)
    case = _case(energy_spread_frac=1e-3)

    def peak_bytes(max_histories):
        settings = replace(
            _SETTINGS,
            min_histories=40,
            max_histories=max_histories,
            brem_max_histories=max_histories,
            block_histories=20,
            min_effective_histories=0.0,
            max_variance_of_variance=1.0,
        )
        tracemalloc.start()
        try:
            transport = transport_case_adaptive(case, settings, transport_core="per-electron")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak, transport["adaptive_sampling"]["realized_electrons"]

    peak_bytes(1_000)  # one-off table loads and compilation
    small, n_small = peak_bytes(100_000)
    large, n_large = peak_bytes(10_000_000)
    assert n_small == n_large
    # A dense cutoff array alone is 80 MB here.
    assert large - small < 16 * 1024**2


def test_adaptive_run_with_spread_is_the_fixed_run():
    """Piecewise cutoffs and chunked energy range leave the output bit-identical."""
    case = _case(energy_spread_frac=1e-3)
    settings = replace(_SETTINGS, min_effective_histories=0.0, max_variance_of_variance=1.0)
    result, stats = run_case_adaptive(case, settings, transport_core="per-electron")
    _assert_identical(result, _fixed(case, stats["realized_electrons"]))


def test_shell_tables_build_once_for_a_multi_block_case(monkeypatch):
    builds = []
    real = shell_transport.build_shell_inelastic_tables

    def counting(*args, **kwargs):
        builds.append(args[3:5])
        return real(*args, **kwargs)

    requests = []
    cached = shell_transport.cached_shell_inelastic_tables

    def counting_request(*args, **kwargs):
        requests.append(args[3:5])
        return cached(*args, **kwargs)

    monkeypatch.setattr(shell_transport, "build_shell_inelastic_tables", counting)
    monkeypatch.setattr(shell_transport, "cached_shell_inelastic_tables", counting_request)
    shell_transport._table_cache.clear()
    case = _case()
    assert case.get("inelastic_model") == "shell-soft-hard"
    settings = AdaptiveSettings(
        target_rse=1e-4, min_histories=20, max_histories=60, block_histories=10
    )
    with pytest.warns(UserWarning):
        run_case_adaptive(case, settings, transport_core="per-electron")
    assert len(requests) >= 3  # one request per transport block
    assert len(builds) == 1  # one table build


def test_shell_table_cache_is_keyed_and_bounded():
    shell_transport._table_cache.clear()
    log_e = np.log(np.array([1.0, 10.0, 100.0]))
    request = ("silicon", 50.0, [(log_e, log_e * 0.0)], 2.0, 20.0)

    class _Tables:
        pass

    built = []

    def fake(*args, **kwargs):
        built.append(args)
        return _Tables()

    original = shell_transport.build_shell_inelastic_tables
    shell_transport.build_shell_inelastic_tables = fake
    try:
        first = shell_transport.cached_shell_inelastic_tables(*request)
        assert shell_transport.cached_shell_inelastic_tables(*request) is first
        assert len(built) == 1
        other = shell_transport.cached_shell_inelastic_tables(
            "silicon", 50.0, [(log_e, log_e * 0.0 + 1.0)], 2.0, 20.0
        )
        assert other is not first and len(built) == 2
        for cutoff in range(60, 60 + 2 * shell_transport._TABLE_CACHE_SIZE):
            shell_transport.cached_shell_inelastic_tables(
                "silicon", float(cutoff), request[2], 2.0, 20.0
            )
        assert len(shell_transport._table_cache) == shell_transport._TABLE_CACHE_SIZE
    finally:
        shell_transport.build_shell_inelastic_tables = original
        shell_transport._table_cache.clear()
