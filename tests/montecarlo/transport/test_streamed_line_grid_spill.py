"""Disk-spilled streamed line-grid evidence (#1141).

The spilled grid pass must give the in-memory pass's grid bit for bit: the
out-of-core stable sort and chunked cumulative sum reproduce
``sinc_spacing_from_samples`` exactly, including ties in width, and spilled
production lines are the same arrays. Scratch files never outlive the pass.
"""

import os
import subprocess
from functools import partial

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import line_grid_spill
from pyrite.montecarlo.runner.line_grid_spill import (
    GridSpillPolicy,
    LineGridSpillError,
    SpillStore,
    external_stable_sort,
    prune_stale_spill_dirs,
    sorted_sinc_spacing,
    write_run,
)
from pyrite.montecarlo.runner.streamed_line_grid import (
    BlockGridEvidence,
    resolve_streamed_line_grid,
)
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from pyrite.montecarlo.spectrum.diagnostics import sinc_spacing_from_samples

from .test_streamed_line_grid import (
    _assert_matches,
    _case,
    _clear_grid_cache,
    _joined,
    _measured_policy,
)


@pytest.fixture
def small_sort(monkeypatch):
    """Tiny sort/chunk sizes, so a few hundred rows recurse through buckets."""
    monkeypatch.setattr(line_grid_spill, "SORT_ROWS", 64)
    monkeypatch.setattr(line_grid_spill, "CHUNK_ROWS", 17)
    monkeypatch.setattr(line_grid_spill, "SAMPLE_ROWS", 8)
    monkeypatch.setattr(line_grid_spill, "MAX_BUCKETS", 4)


def _adversarial(rng, kind, size):
    if kind == "distinct":
        width = rng.lognormal(0.0, 2.0, size)
    elif kind == "ties":
        width = rng.choice([0.5, 1.0, 1.0 + 2**-52, 3.0], size)
    elif kind == "one-value":
        width = np.full(size, 2.5)
    elif kind == "one-outlier":
        width = np.full(size, 2.5)
        width[size // 3] = 1e-9
    else:  # descending runs with heavy duplication
        width = np.repeat(np.linspace(9.0, 1.0, 7), -(-size // 7))[:size]
    weight = rng.lognormal(0.0, 3.0, size)
    return width, weight


@pytest.mark.parametrize("kind", ["distinct", "ties", "one-value", "one-outlier", "runs"])
@pytest.mark.parametrize("limit", [0.0, 0.01, 0.5])
def test_spilled_spacing_is_bitwise_the_in_memory_one(tmp_path, small_sort, kind, limit):
    rng = np.random.default_rng(1141)
    width, weight = _adversarial(rng, kind, 700)
    expected = sinc_spacing_from_samples([width.copy(), weight.copy()], aliased_weight_limit=limit)

    store = SpillStore(tmp_path)
    cuts = [0, 1, 1, 90, 333, 512, 700]  # includes an empty block
    runs = [write_run(store, width[a:b], weight[a:b]) for a, b in zip(cuts, cuts[1:], strict=False)]
    ordered = external_stable_sort(store, runs, (width.dtype, weight.dtype))

    order = np.argsort(width, kind="stable")
    np.testing.assert_array_equal(np.fromfile(ordered.width), width[order])
    np.testing.assert_array_equal(np.fromfile(ordered.weight), weight[order])
    assert sorted_sinc_spacing(ordered, aliased_weight_limit=limit) == expected
    # Only the sorted output remains on disk.
    assert sorted(p.name for p in store.root.iterdir()) == sorted(
        [ordered.width.name, ordered.weight.name]
    )
    assert store.peak_bytes <= 2 * 16 * width.size + 16 * 64
    store.close()
    assert not store.root.exists()


def test_spilled_spacing_refuses_an_empty_population(tmp_path):
    store = SpillStore(tmp_path)
    ordered = external_stable_sort(store, [], (np.dtype(float),) * 2)
    with pytest.raises(ValueError, match="no valid segments"):
        sorted_sinc_spacing(ordered)


def _spill_everything(monkeypatch, tmp_path):
    policy = partial(GridSpillPolicy, threshold_bytes=0, directory=tmp_path / "spill")
    monkeypatch.setattr(line_grid_spill, "GridSpillPolicy", policy)
    return tmp_path / "spill"


@pytest.fixture(scope="module", params=["sinc-nyquist", "resonance-local"])
def policy_case(request):
    case = _case()
    if request.param == "resonance-local":
        case = _case(_measured_policy(case))
    return case


@pytest.mark.parametrize("block", [5, 24])
def test_spilled_streamed_grid_matches_the_joined_path(
    policy_case, block, monkeypatch, tmp_path, small_sort
):
    _clear_grid_cache()
    reference = _joined(policy_case)
    _clear_grid_cache()
    spill_dir = _spill_everything(monkeypatch, tmp_path)
    streamed = run_case_streamed(policy_case, block, transport_core="per-electron")
    grid_pass = streamed["_streamed_line_grid_pass"]
    assert grid_pass["transported"] and grid_pass["spilled"]
    assert grid_pass["spill_peak_bytes"] >= grid_pass["evidence_bytes"] > 0
    _assert_matches(streamed, reference)
    assert list(spill_dir.iterdir()) == []


def test_spilled_and_in_memory_grids_are_equal_mid_run(monkeypatch, tmp_path):
    """Spilling that starts after some blocks were held gives the same grid."""
    case = _case(_measured_policy(_case()))
    _clear_grid_cache()
    kwargs = dict(transport_core="per-electron", n_hat=_n_hat(case), groove=None)
    held, held_record, held_pass = resolve_streamed_line_grid(
        case, 4, spill=GridSpillPolicy(threshold_bytes=None), **kwargs
    )
    assert not held_pass["spilled"]
    _clear_grid_cache()
    # Hold the first two blocks, then spill them and every later block.
    monkeypatch.setattr(BlockGridEvidence, "_should_spill", lambda self, seen: seen[0] > 8)
    spilled, record, grid_pass = resolve_streamed_line_grid(
        case, 4, spill=GridSpillPolicy(threshold_bytes=0, directory=tmp_path), **kwargs
    )
    assert grid_pass["spilled"]
    assert grid_pass["evidence_bytes"] == held_pass["evidence_bytes"]
    np.testing.assert_array_equal(spilled, held)
    assert {k: v for k, v in record.items() if k != "cache"} == {
        k: v for k, v in held_record.items() if k != "cache"
    }


def _n_hat(case):
    from pyrite.montecarlo.runner.artifacts import _case_geometry

    return _case_geometry(case)[3]


def test_scratch_is_removed_when_the_pass_fails(monkeypatch, tmp_path):
    spill_dir = _spill_everything(monkeypatch, tmp_path)
    case = _case()
    _clear_grid_cache()

    def failing_resolution(*args, **kwargs):
        assert any(spill_dir.rglob("*.raw"))  # evidence was on disk
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.cxr_grid, "_resolve_policy_grid", failing_resolution)
    with pytest.raises(KeyboardInterrupt):
        run_case_streamed(case, 8, transport_core="per-electron")
    assert list(spill_dir.iterdir()) == []


def test_insufficient_disk_refuses_at_the_first_block(monkeypatch, tmp_path):
    spill_dir = _spill_everything(monkeypatch, tmp_path)
    monkeypatch.setattr(SpillStore, "free_bytes", lambda self: 0)
    blocks = []
    transport = runner._transport_case

    def counted(*args, **kwargs):
        consumer = kwargs["block_consumer"]

        def consume(block):
            blocks.append(int(block["Ne"]))
            consumer(block)

        return transport(*args, **{**kwargs, "block_consumer": consume})

    monkeypatch.setattr(runner, "_transport_case", counted)
    _clear_grid_cache()
    with pytest.raises(LineGridSpillError, match="scratch disk"):
        run_case_streamed(_case(), 4, transport_core="per-electron")
    assert len(blocks) == 1
    assert list(spill_dir.iterdir()) == []


def test_stale_directories_of_dead_processes_are_pruned(tmp_path):
    dead = subprocess.Popen(["true"])
    dead.wait()
    store = SpillStore(tmp_path)
    tag = store.root.name.split("-", 1)[0]
    stale = tmp_path / f"{tag}-{dead.pid}-deadbeef"
    alive = tmp_path / f"{tag}-{os.getppid()}-cafef00d"
    other_host = tmp_path / f"elsewhere-{dead.pid}-deadbeef"
    for path in (stale, alive, other_host):
        path.mkdir()
    assert prune_stale_spill_dirs(tmp_path) == [stale]
    assert alive.exists() and other_host.exists() and store.root.exists()


def test_a_block_that_fails_mid_spill_is_not_counted_twice(monkeypatch, tmp_path):
    """A consumer failure leaves no partial block, so the replay counts it once."""
    spill_dir = _spill_everything(monkeypatch, tmp_path)
    case = _case(_measured_policy(_case()))
    _clear_grid_cache()
    kwargs = dict(transport_core="per-electron", n_hat=_n_hat(case), groove=None)
    held, _, held_pass = resolve_streamed_line_grid(
        case, 4, spill=GridSpillPolicy(threshold_bytes=None), **kwargs
    )
    _clear_grid_cache()
    real_add, failures = BlockGridEvidence.add, []

    def add_failing_once(self, segs):
        if len(failures) < 2 and self._seen[0] in (0, 8):  # first spill, later block
            append = line_grid_spill.LineFile.append

            def broken(lines, populations):
                with lines.path.open("ab") as stream:
                    stream.write(b"partial block")  # a write cut short, as by an OOM
                raise MemoryError("injected")

            monkeypatch.setattr(line_grid_spill.LineFile, "append", broken)
            try:
                real_add(self, segs)
            except MemoryError:
                failures.append(self._seen[0])
            finally:
                monkeypatch.setattr(line_grid_spill.LineFile, "append", append)
        real_add(self, segs)  # the replay

    monkeypatch.setattr(BlockGridEvidence, "add", add_failing_once)
    grid, _, grid_pass = resolve_streamed_line_grid(
        case, 4, spill=line_grid_spill.GridSpillPolicy(), **kwargs
    )
    assert failures == [0, 8]
    assert grid_pass["spilled"]
    assert grid_pass["evidence_bytes"] == held_pass["evidence_bytes"]
    np.testing.assert_array_equal(grid, held)
    assert list(spill_dir.iterdir()) == []


def test_spilled_lines_share_one_descriptor_and_keep_their_bits(tmp_path):
    from pyrite.montecarlo.spectrum.cxr_seeds import ResonancePopulation

    rng = np.random.default_rng(7)
    store = SpillStore(tmp_path)
    lines = line_grid_spill.LineFile(store)
    blocks, records = [], []
    for index in range(1500):  # beyond a 1024-descriptor limit as per-array maps
        size = int(rng.integers(0, 5))
        dtype = np.float32 if index % 2 else np.float64
        population = ResonancePopulation(
            f"r{index % 3}",
            *(rng.random(size).astype(dtype) for _ in range(3)),
        )
        blocks.append(population)
        records += lines.append([population])
    descriptors = len(os.listdir("/proc/self/fd"))
    served = lines.populations(records)
    assert len(os.listdir("/proc/self/fd")) <= descriptors + 1
    for got, want in zip(served, blocks, strict=True):
        assert got.label == want.label
        for field in ("energy_eV", "weight", "width_eV"):
            array = getattr(got, field)
            assert array.dtype == getattr(want, field).dtype
            np.testing.assert_array_equal(array, getattr(want, field))
            assert array.size == 0 or array.ctypes.data % 64 == 0
    assert store.bytes == lines.size
    lines.close()
    store.close()


def test_the_pivot_scan_releases_each_chunk(tmp_path, monkeypatch):
    """The pivot sample must not pin chunks: that held 8 B/row (job 13619)."""
    import gc
    import weakref

    store = SpillStore(tmp_path)
    rng = np.random.default_rng(3)
    runs = [write_run(store, rng.random(50), rng.random(50)) for _ in range(6)]
    alive = []
    real = line_grid_spill.iter_run_chunks

    def tracked(runs_, chunk_rows=None):
        for width, weight in real(runs_, chunk_rows):
            gc.collect()
            alive.append(sum(ref() is not None for ref in refs))
            refs.append(weakref.ref(width))
            yield width, weight
            del width, weight

    refs = []
    monkeypatch.setattr(line_grid_spill, "iter_run_chunks", tracked)
    monkeypatch.setattr(line_grid_spill, "SAMPLE_ROWS", 16)
    lowest, highest, sample = line_grid_spill._scan(runs, 300)
    assert max(alive) <= 1  # only the loop variable's previous chunk
    assert lowest <= sample.min() and sample.max() <= highest
