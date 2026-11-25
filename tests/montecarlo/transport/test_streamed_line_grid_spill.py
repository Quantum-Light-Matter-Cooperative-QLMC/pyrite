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
from pyrite.paths import cache_dir

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


# --- memory-aware default threshold (#1453) -------------------------------

GIB = 1024**3


def _host(monkeypatch, *, available, cgroup=None):
    """A host with mocked ``MemAvailable`` and cgroup headroom, in bytes."""
    from pyrite.montecarlo.transport import host_memory

    monkeypatch.delenv("PYRITE_MC_HOST_LIMIT_BYTES", raising=False)
    monkeypatch.setattr(
        host_memory.psutil, "virtual_memory", lambda: type("VM", (), {"available": available})()
    )
    monkeypatch.setattr(host_memory, "_cgroup_headroom_bytes", lambda: cgroup)


@pytest.mark.parametrize(
    ("available", "cgroup", "expected"),
    [
        (64 * GIB, None, int(0.25 * (62 * GIB))),  # roomy host: a quarter of headroom
        (64 * GIB, 16 * GIB, int(0.25 * (16 * GIB - 1.6 * GIB))),  # tighter cgroup wins
        (4 * GIB, None, 2 * GIB),  # constrained host: the 2 GiB floor
        (64 * GIB, 1 * GIB, 2 * GIB),  # tiny cgroup: floor
        (0, None, 2 * GIB),
    ],
)
def test_default_threshold_follows_available_memory(monkeypatch, available, cgroup, expected):
    _host(monkeypatch, available=available, cgroup=cgroup)
    assert line_grid_spill.default_spill_threshold_bytes() == expected
    assert GridSpillPolicy().threshold_bytes == expected
    assert expected >= line_grid_spill.MIN_SPILL_THRESHOLD_BYTES == 2 * GIB


def test_default_threshold_honors_the_host_limit_env(monkeypatch):
    _host(monkeypatch, available=64 * GIB)
    monkeypatch.setenv("PYRITE_MC_HOST_LIMIT_BYTES", str(24 * GIB))
    threshold = line_grid_spill.default_spill_threshold_bytes()
    assert line_grid_spill.MIN_SPILL_THRESHOLD_BYTES < threshold <= 0.25 * 24 * GIB


@pytest.mark.parametrize(
    "failure", [OSError("no /proc/meminfo"), FileNotFoundError(), ValueError()]
)
def test_unreadable_memory_falls_back_to_the_floor(monkeypatch, failure):
    from pyrite.montecarlo.transport import host_memory

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(host_memory.psutil, "virtual_memory", broken)
    assert line_grid_spill.default_spill_threshold_bytes() == 2 * GIB


def test_malformed_host_limit_env_falls_back_to_the_floor(monkeypatch):
    _host(monkeypatch, available=64 * GIB)
    monkeypatch.setenv("PYRITE_MC_HOST_LIMIT_BYTES", "not-a-number")
    assert line_grid_spill.default_spill_threshold_bytes() == 2 * GIB


def test_missing_cgroup_files_leave_meminfo_alone(monkeypatch, tmp_path):
    from pyrite.montecarlo.transport import host_memory

    _host(monkeypatch, available=64 * GIB)
    monkeypatch.undo()  # keep the cgroup reader real, over a root that is absent
    monkeypatch.delenv("PYRITE_MC_HOST_LIMIT_BYTES", raising=False)
    monkeypatch.setattr(host_memory, "Path", lambda *parts: tmp_path / "absent")
    monkeypatch.setattr(
        host_memory.psutil, "virtual_memory", lambda: type("VM", (), {"available": 40 * GIB})()
    )
    assert host_memory._cgroup_headroom_bytes() is None
    assert line_grid_spill.default_spill_threshold_bytes() == int(0.25 * 38 * GIB)


@pytest.mark.parametrize(
    ("mib", "directory", "expected"),
    [(0, "d", 0), (7, "d", 7 * 1024**2), (7, None, 7 * 1024**2), (None, "d", "default")],
)
def test_explicit_spill_settings_override_the_memory_default(
    monkeypatch, tmp_path, mib, directory, expected
):
    import pyrite as pr
    from pyrite.montecarlo.runner.transport_options import stream_spill_policy

    _host(monkeypatch, available=512 * GIB)  # a huge host changes nothing explicit
    directory = None if directory is None else str(tmp_path / directory)
    numerics = pr.Numerics(
        execution=pr.config.KernelTuning(
            stream_block_electrons=5, stream_spill_dir=directory, stream_spill_mib=mib
        )
    )
    policy = stream_spill_policy(numerics)
    assert policy.threshold_bytes == (
        line_grid_spill.default_spill_threshold_bytes() if expected == "default" else expected
    )
    assert policy.directory == directory


def test_no_spill_settings_keep_the_default_policy():
    import pyrite as pr
    from pyrite.montecarlo.runner.transport_options import stream_spill_policy

    assert stream_spill_policy(pr.Numerics(execution=pr.config.KernelTuning())) is None


def test_default_spills_on_a_constrained_host_and_not_on_a_roomy_one(monkeypatch):
    """Same evidence, same grid: it stays in memory only where the host has room.

    The floor is scaled down so a few kilobytes of evidence stand in for the
    gigabytes of a low-tilt run. The default policy is built under a mocked
    memory probe; transport itself runs on the real host.
    """
    monkeypatch.setattr(line_grid_spill, "MIN_SPILL_THRESHOLD_BYTES", 2048)
    case = _case(_measured_policy(_case()))
    kwargs = dict(transport_core="per-electron", n_hat=_n_hat(case), groove=None)

    def resolve(available):
        with monkeypatch.context() as mocked:
            _host(mocked, available=available)
            policy = GridSpillPolicy()  # what a ``spill=None`` pass builds
        _clear_grid_cache()
        return resolve_streamed_line_grid(case, 4, spill=policy, **kwargs)

    tight, tight_record, tight_pass = resolve(1 << 12)  # headroom below the floor
    assert tight_pass["spilled"] and tight_pass["evidence_bytes"] > 2048
    assert list((cache_dir() / "line-grid-spill").iterdir()) == []  # removed on success
    roomy, roomy_record, roomy_pass = resolve(64 * GIB)  # same evidence, held
    assert not roomy_pass["spilled"] and roomy_pass["spill_peak_bytes"] == 0
    assert roomy_pass["evidence_bytes"] == tight_pass["evidence_bytes"]

    np.testing.assert_array_equal(tight, roomy)
    assert {k: v for k, v in tight_record.items() if k != "cache"} == {
        k: v for k, v in roomy_record.items() if k != "cache"
    }
    assert not any((cache_dir() / "line-grid-spill").rglob("*"))


def test_forced_small_threshold_is_bitwise_the_in_memory_spacing(tmp_path, small_sort):
    """A forced tiny threshold sends the pass through the external sort."""
    rng = np.random.default_rng(1453)
    width, weight = _adversarial(rng, "ties", 500)
    expected = sinc_spacing_from_samples([width.copy(), weight.copy()], aliased_weight_limit=0.01)
    store = SpillStore(tmp_path)
    runs = [write_run(store, width[a:b], weight[a:b]) for a, b in [(0, 200), (200, 500)]]
    ordered = external_stable_sort(store, runs, (width.dtype, weight.dtype))
    assert sorted_sinc_spacing(ordered, aliased_weight_limit=0.01) == expected
    store.close()


# --- host admission of the in-memory spacing sort (#1453) -----------------


def _refuse_sort(monkeypatch):
    """Admission that refuses the spacing sort only; returns the asked sorts."""
    from pyrite.montecarlo.transport import host_memory

    asked = []
    admit = host_memory.admit_host_allocation

    def refuse(required, phase):
        if phase != "streamed line-grid spacing sort":
            return admit(required, phase)
        asked.append((required, phase))
        raise host_memory.HostMemoryResourceError(phase, required, 0)

    monkeypatch.setattr(host_memory, "admit_host_allocation", refuse)
    return asked


def test_refused_sort_admission_spills_to_the_same_grid(monkeypatch):
    from pyrite.montecarlo.spectrum.diagnostics import sinc_spacing_sort_bytes

    case = _case(_measured_policy(_case()))
    kwargs = dict(transport_core="per-electron", n_hat=_n_hat(case), groove=None)
    held_policy = GridSpillPolicy(threshold_bytes=1 << 40)  # never reached by size
    _clear_grid_cache()
    held, held_record, held_pass = resolve_streamed_line_grid(case, 4, spill=held_policy, **kwargs)
    assert not held_pass["spilled"]

    asked = _refuse_sort(monkeypatch)
    _clear_grid_cache()
    spilled, record, grid_pass = resolve_streamed_line_grid(case, 4, spill=held_policy, **kwargs)
    assert grid_pass["spilled"] and grid_pass["spill_peak_bytes"] > 0
    rows = grid_pass["evidence_bytes"]  # sinc bytes only: this policy keeps lines too
    assert len(asked) == 1 and asked[0][1] == "streamed line-grid spacing sort"
    assert 0 < asked[0][0] <= sinc_spacing_sort_bytes(rows // 16 + 1)
    np.testing.assert_array_equal(spilled, held)
    assert {k: v for k, v in record.items() if k != "cache"} == {
        k: v for k, v in held_record.items() if k != "cache"
    }
    assert list((cache_dir() / "line-grid-spill").iterdir()) == []


def test_refused_sort_admission_cannot_spill_when_spilling_is_off(monkeypatch):
    case = _case()
    kwargs = dict(transport_core="per-electron", n_hat=_n_hat(case), groove=None)
    asked = _refuse_sort(monkeypatch)
    _clear_grid_cache()
    _, _, grid_pass = resolve_streamed_line_grid(
        case, 4, spill=GridSpillPolicy(threshold_bytes=None), **kwargs
    )
    assert not grid_pass["spilled"] and asked == []


def test_selection_route_gives_the_joined_grid_held_and_spilled(monkeypatch, tmp_path):
    """Spacing by selection (#1453) feeds the same grid on held blocks and spilled runs."""
    from pyrite.montecarlo.spectrum import spacing_selection

    monkeypatch.setattr(spacing_selection, "SELECT_MIN_ROWS", 1)
    monkeypatch.setattr(spacing_selection, "SELECT_CHUNK_ROWS", 64)
    monkeypatch.setattr(spacing_selection, "SAMPLE_STRIDE", 1)
    taken = []
    select = spacing_selection.sinc_spacing_by_selection

    def spy(source, limit):
        result = select(source, limit)
        taken.append(result is not None)
        return result

    monkeypatch.setattr(spacing_selection, "sinc_spacing_by_selection", spy)
    case = _case()
    kwargs = dict(transport_core="per-electron", n_hat=_n_hat(case), groove=None)
    _clear_grid_cache()
    reference = _joined(case)  # the joined resolver also reduces by selection
    _clear_grid_cache()
    held, _, held_pass = resolve_streamed_line_grid(
        case, 4, spill=GridSpillPolicy(threshold_bytes=None), **kwargs
    )
    _clear_grid_cache()
    spilled, _, spill_pass = resolve_streamed_line_grid(
        case, 4, spill=GridSpillPolicy(threshold_bytes=0, directory=tmp_path), **kwargs
    )
    assert not held_pass["spilled"] and spill_pass["spilled"]
    assert taken and all(taken)  # selection answered every reduction, no sort fallback
    np.testing.assert_array_equal(held, spilled)
    np.testing.assert_array_equal(held, reference["E_grid"])
    assert not any(tmp_path.rglob("*.raw"))
