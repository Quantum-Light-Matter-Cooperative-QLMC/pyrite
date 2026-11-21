"""Replay retention/identity contracts, with no transport or expensive scoring."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._util.hashing import canonical_json_digest, file_sha256
from pyrite.perf._track_capture import TrackCapture, load_index, read_shard, source_fingerprint

HERE = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
REPLAY = runpy.run_path(str(HERE / "replay_cone.py"))


def tracks(n=3, ids=(0, 2)):
    rows = len(ids)
    return dict(
        Ne=n,
        elec_id=np.array(ids, dtype=np.int64),
        L_ang=np.ones(rows),
        E_keV=np.full(rows, 100.0),
        E_repr_keV=np.full(rows, 99.0),
        r_mid=np.zeros((rows, 3)),
        v_hat=np.tile([0.0, 0.0, 1.0], (rows, 1)),
        thickness_ang=100.0,
        crystal_width_ang=1000.0,
        crystal_height_ang=1000.0,
        radiative=dict(
            model="bremslib-soft-hard",
            cutoff_eV=1000.0,
            bremslib_tables=(("Si", 14, "key", "digest"),),
        ),
    )


def capture(tmp_path):
    root = tmp_path / "tracks"
    writer = TrackCapture(root, 5, dict(seed=301, cone={}, context={}), 1_000_000)
    writer.append(tracks(), 0, 3)
    writer.append(tracks(2, ()), 3, 2)
    writer.finish()
    return root


def resign(root, record):
    record.pop("manifest_sha256", None)
    record["manifest_sha256"] = canonical_json_digest(record)
    (root / "capture.json").write_text(json.dumps(record))


def test_capture_preserves_dtypes_rows_and_all_zero_score_primaries(tmp_path):
    root = capture(tmp_path)
    index = load_index(root, 1_000_000)
    assert index["histories"] == 5 and not index["timing_comparison_eligible"]
    segments, ids = read_shard(root, index["shards"][0])
    np.testing.assert_array_equal(ids, [0, 1, 2])
    original = tracks()
    for key in ("elec_id", "L_ang", "E_repr_keV", "r_mid", "v_hat"):
        np.testing.assert_array_equal(segments[key], original[key])
        assert segments[key].dtype == original[key].dtype
    assert segments["radiative"] == original["radiative"]
    empty, ids = read_shard(root, index["shards"][1])
    assert empty["Ne"] == 2 and empty["elec_id"].size == 0
    np.testing.assert_array_equal(ids, [3, 4])


def test_partial_and_over_budget_captures_never_publish_completion(tmp_path):
    root = tmp_path / "partial"
    writer = TrackCapture(root, 5, {}, 1_000_000)
    writer.append(tracks(), 0, 3)
    with pytest.raises(ValueError, match="partial"):
        writer.finish()
    assert not (root / "capture.json").exists()
    with pytest.raises(FileNotFoundError):
        load_index(root, 1_000_000)
    small = TrackCapture(tmp_path / "small", 3, {}, 100)
    with pytest.raises(ValueError, match="ceiling"):
        small.append(tracks(), 0, 3)
    assert not (small.root / "capture.json").exists()
    assert not list(small.root.glob("*.npz"))


def test_no_repeated_shards_or_overwrite(tmp_path):
    writer = TrackCapture(tmp_path / "tracks", 3, {}, 1_000_000)
    with pytest.raises(ValueError, match="contiguous"):
        writer.append(tracks(), 1, 3)
    writer.append(tracks(), 0, 3)
    with pytest.raises(ValueError, match="contiguous"):
        writer.append(tracks(), 0, 3)
    writer.finish()
    with pytest.raises(ValueError, match="completed"):
        writer.finish()
    with pytest.raises(FileExistsError):
        TrackCapture(writer.root, 3, {}, 1_000_000)


@pytest.mark.parametrize(
    "damage", ["bytes", "missing", "offset", "extra", "manifest", "population"]
)
def test_capture_preflight_rejects_incomplete_or_corrupt_inputs(tmp_path, damage):
    root = capture(tmp_path)
    record = json.loads((root / "capture.json").read_text())
    path = root / record["shards"][0]["file"]
    if damage == "bytes":
        path.write_bytes(b"damaged")
    elif damage == "missing":
        path.unlink()
    elif damage == "extra":
        (root / "tracks-123456.npz").write_bytes(b"extra")
    elif damage == "manifest":
        record["histories"] += 1
        (root / "capture.json").write_text(json.dumps(record))
    else:
        if damage == "offset":
            record["shards"][1]["offset"] = 2
        else:
            record["histories"] = 6
        resign(root, record)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_index(root, 1_000_000)


@pytest.mark.parametrize("damage", ["id", "nan", "geometry", "flux", "radiative"])
def test_unserializable_or_unaccounted_physics_is_rejected(tmp_path, damage):
    row = tracks()
    if damage == "id":
        row["elec_id"][0] = 3
    elif damage == "nan":
        row["E_keV"][0] = np.nan
    elif damage == "geometry":
        row["thickness_ang"] = -1
    elif damage == "flux":
        row["channeling_flux"] = np.ones((2, 1))
    else:
        row.pop("radiative")
    writer = TrackCapture(tmp_path / "tracks", 3, {}, 1_000_000)
    with pytest.raises(ValueError):
        writer.append(row, 0, 3)
    assert not (writer.root / "capture.json").exists()


def test_replay_pairs_exact_primaries_and_keeps_misses(tmp_path):
    root = capture(tmp_path)
    index = load_index(root, 1_000_000)
    seen = []

    def scorer(segments, n, _cone, ctx):
        seen.append((n, segments["elec_id"].copy()))
        return dict(
            cone_source_photons_ge10kev=np.bincount(
                segments["elec_id"], weights=segments["L_ang"], minlength=n
            )
            * ctx["energy_order"]
        )

    arrays = REPLAY["replay"](root, index, {}, scorer, 100, 1_000_000)
    np.testing.assert_array_equal(arrays["primary_ids"], np.arange(5))
    np.testing.assert_array_equal(arrays["point-e8"], [8, 0, 8, 0, 0])
    np.testing.assert_array_equal(
        arrays["difference-point-e8-minus-integrated-L3"], [-24, 0, -24, 0, 0]
    )
    assert len(seen) == 2 * len(REPLAY["rules"]())
    with pytest.raises(ValueError, match="ceiling"):
        REPLAY["replay"](root, index, {}, scorer, 100, 1)


def test_audit_only_needs_no_scorer_tables_or_cuda(tmp_path, monkeypatch):
    root = capture(tmp_path)
    output = tmp_path / "audit.json"
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        sys, "argv", ["replay_cone.py", str(root), "--audit-only", "--output", str(output)]
    )
    REPLAY["main"]()
    result = json.loads(output.read_text())
    assert result["audit_only"] and result["numerical_gate"] == "unmeasured"
    assert not output.with_suffix(".npz").exists()


def test_changed_payload_after_preflight_and_reordered_ids_are_rejected(tmp_path):
    root = capture(tmp_path)
    index = load_index(root, 1_000_000)
    shard = index["shards"][0]
    path = root / shard["file"]
    with np.load(path) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["primary_ids"] = np.array([0, 2, 1])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="changed"):
        read_shard(root, shard)
    shard["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="incident IDs"):
        read_shard(root, shard)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--estimator", "continuous", "--no-audit"],
        ["--estimator", "continuous", "--tables-only"],
        ["--estimator", "continuous", "--capture-max-bytes", "0"],
    ],
)
def test_invalid_capture_requests_fail_before_tables_or_cuda(tmp_path, monkeypatch, extra):
    harness = HERE / "pyrite_bench.py"
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(harness),
            "si-300keV-100um",
            "--histories",
            "5",
            "--seed",
            "301",
            "--output",
            str(tmp_path / "sample.json"),
            "--capture-tracks",
            str(tmp_path / "tracks"),
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(harness))["main"]()
    assert error.value.code == 2
    assert not (tmp_path / "tracks").exists()


def test_harness_captures_only_warm_request_with_global_block_ids(tmp_path, monkeypatch):
    """Fake transport exercises the actual benchmark loop; no GPU or histories run."""
    from pyrite.montecarlo import transport
    from pyrite.perf import _benchmark_tables, detector_scoring
    from pyrite.xsgen.bremslib import tables as bremslib
    from pyrite.xsgen.elsepa import catalog as elsepa
    from pyrite.xsgen.sbethe import catalog as sbethe

    seeds = []

    def simulate(**kwargs):
        n = kwargs["Ne"]
        seed = kwargs["seed"]
        seeds.append(seed)
        row = tracks(n, tuple(range(n)))
        row["L_ang"][:] = seed
        row["thickness_ang"] = kwargs["thickness_ang"]
        row["crystal_width_ang"] = kwargs["crystal_width_mm"] * 1e7
        row["crystal_height_ang"] = kwargs["crystal_height_mm"] * 1e7
        row["event_kind"] = np.zeros(n, dtype=np.int64)
        return row

    def no_op(*a, **kw):
        pass

    pool = SimpleNamespace(set_limit=no_op, free_all_blocks=no_op)
    cuda = SimpleNamespace(
        Device=lambda: SimpleNamespace(synchronize=no_op),
        runtime=SimpleNamespace(getDeviceProperties=lambda _: dict(name=b"fake-test-device")),
    )
    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(cuda=cuda, get_default_memory_pool=lambda: pool)
    )
    monkeypatch.setattr(transport, "_simulate_trajectories", simulate)
    monkeypatch.setattr(
        elsepa, "resolve_layer_tables", lambda _: (SimpleNamespace(arrays={}, tables=[]),)
    )
    monkeypatch.setattr(
        sbethe,
        "resolve_composition_table",
        lambda *_: SimpleNamespace(arrays=lambda: {}, key="stopping", digest="digest"),
    )
    monkeypatch.setattr(
        bremslib, "load_bremsstrahlung_tables", lambda _: {"Si": SimpleNamespace(key="brem")}
    )
    monkeypatch.setattr(_benchmark_tables, "select_elastic_tables", lambda entries, *_: entries)
    monkeypatch.setattr(_benchmark_tables, "select_stopping_table", lambda baseline, *_: baseline)
    monkeypatch.setattr(_benchmark_tables, "benchmark_table_audit", lambda *_: {})
    monkeypatch.setattr(detector_scoring, "absorption_edges_keV", lambda _: [])
    harness = HERE / "pyrite_bench.py"
    main = runpy.run_path(str(harness))["main"]
    monkeypatch.setitem(
        main.__globals__, "block_scores", lambda r, n, e: dict(transmitted=np.ones(n))
    )
    monkeypatch.setitem(
        main.__globals__,
        "continuous_cone_scores",
        lambda r, n, cone, ctx: dict(cone_source_photons_ge10kev=np.zeros(n)),
    )
    monkeypatch.setitem(main.__globals__, "spectrum_grids", lambda *a: {})
    output = tmp_path / "sample.json"
    root = tmp_path / "tracks"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(harness),
            "si-300keV-100um",
            "--histories",
            "5",
            "--seed",
            "301",
            "--block",
            "2",
            "--score-rows",
            "1",
            "--estimator",
            "continuous",
            "--capture-tracks",
            str(root),
            "--output",
            str(output),
        ],
    )
    main()
    index = load_index(root, 1_000_000)
    assert {301, 1000301, 2000301} == set(seeds)
    assert index["metadata"]["seed"] == 301
    assert index["histories"] == len(index["shards"]) == 5
    assert [s["offset"] for s in index["shards"]] == list(range(5))
    for shard in index["shards"]:
        row, _ = read_shard(root, shard)
        np.testing.assert_array_equal(row["L_ang"], [301])
    result = json.loads(output.read_text())
    assert result["purpose"] == "accuracy-only"
    assert result["track_capture"]["manifest_sha256"] == index["manifest_sha256"]


@pytest.mark.parametrize("damage", [None, "sources", "tables"])
def test_replay_command_checks_identity_and_retains_raw_differences(tmp_path, monkeypatch, damage):
    from pyrite.xsgen.bremslib import tables as bremslib

    metadata = dict(
        cone={},
        context=dict(composition=[["Si", 1.0]]),
        pyrite_python_source_sha256=source_fingerprint(),
        harness_sha256=file_sha256(HERE / "pyrite_bench.py"),
    )
    if damage == "sources":
        metadata["pyrite_python_source_sha256"] = "0" * 64
    root = tmp_path / "tracks"
    writer = TrackCapture(root, 3, metadata, 1_000_000)
    writer.append(tracks(), 0, 3)
    writer.finish()
    table = SimpleNamespace(
        atomic_number=14, key="key", digest="wrong" if damage == "tables" else "digest"
    )
    monkeypatch.setattr(bremslib, "load_bremsstrahlung_tables", lambda _: {"Si": table})

    def scorer(rows, n, cone, ctx):
        return dict(
            cone_source_photons_ge10kev=np.bincount(rows["elec_id"], minlength=n)
            * ctx["energy_order"]
        )

    monkeypatch.setattr(runpy, "run_path", lambda _: dict(continuous_cone_scores=scorer))
    output = tmp_path / "replay.json"
    monkeypatch.setattr(sys, "argv", ["replay_cone.py", str(root), "--output", str(output)])
    if damage:
        with pytest.raises(ValueError, match="source|tables differ"):
            REPLAY["main"]()
        assert not output.exists()
        assert not output.with_suffix(".npz").exists()
    else:
        REPLAY["main"]()
        result = json.loads(output.read_text())
        assert result["numerical_gate"] == "unmeasured" and not result["timing_comparison_eligible"]
        assert len(result["rules"]) == 10 and len(result["contrasts"]) == 9
        assert result["raw_scores_sha256"] == file_sha256(output.with_suffix(".npz"))
        zero = result["observables"]["difference-angular-L1-L2"]
        assert zero["precision_status"] == "unresolved"
        with np.load(output.with_suffix(".npz")) as arrays:
            np.testing.assert_array_equal(arrays["primary_ids"], [0, 1, 2])
