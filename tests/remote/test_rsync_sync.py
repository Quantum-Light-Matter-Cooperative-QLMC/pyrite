"""rsync code-sync transport (#287): argv shape, fallback, stamp ordering.

The end-to-end tests run real ``rsync``/``cp``/``sh`` against a local directory
standing in for the box: host prefixes are stripped, ssh commands run under
``sh -c``. ``subprocess.check_call`` bypasses the conftest network guard, which
only wraps ``subprocess.run``.
"""

import shlex
import shutil
import subprocess
import tarfile

import pytest

from pyrite import _energy_grid_artifacts as artifacts
from pyrite.remote import config, transport

# Exact payload/argv pins assume no user-layer profiles ride along.
pytestmark = pytest.mark.usefixtures("empty_user_catalog")

HOST = "box"


def _tree(root):
    """A small checkout: a package, a data file, a top-level file."""
    pkg = root / "src" / "pyrite"
    pkg.mkdir(parents=True)
    (pkg / "keep.py").write_text("KEEP = 1\n")
    (pkg / "gone.py").write_text("GONE = 1\n")
    (pkg / "data.bin").write_bytes(b"\x00\x01")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "keep.cpython-314.pyc").write_bytes(b"bytecode")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (root / "cfg").mkdir()
    (root / "cfg" / "nested.toml").write_text("n = 1\n")
    return pkg


def _configure(monkeypatch, local, remote_dir, mode="auto"):
    monkeypatch.setattr(config, "LOCAL_ROOT", local)
    monkeypatch.setattr(
        config, "SYNC_PATHS", ["src", "pyproject.toml", "cfg/nested.toml", "missing-dir"]
    )
    monkeypatch.setattr(config, "HOST", HOST)
    monkeypatch.setattr(config, "REMOTE_DIR", str(remote_dir))
    monkeypatch.setenv("PYRITE_SSH_MUX", "0")
    monkeypatch.setenv("PYRITE_SYNC_TRANSPORT", mode)


def _strip_host(arg):
    return arg.removeprefix(f"{HOST}:")


def _fake_box(monkeypatch, *, remote_rsync=True, fail_rsync=False):
    """Route transport calls to the local filesystem; return the call log."""
    calls = []

    def capture(command):
        if "squeue" in command:
            return ""
        if not remote_rsync:
            command = command.replace("command -v rsync", "false")
        return subprocess.check_output(["sh", "-c", command], text=True)

    def run(argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "rsync":
            if fail_rsync:
                raise subprocess.CalledProcessError(23, argv)
            dash_e = argv.index("-e")
            local = [_strip_host(a) for a in argv[:dash_e] + argv[dash_e + 2 :]]
            subprocess.check_call(local)
        elif argv[0] == "scp":
            subprocess.check_call(["cp", argv[-2], shlex.split(_strip_host(argv[-1]))[0]])
        elif argv[0] == "ssh":
            subprocess.check_call(["sh", "-c", argv[-1]])

    monkeypatch.setattr(transport, "_ssh_capture", capture)
    monkeypatch.setattr(transport, "_run", run)
    return calls


def _stamp_digest(remote_dir):
    text = (remote_dir / config.SYNC_STAMP_NAME).read_text()
    return next(line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("code_"))


needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")


@needs_rsync
def test_rsync_mirrors_deletions_and_spares_box_state(tmp_path, monkeypatch):
    local, box = tmp_path / "local", tmp_path / "box"
    pkg = _tree(local)
    box.mkdir()
    for keep in ("checkpoints/run/out.npz", "jobs/j1/meta", "xsgen/tables/t.npz", ".venv/x"):
        (box / keep).parent.mkdir(parents=True, exist_ok=True)
        (box / keep).write_text("box state")
    _configure(monkeypatch, local, box)
    calls = _fake_box(monkeypatch)

    transport.sync_code()
    assert [c[0] for c in calls] == ["rsync", "rsync", "ssh"]
    assert (box / "src/pyrite/gone.py").is_file()
    assert (box / "cfg/nested.toml").read_text() == "n = 1\n"  # -R keeps the subpath
    assert not (box / "src/pyrite/__pycache__").exists()  # caches never ship
    kept_inode = (box / "src/pyrite/data.bin").stat().st_ino

    (pkg / "gone.py").unlink()
    (pkg / "keep.py").write_text("KEEP = 2\n")
    transport.sync_code()

    assert not (box / "src/pyrite/gone.py").exists()
    assert (box / "src/pyrite/keep.py").read_text() == "KEEP = 2\n"
    assert (box / "src/pyrite/data.bin").stat().st_ino == kept_inode  # untouched
    for keep in ("checkpoints/run/out.npz", "jobs/j1/meta", "xsgen/tables/t.npz", ".venv/x"):
        assert (box / keep).read_text() == "box state"


@needs_rsync
def test_rsync_and_tar_stamp_the_same_digest(tmp_path, monkeypatch):
    local = tmp_path / "local"
    _tree(local)
    digests = {}
    for mode in ("rsync", "tar"):
        box = tmp_path / mode
        box.mkdir()
        _configure(monkeypatch, local, box, mode)
        calls = _fake_box(monkeypatch)
        transport.sync_code()
        digests[mode] = _stamp_digest(box)
        assert calls[0][0] == ("rsync" if mode == "rsync" else "scp")
    assert (
        digests["rsync"] == digests["tar"] == transport._payload_digest(transport._sync_entries())
    )
    assert sorted(p.relative_to(tmp_path / "rsync") for p in (tmp_path / "rsync").rglob("*")) == (
        sorted(p.relative_to(tmp_path / "tar") for p in (tmp_path / "tar").rglob("*"))
    )


@needs_rsync
def test_rsync_keeps_box_cache_in_live_dirs_but_drops_stale_cache_husks(tmp_path, monkeypatch):
    local, box = tmp_path / "local", tmp_path / "box"
    _tree(local)
    for cache in ("src/pyrite/__pycache__/keep.pyc", "src/pyrite/old_pkg/__pycache__/m.pyc"):
        (box / cache).parent.mkdir(parents=True)
        (box / cache).write_bytes(b"box bytecode")
    _configure(monkeypatch, local, box)
    _fake_box(monkeypatch)

    transport.sync_code()

    assert (box / "src/pyrite/__pycache__/keep.pyc").is_file()
    assert not (box / "src/pyrite/old_pkg").exists()


@needs_rsync
def test_rsync_protects_delta_managed_and_legacy_dataset_paths(tmp_path, monkeypatch):
    from pyrite.datasets import DATASETS

    local, box = tmp_path / "local", tmp_path / "box"
    _tree(local)
    remote_artifact = box / "src/pyrite/data/catalog/energy-grid-artifacts/aa/old.json"
    legacy = box / transport._LEGACY_DATASET_DIR / next(iter(DATASETS.values())).filename
    for path in (remote_artifact, legacy):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("box only")
    _configure(monkeypatch, local, box)
    _fake_box(monkeypatch)

    transport.sync_code()

    assert remote_artifact.read_text() == "box only"
    assert legacy.read_text() == "box only"


def test_rsync_ships_only_missing_artifacts_through_the_delta_tar(tmp_path, monkeypatch):
    local, box = tmp_path / "local", tmp_path / "box"
    _tree(local)
    store = local / "src/pyrite/data/catalog/energy-grid-artifacts"
    identity = artifacts.artifact_identity(
        "hbn",
        [{"energy_keV": 30, "start_eV": 10, "stop_eV": 50, "num": 4}],
        {"stop_eV": 100, "step_eV": 10},
        [30],
    )
    missing = artifacts.write_artifact(store, identity)
    _configure(monkeypatch, local, box)
    monkeypatch.setattr(transport, "_local_rsync_blocker", lambda _entries: None)
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda c: "" if "squeue" in c else f"{transport._RSYNC_PROBE_MARK}\nrsync\n",
    )
    calls, archived = [], []

    def run(argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "scp":
            with tarfile.open(argv[-2], "r:gz") as bundle:
                archived.extend(bundle.getnames())

    monkeypatch.setattr(transport, "_run", run)

    transport.sync_code()

    assert [c[0] for c in calls] == ["rsync", "rsync", "scp", "ssh"]
    assert archived == [
        f"src/pyrite/data/catalog/energy-grid-artifacts/{missing.digest[:2]}/{missing.digest}.json"
    ]
    src_call = next(c for c in calls if c[0] == "rsync" and c[-1].endswith("/src/"))
    assert "--exclude=/pyrite/data/catalog/energy-grid-artifacts/" in src_call
    finish = calls[-1][-1]
    assert finish.index("tar xzf") < finish.index("cp -a") < finish.index("printf %s")


def _rsync_only(tmp_path, monkeypatch, **box):
    local = tmp_path / "local"
    _tree(local)
    _configure(monkeypatch, local, tmp_path / "box")
    monkeypatch.setattr(transport, "_local_rsync_blocker", lambda _entries: None)
    probe = f"{transport._RSYNC_PROBE_MARK}\nrsync\n" if box.get("remote_rsync", True) else ""
    monkeypatch.setattr(transport, "_ssh_capture", lambda c: "" if "squeue" in c else probe)
    calls = []
    monkeypatch.setattr(transport, "_run", lambda argv, **_kw: calls.append(argv))
    return calls


def test_rsync_argv_shape_per_sync_path(tmp_path, monkeypatch):
    calls = _rsync_only(tmp_path, monkeypatch)

    transport.sync_code()

    rsyncs = [c for c in calls if c[0] == "rsync"]
    remote = tmp_path / "box"
    assert [c[-1] for c in rsyncs] == [f"{HOST}:{remote}/src/", f"{HOST}:{remote}/"]
    src, files = rsyncs
    assert src[:4] == ["rsync", "-s", "-e", "ssh"]
    for flag in ("-a", "--checksum", "--delay-updates", "--delete", "--filter=-p __pycache__/"):
        assert flag in src
    assert src[-2] == f"{tmp_path / 'local' / 'src'}/"
    assert "--delete" not in files and "-R" in files
    assert files[-3:-1] == [
        f"{tmp_path / 'local'}/./pyproject.toml",
        f"{tmp_path / 'local'}/./cfg/nested.toml",
    ]
    assert calls[-1][0] == "ssh" and "tar xzf" not in calls[-1][-1]


def test_rsync_e_reuses_the_control_master(tmp_path, monkeypatch):
    calls = _rsync_only(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "ssh_mux_options", lambda: ["-o", "ControlPath=/s p/%C"])

    transport.sync_code()

    assert calls[0][calls[0].index("-e") + 1] == "ssh -o 'ControlPath=/s p/%C'"


def test_delete_never_targets_box_state(tmp_path, monkeypatch):
    calls = _rsync_only(tmp_path, monkeypatch)

    transport.sync_code()

    remote = tmp_path / "box"
    deleting = [c[-1] for c in calls if c[0] == "rsync" and "--delete" in c]
    allowed = {f"{HOST}:{remote}/{p}/" for p in ("src", "checks", "vendor", "external-catalog")}
    assert deleting and set(deleting) <= allowed
    for state in ("", "checkpoints", "jobs", ".venv", config.SYNC_STAMP_NAME, "xsgen/tables"):
        assert f"{HOST}:{remote}/{state}".rstrip("/") + "/" not in deleting


def test_rsync_failure_leaves_the_old_stamp(tmp_path, monkeypatch):
    calls = _rsync_only(tmp_path, monkeypatch)

    def run(argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "rsync" and "-R" in argv:  # the plain-file call, after src/
            raise subprocess.CalledProcessError(23, argv)

    monkeypatch.setattr(transport, "_run", run)

    with pytest.raises(subprocess.CalledProcessError):
        transport.sync_code()

    assert [c[0] for c in calls] == ["rsync", "rsync"]  # no stamp-writing ssh


def test_external_catalog_dir_is_mirrored_through_an_include_filter(tmp_path, monkeypatch):
    calls = _rsync_only(tmp_path, monkeypatch)
    catalog = tmp_path / "cat"
    rules = []
    monkeypatch.setattr(config, "external_catalog_selected", lambda: True)
    monkeypatch.setattr(
        transport,
        "_sync_entries",
        lambda: [
            ("external-catalog/catalog.toml", catalog / "catalog.toml"),
            ("external-catalog/cifs/a.cif", catalog / "cifs/a.cif"),
        ],
    )
    monkeypatch.setattr(transport, "_payload_digest", lambda _entries: "d" * 64)
    import pyrite._catalog_layout as layout

    monkeypatch.setattr(layout, "selected_catalog", lambda: catalog)
    catalog.mkdir()

    def run(argv, **_kwargs):
        calls.append(argv)
        merge = [a for a in argv if a.startswith("--filter=merge ")]
        if merge:
            rules.append(open(merge[0].split(" ", 1)[1]).read())

    monkeypatch.setattr(transport, "_run", run)

    transport.sync_code()

    cat_call = next(c for c in calls if c[-1].endswith("/external-catalog/"))
    assert "--delete-excluded" in cat_call and cat_call[-2] == f"{catalog}/"
    assert rules == ["+ /catalog.toml\n+ /cifs/\n+ /cifs/a.cif\n- *\n"]
    assert "rm -rf external-catalog.toml user-catalog && printf %s" in calls[-1][-1]


@pytest.mark.parametrize(
    ("mode", "blocker", "remote_rsync", "expected"),
    [
        ("auto", None, True, "rsync"),
        ("auto", None, False, "scp"),
        ("auto", "rsync not found locally", True, "scp"),
        ("auto", "CRLF line endings in src/a.py", True, "scp"),
        ("tar", None, True, "scp"),
        ("rsync", None, True, "rsync"),
    ],
)
def test_transport_selection_matrix(tmp_path, monkeypatch, mode, blocker, remote_rsync, expected):
    calls = _rsync_only(tmp_path, monkeypatch, remote_rsync=remote_rsync)
    monkeypatch.setenv("PYRITE_SYNC_TRANSPORT", mode)
    monkeypatch.setattr(transport, "_local_rsync_blocker", lambda _entries: blocker)
    probes = []
    real_capture = transport._ssh_capture
    monkeypatch.setattr(transport, "_ssh_capture", lambda c: probes.append(c) or real_capture(c))

    transport.sync_code()

    assert calls[0][0] == expected
    probed = any(transport._RSYNC_PROBE_MARK in c for c in probes)
    assert probed == (mode != "tar" and blocker is None)


@pytest.mark.parametrize(
    ("blocker", "remote_rsync"), [("rsync not found locally", True), (None, False)]
)
def test_forced_rsync_refuses_rather_than_falling_back(
    tmp_path, monkeypatch, blocker, remote_rsync
):
    calls = _rsync_only(tmp_path, monkeypatch, remote_rsync=remote_rsync)
    monkeypatch.setenv("PYRITE_SYNC_TRANSPORT", "rsync")
    monkeypatch.setattr(transport, "_local_rsync_blocker", lambda _entries: blocker)

    with pytest.raises(SystemExit, match="PYRITE_SYNC_TRANSPORT=rsync"):
        transport.sync_code()
    assert calls == []


def test_invalid_transport_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("PYRITE_SYNC_TRANSPORT", "ftp")
    with pytest.raises(SystemExit, match="PYRITE_SYNC_TRANSPORT"):
        transport._sync_transport_mode()


def test_local_blockers(tmp_path, monkeypatch):
    lf, crlf = tmp_path / "a.py", tmp_path / "b.py"
    lf.write_bytes(b"A = 1\n")
    crlf.write_bytes(b"B = 1\r\n")
    monkeypatch.setattr(transport.shutil, "which", lambda _name: "/usr/bin/rsync")
    assert transport._local_rsync_blocker([("src/a.py", lf)]) is None
    assert "CRLF" in transport._local_rsync_blocker([("src/b.py", crlf)])
    assert "rsync-safe" in transport._local_rsync_blocker([("external-catalog/x[1].cif", lf)])
    monkeypatch.setattr(transport.shutil, "which", lambda _name: None)
    assert transport._local_rsync_blocker([]) == "rsync not found locally"


def test_verbose_reports_the_chosen_transport(tmp_path, monkeypatch, capsys):
    _rsync_only(tmp_path, monkeypatch)
    with transport.verbose_ssh(True):
        transport.sync_code()
    assert "sync transport: rsync" in capsys.readouterr().err

    monkeypatch.setenv("PYRITE_SYNC_TRANSPORT", "tar")
    with transport.verbose_ssh(True):
        transport.sync_code()
    assert "sync transport: tar (PYRITE_SYNC_TRANSPORT=tar)" in capsys.readouterr().err
