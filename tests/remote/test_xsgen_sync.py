"""Remote sync of xsgen tables: inventory, skip-if-present, tamper, preflight."""

import hashlib
import tarfile

import pytest

from pyrite.remote import _queue_scripts, config, transport

KEY_A = "a" * 64
KEY_B = "b" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tables(tmp_path, keys=(KEY_A, KEY_B)):
    root = tmp_path / "tables"
    root.mkdir()
    found = {}
    for key in keys:
        (root / f"{key}.npz").write_bytes(b"payload-" + key.encode())
        (root / f"{key}.json").write_bytes(b'{"key": "' + key.encode() + b'"}')
        found[f"{key}.npz"] = root / f"{key}.npz"
        found[f"{key}.json"] = root / f"{key}.json"
    return found


def _digests(found):
    return {name: _sha(path.read_bytes()) for name, path in found.items()}


def test_inventory_parsing_keeps_only_table_names_and_fails_closed():
    out = f"{'c' * 64}  ./{KEY_A}.npz\n{'d' * 64}  ./notes.txt\n\n{'e' * 64} *./{KEY_A}.json\n"
    assert transport._parse_xsgen_inventory(out) == {
        f"{KEY_A}.npz": "c" * 64,
        f"{KEY_A}.json": "e" * 64,
    }
    with pytest.raises(SystemExit, match="invalid remote xsgen table inventory"):
        transport._parse_xsgen_inventory("garbage\n")


def test_one_round_trip_carries_both_inventories(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_DIR", "/remote")
    digest = "9" * 64
    artifact = f"{digest}  /remote/src/pyrite/data/catalog/energy-grid-artifacts/99/{digest}.json\n"
    commands = []

    def capture(command):
        commands.append(command)
        return artifact + transport._XSGEN_SECTION_MARK + f"\n{'c' * 64}  ./{KEY_A}.npz\n"

    monkeypatch.setattr(transport, "_ssh_capture", capture)

    artifacts, tables = transport._remote_inventories(artifacts=True, tables=True)

    assert len(commands) == 1
    assert "energy-grid-artifacts" in commands[0] and "/remote/xsgen/tables" in commands[0]
    assert ".local/share}/pyrite/xsgen/tables" in commands[0]  # legacy migration
    assert artifacts == frozenset({digest})
    assert tables == {f"{KEY_A}.npz": "c" * 64}


def test_missing_section_marker_fails_closed(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: "")
    with pytest.raises(SystemExit, match="invalid remote xsgen table inventory"):
        transport._remote_inventories(artifacts=False, tables=True)


def test_ships_only_missing_or_differing_pairs_payload_first(tmp_path):
    local = _tables(tmp_path)
    digests = _digests(local)
    remote = {  # A is intact, B's payload is truncated and its manifest absent
        f"{KEY_A}.npz": digests[f"{KEY_A}.npz"],
        f"{KEY_A}.json": digests[f"{KEY_A}.json"],
        f"{KEY_B}.npz": "0" * 64,
    }

    shipped = transport._xsgen_to_ship(local, digests, remote)

    assert [arc for arc, _ in shipped] == [
        f"xsgen/tables/{KEY_B}.npz",
        f"xsgen/tables/{KEY_B}.json",
    ]


def test_second_sync_transfers_no_table_bytes(tmp_path):
    local = _tables(tmp_path)
    digests = _digests(local)
    assert transport._xsgen_to_ship(local, digests, dict(digests)) == []


def test_tampered_remote_file_is_replaced(tmp_path):
    local = _tables(tmp_path)
    digests = _digests(local)
    remote = dict(digests)
    remote[f"{KEY_A}.json"] = _sha(b"edited on the box")

    shipped = transport._xsgen_to_ship(local, digests, remote)

    assert [arc for arc, _ in shipped] == [
        f"xsgen/tables/{KEY_A}.npz",
        f"xsgen/tables/{KEY_A}.json",
    ]


def test_tables_digest_is_order_free_and_content_sensitive(tmp_path):
    digests = _digests(_tables(tmp_path))
    reordered = dict(reversed(list(digests.items())))
    assert transport._xsgen_tables_digest(digests) == transport._xsgen_tables_digest(reordered)
    changed = {**digests, f"{KEY_A}.npz": "0" * 64}
    assert transport._xsgen_tables_digest(changed) != transport._xsgen_tables_digest(digests)


def test_stamp_renders_tables_digest_only_when_present():
    base = transport.SyncStamp("d" * 64, "rev", False, "t", "s")
    assert "code_tables_digest" not in base.render()
    with_tables = transport.SyncStamp("d" * 64, "rev", False, "t", "s", "f" * 64)
    assert with_tables.render().endswith(f"code_tables_digest: {'f' * 64}\n")


def _sync(tmp_path, monkeypatch, local, remote):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "m.py").write_text("X = 1\n")
    monkeypatch.setattr(config, "LOCAL_ROOT", root)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(config, "REMOTE_DIR", "/remote")
    monkeypatch.setattr(transport, "_local_xsgen_tables", lambda: local)
    captured = []

    def capture(command):
        if "squeue" in command:
            return ""
        captured.append(command)
        body = "".join(f"{d}  ./{n}\n" for n, d in remote.items())
        return transport._XSGEN_SECTION_MARK + "\n" + body

    monkeypatch.setattr(transport, "_ssh_capture", capture)
    archived, commands = [], []

    def fake_run(argv, **kwargs):
        commands.append(argv)
        if argv[0] == "scp":
            with tarfile.open(argv[1], "r:gz") as bundle:
                archived.extend(bundle.getnames())

    monkeypatch.setattr(transport, "_run", fake_run)
    transport.sync_code()
    return archived, commands, captured


def test_sync_ships_missing_tables_once_and_stamps_the_inventory(tmp_path, monkeypatch):
    local = _tables(tmp_path)
    digests = _digests(local)

    archived, commands, captured = _sync(tmp_path, monkeypatch, local, {})
    assert sorted(n for n in archived if n.startswith("xsgen/")) == sorted(
        f"xsgen/tables/{name}" for name in local
    )
    assert len(captured) == 1
    stamp = f"code_tables_digest: {transport._xsgen_tables_digest(digests)}"
    assert stamp in commands[-1][3]

    archived, _, captured = _sync(tmp_path, monkeypatch, local, digests)
    assert not [n for n in archived if n.startswith("xsgen/")]
    assert len(captured) == 1


def test_sync_refuses_when_a_pinned_table_is_missing_locally(monkeypatch):
    from pyrite.xsgen import verify

    monkeypatch.setattr(
        verify,
        "pinned_tables",
        lambda codes=verify.CODES: [("elsepa", "Z=29", "f" * 64, "0" * 64)],
    )
    monkeypatch.setattr("pyrite.xsgen.verify.resolve", lambda key: None)

    with pytest.raises(SystemExit, match=r"pyrite tables fetch elsepa --archive PATH"):
        transport._real_local_xsgen_tables()


def test_local_tables_skip_packaged_tier_and_unpaired_files(tmp_path, monkeypatch):
    from pyrite.xsgen import store, verify

    user = tmp_path / "user"
    user.mkdir()
    (user / f"{KEY_A}.npz").write_bytes(b"x")
    (user / f"{KEY_A}.json").write_bytes(b"{}")
    (user / f"{KEY_B}.json").write_bytes(b"{}")  # no payload
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / f"{'c' * 64}.npz").write_bytes(b"x")
    (packaged / f"{'c' * 64}.json").write_bytes(b"{}")
    monkeypatch.setattr(verify, "pinned_tables", lambda codes=verify.CODES: [])
    monkeypatch.setattr(store, "search_dirs", lambda: (user, packaged))
    monkeypatch.setattr(store, "packaged_table_dir", lambda: packaged)

    found = transport._real_local_xsgen_tables()

    assert sorted(found) == [f"{KEY_A}.json", f"{KEY_A}.npz"]


def test_job_script_preflights_tables_and_fails_fast_before_the_sweep():
    block = _queue_scripts._tables_preflight_block()
    assert "pyrite tables verify" in block
    assert 'echo "FAILED (tables)' in block
    assert "exit 1" in block
    assert "PYRITE_HOME=" in block


@pytest.mark.parametrize("chunked", [False, True])
def test_generated_script_stops_in_preflight_before_any_transport(monkeypatch, tmp_path, chunked):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    jobdir = tmp_path / "jobs" / "j"
    jobdir.mkdir(parents=True)
    ran = tmp_path / "ran"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = sync ]; then exit 0; fi\n'
        'case "$*" in *"tables verify"*) echo "missing: elsepa Z=6"; exit 1 ;; esac\n'
        f'echo transport >> "{ran.as_posix()}"\n'
    )
    fake_uv.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "REMOTE_UV", fake_uv.as_posix())
    from pyrite import remote

    script = (
        remote._chunked_queue_script("j", ["hopg"], False, None, 10)
        if chunked
        else remote._queue_script("j", ["hopg"], quick=False, workers=None)
    )

    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 1
    assert (jobdir / "state").read_text().startswith("FAILED (tables)")
    assert "missing: elsepa Z=6" in (jobdir / "log").read_text()
    assert not ran.exists()
    assert not (jobdir / ".tables_ok").exists()
