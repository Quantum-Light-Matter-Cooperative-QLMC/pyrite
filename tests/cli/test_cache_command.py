"""``pyrite cache`` status/clear streams, preview, and scope."""

import json

import pytest

from pyrite import _jit_cache
from pyrite.cli.commands import cache as cache_command
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def managed(monkeypatch, tmp_path):
    root = tmp_path / "user-cache"
    monkeypatch.setattr(_jit_cache, "cache_dir", lambda: root)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(root / "numba"))
    monkeypatch.setenv("CUPY_CACHE_DIR", str(root / "cupy"))
    (root / "cupy").mkdir(parents=True)
    (root / "cupy" / "abc.cubin").write_bytes(b"0123")
    return root


def test_status_json_reports_locations_and_sizes(managed):
    result = invoke(cache_command.command, ["status", "--backend", "cupy", "-o", "json"])
    assert_clean_result(result)
    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.cache.status.v1"
    (location,) = envelope["payload"]["locations"]
    assert location | {"path": None} == {
        "backend": "cupy",
        "path": None,
        "source": "managed",
        "active": True,
        "owned": True,
        "exists": True,
        "files": 1,
        "bytes": 4,
    }


def test_status_table_lists_every_location(managed):
    result = invoke(cache_command.command, ["status"])
    assert_clean_result(result)
    lines = result.stdout.splitlines()
    assert lines[0].split()[:3] == ["BACKEND", "SOURCE", "STATE"]
    assert [line.split()[:3] for line in lines[1:4]] == [
        ["numba", "managed", "active"],
        ["numba", "in-tree", "inactive"],
        ["cupy", "managed", "active"],
    ]


def test_clear_previews_without_tty_then_deletes_with_yes(managed):
    result = invoke(cache_command.command, ["clear", "--backend", "cupy"])
    assert_clean_result(result)
    assert f"delete cupy tree {managed / 'cupy'}" in result.stdout
    assert "preview only" in result.stdout
    assert (managed / "cupy").exists()

    result = invoke(cache_command.command, ["clear", "--backend", "cupy", "--yes"])
    assert_clean_result(result)
    assert "removed 1 cache path(s)" in result.stdout
    assert not (managed / "cupy").exists()

    result = invoke(cache_command.command, ["clear", "--backend", "cupy", "--yes"])
    assert_clean_result(result, stdout="no PyRITE-owned compiled kernels to clear\n")


def test_clear_skips_user_cupy_cache_on_stderr(managed, monkeypatch, tmp_path):
    user = tmp_path / "user-cupy"
    user.mkdir()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(user))
    result = invoke(cache_command.command, ["clear", "--backend", "cupy", "--yes"])
    assert result.exit_code == 0
    assert f"skip cupy {user}: not PyRITE-owned" in result.stderr
    assert result.stdout == "no PyRITE-owned compiled kernels to clear\n"
    assert user.exists()
