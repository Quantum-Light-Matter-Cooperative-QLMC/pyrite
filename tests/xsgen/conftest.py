"""Shared fixtures for the external-code generator tests.

No Fortran here. Everything below drives the subprocess, scratch-directory,
and store layers against a *fake* binary, because CI cannot compile or run the
real codes -- tests that do are marked ``extern_codes``.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

#: The scratch/subprocess layer runs an executable script with a shebang, which
#: Windows does not honour. Every other layer is platform-independent.
posix_only = pytest.mark.skipif(os.name != "posix", reason="needs POSIX exec semantics")


@pytest.fixture
def fake_binary(tmp_path: Path):
    """Return a factory making an executable Python script.

    The script stands in for ELSEPA's ``elscata`` or SBETHE's ``sbethe``: it
    runs in whatever directory it is given, reads standard input, and writes
    fixed-name files into its working directory -- which is exactly the
    behaviour that makes per-run isolation a correctness requirement.
    """

    def make(body: str, name: str = "fake") -> Path:
        path = tmp_path / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IRUSR)
        return path

    return make


@pytest.fixture
def isolated_dirs(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    """Point the user cache and user data directories at ``tmp_path``.

    ``pyrite.paths`` resolves both through ``platformdirs`` at call time, so
    patching the two lookups redirects :func:`pyrite.xsgen.store.user_table_dir`
    and :func:`pyrite.xsgen._run.scratch_root` together, without either module
    needing a test-only seam.
    """
    cache = tmp_path / "cache"
    data = tmp_path / "data"
    monkeypatch.setattr("pyrite.paths.user_cache_path", lambda *a, **k: cache)
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: data)
    return {"cache": cache, "data": data}


@pytest.fixture
def packaged_dir(monkeypatch, tmp_path: Path) -> Path:
    """Redirect the packaged table tier to a writable directory.

    The real packaged tier lives inside the installed wheel and holds nothing
    yet (tables ship with M4/M6 of #161), so two-tier resolution is otherwise
    untestable.
    """
    root = tmp_path / "packaged"
    monkeypatch.setattr("pyrite.xsgen.store.data_dir", lambda: root)
    return root / "xsgen" / "tables"
