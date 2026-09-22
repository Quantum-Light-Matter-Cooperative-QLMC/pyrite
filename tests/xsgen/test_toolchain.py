"""Compiler discovery and cached builds.

gfortran is an optional runtime dependency (D9). Its absence is a normal
condition on a fresh machine, and must produce an actionable message rather
than a traceback or -- the thing D9 forbids -- a quiet substitution.

The build cache is keyed on the source digest, so a patched tree is rebuilt
and an unchanged one is compiled once per machine.
"""

import stat
import sys
from pathlib import Path

import pytest

from pyrite.xsgen._errors import BuildError, ToolchainUnavailableError
from pyrite.xsgen.sources import code_spec, resolve_source
from pyrite.xsgen.toolchain import (
    Toolchain,
    build,
    build_command,
    build_root,
    find_toolchain,
)

pytestmark = pytest.mark.usefixtures("isolated_dirs")


@pytest.fixture
def elsepa_tree(tmp_path: Path) -> Path:
    root = tmp_path / "elsepa-2020"
    (root / "database").mkdir(parents=True)
    for name in code_spec("elsepa").digest_sources:
        (root / name).write_text(f"      PROGRAM {name}\n", encoding="utf-8")
    return root


@pytest.fixture
def fake_compiler(tmp_path: Path, monkeypatch):
    """A stand-in compiler that reports a version and writes its ``-o`` target."""

    def make(*, succeeds: bool = True) -> Path:
        path = tmp_path / "bin" / "fakefc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"#!{sys.executable}\n"
            "import sys, pathlib\n"
            "if '--version' in sys.argv:\n"
            "    print('Fake Fortran (test) 1.0'); raise SystemExit(0)\n"
            f"if not {succeeds!r}:\n"
            "    sys.stderr.write('fatal: unsupported construct\\n'); raise SystemExit(1)\n"
            "target = sys.argv[sys.argv.index('-o') + 1]\n"
            "pathlib.Path(target).write_text('built', encoding='utf-8')\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PYRITE_XSGEN_FC", str(path))
        return path

    return make


def test_an_absent_compiler_names_what_to_install(monkeypatch):
    monkeypatch.delenv("PYRITE_XSGEN_FC", raising=False)
    monkeypatch.setattr("pyrite.xsgen.toolchain.shutil.which", lambda name: None)

    with pytest.raises(ToolchainUnavailableError) as excinfo:
        find_toolchain()
    message = str(excinfo.value)
    assert "gfortran" in message
    assert "apt install gfortran" in message
    assert "cannot substitute a model" in message, "D9: no silent surrogate"


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX exec semantics")
def test_the_compiler_override_is_honoured_and_its_version_recorded(fake_compiler):
    compiler = fake_compiler()
    toolchain = find_toolchain()
    assert toolchain.compiler == compiler
    assert toolchain.version == "Fake Fortran (test) 1.0"


def test_the_build_command_is_reproducible_by_hand(elsepa_tree, tmp_path):
    source = resolve_source("elsepa", elsepa_tree)
    toolchain = Toolchain(Path("/usr/bin/gfortran"), "GNU Fortran 15.2.0")
    argv = build_command(source, code_spec("elsepa").programs["elscata"], toolchain, tmp_path / "e")

    assert argv[0] == "/usr/bin/gfortran"
    # Fixed-form F77-era sources: modern gfortran rejects them without these.
    assert "-std=legacy" in argv and "-fallow-argument-mismatch" in argv
    # ``elscata.f`` alone. It opens with INCLUDE lines pulling in the other two,
    # so naming all three compiles those bodies twice and the link fails on
    # dozens of duplicate symbols -- which is what the real tree does.
    assert argv[-1] == str(elsepa_tree / "elscata.f")
    assert str(elsepa_tree / "radial.f") not in argv
    # ...and the include path that lets those INCLUDE lines resolve, since the
    # build runs from the cache directory rather than from the tree.
    assert f"-I{elsepa_tree}" in argv


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX exec semantics")
def test_a_built_binary_is_cached_per_source_digest(elsepa_tree, fake_compiler):
    fake_compiler()
    source = resolve_source("elsepa", elsepa_tree)

    first = build(source, "elscata")
    assert first.is_file()
    assert build_root() in first.parents

    first.write_text("marker", encoding="utf-8")
    assert build(source, "elscata").read_text(encoding="utf-8") == "marker", "cache reused"

    # Patching a source is a different code, so it must not serve the old binary.
    (elsepa_tree / "radial.f").write_text("      PROGRAM patched\n", encoding="utf-8")
    rebuilt = build(resolve_source("elsepa", elsepa_tree), "elscata")
    assert rebuilt != first
    assert rebuilt.read_text(encoding="utf-8") == "built"


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX exec semantics")
def test_force_rebuilds_an_unchanged_tree(elsepa_tree, fake_compiler):
    fake_compiler()
    source = resolve_source("elsepa", elsepa_tree)
    binary = build(source, "elscata")
    binary.write_text("stale", encoding="utf-8")

    assert build(source, "elscata", force=True).read_text(encoding="utf-8") == "built"


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX exec semantics")
def test_a_failed_build_reports_the_command_and_leaves_no_binary(elsepa_tree, fake_compiler):
    fake_compiler(succeeds=False)
    source = resolve_source("elsepa", elsepa_tree)

    with pytest.raises(BuildError) as excinfo:
        build(source, "elscata")
    message = str(excinfo.value)
    assert "unsupported construct" in message
    assert "-std=legacy" in message, "the exact command must be reproducible"

    # A half-built artefact at the cache path would be served as a success.
    digest_dirs = list(build_root().glob("elsepa-*"))
    assert all(not (path / "elscata").exists() for path in digest_dirs)


def test_a_code_with_no_programs_says_so(tmp_path):
    root = tmp_path / "BremsLib_v2.0.8"
    library = root / "BremsLib_v2.0.8" / "SDCS"
    library.mkdir(parents=True)
    (library / "SDCS_1.txt").write_text("", encoding="utf-8")
    source = resolve_source("bremslib", root)

    with pytest.raises(KeyError, match="read, not run"):
        build(source, "brems")
