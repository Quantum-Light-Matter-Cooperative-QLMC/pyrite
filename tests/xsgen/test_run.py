"""Scratch-directory isolation for external-code runs.

The behaviour frozen here is the one the design calls a correctness
requirement rather than a convenience: every run gets its own working
directory. All three codes write fixed output names into the process working
directory and read their databases by relative path, so runs sharing a
directory overwrite each other. SBETHE is the sharp case -- its ``<mname>.mat``
cache is read back in preference to the prompts whenever it exists, so a
sibling's stale ``.mat`` silently substitutes the wrong material and the run
still exits 0.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pyrite.xsgen._errors import RunError
from pyrite.xsgen._run import run_program, scratch_dir, scratch_root

from .conftest import posix_only

pytestmark = pytest.mark.usefixtures("isolated_dirs")

#: Writes one fixed-name output whose content is whatever arrived on stdin --
#: the shape every one of the three codes has.
_ECHO_TO_FIXED_NAME = """
import sys, pathlib
pathlib.Path("out.dat").write_text(sys.stdin.read(), encoding="utf-8")
"""


def test_scratch_dir_provides_a_fresh_empty_directory(isolated_dirs):
    with scratch_dir() as first, scratch_dir() as second:
        assert first != second
        assert first.is_dir() and second.is_dir()
        assert list(first.iterdir()) == []
        assert first.parent == scratch_root() == isolated_dirs["cache"] / "xsgen" / "scratch"


def test_scratch_dir_links_data_directories_under_their_open_name(tmp_path):
    database = tmp_path / "tree" / "database"
    database.mkdir(parents=True)
    (database / "z_029.dat").write_text("copper", encoding="utf-8")

    with scratch_dir({"database": database}) as workdir:
        # The program opens './database/...' relative to its cwd, so what
        # matters is that the *relative* path resolves to the tree's bytes.
        assert (workdir / "database" / "z_029.dat").read_text(encoding="utf-8") == "copper"
        assert (workdir / "database").is_symlink()


def test_scratch_dir_removes_itself_on_success():
    with scratch_dir() as workdir:
        (workdir / "scratch.txt").write_text("x", encoding="utf-8")
    assert not workdir.exists()


def test_scratch_dir_removes_itself_on_failure_by_default():
    with pytest.raises(RuntimeError):
        with scratch_dir() as workdir:
            held = workdir
            raise RuntimeError("the program blew up")
    assert not held.exists()


def test_scratch_dir_keeps_the_directory_only_when_the_body_fails():
    with scratch_dir(keep_on_failure=True) as succeeded:
        pass
    assert not succeeded.exists(), "a successful run must still clean up"

    with pytest.raises(RuntimeError):
        with scratch_dir(keep_on_failure=True) as failed:
            held = failed
            raise RuntimeError("the program blew up")
    assert held.is_dir(), "a failed run is kept for inspection"


def test_missing_data_directory_fails_before_the_program_starts(tmp_path):
    absent = tmp_path / "tree" / "sdbase"
    with pytest.raises(RunError) as excinfo:
        with scratch_dir({"sdbase": absent}):
            pytest.fail("the context body must not run")
    message = str(excinfo.value)
    assert "sdbase" in message
    assert str(absent) in message
    assert "pyrite tables fetch" in message


@posix_only
def test_run_program_collects_named_outputs(fake_binary):
    binary = fake_binary(_ECHO_TO_FIXED_NAME)
    result = run_program(binary, stdin_text="IZ 29\n", outputs=("out.dat",))

    assert result.returncode == 0
    assert result.text("out.dat") == "IZ 29\n"
    assert not result.workdir.exists(), "the scratch directory is gone on success"


@posix_only
def test_run_program_exposes_data_directories_to_the_program(fake_binary, tmp_path):
    database = tmp_path / "tree" / "database"
    database.mkdir(parents=True)
    (database / "pinned.txt").write_text("from the tree", encoding="utf-8")
    binary = fake_binary(
        "import pathlib\n"
        'pathlib.Path("out.dat").write_bytes('
        'pathlib.Path("database/pinned.txt").read_bytes())\n'
    )

    result = run_program(binary, data_dirs={"database": database}, outputs=("out.dat",))

    assert result.text("out.dat") == "from the tree"
    assert result.link_modes == {"database": "symlink"}


@posix_only
def test_run_program_writes_requested_input_files(fake_binary):
    binary = fake_binary(
        "import pathlib\n"
        'pathlib.Path("out.dat").write_bytes(pathlib.Path("deck.in").read_bytes())\n'
    )
    result = run_program(binary, input_files={"deck.in": "MUFFIN 1\n"}, outputs=("out.dat",))
    assert result.text("out.dat") == "MUFFIN 1\n"


@posix_only
def test_nonzero_exit_raises_carrying_the_program_diagnostics(fake_binary):
    binary = fake_binary('import sys\nsys.stderr.write("DHFS did not converge\\n")\nsys.exit(3)\n')
    with pytest.raises(RunError) as excinfo:
        run_program(binary)
    assert "exited 3" in str(excinfo.value)
    assert "DHFS did not converge" in str(excinfo.value)


@posix_only
def test_zero_exit_without_the_expected_output_is_a_failure(fake_binary):
    """A code that exits 0 having written nothing has failed.

    ELSEPA does this when the deck is accepted but the partial-wave series
    does not converge. Returning an empty result would send the caller on to
    store an empty table.
    """
    binary = fake_binary('print("run completed")\n')
    with pytest.raises(RunError) as excinfo:
        run_program(binary, outputs=("dcs.dat",))
    assert "wrote no dcs.dat" in str(excinfo.value)


@posix_only
def test_a_missing_binary_is_reported_before_any_scratch_setup(tmp_path):
    with pytest.raises(RunError, match="does not exist"):
        run_program(tmp_path / "never-built")


@posix_only
def test_concurrent_runs_writing_one_fixed_name_do_not_collide(fake_binary):
    """The SBETHE hazard, frozen.

    Two runs of the same program, at the same time, each writing the same
    fixed output name. Without per-run isolation one overwrites the other and
    at least one caller receives the wrong material's numbers while both exit
    0 -- a silent wrong answer, not a crash.
    """
    binary = fake_binary(
        "import sys, pathlib, time\n"
        "payload = sys.stdin.read()\n"
        # Write, pause, then read back: the pause widens the window in which a
        # sibling sharing the directory would clobber this run's file.
        'pathlib.Path("mos2.mat").write_text(payload, encoding="utf-8")\n'
        "time.sleep(0.2)\n"
        'pathlib.Path("stp.dat").write_text('
        'pathlib.Path("mos2.mat").read_text(encoding="utf-8"), encoding="utf-8")\n'
    )

    def run(material: str):
        return run_program(binary, stdin_text=material, outputs=("stp.dat",))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["MoS2", "silicon"]))

    assert [result.text("stp.dat") for result in results] == ["MoS2", "silicon"]
    assert results[0].workdir != results[1].workdir


@posix_only
def test_a_failed_run_names_the_directory_it_kept(fake_binary):
    binary = fake_binary("import sys\nsys.exit(1)\n")
    with pytest.raises(RunError) as excinfo:
        run_program(binary, keep_on_failure=True)
    message = str(excinfo.value)
    assert "scratch directory kept at" in message
    kept = Path(message.split("scratch directory kept at ")[1].splitlines()[0])
    assert kept.is_dir()
