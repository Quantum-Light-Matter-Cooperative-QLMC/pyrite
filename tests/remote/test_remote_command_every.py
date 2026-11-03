"""Remote every tests."""

import subprocess

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: remote._queue_script("j", ["hopg"], False, None), id="scan-monolithic"
        ),
        pytest.param(
            lambda: remote._chunked_queue_script("j", ["hopg"], False, None, 10),
            id="scan-chunked",
        ),
        pytest.param(lambda: remote._zhai_queue_script("j", 1, 1, 1, 0, False), id="zhai"),
        pytest.param(
            lambda: remote._rebrem_queue_script("j", ["hopg"], None, None, False),
            id="rebrem-monolithic",
        ),
        pytest.param(
            lambda: remote._rebrem_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="rebrem-chunked",
        ),
        pytest.param(
            lambda: scripts._recxr_queue_script("j", ["hopg"], None, None, False),
            id="reline-monolithic",
        ),
        pytest.param(
            lambda: scripts._recxr_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="reline-chunked",
        ),
    ],
)
def test_every_generated_bash_payload_quotes_hostile_remote_dir(monkeypatch, build):
    hostile = '/safe"; SENTINEL_REMOTE_DIR; #'
    monkeypatch.setattr(config, "REMOTE_DIR", hostile)

    script = build()

    expected_jobdir = config.shell_word(config.remote_path(config.JOBS_SUBDIR, "j"))
    assert f"JOBDIR={expected_jobdir}" in script
    assert "timing: uv sync %d.%03d s" in script
    assert "sync --package pyrite-mc --no-default-groups --extra" in script
    syntax = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: scripts._release_checkpoint_stems_command("j", ["hopg"]),
            id="release-stems",
        ),
        pytest.param(lambda: scripts._release_job_reservations_command("j"), id="release-job"),
        pytest.param(
            lambda: scripts._reserve_checkpoint_stems_command("j", ["hopg"]),
            id="reserve",
        ),
        pytest.param(lambda: scripts._submit_slurm_command("j", ["hopg"]), id="submit"),
        pytest.param(
            lambda: scripts._write_job_script_command(
                config.remote_path(config.JOBS_SUBDIR, "j"), "job: j\n"
            ),
            id="upload",
        ),
        pytest.param(
            lambda: scripts._clear_checkpoint_stems_command("j", ["hopg"]),
            id="clear",
        ),
        pytest.param(lambda: scripts._reap_job_command("j"), id="reap"),
        pytest.param(lambda: viewer._status_remote_command('JOB="j"', 0), id="status"),
    ],
)
def test_every_remote_bash_command_family_quotes_hostile_remote_dir(monkeypatch, build):
    hostile = '/safe"; SENTINEL_REMOTE_DIR; #'
    monkeypatch.setattr(config, "REMOTE_DIR", hostile)

    command = build()

    assert '\\"; SENTINEL_REMOTE_DIR' in command or "'/safe\"; SENTINEL_REMOTE_DIR" in command
    syntax = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: remote._queue_script("j", ["hopg"], False, None), id="scan-monolithic"
        ),
        pytest.param(
            lambda: remote._chunked_queue_script("j", ["hopg"], False, None, 10),
            id="scan-chunked",
        ),
        pytest.param(lambda: remote._zhai_queue_script("j", 1, 1, 1, 0, False), id="zhai"),
        pytest.param(
            lambda: remote._rebrem_queue_script("j", ["hopg"], None, None, False),
            id="rebrem-monolithic",
        ),
        pytest.param(
            lambda: remote._rebrem_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="rebrem-chunked",
        ),
        pytest.param(
            lambda: scripts._recxr_queue_script("j", ["hopg"], None, None, False),
            id="reline-monolithic",
        ),
        pytest.param(
            lambda: scripts._recxr_chunked_queue_script("j", ["hopg"], None, None, False, 10),
            id="reline-chunked",
        ),
    ],
)
def test_every_generated_bash_payload_rejects_remote_uv_program_text(monkeypatch, build):
    monkeypatch.setattr(config, "REMOTE_UV", "uv; SENTINEL_REMOTE_UV #")

    with pytest.raises(SystemExit, match="PYRITE_REMOTE_UV"):
        build()


@pytest.mark.parametrize("chunked", [False, True])
def test_queue_fallback_policy_is_forwarded(chunked):
    def build(allow):
        if chunked:
            return _queue_scripts._chunked_queue_script(
                "j", ["hopg"], False, None, 10, allow_physics_fallback=allow
            )
        return _queue_scripts._queue_script(
            "j", ["hopg"], False, None, allow_physics_fallback=allow
        )

    strict = build(False)
    allowed = build(True)
    assert "PYRITE_STRICT_PHYSICS_FALLBACK=1" in strict
    assert "--allow-physics-fallback" not in strict
    assert "PYRITE_STRICT_PHYSICS_FALLBACK=1" not in allowed
    assert "--allow-physics-fallback" in allowed
    assert "--require bremslib,elsepa" in strict
    assert "--require elsepa" in allowed
