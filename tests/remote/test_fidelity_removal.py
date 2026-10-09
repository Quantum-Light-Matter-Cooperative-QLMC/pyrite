"""`--fidelity` was removed at 0.6.0 (issue #387): generated argv no longer carries it."""

import pytest

from pyrite._env import GENERATED_INVOCATION_ENV
from pyrite.remote import _queue_scripts, scripts
from pyrite.runs import scan


def test_slurm_wrapper_marks_payload_as_generated_invocation():
    script = scripts._slurm_batch_script("job1", "echo payload", job_name="fidelity-test")

    export = f"export {GENERATED_INVOCATION_ENV}=1"
    assert export in script
    assert script.index(export) < script.index("echo payload")


@pytest.mark.parametrize("builder", [_queue_scripts._rebrem_flags, _queue_scripts._reline_flags])
def test_remote_recompute_no_longer_forwards_fidelity(builder):
    # The box-side recompute reads fidelity from checkpoint metadata.
    assert "--fidelity" not in builder(None, None, False)
    assert builder(None, None, True) == " --redo-all"


def test_queue_script_no_longer_forwards_fidelity():
    script = _queue_scripts._queue_script("j", ["mos2"], quick=False, workers=None)

    assert "--fidelity" not in script


def test_nsys_reexec_marks_child_as_generated_invocation(monkeypatch, tmp_path):
    monkeypatch.delenv(GENERATED_INVOCATION_ENV, raising=False)
    monkeypatch.delenv("PYRITE_MC_NSYS", raising=False)
    monkeypatch.setattr(scan.shutil, "which", lambda _name: "/usr/bin/nsys")

    class _Execd(Exception):
        pass

    def fake_execvp(_file, argv):
        raise _Execd(argv)

    monkeypatch.setattr(scan.os, "execvp", fake_execvp)
    with pytest.raises(_Execd) as execd:
        scan._reexec_under_nsys(
            catalog_profile="sub_100keV",
            material="hopg",
            performance_profile="sub_100keV",
            performance_dir=tmp_path,
            performance_interval=5.0,
            workers=0,
            quick=False,
            n_families=None,
        )

    argv = execd.value.args[0]
    assert "--fidelity" not in argv
    assert scan.os.environ[GENERATED_INVOCATION_ENV] == "1"
