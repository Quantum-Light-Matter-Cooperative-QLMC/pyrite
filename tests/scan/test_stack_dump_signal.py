"""The scan entry lets SIGUSR1 print Python stacks without stopping the run."""

import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="needs SIGUSR1")


def test_sigusr1_dumps_every_thread_and_keeps_running():
    code = (
        "import os, signal, time\n"
        "from pyrite._entry.scan import enable_stack_dump_signal\n"
        "enable_stack_dump_signal()\n"
        "os.kill(os.getpid(), signal.SIGUSR1)\n"
        "time.sleep(0.2)\n"
        "print('still running')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "still running"
    assert "most recent call first" in result.stderr


def test_handler_is_visible_in_the_caught_signal_mask():
    code = (
        "from pyrite._entry.scan import enable_stack_dump_signal\n"
        "enable_stack_dump_signal()\n"
        "mask = next(l for l in open('/proc/self/status') if l.startswith('SigCgt:'))\n"
        "print((int(mask.split()[1], 16) >> (int(__import__('signal').SIGUSR1) - 1)) & 1)\n"
    )
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.stdout.strip() == "1", result.stderr


def test_py_spy_status_wrapper_records_and_returns_the_child_status(tmp_path):
    from pyrite.perf import py_spy

    status_file = tmp_path / "hopg.py-spy.status"
    command = py_spy.status_wrapped(
        str(status_file), [sys.executable, "-c", "import sys; sys.exit(3)"]
    )
    assert subprocess.run(command, timeout=120).returncode == 3
    assert py_spy.read_status(status_file) == 3
    assert py_spy.read_status(tmp_path / "missing") == 1
