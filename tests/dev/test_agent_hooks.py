"""Agent hook scripts under ``.agents/hooks`` block, allow or report as intended."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / ".agents" / "hooks" / "guard_bash.sh"
RUFF_HOOK = ROOT / ".agents" / "hooks" / "ruff_after_edit.sh"

needs_posix_timeout = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("timeout") is None,
    reason="the edit hook needs a POSIX shell and timeout(1)",
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="hook scripts need bash and jq",
)


def _guard(command: str) -> int:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run(
        ["bash", str(GUARD)], input=payload, text=True, capture_output=True, check=False
    ).returncode


@pytest.mark.parametrize(
    "command",
    [
        "pkill -f pytest",
        "ls; pkill python",
        "sudo killall python",
        "echo $(pkill x)",
        "git add .",
        "git add -A",
        "git -C /tmp/repo add -u",
        "git push --force",
        "git push -f origin issue-1-x",
        "git push origin HEAD:main",
        "git push origin main",
    ],
)
def test_guard_blocks_risky_commands(command: str) -> None:
    assert _guard(command) == 2


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "never pkill by pattern"',
        "kill 1234",
        "git add -- src/a.py tests/b.py",
        "git add src/.gitignore",
        "git push --force-with-lease=b:abc123 origin b",
        "git push -u origin issue-12-main-fix",
        "git fetch origin main && git rebase origin/main",
    ],
)
def test_guard_allows_ordinary_commands(command: str) -> None:
    assert _guard(command) == 0


def _ruff_hook(project: Path, pixi_script: str) -> subprocess.CompletedProcess[str]:
    """Run the edit hook on one project file with a stand-in ``pixi``."""
    pixi = project / "bin" / "pixi"
    pixi.parent.mkdir()
    pixi.write_text(f"#!/usr/bin/env bash\n{pixi_script}\n", encoding="utf-8")
    pixi.chmod(0o755)
    source = project / "module.py"
    source.write_text("x = 1\n", encoding="utf-8")
    env = os.environ | {
        "CLAUDE_PROJECT_DIR": str(project),
        "PATH": f"{pixi.parent}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        ["bash", str(RUFF_HOOK)],
        input=json.dumps({"tool_input": {"file_path": str(source)}}),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@needs_posix_timeout
def test_ruff_hook_reports_a_stale_lock(tmp_path: Path) -> None:
    result = _ruff_hook(tmp_path, "echo 'lock file not up-to-date' >&2; exit 1")
    assert result.returncode == 0
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "did not format" in context
    assert "lock file not up-to-date" in context


@needs_posix_timeout
def test_ruff_hook_is_silent_when_the_file_is_clean(tmp_path: Path) -> None:
    result = _ruff_hook(tmp_path, "exit 0")
    assert (result.returncode, result.stdout) == (0, "")


def test_guard_fails_open_on_unparsable_input() -> None:
    result = subprocess.run(
        ["bash", str(GUARD)], input="not json", text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
