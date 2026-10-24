"""Agent hook scripts under ``.agents/hooks`` block or allow the intended commands."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / ".agents" / "hooks" / "guard_bash.sh"

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


def test_guard_fails_open_on_unparsable_input() -> None:
    result = subprocess.run(
        ["bash", str(GUARD)], input="not json", text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
