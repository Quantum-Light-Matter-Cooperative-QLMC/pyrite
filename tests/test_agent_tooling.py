"""Tests for portable repository agent tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cxr_mc import _dev


@pytest.fixture
def dev_module():
    return _dev


@pytest.fixture
def sweep_guard_module():
    path = Path(__file__).parents[1] / ".claude" / "hooks" / "guard_local_sweep.py"
    spec = importlib.util.spec_from_file_location("cxr_mc_sweep_guard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_skill(root: Path, name: str, description: str | None = None) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    if description is None:
        description = f"Use when working with {name}."
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# Missing or unterminated frontmatter\n", "missing or unterminated YAML frontmatter"),
        ("---\nname: sample\n---\n", "missing description"),
        (
            "---\nname: sample\ndescription: Does sample work.\n---\n",
            "description must start with 'Use when'",
        ),
    ],
)
def test_skill_validation_rejects_malformed_required_frontmatter(
    dev_module, tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "sample" / "SKILL.md"
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")

    with pytest.raises(dev_module.AgentToolingError, match=message):
        dev_module.validate_skill_file(path)


def test_skill_validation_requires_name_to_match_directory(dev_module, tmp_path: Path) -> None:
    path = write_skill(tmp_path, "sample")
    path.write_text(path.read_text().replace("name: sample", "name: other"))

    with pytest.raises(dev_module.AgentToolingError, match="must match directory"):
        dev_module.validate_skill_file(path)


def test_skill_tree_requires_skill_file_in_every_top_level_directory(
    dev_module, tmp_path: Path
) -> None:
    canonical = tmp_path / "canonical"
    mirror = tmp_path / "mirror"
    (canonical / "missing").mkdir(parents=True)
    (canonical / "missing" / "reference.txt").write_text("reference\n")
    (mirror / "missing").mkdir(parents=True)
    (mirror / "missing" / "reference.txt").write_text("reference\n")

    with pytest.raises(dev_module.AgentToolingError, match="missing SKILL.md"):
        dev_module.check_skill_trees(canonical, mirror)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing from mirror"),
        ("different", "byte-different"),
        ("extra", "unexpected mirror files"),
    ],
)
def test_check_skills_rejects_mirror_drift(
    dev_module, tmp_path: Path, mutation: str, message: str
) -> None:
    canonical = tmp_path / "canonical"
    mirror = tmp_path / "mirror"
    source = write_skill(canonical, "sample")
    mirrored = write_skill(mirror, "sample")
    if mutation == "missing":
        mirrored.unlink()
    elif mutation == "different":
        mirrored.write_text(mirrored.read_text() + "stale\n")
    else:
        extra = mirror / "extra.txt"
        extra.write_text("unexpected\n")

    assert source.is_file()
    with pytest.raises(dev_module.AgentToolingError, match=message):
        dev_module.check_skill_trees(canonical, mirror)


def test_sync_skills_replaces_stale_mirror_with_regular_files(dev_module, tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    mirror = tmp_path / "mirror"
    source = write_skill(canonical, "sample")
    (canonical / "sample" / "reference.txt").write_text("canonical\n")
    write_skill(mirror, "stale")

    dev_module.sync_skill_trees(canonical, mirror)

    mirrored = mirror / "sample" / "SKILL.md"
    assert mirrored.read_bytes() == source.read_bytes()
    assert not mirrored.is_symlink()
    assert not (mirror / "stale").exists()
    dev_module.check_skill_trees(canonical, mirror)


def test_sync_skills_preserves_mirror_when_canonical_skill_is_invalid(
    dev_module, tmp_path: Path
) -> None:
    canonical = tmp_path / "canonical"
    mirror = tmp_path / "mirror"
    invalid = write_skill(canonical, "sample")
    invalid.write_text("# missing frontmatter\n")
    existing = write_skill(mirror, "existing")
    before = existing.read_bytes()

    with pytest.raises(
        dev_module.AgentToolingError, match="missing or unterminated YAML frontmatter"
    ):
        dev_module.sync_skill_trees(canonical, mirror)

    assert existing.read_bytes() == before


def test_repository_skills_are_valid_and_exactly_mirrored(dev_module) -> None:
    dev_module.check_skill_trees(dev_module.AGENT_SKILLS_DIR, dev_module.CLAUDE_SKILLS_DIR)


def test_claude_project_memory_imports_shared_instructions() -> None:
    claude_md = Path(__file__).parents[1] / "CLAUDE.md"

    assert "@AGENTS.md" in claude_md.read_text(encoding="utf-8")


def test_agent_session_start_syncs_optional_dependencies() -> None:
    root = Path(__file__).parents[1]
    expected = "uv sync --all-groups --extra viz-render"
    cache_prefix = "rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache"
    claude = json.loads((root / ".claude" / "settings.json").read_text())
    codex = json.loads((root / ".codex" / "hooks.json").read_text())

    claude_command = claude["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    codex_command = codex["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    claude_hook_script = (root / ".claude" / "hooks" / "sync_local_backend.sh").read_text()

    assert "sync_local_backend.sh" in claude_command
    assert expected in claude_hook_script
    assert expected in codex_command
    assert cache_prefix in claude_hook_script
    assert cache_prefix in codex_command


@pytest.mark.parametrize(
    "command",
    [
        "uv run cxr run standard -m hopg",
        "rtk env UV_CACHE_DIR=/tmp/cache uv run cxr run standard -m hopg",
        "git status && uv run cxr run standard -m hopg",
        "(uv run cxr run standard -m hopg)",
    ],
)
def test_sweep_guard_blocks_local_scan(sweep_guard_module, command: str) -> None:
    assert sweep_guard_module._local_scan(command)


@pytest.mark.parametrize(
    "command",
    [
        "uv run cxr run standard -m hopg --remote",
        "uv run cxr run standard -m hopg --remote=qlmc",
        "uv run cxr run standard -m hopg -R",
        "uv run cxr run standard -m hopg -Rqlmc",
        "uv run cxr remote run standard -m hopg",
        "uv run cxr run --help",
        "echo 'uv run cxr run standard -m hopg'",
        "rg 'cxr run' README.md",
        "git status",
    ],
)
def test_sweep_guard_allows_safe_commands(sweep_guard_module, command: str) -> None:
    assert not sweep_guard_module._local_scan(command)


@pytest.mark.parametrize(
    "command",
    [
        "CXR_LOCAL_SWEEP_OK=1 uv run cxr run standard -m hopg",
        "rtk env CXR_LOCAL_SWEEP_OK=1 uv run cxr run standard -m hopg",
        "env CXR_LOCAL_SWEEP_OK=true uv run cxr run standard -m hopg",
    ],
)
def test_sweep_guard_inline_override_opts_out_a_detected_run(
    sweep_guard_module, monkeypatch, command: str
) -> None:
    monkeypatch.delenv("CXR_LOCAL_SWEEP_OK", raising=False)
    # Still detected as a local run -- the detector is unchanged ...
    assert sweep_guard_module._local_scan(command)
    # ... but the inline opt-in suppresses the block.
    assert sweep_guard_module._override_active(command)


def test_sweep_guard_ambient_override_opts_out(sweep_guard_module, monkeypatch) -> None:
    command = "uv run cxr run standard -m hopg"
    monkeypatch.setenv("CXR_LOCAL_SWEEP_OK", "1")
    assert sweep_guard_module._override_active(command)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_sweep_guard_falsey_override_still_blocks(
    sweep_guard_module, monkeypatch, value: str
) -> None:
    monkeypatch.delenv("CXR_LOCAL_SWEEP_OK", raising=False)
    command = f"CXR_LOCAL_SWEEP_OK={value} uv run cxr run standard -m hopg"
    assert sweep_guard_module._local_scan(command)
    assert not sweep_guard_module._override_active(command)
