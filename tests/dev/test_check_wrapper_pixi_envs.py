"""``checks/`` SLURM wrappers install and run the pixi environment of their node (#1488).

After the uv-to-pixi migration (#1462) several wrappers kept uv-only sync flags,
which ``pixi install`` rejects, or hard-coded ``-e amd`` after sourcing
``checks/rocm/env.sh``, which installs the node's own environment
(``$PYRITE_PIXI_ENV``), so an NVIDIA node then ran an environment it never
installed. Static checks only: no wrapper is executed.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "checks"

#: uv sync/pip flags with no ``pixi install``/``pixi reinstall`` equivalent.
_UV_ONLY = re.compile(r"--(extra|group|reinstall-package|no-cache)\b")
_PIXI_SYNC = re.compile(r"\bpixi\s+(install|reinstall)\b|\bpyrite_gpu_sync\b")
_HARD_CODED_GPU_ENV = re.compile(r"(-e|--environment)[ =](amd|nvidia)\b")
_AMD_ENV = re.compile(r"(-e|--environment)[ =]amd\b")
_AMD_PINNED = re.compile(r"^#SBATCH\s+(-p\s*|--partition=)gpu-amd\b", re.MULTILINE)


def _wrappers() -> list[Path]:
    return sorted(p for p in CHECKS.rglob("*") if p.suffix in {".sbatch", ".sh"})


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Non-comment lines, with SLURM ``#SBATCH`` directives treated as comments."""
    return [
        (number, line)
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]


def _sources_env_sh(path: Path) -> bool:
    return any("checks/rocm/env.sh" in line for _, line in _code_lines(path))


WRAPPERS = _wrappers()


def test_wrappers_found() -> None:
    assert any(_sources_env_sh(p) for p in WRAPPERS)


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: str(p.relative_to(CHECKS)))
def test_no_uv_only_flags_on_pixi_sync(path: Path) -> None:
    bad = [
        f"{number}: {line.strip()}"
        for number, line in _code_lines(path)
        if _PIXI_SYNC.search(line) and _UV_ONLY.search(line)
    ]
    assert not bad, "uv-only flags passed to pixi:\n" + "\n".join(bad)


@pytest.mark.parametrize(
    "path",
    [p for p in WRAPPERS if _sources_env_sh(p)],
    ids=lambda p: str(p.relative_to(CHECKS)),
)
def test_env_sh_wrappers_use_node_environment(path: Path) -> None:
    bad = [
        f"{number}: {line.strip()}"
        for number, line in _code_lines(path)
        if _HARD_CODED_GPU_ENV.search(line)
    ]
    assert not bad, 'use -e "$PYRITE_PIXI_ENV" after checks/rocm/env.sh:\n' + "\n".join(bad)


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: str(p.relative_to(CHECKS)))
def test_amd_environment_only_in_amd_pinned_wrappers(path: Path) -> None:
    if _AMD_PINNED.search(path.read_text()):
        return
    bad = [
        f"{number}: {line.strip()}" for number, line in _code_lines(path) if _AMD_ENV.search(line)
    ]
    assert not bad, "-e amd in a wrapper not pinned to gpu-amd:\n" + "\n".join(bad)
