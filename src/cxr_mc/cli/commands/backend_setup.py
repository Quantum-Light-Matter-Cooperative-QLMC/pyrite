"""``cxr setup``: detect GPU hardware and persist ``CXR_MC_BACKEND`` in ``.env``.

Detection is OS-tooling only (``nvidia-smi``, ``rocm-smi``/``rocminfo``,
``clinfo``/``sycl-ls``/``lspci``, device nodes) and never assumes a vendor
Python package (``cupy``, ``dpnp``/``dpctl``) is installed -- those extras are
what this command is meant to help a user *decide* to install, so requiring
them first would defeat the point. See ``montecarlo._backend.select_backend``
for the runtime-side counterpart this command does not touch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from .. import _core as _cli_core

_ENV_KEY = "CXR_MC_BACKEND"

# Matches README.md's install table; printed as a follow-up instruction, never
# run automatically (see agentdocs/tasks/feature/backend-autodetect/README.md decision 2).
_VENDOR_EXTRA_INSTALL = {
    "cuda": "uv sync --extra nvidia",
    "rocm": "CUPY_INSTALL_USE_HIP=1 uv sync --extra amd",
    "sycl": "uv sync --extra intel",
}


@dataclass(frozen=True)
class DetectionResult:
    """One accelerator vendor's detected presence and how it was detected."""

    vendor: str
    backend: str
    reason: str


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _has_device_nodes(pattern: str) -> bool:
    dev = Path("/dev")
    if not dev.is_dir():
        return False
    try:
        return any(dev.glob(pattern))
    except OSError:
        return False


def _run_text(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def detect_nvidia() -> DetectionResult | None:
    """NVIDIA presence: ``nvidia-smi`` on PATH, else ``/dev/nvidia*`` nodes."""
    if _has_tool("nvidia-smi"):
        return DetectionResult("nvidia", "cuda", "nvidia-smi on PATH")
    if _has_device_nodes("nvidia*"):
        return DetectionResult("nvidia", "cuda", "/dev/nvidia* device node")
    return None


def detect_amd() -> DetectionResult | None:
    """AMD presence: ``rocm-smi``/``rocminfo`` on PATH, else ``/dev/kfd``."""
    if _has_tool("rocm-smi"):
        return DetectionResult("amd", "rocm", "rocm-smi on PATH")
    if _has_tool("rocminfo"):
        return DetectionResult("amd", "rocm", "rocminfo on PATH")
    if _has_device_nodes("kfd"):
        return DetectionResult("amd", "rocm", "/dev/kfd device node")
    return None


def detect_intel() -> DetectionResult | None:
    """Intel presence: ``sycl-ls``/``clinfo`` reporting an Intel device, else
    ``lspci`` listing an Intel VGA controller. No dedicated device node
    convention exists for Intel GPUs, unlike NVIDIA/AMD."""
    for tool, reason in (("sycl-ls", "sycl-ls output"), ("clinfo", "clinfo output")):
        if _has_tool(tool):
            output = _run_text([tool])
            if output and "intel" in output.lower():
                return DetectionResult("intel", "sycl", reason)
    if _has_tool("lspci"):
        output = _run_text(["lspci"])
        if output:
            for line in output.splitlines():
                if "VGA compatible controller" in line and "Intel" in line:
                    return DetectionResult("intel", "sycl", "lspci VGA controller")
    return None


def detect_all() -> list[DetectionResult]:
    """Detected accelerators in priority order (NVIDIA, AMD, Intel)."""
    return [
        result for result in (detect_nvidia(), detect_amd(), detect_intel()) if result is not None
    ]


def _repo_root() -> Path:
    """Best-effort repo root: three levels above this module in a source
    checkout (``src/cxr_mc/cli/backend_setup.py`` -> repo root), falling back
    to the current working directory for non-editable installs."""
    module_path = Path(__file__).resolve()
    try:
        root = module_path.parents[3]
    except IndexError:
        return Path.cwd()
    if (root / "pyproject.toml").is_file():
        return root
    return Path.cwd()


def _env_path() -> Path:
    return _repo_root() / ".env"


def _interactive() -> bool:
    """Whether stdin is a TTY we may safely prompt on (isolated for tests)."""
    return sys.stdin.isatty()


def read_existing_backend(env_path: Path) -> str | None:
    """Return the current ``CXR_MC_BACKEND`` value in ``env_path``, if set."""
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == _ENV_KEY:
            return value.strip()
    return None


def write_backend(env_path: Path, backend: str) -> None:
    """Create, in-place update, or append ``CXR_MC_BACKEND=<backend>`` in
    ``env_path``. Preserves every other line untouched."""
    if not env_path.is_file():
        env_path.write_text(f"{_ENV_KEY}={backend}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.partition("=")[0].strip()
        if key == _ENV_KEY:
            lines[index] = f"{_ENV_KEY}={backend}\n"
            env_path.write_text("".join(lines), encoding="utf-8")
            return

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"{_ENV_KEY}={backend}\n")
    env_path.write_text("".join(lines), encoding="utf-8")


def _print_extra_instructions(backend: str) -> None:
    install_cmd = _VENDOR_EXTRA_INSTALL.get(backend)
    if install_cmd is None:
        return
    click.echo(
        f"writing .env alone does not install the accelerator package; run "
        f"`{install_cmd}` (a new Claude Code session does this automatically via "
        "`.claude/hooks/sync_local_backend.sh`)."
    )


@click.command(
    "setup",
    help=(
        "Detect installed GPU hardware and write CXR_MC_BACKEND to repo-root .env.\n\n"
        "Probes OS-level tooling only (nvidia-smi, rocm-smi/rocminfo, clinfo/sycl-ls/"
        "lspci) -- no vendor Python package (cupy, dpnp/dpctl) needs to be installed "
        "first. Prompts interactively to opt into an accelerator; defaults to cpu if "
        "none is detected, declined, or the session is non-interactive. A no-op once "
        "CXR_MC_BACKEND is already set in .env, unless --force is given."
    ),
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Accept the top detected accelerator without an interactive prompt.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-run detection and overwrite an existing CXR_MC_BACKEND value in .env.",
)
def command(yes: bool, force: bool) -> None:
    env_path = _env_path()
    existing = read_existing_backend(env_path)
    if existing is not None and not force:
        click.echo(
            f"{_cli_core.paint(_ENV_KEY, 'inactive')} already set to {existing!r} in "
            f"{env_path}; nothing to do (use --force to re-detect)."
        )
        return

    candidates = detect_all()
    interactive = _interactive()

    chosen = "cpu"
    if not candidates:
        click.echo("no supported GPU hardware detected; defaulting to CXR_MC_BACKEND=cpu.")
    else:
        top = candidates[0]
        if yes:
            chosen = top.backend
            click.echo(f"detected {top.vendor} GPU ({top.reason}); using backend={top.backend}.")
        elif interactive:
            accept = click.confirm(
                f"Detected {top.vendor} GPU via {top.reason}. Enable GPU acceleration "
                f"(backend={top.backend})?",
                default=True,
                err=True,
            )
            chosen = top.backend if accept else "cpu"
        else:
            click.echo(
                f"detected {top.vendor} GPU ({top.reason}) but this session is "
                "non-interactive; defaulting to CXR_MC_BACKEND=cpu. Re-run `cxr setup "
                "-y` to accept it without a prompt.",
                err=True,
            )

    write_backend(env_path, chosen)
    click.echo(f"{_cli_core.paint('wrote', 'done')} {_ENV_KEY}={chosen} to {env_path}")
    _print_extra_instructions(chosen)


__all__ = ["command", "detect_all", "detect_amd", "detect_intel", "detect_nvidia"]
