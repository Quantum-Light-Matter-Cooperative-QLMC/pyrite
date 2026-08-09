"""Environment + configuration constants for the remote job subsystem."""

import os
import re
import shlex
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

# Compatibility override for tests and callers that historically patched this
# module global. Normal resolution is dynamic so environment and store changes
# made before an invocation are observed.
HOST: str | None = None
REMOTE_DIR = os.environ.get("CXR_REMOTE_DIR", "/home/aamador/dev/cxr-mc")
REMOTE_UV = os.environ.get("CXR_REMOTE_UV", "/home/aamador/.local/bin/uv")
REMOTE_GPU_VENDOR = os.environ.get("CXR_REMOTE_GPU_VENDOR", "nvidia")
SLURM_PARTITION = "gpu"
SLURM_GPUS = 1
SLURM_CPUS_PER_MATERIAL = 8
SLURM_TIME = "UNLIMITED"
# Default 1: the box has one GPU (SLURM_GPUS=1), and >1 co-tenant `cxr run`
# processes time-slice the card while the runner's own CPU-pool/GPU pipeline
# already overlaps the two phases -- contention for no throughput win, plus
# VRAM-pool oversubscription. Opt in via parallel_materials; the queue script
# exports CXR_MC_GPU_SHARE so co-tenant pool caps sum to CXR_MC_GPU_POOL_FRAC.
DEFAULT_PARALLEL_MATERIALS = 1
MAX_PARALLEL_MATERIALS = 4
# repo root = three levels up from src/cxr_mc/remote/config.py. The package facade orchestrates
# the *checkout* (it tars the working tree up to the box), so it resolves paths
# against the repo root, not its own package dir.
LOCAL_ROOT = Path(__file__).resolve().parents[3]
MATS_FILE = LOCAL_ROOT / "mats_to_sim.toml"

# detached-job bookkeeping lives under <REMOTE_DIR>/jobs/<jobid>/ on the box
# (gitignored there): run.sh, meta, state, log. One subdir per `start`.
JOBS_SUBDIR = "jobs"
RESERVATIONS_SUBDIR = "reservations"

# The Zhai reproduction job has no crystal key of its own, but reusing the
# existing material-stem bookkeeping (_refuse_if_busy / _live_jobs /
# stop_jobs) needs one to key off of -- this is that synthetic token.
ZHAI_STEM = "zhai"

# what `sync` ships up: the code that changes (the src/ package now also carries
# data/, so it travels too, and the box invokes its entry shims via
# `python -m cxr_mc._entry.<name>`), plus checks/ and pyproject.toml. Not
# checkpoints/ (the output we pull back the other way).
SYNC_PATHS = [
    "src",
    "checks",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "mats_to_sim.toml",
]

# text extensions whose CRLF is normalized to LF before tarring (see _add_to_tar):
# the laptop is Windows so its working files are CRLF, and shipping those over the
# box's LF checkout dirties `git status` there even though content is identical.
TEXT_EXTS = {".py", ".toml", ".cfg", ".ini", ".txt", ".md", ".csv"}


_HOST_ALIAS_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_EXECUTABLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_ABS_EXECUTABLE_RE = re.compile(r"/[A-Za-z0-9._+/-]+")


def _reject_controls(field: str, value: str) -> None:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise SystemExit(
            f"invalid {field}={value!r}: control characters are not allowed; "
            f"set {field} to one printable value"
        )


def validate_remote_target(value: str) -> str:
    """Validate and return one SSH-config host alias."""
    if _HOST_ALIAS_RE.fullmatch(value) is None:
        raise ValueError(
            "expected host alias containing only letters, digits, dots, underscores, "
            "or hyphens, without a leading dash"
        )
    return value


def remote_host() -> str:
    """Return the validated effective SSH-config host alias."""
    from ..cli import _config as cli_config

    try:
        resolved = cli_config.resolve("remote.target", HOST)
        return validate_remote_target(resolved.value)
    except (ValueError, cli_config.ConfigError) as exc:
        raise SystemExit(f"invalid CXR_REMOTE_HOST: {exc}") from exc


@contextmanager
def override_remote_host(value: str | None):
    """Temporarily apply one validated per-call remote target override."""
    global HOST
    if value is None:
        yield
        return
    previous = HOST
    HOST = value
    try:
        remote_host()
        yield
    finally:
        HOST = previous


def remote_dir() -> str:
    """Return validated absolute POSIX checkout path."""
    value = REMOTE_DIR
    if not isinstance(value, str):
        raise SystemExit(f"invalid CXR_REMOTE_DIR={value!r}: expected absolute POSIX path")
    _reject_controls("CXR_REMOTE_DIR", value)
    if not value or not PurePosixPath(value).is_absolute():
        raise SystemExit(
            f"invalid CXR_REMOTE_DIR={value!r}: expected absolute POSIX path "
            "(for example /home/user/dev/cxr-mc)"
        )
    return value


def remote_uv() -> str:
    """Return validated executable name or absolute POSIX executable path."""
    value = REMOTE_UV
    if not isinstance(value, str):
        raise SystemExit(
            f"invalid CXR_REMOTE_UV={value!r}: expected executable name or absolute POSIX path"
        )
    _reject_controls("CXR_REMOTE_UV", value)
    if not value or (
        _ABS_EXECUTABLE_RE.fullmatch(value) is None and _EXECUTABLE_RE.fullmatch(value) is None
    ):
        raise SystemExit(
            f"invalid CXR_REMOTE_UV={value!r}: expected executable name such as 'uv' "
            "or absolute POSIX path, not shell program text"
        )
    return value


def remote_gpu_vendor() -> str:
    """Return validated remote accelerator vendor capability selector."""

    value = str(REMOTE_GPU_VENDOR).strip().lower()
    if value not in {"nvidia", "amd", "intel"}:
        raise SystemExit(
            f"invalid CXR_REMOTE_GPU_VENDOR={REMOTE_GPU_VENDOR!r}: expected nvidia, amd, or intel"
        )
    return value


def remote_path(*parts: str) -> str:
    """Build raw remote path from validated checkout root and trusted components."""
    root = remote_dir().rstrip("/") or "/"
    suffix = "/".join(part.strip("/") for part in parts)
    return f"{root}/{suffix}" if suffix and root != "/" else f"{root}{suffix}"


def shell_word(value: str) -> str:
    """Render one value as one POSIX shell word."""
    escaped = value.replace("\\", "\\\\")
    for char in ('"', "$", "`"):
        escaped = escaped.replace(char, f"\\{char}")
    return f'"{escaped}"'


def shell_single_word(value: str) -> str:
    """Render one value as one single-quoted POSIX shell word."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def shell_arg(value: str) -> str:
    """Render shell argument, adding quoting only when syntax requires it."""
    return shlex.quote(value)


def shell_remote_dir() -> str:
    return shell_word(remote_dir())


def shell_remote_uv() -> str:
    return shell_word(remote_uv())


def shell_remote_path(*parts: str) -> str:
    return shell_word(remote_path(*parts))


def sbatch_remote_path(*parts: str) -> str:
    """Render path for one unquoted ``#SBATCH`` directive value."""
    value = remote_path(*parts)
    if any(char.isspace() for char in value) or "#" in value:
        raise SystemExit(
            f"invalid CXR_REMOTE_DIR={REMOTE_DIR!r}: SLURM output paths cannot contain "
            "whitespace or '#'; choose a checkout path safe for #SBATCH directives"
        )
    return value


def scp_remote_path(path: str) -> str:
    """Render validated host plus one quoted remote path for SCP argv."""
    _reject_controls("remote SCP path", path)
    if not PurePosixPath(path).is_absolute():
        raise SystemExit(f"invalid remote SCP path={path!r}: expected absolute POSIX path")
    return f"{remote_host()}:{shell_arg(path)}"
