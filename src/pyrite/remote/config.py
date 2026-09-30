"""Environment + configuration constants for the remote job subsystem."""

import os
import re
import shlex
import subprocess
from contextlib import contextmanager
from functools import cache
from pathlib import Path, PurePosixPath

from .._env import env_value

# Compatibility override for tests and callers that historically patched this
# module global. Normal resolution is dynamic so environment and store changes
# made before an invocation are observed.
HOST: str | None = None
# A leading ``~/`` is the remote login home, resolved once per host over ssh (see
# ``_expand_remote_home``): every downstream script and scp path sees an absolute
# path, and the defaults work on any box without configuration.
REMOTE_DIR = env_value("PYRITE_REMOTE_DIR", "~/pyrite")
REMOTE_UV = env_value("PYRITE_REMOTE_UV", "~/.local/bin/uv")
# SLURM target profile overrides, same compatibility role as ``HOST``: ``None``
# resolves ``remote.gpu_vendor`` / ``remote.partition`` / ``remote.nodelist`` /
# ``remote.gres`` through the shared config precedence (see ``console.config``).
REMOTE_GPU_VENDOR: str | None = None
SLURM_PARTITION: str | None = None
SLURM_NODELIST: str | None = None
SLURM_GRES: str | None = None
SLURM_GPUS = 1
SLURM_CPUS_PER_MATERIAL = 8
SLURM_TIME = "UNLIMITED"
# Default 1: the box has one GPU (SLURM_GPUS=1), and >1 co-tenant `pyrite run`
# processes time-slice the card while the runner's own CPU-pool/GPU pipeline
# already overlaps the two phases -- contention for no throughput win, plus
# VRAM-pool oversubscription. Opt in via parallel_materials; the queue script
# exports PYRITE_MC_GPU_SHARE so co-tenant pool caps sum to PYRITE_MC_GPU_POOL_FRAC.
DEFAULT_PARALLEL_MATERIALS = 1
MAX_PARALLEL_MATERIALS = 4
# repo root = three levels up from src/pyrite/remote/config.py. The package facade orchestrates
# the *checkout* (it tars the working tree up to the box), so it resolves paths
# against the repo root, not its own package dir.
LOCAL_ROOT = Path(__file__).resolve().parents[3]

# detached-job bookkeeping lives under <REMOTE_DIR>/jobs/<jobid>/ on the box
# (gitignored there): run.sh, meta, state, log. One subdir per `start`.
JOBS_SUBDIR = "jobs"
RESERVATIONS_SUBDIR = "reservations"

# `sync` records what it just unpacked in <REMOTE_DIR>/.pyrite-sync: the payload
# content digest plus revision evidence. It is the only code-identity marker the
# box has -- the remote checkout is an exported tree, not a repository -- so a
# later sync can tell whether a live job is running that same code, and evidence
# collected there can name a revision instead of shelling out to git.
SYNC_STAMP_NAME = ".pyrite-sync"

# The Zhai reproduction job has no crystal key of its own, but reusing the
# existing material-stem bookkeeping (_refuse_if_busy / _live_jobs /
# stop_jobs) needs one to key off of -- this is that synthetic token.
ZHAI_STEM = "zhai"

# what `sync` ships up: the code that changes (the src/ package now also carries
# data/, so it travels too, and the box invokes its entry shims via
# `python -m pyrite._entry.<name>`), plus checks/ and pyproject.toml. Not
# checkpoints/ (the output we pull back the other way).
SYNC_PATHS = [
    "src",
    "checks",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE.txt",
    "THIRD-PARTY-NOTICES.md",
    # Generator sources (ADR-0014): lets the box run `pyrite tables generate`.
    "vendor",
]


def remote_catalog_path() -> str | None:
    """Return the staged catalog path when a non-bundled catalog is selected."""
    from .._catalog_layout import bundled_catalog, selected_catalog

    source = selected_catalog()
    if source == bundled_catalog().resolve():
        return None
    name = "external-catalog" if source.is_dir() else "external-catalog.toml"
    return remote_path(name)


def remote_runtime_env() -> str:
    """Environment assignments for direct remote PyRITE invocations."""
    assignments = [f"PYRITE_HOME={shell_word(remote_dir())}"]
    catalog = remote_catalog_path()
    if catalog is not None:
        assignments.append(f"PYRITE_CATALOG={shell_word(catalog)}")
    return " ".join(assignments)


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
    from ..console import config as cli_config

    try:
        resolved = cli_config.resolve("remote.target", HOST)
        if not resolved.value:
            raise SystemExit(
                "remote target is not configured; set remote.target with "
                "`pyrite config set remote.target HOST` or set PYRITE_REMOTE_HOST"
            )
        return validate_remote_target(resolved.value)
    except (ValueError, cli_config.ConfigError) as exc:
        raise SystemExit(f"invalid PYRITE_REMOTE_HOST: {exc}") from exc


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


def ssh_mux_options() -> list[str]:
    """OpenSSH options that share one persistent connection per host.

    Every remote command otherwise pays a full connect (through the lab's
    ``cloudflared`` proxy, ~4 s); with a control master only the first does.
    The socket directory is deliberately short: ``ControlPath`` is capped near
    108 bytes. Disable with ``PYRITE_SSH_MUX=0``; skipped off POSIX, where
    OpenSSH has no control sockets.
    """
    if os.name != "posix" or env_value("PYRITE_SSH_MUX", "1") == "0":
        return []
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime and os.path.isdir(runtime) else Path.home() / ".cache"
    sockets = base / "pyrite-ssh"
    try:
        sockets.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        return []
    return [
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={sockets}/%C",
        "-o",
        "ControlPersist=10m",
    ]


def ssh_argv(*args: str) -> list[str]:
    """Return an ``ssh`` command line with connection sharing enabled."""
    return ["ssh", *ssh_mux_options(), *args]


def scp_argv(*args: str) -> list[str]:
    """Return an ``scp`` command line with connection sharing enabled."""
    return ["scp", *ssh_mux_options(), *args]


@cache
def _remote_home(host: str) -> str:
    """Return the absolute login home of ``host`` (one ssh round trip, cached)."""
    try:
        result = subprocess.run(
            ssh_argv("-n", "-o", "BatchMode=yes", host, 'printf %s "$HOME"'),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"cannot resolve '~' on remote host {host!r}: {exc}") from exc
    home = result.stdout.strip()
    if result.returncode != 0 or not PurePosixPath(home).is_absolute():
        raise SystemExit(
            f"cannot resolve '~' on remote host {host!r} (ssh exit {result.returncode}); "
            "set PYRITE_REMOTE_DIR / PYRITE_REMOTE_UV to absolute paths"
        )
    _reject_controls("remote $HOME", home)
    return home


def _expand_remote_home(value: str) -> str:
    """Replace a leading ``~`` or ``~/`` with the remote login home."""
    if value != "~" and not value.startswith("~/"):
        return value
    return _remote_home(remote_host()).rstrip("/") + value[1:]


def remote_dir() -> str:
    """Return validated absolute POSIX checkout path (``~/`` expanded remotely)."""
    value = REMOTE_DIR
    if not isinstance(value, str):
        raise SystemExit(f"invalid PYRITE_REMOTE_DIR={value!r}: expected absolute POSIX path")
    _reject_controls("PYRITE_REMOTE_DIR", value)
    value = _expand_remote_home(value)
    if not value or not PurePosixPath(value).is_absolute():
        raise SystemExit(
            f"invalid PYRITE_REMOTE_DIR={value!r}: expected absolute POSIX path "
            "or ~/... (for example ~/pyrite)"
        )
    return value


def remote_uv() -> str:
    """Return validated executable name or absolute POSIX path (``~/`` expanded remotely)."""
    value = REMOTE_UV
    if not isinstance(value, str):
        raise SystemExit(
            f"invalid PYRITE_REMOTE_UV={value!r}: expected executable name or absolute POSIX path"
        )
    _reject_controls("PYRITE_REMOTE_UV", value)
    value = _expand_remote_home(value)
    if not value or (
        _ABS_EXECUTABLE_RE.fullmatch(value) is None and _EXECUTABLE_RE.fullmatch(value) is None
    ):
        raise SystemExit(
            f"invalid PYRITE_REMOTE_UV={value!r}: expected executable name such as 'uv' "
            "or absolute POSIX path, not shell program text"
        )
    return value


_SLURM_SETTINGS = {
    # key: (environment name, value pattern, what a valid value looks like)
    "remote.gpu_vendor": (
        "PYRITE_REMOTE_GPU_VENDOR",
        re.compile(r"nvidia|amd|intel"),
        "nvidia, amd, or intel",
    ),
    "remote.partition": (
        "PYRITE_REMOTE_PARTITION",
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*"),
        "a SLURM partition name such as gpu-amd",
    ),
    "remote.nodelist": (
        "PYRITE_REMOTE_NODELIST",
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._,\[\]-]*"),
        "a SLURM node list such as qlmc-ace or node[01-02], or 'any'",
    ),
    "remote.gres": (
        "PYRITE_REMOTE_GRES",
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._:,-]*"),
        "a SLURM gres string such as gpu:1 or gpu:radeon8060s:1",
    ),
}
# ``remote.nodelist`` value meaning "let SLURM pick any node in the partition".
NODELIST_ANY = "any"


def validate_slurm_setting(key: str, value: str) -> str:
    """Validate one SLURM profile value; accepted values are safe unquoted in ``#SBATCH``."""
    _env_name, pattern, expected = _SLURM_SETTINGS[key]
    if key == "remote.gpu_vendor":
        value = value.strip().lower()
    if pattern.fullmatch(value) is None:
        raise ValueError(f"expected {expected}")
    return value


def _slurm_setting(key: str, override: str | None) -> str:
    from ..console import config as cli_config

    env_name = _SLURM_SETTINGS[key][0]
    try:
        value = validate_slurm_setting(key, cli_config.resolve(key, override).value)
    except (ValueError, cli_config.ConfigError) as exc:
        raise SystemExit(f"invalid {key} ({env_name}): {exc}") from exc
    return "" if key == "remote.nodelist" and value == NODELIST_ANY else value


def remote_gpu_vendor() -> str:
    """Return validated remote accelerator vendor (``remote.gpu_vendor``)."""
    return _slurm_setting("remote.gpu_vendor", REMOTE_GPU_VENDOR)


def slurm_partition() -> str:
    """Return validated SLURM partition for new jobs (``remote.partition``)."""
    return _slurm_setting("remote.partition", SLURM_PARTITION)


def slurm_nodelist() -> str | None:
    """Return validated SLURM node selection (``remote.nodelist``), or None for any node."""
    return _slurm_setting("remote.nodelist", SLURM_NODELIST) or None


def slurm_gres() -> str:
    """Return validated SLURM generic-resource request (``remote.gres``)."""
    return _slurm_setting("remote.gres", SLURM_GRES)


def remote_path(*parts: str) -> str:
    """Build raw remote path from validated checkout root and trusted components."""
    root = remote_dir().rstrip("/") or "/"
    suffix = "/".join(part.strip("/") for part in parts)
    return f"{root}/{suffix}" if suffix and root != "/" else f"{root}{suffix}"


def remote_sync_stamp_path() -> str:
    """Absolute path of the code-identity stamp `sync` writes on the box."""
    return remote_path(SYNC_STAMP_NAME)


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
            f"invalid PYRITE_REMOTE_DIR={REMOTE_DIR!r}: SLURM output paths cannot contain "
            "whitespace or '#'; choose a checkout path safe for #SBATCH directives"
        )
    return value


def scp_remote_path(path: str) -> str:
    """Render validated host plus one quoted remote path for SCP argv."""
    _reject_controls("remote SCP path", path)
    if not PurePosixPath(path).is_absolute():
        raise SystemExit(f"invalid remote SCP path={path!r}: expected absolute POSIX path")
    return f"{remote_host()}:{shell_arg(path)}"
