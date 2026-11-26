"""Environment + configuration constants for the remote job subsystem."""

import os
import re
import shlex
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path, PurePosixPath

from .._util.env import env_value
from ..console.outputs import LEGACY_LAYOUT, OUTPUT_DIRNAME
from ..paths import cache_dir

# Compatibility override for tests and callers that historically patched this
# module global. Normal resolution is dynamic so environment and store changes
# made before an invocation are observed.
HOST: str | None = None
# A leading ``~/`` is the remote login home, resolved once per host over ssh (see
# ``_expand_remote_home``): every downstream script and scp path sees an absolute
# path, and the defaults work on any box without configuration.
REMOTE_DIR = env_value("PYRITE_REMOTE_DIR", "~/pyrite")
REMOTE_PIXI = env_value("PYRITE_REMOTE_PIXI", "~/.pixi/bin/pixi")
# SLURM target profile overrides, same compatibility role as ``HOST``: ``None``
# resolves ``remote.gpu_vendor`` / ``remote.partition`` / ``remote.nodelist`` /
# ``remote.gres`` / ``remote.account`` / ``remote.constraint`` /
# ``remote.max_time`` / ``remote.qos`` / ``remote.mail_type`` / ``remote.mail_user`` /
# ``remote.sbatch_extra`` through the shared config precedence (see ``console.config``).
REMOTE_GPU_VENDOR: str | None = None
SLURM_PARTITION: str | None = None
SLURM_NODELIST: str | None = None
SLURM_GRES: str | None = None
SLURM_ACCOUNT: str | None = None
SLURM_CONSTRAINT: str | None = None
SLURM_MAX_TIME: str | None = None
SLURM_QOS: str | None = None
SLURM_MAIL_TYPE: str | None = None
SLURM_MAIL_USER: str | None = None
SLURM_SBATCH_EXTRA: str | None = None
HOST_EXEC: str | None = None
SLURM_NICE: str | None = None
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
# against the repo root, not its own package dir. Source-tree inputs only: under
# an installed package this is inside site-packages, so no output may be written
# here -- outputs resolve through ``console.outputs.output_dir``.
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

# flock target serializing every writer of the shared checkout. `sync` holds it
# exclusively from the live-job check through the stamp; a job's startup takes it
# shared, so a job never begins importing mid-sync. Seconds a holder is awaited.
SYNC_LOCK_NAME = ".pyrite-sync.lock"
SYNC_LOCK_WAIT_SECONDS = 300

# The Zhai reproduction job has no crystal key of its own, but reusing the
# existing material-stem bookkeeping (_refuse_if_busy / _live_jobs /
# stop_jobs) needs one to key off of -- this is that synthetic token.
ZHAI_STEM = "zhai"

# what `sync` ships up: the code that changes (the src/ package now also carries
# data/, so it travels too, and the box invokes its entry shims via
# `python -m pyrite._entry.<name>`), plus checks/ and pyproject.toml. Not
# pyrite-output/ (the output we pull back the other way).
SYNC_PATHS = [
    "src",
    "checks",
    "pyproject.toml",
    "pixi.lock",
    "README.md",
    "PYPI.md",
    "LICENSE.txt",
    "THIRD-PARTY-NOTICES.md",
    "LICENSE-SVML.md",
    # Generator sources (ADR-0014): lets the box run `pyrite tables generate`.
    "vendor",
]


# The bundled catalog inside the synced checkout (``src`` is in SYNC_PATHS).
REMOTE_BUNDLED_CATALOG = "src/pyrite/data/catalog"


def external_catalog_selected() -> bool:
    """Whether the selected catalog is staged rather than the bundled one."""
    from .._catalog_layout import bundled_catalog, selected_catalog

    return selected_catalog().resolve() != bundled_catalog().resolve()


def remote_catalog_path() -> str:
    """Return the remote catalog path matching the local selection.

    Always explicit: the bundled selection maps to the synced checkout's
    bundled catalog, so a catalog configured or staged on the box never wins.
    """
    from .._catalog_layout import selected_catalog

    if not external_catalog_selected():
        return remote_path(REMOTE_BUNDLED_CATALOG)
    name = "external-catalog" if selected_catalog().is_dir() else "external-catalog.toml"
    return remote_path(name)


# The local user catalog layer (profiles read over the bundled catalog) inside
# the synced checkout; shipped only while the bundled catalog is selected.
REMOTE_USER_CATALOG = "user-catalog"


def remote_user_catalog_path() -> str:
    """Return the remote user-layer path; always explicit, so the box's own never wins."""
    return remote_path(REMOTE_USER_CATALOG)


def remote_runtime_env() -> str:
    """Environment assignments for direct remote PyRITE invocations."""
    return " ".join(
        (
            f"PYRITE_DATA_DIR={shell_word(remote_dir())}",
            f"PYRITE_CACHE_DIR={shell_word(remote_path('cache'))}",
            f"PYRITE_HOME={shell_word(remote_dir())}",
            f"PYRITE_CATALOG={shell_word(remote_catalog_path())}",
            f"PYRITE_USER_CATALOG={shell_word(remote_user_catalog_path())}",
        )
    )


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


def _legacy_remote_host() -> str:
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
        validate_remote_target(value)
        selected = _selected_target()
        if selected is None:
            yield
        else:
            with use_target(replace(selected, host=value)):
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
    sockets = (
        Path(runtime) / "pyrite-ssh" if runtime and os.path.isdir(runtime) else cache_dir() / "ssh"
    )
    if len(str(sockets).encode()) + len("/%C") >= 100:
        sockets = Path("/tmp") / f"pyrite-ssh-{os.getuid()}"
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


def rsync_argv(*args: str) -> list[str]:
    """Return an ``rsync`` command line whose ssh shares the control master.

    ``-s`` sends paths through the rsync protocol rather than the remote shell,
    so remote paths are passed raw (see ``rsync_remote_path``), never quoted.
    """
    return ["rsync", "-s", "-e", shlex.join(["ssh", *ssh_mux_options()]), *args]


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
            "set PYRITE_REMOTE_DIR / PYRITE_REMOTE_PIXI to absolute paths"
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
    value = _target_field("remote_dir", REMOTE_DIR)
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


def remote_pixi() -> str:
    """Return validated pixi executable name or absolute POSIX path (``~/`` expanded remotely)."""
    value = _target_field("pixi", REMOTE_PIXI)
    if not isinstance(value, str):
        raise SystemExit(
            f"invalid PYRITE_REMOTE_PIXI={value!r}: expected executable name or absolute POSIX path"
        )
    _reject_controls("PYRITE_REMOTE_PIXI", value)
    value = _expand_remote_home(value)
    if not value or (
        _ABS_EXECUTABLE_RE.fullmatch(value) is None and _EXECUTABLE_RE.fullmatch(value) is None
    ):
        raise SystemExit(
            f"invalid PYRITE_REMOTE_PIXI={value!r}: expected executable name such as 'pixi' "
            "or absolute POSIX path, not shell program text"
        )
    return value


# ``sbatch --mail-type`` events; NONE suppresses the others.
_MAIL_TYPES = (
    "NONE",
    "BEGIN",
    "END",
    "FAIL",
    "REQUEUE",
    "ALL",
    "INVALID_DEPEND",
    "STAGE_OUT",
    "TIME_LIMIT",
    "TIME_LIMIT_90",
    "TIME_LIMIT_80",
    "TIME_LIMIT_50",
)
_MAIL_TYPE = f"(?:{'|'.join(sorted(_MAIL_TYPES, key=len, reverse=True))})"
# One ``remote.sbatch_extra`` item: a long option, optionally ``=value``, whose
# characters are safe unquoted in an ``#SBATCH`` line.
_SBATCH_ITEM = r"--[a-z][a-z0-9-]*(?:=[A-Za-z0-9_./:,@%+|&\[\]()*-]+)?"
# Options ``remote.sbatch_extra`` may not set: PyRITE writes them itself (job
# bookkeeping, chunk timing, resources) or a dedicated setting owns them.
_SBATCH_OWNED = {
    "job-name": None,
    "output": None,
    "error": None,
    "time": "remote.max_time or --chunk-minutes",
    "signal": None,
    "array": None,
    "nodes": None,
    "ntasks": None,
    "ntasks-per-node": None,
    "cpus-per-task": None,
    "mem-per-cpu": None,
    "parsable": None,
    "wait": None,
    "test-only": None,
    "partition": "remote.partition",
    "nodelist": "remote.nodelist",
    "gres": "remote.gres",
    "account": "remote.account",
    "constraint": "remote.constraint",
    "qos": "remote.qos",
    "mail-type": "remote.mail_type",
    "mail-user": "remote.mail_user",
    "nice": "remote.nice",
}

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
    # The optional settings below omit their #SBATCH directive when empty.
    "remote.account": (
        "PYRITE_REMOTE_ACCOUNT",
        re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._-]*)?"),
        "a SLURM account name such as pi_lab, or empty for the site default",
    ),
    "remote.constraint": (
        "PYRITE_REMOTE_CONSTRAINT",
        # SLURM feature expression: names joined by & | , with [] () and *count.
        re.compile(r"(?:[A-Za-z0-9_.:\[\]()|&,*-]+)?"),
        "a SLURM feature expression such as 'A100|H200', or empty for none",
    ),
    "remote.max_time": (
        "PYRITE_REMOTE_MAX_TIME",
        re.compile(r"(?:\d+(?:-\d+)?(?::\d+){0,2})?"),
        "a SLURM time limit such as 1-00:00:00 or 1440, or empty for no cap",
    ),
    "remote.qos": (
        "PYRITE_REMOTE_QOS",
        re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._-]*)?"),
        "a SLURM QOS name such as normal or highp, or empty for the site default",
    ),
    "remote.mail_type": (
        "PYRITE_REMOTE_MAIL_TYPE",
        re.compile(rf"(?:{_MAIL_TYPE}(?:,{_MAIL_TYPE})*)?"),
        "comma-separated SLURM mail events such as END,FAIL "
        f"({', '.join(_MAIL_TYPES)}), or empty for no email",
    ),
    "remote.mail_user": (
        "PYRITE_REMOTE_MAIL_USER",
        re.compile(r"(?:[A-Za-z0-9._%+-]+(?:@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)?)?"),
        "an email address or cluster username, or empty for the submitting user",
    ),
    "remote.sbatch_extra": (
        "PYRITE_REMOTE_SBATCH_EXTRA",
        re.compile(rf"(?:{_SBATCH_ITEM}(?: {_SBATCH_ITEM})*)?"),
        "space-separated sbatch long options such as '--reservation=lab --exclude=node3', "
        "or empty for none",
    ),
    # Not a batch directive: a plain-word command prefix (no shell operators or
    # quotes), so it is safe unquoted in the ssh command line.
    "remote.nice": (
        "PYRITE_REMOTE_NICE",
        re.compile(r"\d{1,9}"),
        "a non-negative SLURM nice value such as 10000, or 0 for none",
    ),
    "remote.host_exec": (
        "PYRITE_REMOTE_HOST_EXEC",
        re.compile(r"(?:[A-Za-z0-9_./:=,@%+-]+(?: [A-Za-z0-9_./:=,@%+-]+)*)?"),
        "a command prefix such as 'srun --time=00:30:00 --mem=8G', or empty",
    ),
}
# ``remote.nodelist`` value meaning "let SLURM pick any node in the partition".
NODELIST_ANY = "any"


def validate_slurm_setting(key: str, value: str) -> str:
    """Validate one SLURM profile value; accepted values are safe unquoted in ``#SBATCH``."""
    _env_name, pattern, expected = _SLURM_SETTINGS[key]
    if key == "remote.gpu_vendor":
        value = value.strip().lower()
    elif key == "remote.mail_type":
        value = value.replace(" ", "").upper()
    elif key == "remote.sbatch_extra":
        value = " ".join(value.split())
    if pattern.fullmatch(value) is None:
        raise ValueError(f"expected {expected}")
    if key == "remote.sbatch_extra":
        _reject_owned_sbatch_options(value)
    return value


def _reject_owned_sbatch_options(value: str) -> None:
    for item in value.split():
        name = item[2:].partition("=")[0]
        if name in _SBATCH_OWNED:
            owner = _SBATCH_OWNED[name]
            hint = f"; set {owner} instead" if owner else "; PyRITE sets it for job bookkeeping"
            raise ValueError(f"--{name} is not allowed in sbatch_extra{hint}")


def _slurm_setting(key: str, override: str | None) -> str:
    from ..console import config as cli_config

    selected = _selected_target()
    if selected is not None:
        value = getattr(selected, key.split(".")[1])
        return "" if key == "remote.nodelist" and value == NODELIST_ANY else value
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


def slurm_account() -> str | None:
    """Return validated SLURM account (``remote.account``), or None for the site default."""
    return _slurm_setting("remote.account", SLURM_ACCOUNT) or None


def slurm_constraint() -> str | None:
    """Return validated SLURM node-feature expression (``remote.constraint``), or None."""
    return _slurm_setting("remote.constraint", SLURM_CONSTRAINT) or None


def slurm_max_time() -> str | None:
    """Return the target's SLURM wall-time cap (``remote.max_time``), or None for no cap."""
    return _slurm_setting("remote.max_time", SLURM_MAX_TIME) or None


def slurm_qos() -> str | None:
    """Return validated SLURM quality of service (``remote.qos``), or None for the default."""
    return _slurm_setting("remote.qos", SLURM_QOS) or None


def slurm_sbatch_extra() -> tuple[str, ...]:
    """Return extra validated ``#SBATCH`` options (``remote.sbatch_extra``)."""
    return tuple(_slurm_setting("remote.sbatch_extra", SLURM_SBATCH_EXTRA).split())


def slurm_mail_type() -> str | None:
    """Return SLURM email events (``remote.mail_type``), or None for no notifications."""
    return _slurm_setting("remote.mail_type", SLURM_MAIL_TYPE) or None


def slurm_mail_user() -> str | None:
    """Return the SLURM email recipient (``remote.mail_user``), or None for the submitter."""
    return _slurm_setting("remote.mail_user", SLURM_MAIL_USER) or None


def slurm_nice() -> int:
    """Return the ``--nice`` for chunked/background submissions (``remote.nice``)."""
    return int(_slurm_setting("remote.nice", SLURM_NICE))


def sbatch_submit_command(*, nice: bool = False) -> str:
    """Return ``sbatch --parsable``, plus the target's ``--nice`` when ``nice`` and nonzero.

    The default 10000 lets a shared single-GPU box run others' jobs between
    slices. On an age-weighted queue it instead costs queue position: on a
    site where age weight W saturates at max age A, it is 10000/W of A.
    """
    value = slurm_nice() if nice else 0
    return f"sbatch --parsable --nice={value}" if value else "sbatch --parsable"


def host_exec() -> str | None:
    """Return the prefix for host-side PyRITE Python (``remote.host_exec``), or None."""
    return _slurm_setting("remote.host_exec", HOST_EXEC) or None


def host_run_command(extra_env: str = "") -> str:
    """Return ``<runtime env> <pixi> run --as-is -e <env>`` for host-side PyRITE outside a job.

    With ``host_exec`` (e.g. ``srun --time=00:30:00``) the command runs inside
    that allocation instead of on the login node; ``/usr/bin/env`` carries the
    runtime variables into the launched task regardless of the scheduler's
    export policy. It is absolute because launchers resolve a bare ``env``
    through the user's PATH, where some installers leave a
    non-executable ``~/.local/bin/env`` shell snippet that shadows the real one.
    """
    assignments = f"{remote_runtime_env()} {extra_env}".rstrip()
    command = f"{assignments} {shell_remote_run()}"
    prefix = host_exec()
    return f"{prefix} /usr/bin/env {command}" if prefix else command


def slurm_time_seconds(value: str) -> int | None:
    """Return SLURM ``--time`` syntax in seconds; None for ``UNLIMITED``/``infinite``.

    Accepts ``M``, ``M:S``, ``H:M:S``, ``D-H``, ``D-H:M`` and ``D-H:M:S``.
    """
    if value.upper() in ("UNLIMITED", "INFINITE"):
        return None
    days, _, clock = value.rpartition("-")
    parts = [int(part) for part in clock.split(":")]
    if days:
        hours, minutes, seconds = (parts + [0, 0])[:3]
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours, (minutes, seconds) = 0, (parts + [0])[:2]
    return ((int(days or 0) * 24 + hours) * 60 + minutes) * 60 + seconds


def remote_path(*parts: str) -> str:
    """Build raw remote path from validated checkout root and trusted components."""
    root = remote_dir().rstrip("/") or "/"
    suffix = "/".join(part.strip("/") for part in parts)
    return f"{root}/{suffix}" if suffix and root != "/" else f"{root}{suffix}"


def remote_output_path(kind: str, *parts: str) -> str:
    """Box output path (cache is a compatibility alias into remote user cache)."""
    if kind == "cache":
        return remote_path("cache", *parts)
    return remote_path(OUTPUT_DIRNAME, kind, *parts)


def remote_output_migration_command() -> str:
    """Shell moving legacy box outputs into the output tree; run from REMOTE_DIR.

    Same rules as :func:`pyrite.console.outputs.migrate_legacy_outputs`: move
    only a directory holding more than ``.gitkeep`` files, never over an
    existing destination. Failed moves block further output operations.
    """
    moves = " && ".join(
        f"_pyrite_migrate {shell_arg(legacy)} {shell_arg(f'{OUTPUT_DIRNAME}/{target}')}"
        for legacy, target in LEGACY_LAYOUT
    )
    return (
        '_pyrite_migrate() { [ -d "$1" ] && [ ! -L "$1" ] || return 0; '
        '[ -n "$(find "$1" ! -type d ! -name .gitkeep -print -quit)" ] || return 0; '
        'if [ -e "$2" ] || [ -L "$2" ]; then '
        'echo "warning: legacy $1 ignored; $2 already exists. Merge or remove the legacy directory." >&2; '
        "return 0; fi; "
        'mkdir -p "$(dirname "$2")" || return 1; '
        'if [ "$(stat -c %d "$1")" = "$(stat -c %d "$(dirname "$2")")" ]; then '
        'mv -T --no-clobber -- "$1" "$2" && [ ! -e "$1" ] || return 1; '
        'else _pyrite_tmp=$(mktemp -d "$(dirname "$2")/.pyrite-migrate.XXXXXX") || return 1; '
        'if cp -a -- "$1" "$_pyrite_tmp/data" '
        '&& diff -r --no-dereference -- "$1" "$_pyrite_tmp/data" >&2 '
        '&& mv -T --no-clobber -- "$_pyrite_tmp/data" "$2" '
        '&& [ ! -e "$_pyrite_tmp/data" ]; then '
        'rm -rf -- "$1" "$_pyrite_tmp" || return 1; '
        'else rm -rf -- "$_pyrite_tmp"; return 1; fi; fi; '
        'echo "moved $1 -> $2" >&2; }; '
        f"{moves} || exit 1; "
        'for _pyrite_job in jobs/*; do [ -d "$_pyrite_job" ] || continue; '
        f'_pyrite_migrate "$_pyrite_job/performance" "{OUTPUT_DIRNAME}/performance/${{_pyrite_job##*/}}" || exit 1; '
        "done"
    )


def remote_sync_stamp_path() -> str:
    """Absolute path of the code-identity stamp `sync` writes on the box."""
    return remote_path(SYNC_STAMP_NAME)


def remote_sync_lock_path() -> str:
    """Absolute path of the flock file guarding the shared checkout."""
    return remote_path(SYNC_LOCK_NAME)


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


def shell_remote_pixi() -> str:
    return shell_word(remote_pixi())


def shell_remote_run() -> str:
    """``<pixi> run --as-is -e <env>``: run in the installed job environment, never re-solve."""
    return f"{shell_remote_pixi()} run --as-is -e {remote_environment()}"


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


def rsync_remote_path(path: str) -> str:
    """Render validated data host plus one raw absolute remote path for ``rsync -s``.

    The host is :func:`transfer_host`: bulk data goes to the target's transfer
    node when one is set, else to the login host.
    """
    _reject_controls("remote rsync path", path)
    if not PurePosixPath(path).is_absolute():
        raise SystemExit(f"invalid remote rsync path={path!r}: expected absolute POSIX path")
    return f"{transfer_host()}:{path}"


def scp_remote_path(path: str) -> str:
    """Render validated data host plus one quoted remote path for SCP argv."""
    _reject_controls("remote SCP path", path)
    if not PurePosixPath(path).is_absolute():
        raise SystemExit(f"invalid remote SCP path={path!r}: expected absolute POSIX path")
    return f"{transfer_host()}:{shell_arg(path)}"


# Invocation-local selection keeps concurrent callers and nested commands isolated.
_TARGET: ContextVar[object] = ContextVar("pyrite_remote_target", default=None)


def _selected_target():
    from .targets import resolve_target, selected_name

    selected = _TARGET.get()
    if selected is not None:
        return selected
    name = selected_name()
    return resolve_target(name) if name else None


def _target_field(name, fallback):
    selected = _selected_target()
    return getattr(selected, name) if selected is not None else fallback


def remote_host() -> str:
    selected = _selected_target()
    return selected.host if selected is not None else _legacy_remote_host()


_ENVIRONMENT_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


def parse_environment(value: str) -> str:
    """Validate a pixi environment name (``[tool.pixi.environments]``); empty means the vendor's.

    The name is safe unquoted in a shell line.
    """
    value = value.strip()
    if value and _ENVIRONMENT_RE.fullmatch(value) is None:
        raise ValueError(
            f"invalid target environment: {value!r} is not a pixi environment name "
            "(lowercase letters, digits and -)"
        )
    return value


def remote_environment() -> str:
    """pixi environment jobs install and run: the target's ``environment``, else its GPU vendor's."""
    return parse_environment(_target_field("environment", "")) or remote_gpu_vendor()


def shell_remote_env_python() -> str:
    """The job environment's interpreter, for launchers (nsys, py-spy) that must exec it directly."""
    return shell_remote_path(".pixi", "envs", remote_environment(), "bin", "python")


#: Stored-target keys from the uv era (#1462), mapped to the field that replaced them.
LEGACY_TARGET_FIELDS = {"uv": "pixi", "extras": "environment"}


def transfer_host() -> str:
    """SSH alias that carries bulk data: the target's ``transfer_host``, else the login host.

    A transfer node (a site DTN) only moves files and reads digests; SLURM,
    locks and PyRITE Python stay on :func:`remote_host`. Both hosts must see
    ``remote_dir`` at one absolute path; a ``~/`` checkout is checked by
    comparing the two login homes (one cached ssh round trip each).
    """
    login = remote_host()
    selected = _selected_target()
    data = getattr(selected, "transfer_host", "") if selected is not None else ""
    if not data or data == login:
        return login
    raw = _target_field("remote_dir", REMOTE_DIR)
    if (raw == "~" or raw.startswith("~/")) and _remote_home(data) != _remote_home(login):
        raise SystemExit(
            f"transfer host {data!r} and login host {login!r} have different homes "
            f"({_remote_home(data)} vs {_remote_home(login)}), so {raw!r} is not one "
            "shared directory; set the target's remote_dir to an absolute path both mount"
        )
    return data


def uses_transfer_host() -> bool:
    """Whether bulk data goes to a host other than the login host.

    Decided from the selected target alone when it names no transfer host, so
    callers that only branch on it contact no host and need none configured.
    """
    if not getattr(_selected_target(), "transfer_host", ""):
        return False
    return transfer_host() != remote_host()


@contextmanager
def use_target(target=None):
    """Bind one resolved target for the lifetime of an invocation."""
    from .targets import resolve_target

    token = _TARGET.set(resolve_target(target))
    try:
        yield _TARGET.get()
    finally:
        _TARGET.reset(token)


@dataclass(frozen=True)
class RemoteTarget:
    """Immutable SSH/SLURM destination. Paths use the remote host's POSIX home.

    Construction validates without network I/O. ``~/`` paths expand when used.
    ``nodelist='any'`` lets SLURM choose. Targets do not enter physics identities.
    """

    host: str
    remote_dir: str = "~/pyrite"
    pixi: str = "~/.pixi/bin/pixi"
    gpu_vendor: str = "nvidia"
    partition: str = "gpu"
    nodelist: str = "any"
    gres: str = "gpu:1"
    description: str = ""
    name: str | None = None
    # Optional site policy; empty omits the directive (appended for positional compatibility).
    account: str = ""
    constraint: str = ""
    max_time: str = ""
    host_exec: str = ""
    nice: str = "10000"
    # SSH alias of a data-transfer node sharing remote_dir; empty uses ``host``.
    transfer_host: str = ""
    # SLURM email notifications; empty mail_type sends none, empty mail_user is the submitter.
    mail_type: str = ""
    mail_user: str = ""
    qos: str = ""
    # Additional ``#SBATCH`` long options; options PyRITE owns are rejected.
    sbatch_extra: str = ""
    # pixi environment each job installs and runs in; empty uses the GPU
    # vendor's environment (``nvidia``/``amd``/``intel``).
    environment: str = ""

    def __post_init__(self):
        from .targets import validate_name

        for field in (
            "host",
            "remote_dir",
            "pixi",
            "gpu_vendor",
            "partition",
            "nodelist",
            "gres",
            "description",
            "account",
            "constraint",
            "max_time",
            "host_exec",
            "nice",
            "transfer_host",
            "mail_type",
            "mail_user",
            "qos",
            "sbatch_extra",
            "environment",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError(f"invalid target {field}: expected printable string")
        validate_remote_target(self.host)
        if self.transfer_host:
            validate_remote_target(self.transfer_host)
        object.__setattr__(self, "environment", parse_environment(self.environment))
        if self.name is not None:
            validate_name(self.name)
        if not (PurePosixPath(self.remote_dir).is_absolute() or self.remote_dir.startswith("~/")):
            raise ValueError("remote_dir must be an absolute POSIX path or ~/path")
        pixi = self.pixi[1:] if self.pixi.startswith("~/") else self.pixi
        if not (_EXECUTABLE_RE.fullmatch(pixi) or _ABS_EXECUTABLE_RE.fullmatch(pixi)):
            raise ValueError("pixi must be an executable name, absolute POSIX path or ~/path")
        for field in (
            "gpu_vendor",
            "partition",
            "nodelist",
            "gres",
            "account",
            "constraint",
            "max_time",
            "host_exec",
            "nice",
            "mail_type",
            "mail_user",
            "qos",
            "sbatch_extra",
        ):
            object.__setattr__(
                self, field, validate_slurm_setting(f"remote.{field}", getattr(self, field))
            )
