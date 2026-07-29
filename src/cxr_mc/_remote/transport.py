"""SSH/SCP transport primitives and code-sync for the remote subsystem."""

import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from . import config, presentation

_TRACE_ARG_LIMIT = 100
_SYNC_EXCLUDED_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_SYNC_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _run(cmd, **kw):
    parts = [
        a if len(a) <= _TRACE_ARG_LIMIT else a[:_TRACE_ARG_LIMIT] + "...<truncated>" for a in cmd
    ]
    print("+", " ".join(parts), flush=True)
    subprocess.run(cmd, check=True, **kw)


def _ssh_capture(remote_cmd):
    """Run a remote command over ssh and return its stdout (text). Prints the
    box's stderr and aborts on a nonzero exit.

    Decode as UTF-8 explicitly: the Linux box emits UTF-8 (job logs carry tqdm's
    block-glyph progress bars, e.g. `████▌`), but `text=True` alone would decode
    with the Windows locale (cp1252), which chokes on those bytes -- so `status`,
    `jobs`, and `logs` would crash mid-read. `errors="replace"` keeps any stray
    non-UTF-8 byte from aborting the whole command."""
    r = subprocess.run(
        ["ssh", "-n", config.remote_host(), remote_cmd],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ssh command failed (exit {r.returncode})")
    return r.stdout


def _ssh_download(remote_cmd: str, destination: Path) -> None:
    """Stream one remote command's stdout directly into a local file."""
    destination = Path(destination)
    try:
        with destination.open("wb") as output:
            _run(["ssh", "-n", config.remote_host(), remote_cmd], stdout=output)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _local_sha256(path: Path) -> str | None:
    """Return a checkpoint's SHA-256 digest, or ``None`` when it is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha256(path: str) -> str:
    """Return the SHA-256 digest of one remote checkpoint artifact."""
    output = _ssh_capture(f"sha256sum {config.shell_arg(path)}").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\S+", output)
    if match is None:
        raise SystemExit(f"invalid sha256sum output for remote artifact {path!r}")
    return match.group(1).lower()


def _check_shell_tokens(tokens):
    """Reject tokens that are unsafe to interpolate into remote shell commands."""
    bad = [token for token in tokens if not presentation._SHELL_TOKEN_RE.fullmatch(token)]
    if bad:
        raise SystemExit(
            f"invalid remote shell token(s) {bad}: expected only letters, digits, "
            "underscore, or hyphen"
        )


def _check_materials(materials):
    """Validate runnable material keys for remote run operations."""
    from ..materials import CATALOG

    _check_shell_tokens(materials)
    unknown = [material for material in materials if material not in CATALOG.materials]
    if unknown:
        raise SystemExit(
            f"unknown material(s): {', '.join(unknown)}; valid: {', '.join(CATALOG.material_keys)}"
        )


def _add_to_tar(tar, local, arcname):
    """Add one local file to the tar. Text files (TEXT_EXTS) get CRLF->LF so the
    box receives LF-clean content -- no cosmetic `git status` diff there, so a
    later `git pull` isn't blocked. Other files are added verbatim."""
    if local.suffix.lower() in config.TEXT_EXTS:
        data = local.read_bytes().replace(b"\r\n", b"\n")
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mtime = int(local.stat().st_mtime)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    else:
        tar.add(local, arcname=arcname)


def _sync_ignored(path: Path) -> bool:
    """Whether a generated local artifact must stay out of code-sync archives."""
    return (
        any(part in _SYNC_EXCLUDED_DIRS for part in path.parts)
        or path.suffix.lower() in _SYNC_EXCLUDED_SUFFIXES
    )


def sync_code():
    """Tar SYNC_PATHS up (CRLF->LF normalized for text, via _add_to_tar) and
    extract them over the repo on the box. Generated interpreter/tool caches
    are excluded: they are host-specific, unnecessary, and expensive to gzip.

    Normalizing line endings here keeps the edit-locally / run-remotely loop --
    it ships the current WORKING tree (no commit required) yet stays LF-clean, so
    the box's `git status` doesn't flag every synced .py as modified and a later
    `git pull` there isn't blocked. (The committed-state alternative -- `git push`
    from the laptop + `git fetch && git reset --hard origin/<branch>` on the box --
    would force a commit before every run, which this loop is built to avoid.)"""
    with tempfile.TemporaryDirectory() as td:
        tarpath = os.path.join(td, "cxr_code.tgz")
        with tarfile.open(tarpath, "w:gz") as t:
            for p in config.SYNC_PATHS:
                local = config.LOCAL_ROOT / p
                if not local.exists():
                    continue
                if local.is_dir():
                    for f in sorted(local.rglob("*")):
                        relative = f.relative_to(local)
                        if f.is_file() and not _sync_ignored(relative):
                            arc = (Path(p) / relative).as_posix()
                            _add_to_tar(t, f, arc)
                else:
                    _add_to_tar(t, local, p)
        _run(["scp", tarpath, config.scp_remote_path("/tmp/cxr_code.tgz")])
    # -n: redirect ssh's stdin from null. Without it, ssh.exe inherits the
    # interactive console stdin and its stdin-forwarding thread never sees EOF,
    # so the client hangs after the remote command (tar) has already exited.
    _run(
        [
            "ssh",
            "-n",
            config.remote_host(),
            f"mkdir -p {config.shell_remote_dir()} && cd {config.shell_remote_dir()} "
            "&& tar xzf /tmp/cxr_code.tgz && rm -f /tmp/cxr_code.tgz",
        ]
    )
