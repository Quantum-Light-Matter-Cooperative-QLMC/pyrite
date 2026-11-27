"""Offline dependency packs and separately staged checkout wheels for SSH queues."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from ..paths import cache_dir
from . import config, transport
from .reporting import report


@dataclass(frozen=True)
class PackIdentity:
    lock_hash: str
    environment: str
    platform: str
    code_digest: str

    @property
    def key(self) -> str:
        return f"v1/{self.lock_hash}/{self.environment}/{self.platform}"

    @property
    def wheel_key(self) -> str:
        return f"{self.key}/code/{self.code_digest}"


def lock_selection(lock_bytes: bytes, environment: str) -> tuple[str, str]:
    """Resolve the Linux lock row and reject every omitted non-project dependency."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "pack mode requires PyYAML on the submitter; use the Pixi default env"
        ) from exc
    lock = yaml.safe_load(lock_bytes)
    try:
        rows = lock["environments"][environment]["packages"]
    except KeyError as exc:
        raise SystemExit(f"no locked pixi environment {environment!r}") from exc
    platforms = [name for name in ("linux-64", "linux-64-cuda") if name in rows]
    if len(platforms) != 1:
        raise SystemExit(f"pack mode needs one Linux x86-64 lock row for {environment!r}")
    platform = platforms[0]
    pypi_index = {p["pypi"]: p for p in lock["packages"] if "pypi" in p}
    skipped = []
    for entry in rows[platform]:
        if "pypi" not in entry:
            continue
        location = entry["pypi"]
        package = pypi_index[location]
        if package["name"] == "pyrite-mc" and location in (".", "./"):
            continue
        if not urlsplit(location).path.endswith(".whl"):
            skipped.append(f"{package['name']} ({location})")
    if skipped:
        raise SystemExit(
            "pack mode cannot omit locked non-wheel dependencies: " + ", ".join(skipped)
        )
    return hashlib.sha256(lock_bytes).hexdigest(), platform


def pack_identity() -> PackIdentity:
    environment = config.remote_environment()
    lock = config.LOCAL_ROOT / "pixi.lock"
    if not lock.is_file():
        raise SystemExit("pack mode requires a source checkout containing pixi.lock")
    lock_hash, platform = lock_selection(lock.read_bytes(), environment)
    code_digest = transport._payload_digest(transport._sync_entries())
    return PackIdentity(lock_hash, environment, platform, code_digest)


def _remote_cache(*parts: str) -> str:
    return config.remote_path("cache", "environment-packs", *parts)


def _local_cache(*parts: str) -> Path:
    return cache_dir() / "environment-packs" / Path(*parts)


def _build_dependencies(identity: PackIdentity, output: Path) -> None:
    downloads = _local_cache("downloads")
    downloads.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pixi-pack",
            "-e",
            identity.environment,
            "-p",
            identity.platform,
            "--ignore-pypi-non-wheel",
            "--use-cache",
            str(downloads),
            "--output-file",
            str(output),
            str(config.LOCAL_ROOT / "pyproject.toml"),
        ],
        check=True,
    )
    actual_hash = hashlib.sha256((config.LOCAL_ROOT / "pixi.lock").read_bytes()).hexdigest()
    if actual_hash != identity.lock_hash:
        raise SystemExit("pixi.lock changed while packing; retry submission")


def _build_wheel(identity: PackIdentity, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pyrite-wheel-") as temp:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", temp],
            cwd=config.LOCAL_ROOT,
            check=True,
        )
        wheels = list(Path(temp).glob("*.whl"))
        if len(wheels) != 1 or not wheels[0].name.endswith("-py3-none-any.whl"):
            raise SystemExit("expected one platform-independent PyRITE wheel")
        if transport._payload_digest(transport._sync_entries()) != identity.code_digest:
            raise SystemExit("checkout changed while building its wheel; retry submission")
        with tarfile.open(output, "w") as archive:
            archive.add(wheels[0], arcname=wheels[0].name)


def _cache_file(local: Path, build) -> None:
    if local.is_file() and local.stat().st_size:
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local.parent) as temp:
        output = Path(temp) / local.name
        start = time.monotonic()
        build(output)
        if not output.is_file() or not output.stat().st_size:
            raise SystemExit(f"environment artifact builder produced no {local.name}")
        report(
            f"built {local.name}: {output.stat().st_size} bytes in {time.monotonic() - start:.1f} s",
            file=sys.stderr,
        )
        os.replace(output, local)


def _stage_artifact(local: Path, remote: str, build) -> None:
    host = config.transfer_host()
    quoted = config.shell_word(remote)
    present = transport._ssh_capture(
        f"if test -s {quoted}; then echo present; fi", host=host
    ).strip()
    if present == "present":
        return
    _cache_file(local, build)
    directory = config.shell_word(str(PurePosixPath(remote).parent))
    # Remote paths are POSIX, even when the submitter is on Windows.
    temporary = f"{remote}.{uuid.uuid4().hex}.part"
    temp_word = config.shell_word(temporary)
    transport._ssh_capture(f"mkdir -p {directory}", host=host)
    try:
        transport._run(
            config.scp_argv(str(local), config.scp_remote_path(temporary)),
            label=f"Uploading {local.name}...",
        )
        digest = transport._local_sha256(local)
        transport._ssh_capture(
            f'test "$(sha256sum {temp_word} | cut -d " " -f 1)" = {digest}'
            f" && mv -f -- {temp_word} {quoted}",
            host=host,
        )
    finally:
        try:
            transport._ssh_capture(f"rm -f -- {temp_word}", host=host)
        except (OSError, SystemExit):
            # Best-effort cleanup must not hide the original upload failure.
            pass


def stage_pack() -> None:
    """Stage immutable dependency and checkout artifacts before reserving a job."""
    identity = pack_identity()
    stamp = config.shell_remote_path(config.SYNC_STAMP_NAME)
    lock = config.shell_remote_path("pixi.lock")
    transport._ssh_capture(
        f"test \"$(sed -n 's/^code_digest: //p' {stamp})\" = {identity.code_digest}"
        f' && test "$(sha256sum {lock} | cut -d " " -f 1)" = {identity.lock_hash}'
        ' || { echo "pack mode requires the matching synced checkout; run remote sync" >&2; exit 1; }'
    )
    for key, name, builder in (
        (identity.key, "dependencies.tar", _build_dependencies),
        (identity.wheel_key, "project-wheel.tar", _build_wheel),
    ):
        _stage_artifact(
            _local_cache(key, name),
            _remote_cache(key, name),
            lambda output, builder=builder: builder(identity, output),
        )


def unpack_block() -> str:
    """Render shared-prefix creation and activation, with no online fallback."""
    identity = pack_identity()
    archive = config.shell_word(_remote_cache(identity.key, "dependencies.tar"))
    wheel = config.shell_word(_remote_cache(identity.wheel_key, "project-wheel.tar"))
    prefix = config.shell_word(_remote_cache(identity.wheel_key, "prefix"))
    stamp = config.shell_remote_path(config.SYNC_STAMP_NAME)
    lock = config.shell_remote_path("pixi.lock")
    return f'''export PYRITE_PACK_PREFIX={prefix}
pyrite_pack_failed() {{
  echo "FAILED (pixi unpack) $(date -Is)" > "$JOBDIR/state"
  echo "pack provisioning failed: $*" >> "$JOBDIR/log"
  exit 1
}}
[ "$(sha256sum {lock} | cut -d ' ' -f 1)" = {identity.lock_hash} ] || pyrite_pack_failed "lock hash mismatch"
[ "$(sed -n 's/^code_digest: //p' {stamp})" = {identity.code_digest} ] || pyrite_pack_failed "synced code mismatch"
[ -s {archive} ] && [ -s {wheel} ] || pyrite_pack_failed "missing dependency pack or project wheel"
mkdir -p "$(dirname "$PYRITE_PACK_PREFIX")" || pyrite_pack_failed "cache directory"
pack_start_ns=$(date +%s%N)
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
# Pixi global binaries are user-local; SLURM need not inherit interactive PATH.
export PATH="$PATH:$HOME/.pixi/bin"
(
  flock -w 3600 -x 9 || exit 1
  if [ ! -f "$PYRITE_PACK_PREFIX/.ready" ]; then
    command -v pixi-unpack >/dev/null || {{ echo "pixi-unpack is not on PATH"; exit 1; }}
    rm -rf -- "$PYRITE_PACK_PREFIX" || exit 1
    mkdir -p "$PYRITE_PACK_PREFIX" || exit 1
    work=$(mktemp -d "${{SLURM_TMPDIR:-${{TMPDIR:-/tmp}}}}/pyrite-pack.XXXXXXXX") || exit 1
    trap 'rm -rf -- "$work"' EXIT
    tar -xf {archive} -C "$work" || exit 1
    mkdir -p "$work/pypi" || exit 1
    tar -xf {wheel} -C "$work/pypi" || exit 1
    pixi-unpack --shell bash --output-directory "$PYRITE_PACK_PREFIX" "$work" || exit 1
    [ -x "$PYRITE_PACK_PREFIX/env/bin/python" ] || exit 1
    [ -f "$PYRITE_PACK_PREFIX/activate.sh" ] || exit 1
    : > "$PYRITE_PACK_PREFIX/.ready" || exit 1
  else
    echo "reusing unpacked prefix"
  fi
) 9>"$PYRITE_PACK_PREFIX.lock" >> "$JOBDIR/log" 2>&1 || pyrite_pack_failed "unpack failed; see log"
set +u
source "$PYRITE_PACK_PREFIX/activate.sh" >> "$JOBDIR/log" 2>&1 || pyrite_pack_failed "activation failed"
set -u
export PATH="$PYRITE_PACK_PREFIX/env/bin:$PATH"
"$PYRITE_PACK_PREFIX/env/bin/python" -c 'import pyrite, pathlib, sys; assert pathlib.Path(pyrite.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())' >> "$JOBDIR/log" 2>&1 || pyrite_pack_failed "project import failed"
printf 'environment: pack lock=%s env=%s platform=%s code=%s\\n' {identity.lock_hash} {identity.environment} {identity.platform} {identity.code_digest} >> "$JOBDIR/log"
printf 'timing: pixi unpack %d ms\\n' "$((($(date +%s%N) - pack_start_ns) / 1000000))" >> "$JOBDIR/log"'''
