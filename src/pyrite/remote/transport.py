"""SSH/SCP transport primitives and code-sync for the remote subsystem."""

import hashlib
import io
import os
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..console import dashboard as presentation
from . import config

_TRACE_ARG_LIMIT = 100
_VERBOSE = False
_SYNC_EXCLUDED_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_SYNC_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@contextmanager
def verbose_ssh(enabled: bool):
    """Temporarily toggle raw ssh/scp command echoing for ``_run`` calls."""
    global _VERBOSE
    previous = _VERBOSE
    _VERBOSE = enabled
    try:
        yield
    finally:
        _VERBOSE = previous


def _run(cmd, *, label=None, **kw):
    """Run one ssh/scp command. Prints the raw command line when verbose is on
    (see ``verbose_ssh``); otherwise prints ``label`` if given, or nothing."""
    if _VERBOSE:
        parts = [
            a if len(a) <= _TRACE_ARG_LIMIT else a[:_TRACE_ARG_LIMIT] + "...<truncated>"
            for a in cmd
        ]
        print("+", " ".join(parts), flush=True, file=sys.stderr)
    elif label is not None:
        print(label, flush=True, file=sys.stderr)
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
        config.ssh_argv("-n", config.remote_host(), remote_cmd),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ssh command failed (exit {r.returncode})")
    return r.stdout


def _ssh_download(remote_cmd: str, destination: Path, *, label=None) -> None:
    """Stream one remote command's stdout directly into a local file."""
    destination = Path(destination)
    try:
        with destination.open("wb") as output:
            _run(
                config.ssh_argv("-n", config.remote_host(), remote_cmd), stdout=output, label=label
            )
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
    bad = [token for token in tokens if not presentation.SHELL_TOKEN_RE.fullmatch(token)]
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


def _normalized_text_bytes(local: Path) -> bytes:
    """Return a text payload file exactly as the box receives it (CRLF->LF)."""
    return local.read_bytes().replace(b"\r\n", b"\n")


def _add_to_tar(tar, local, arcname):
    """Add one local file to the tar. Text files (TEXT_EXTS) get CRLF->LF so the
    box receives LF-clean content -- no cosmetic `git status` diff there, so a
    later `git pull` isn't blocked. Other files are added verbatim."""
    if local.suffix.lower() in config.TEXT_EXTS:
        data = _normalized_text_bytes(local)
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


_ENERGY_GRID_ARTIFACT_ROOT = Path("src/pyrite/data/catalog/energy-grid-artifacts")


def _local_energy_grid_artifacts() -> dict[str, Path]:
    """Return verified-name local immutable objects eligible for code sync."""
    root = config.LOCAL_ROOT / _ENERGY_GRID_ARTIFACT_ROOT
    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in root.glob("[0-9a-f][0-9a-f]/*.json"):
        match = re.fullmatch(r"([0-9a-f]{64})\.json", path.name)
        if match is not None and match.group(1).startswith(path.parent.name):
            digest = _local_sha256(path)
            if digest is not None and digest == match.group(1):
                found[digest] = path
    return found


def _energy_grid_inventory_command() -> tuple[str, str]:
    """Return ``(remote_root, shell command)`` listing artifact digests."""
    remote_root = f"{config.remote_dir().rstrip('/')}/{_ENERGY_GRID_ARTIFACT_ROOT.as_posix()}"
    return remote_root, (
        f"if [ -d {config.shell_arg(remote_root)} ]; then "
        f"find {config.shell_arg(remote_root)} -type f -name '*.json' -exec sha256sum {{}} +; fi"
    )


def _parse_energy_grid_inventory(output: str, remote_root: str) -> frozenset[str]:
    """Keep only remote objects whose content digest matches their name."""
    verified: set[str] = set()
    for line in output.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+(.+)", line.strip())
        if match is None:
            raise SystemExit("invalid remote energy-grid artifact inventory")
        actual = match.group(1).lower()
        reported_path = match.group(2)
        expected_path = f"{remote_root}/{actual[:2]}/{actual}.json"
        if reported_path == expected_path:
            verified.add(actual)
    return frozenset(verified)


def _remote_energy_grid_artifacts() -> frozenset[str]:
    """Inventory remotely verified immutable objects in one SSH round trip."""
    remote_root, command = _energy_grid_inventory_command()
    return _parse_energy_grid_inventory(_ssh_capture(command), remote_root)


# --- xsgen tables -----------------------------------------------------------
#
# Tables are immutable ``<key>.npz`` + ``<key>.json`` pairs keyed by content
# identity, like the energy-grid artifacts above. The box's copy is judged by
# file digest, never mtime, so a truncated or edited file is replaced on the
# next sync. ``PYRITE_HOME`` is REMOTE_DIR on the box, so the selected table
# tier there is ``<REMOTE_DIR>/xsgen/tables``.

_XSGEN_TABLE_DIR = "xsgen/tables"
_XSGEN_NAME_RE = re.compile(r"[0-9a-f]{64}\.(?:npz|json)")
_XSGEN_SECTION_MARK = "---pyrite-xsgen-inventory---"


def _local_xsgen_tables() -> dict[str, Path]:
    """Return ``{file name: path}`` for every table file a sync must guarantee.

    That is every locally stored non-packaged table (pinned releases, muffin-tin
    solids, other generated tables; packaged ones already ride with
    ``src/pyrite``). Refuses when a pinned release table is absent or differs
    locally, since nothing could be shipped for it.
    """
    from ..xsgen.store import packaged_table_dir, search_dirs
    from ..xsgen.verify import OK, TABLE_CODES, verify_pinned

    failed = [check for check in verify_pinned(TABLE_CODES) if check.status != OK]
    if failed:
        codes = sorted({check.code for check in failed})
        detail = ", ".join(f"{check.code} {check.label} ({check.status})" for check in failed[:6])
        more = f" and {len(failed) - 6} more" if len(failed) > 6 else ""
        raise SystemExit(
            f"refusing to sync: {len(failed)} pinned xsgen table(s) are not intact locally: "
            f"{detail}{more}.\n"
            + "\n".join(f"  fix: pyrite tables fetch {code} --archive PATH" for code in codes)
        )
    found: dict[str, Path] = {}
    claimed: set[str] = set()
    for root in search_dirs():
        if root == packaged_table_dir() or not root.is_dir():
            continue
        for manifest in sorted(root.glob("*.json")):
            payload = manifest.with_suffix(".npz")
            key = manifest.stem
            if (
                key in claimed
                or not _XSGEN_NAME_RE.fullmatch(manifest.name)
                or not payload.is_file()
            ):
                continue
            claimed.add(key)
            found[payload.name] = payload
            found[manifest.name] = manifest
    return found


def _xsgen_inventory_command() -> str:
    """Shell that migrates legacy-tier tables, then digests the synced tier.

    Migration is non-clobbering (``cp -n``) and payload-before-manifest, so an
    interrupted copy leaves a payload without a manifest, which the store
    ignores. Migrated files are then judged by the same digest inventory as any
    other, so a bad legacy copy is replaced by the shipped one.
    """
    dest = config.shell_arg(f"{config.remote_dir().rstrip('/')}/{_XSGEN_TABLE_DIR}")
    legacy = '"${XDG_DATA_HOME:-$HOME/.local/share}/pyrite/xsgen/tables"'
    return (
        f"mkdir -p {dest} && if [ -d {legacy} ]; then "
        f'for f in {legacy}/*.npz {legacy}/*.json; do [ -f "$f" ] && cp -n "$f" {dest}/; done; '
        f"fi; cd {dest} && "
        "find . -maxdepth 1 -type f \\( -name '*.npz' -o -name '*.json' \\) -exec sha256sum {} +"
    )


def _parse_xsgen_inventory(output: str) -> dict[str, str]:
    """Parse ``sha256sum`` lines into ``{file name: digest}``, failing closed."""
    digests: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(?:\./)?(.+)", line.strip())
        if match is None:
            raise SystemExit("invalid remote xsgen table inventory")
        if _XSGEN_NAME_RE.fullmatch(match.group(2)):
            digests[match.group(2)] = match.group(1).lower()
    return digests


def _remote_inventories(
    *, artifacts: bool, tables: bool, datasets: bool = False
) -> tuple[frozenset[str], dict[str, str], dict[str, str]]:
    """Inventory energy-grid artifacts, xsgen tables and datasets in one SSH round trip.

    Each section runs in its own subshell, so one section's ``cd`` cannot
    move the next.
    """
    commands: list[str] = []
    remote_root = ""
    if artifacts:
        remote_root, command = _energy_grid_inventory_command()
        commands.append(command)
    if tables:
        commands.append(f"echo {_XSGEN_SECTION_MARK}")
        commands.append(f"( {_xsgen_inventory_command()} )")
    if datasets:
        commands.append(f"echo {_DATASET_SECTION_MARK}")
        commands.append(f"( {_dataset_inventory_command()} )")
    if not commands:
        return frozenset(), {}, {}
    output = _ssh_capture("; ".join(commands))
    head, _, dataset_part = output.partition(_DATASET_SECTION_MARK + "\n")
    if datasets and _DATASET_SECTION_MARK not in output:
        raise SystemExit("invalid remote dataset inventory")
    artifact_part, _, table_part = head.partition(_XSGEN_SECTION_MARK + "\n")
    if tables and _XSGEN_SECTION_MARK not in head:
        raise SystemExit("invalid remote xsgen table inventory")
    found_artifacts = (
        _parse_energy_grid_inventory(artifact_part, remote_root) if artifacts else frozenset()
    )
    return (
        found_artifacts,
        _parse_xsgen_inventory(table_part) if tables else {},
        _parse_dataset_inventory(dataset_part) if datasets else {},
    )


# --- fetched datasets (EEDL, EADL) -------------------------------------------
#
# Single pinned files, installed on the box at ``<REMOTE_DIR>/datasets/<name>/``
# (``PYRITE_HOME`` is REMOTE_DIR there). Judged by content digest like the
# tables; a box that has no internet receives them only this way. A box synced
# before #263 still holds them under ``src/pyrite/data/characteristic_cross_sections``
# (code sync overlays, it does not delete), so the inventory first copies those
# into place without clobbering; the digest check then decides what ships.

_DATASET_SECTION_MARK = "---pyrite-dataset-inventory---"
_LEGACY_DATASET_DIR = "src/pyrite/data/characteristic_cross_sections"


def _dataset_arcnames() -> dict[str, str]:
    """Return ``{dataset name: arcname below REMOTE_DIR}``."""
    from ..datasets import DATASETS

    return {name: f"datasets/{name}/{d.filename}" for name, d in DATASETS.items()}


def _local_datasets() -> dict[str, Path]:
    """Return ``{arcname: local path}`` for every pinned dataset a sync ships.

    Refuses when one is missing or differs locally: a run on the box needs
    every one of them, and a box without internet cannot fetch them itself.
    """
    from ..datasets import OK, dataset_path, verify_dataset

    arcnames = _dataset_arcnames()
    failed = [(name, status) for name in arcnames if (status := verify_dataset(name)) != OK]
    if failed:
        detail = ", ".join(f"{name} ({status})" for name, status in failed)
        raise SystemExit(
            f"refusing to sync: pinned dataset(s) are not intact locally: {detail}.\n"
            + "\n".join(f"  fix: pyrite tables fetch {name}" for name, _ in failed)
        )
    return {arc: dataset_path(name) for name, arc in arcnames.items()}


def _dataset_inventory_command() -> str:
    """Shell that migrates legacy code-synced copies, then digests the datasets."""
    from ..datasets import DATASETS

    root = config.shell_arg(config.remote_dir().rstrip("/"))
    migrate = " ".join(
        f"if [ -f {_LEGACY_DATASET_DIR}/{d.filename} ] && [ ! -e datasets/{name}/{d.filename} ]; "
        f"then mkdir -p datasets/{name} && cp -n {_LEGACY_DATASET_DIR}/{d.filename} "
        f"datasets/{name}/{d.filename}; fi;"
        for name, d in DATASETS.items()
    )
    return (
        f"mkdir -p {root} && cd {root} && {migrate} "
        "if [ -d datasets ]; then find datasets -mindepth 2 -maxdepth 2 -type f "
        "-exec sha256sum {} +; fi"
    )


def _parse_dataset_inventory(output: str) -> dict[str, str]:
    """Parse ``sha256sum`` lines into ``{arcname: digest}``, failing closed."""
    known = set(_dataset_arcnames().values())
    digests: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(?:\./)?(.+)", line.strip())
        if match is None:
            raise SystemExit("invalid remote dataset inventory")
        if match.group(2) in known:
            digests[match.group(2)] = match.group(1).lower()
    return digests


def _datasets_to_ship(
    local: dict[str, Path], local_digests: dict[str, str], remote: dict[str, str]
) -> list[tuple[str, Path]]:
    """Return ``(arcname, path)`` for datasets the box lacks or holds differently."""
    return [(arc, local[arc]) for arc in sorted(local) if remote.get(arc) != local_digests[arc]]


def _xsgen_tables_digest(local_digests: dict[str, str]) -> str:
    """Identify the guaranteed table set by names and content digests."""
    digest = hashlib.sha256()
    for name in sorted(local_digests):
        digest.update(f"{name}\0{local_digests[name]}\0".encode())
    return digest.hexdigest()


def _xsgen_to_ship(local: dict[str, Path], local_digests: dict[str, str], remote: dict[str, str]):
    """Return local table files the box lacks or holds with different content.

    A key ships as a pair if either of its files differs, payload first.
    """
    stale_keys = {
        Path(name).stem for name, digest in local_digests.items() if remote.get(name) != digest
    }
    return [
        (f"{_XSGEN_TABLE_DIR}/{key}{suffix}", local[f"{key}{suffix}"])
        for suffix in (".npz", ".json")
        for key in sorted(stale_keys)
    ]


def _sync_entries() -> list[tuple[str, Path]]:
    """Return every ``(arcname, local path)`` pair a code sync would ship.

    Sorted by arcname so both the archive and the payload digest see one
    deterministic order regardless of filesystem traversal order.
    """
    entries: list[tuple[str, Path]] = []
    for p in config.SYNC_PATHS:
        local = config.LOCAL_ROOT / p
        if not local.exists():
            continue
        if local.is_dir():
            for f in sorted(local.rglob("*")):
                relative = f.relative_to(local)
                if f.is_file() and not _sync_ignored(relative):
                    entries.append(((Path(p) / relative).as_posix(), f))
        else:
            entries.append((Path(p).as_posix(), local))
    from .._catalog_layout import bundled_catalog, read_sources, selected_catalog

    catalog = selected_catalog()
    if catalog != bundled_catalog().resolve():
        if not catalog.exists():
            raise SystemExit(f"selected catalog does not exist: {catalog}")
        if catalog.is_dir():
            sources = [catalog / relative for relative, _ in read_sources(catalog)]
            assets = []
            for name in ("cifs", "energy-grid-artifacts"):
                root = catalog / name
                if root.is_dir():
                    assets.extend(
                        file
                        for file in root.rglob("*")
                        if not any(part.startswith(".") for part in file.relative_to(root).parts)
                    )
            for file in sorted((*sources, *assets)):
                if file.is_symlink() or not file.resolve().is_relative_to(catalog):
                    raise SystemExit(f"selected catalog contains unsafe symlink: {file}")
                if file.is_file():
                    entries.append(
                        ((Path("external-catalog") / file.relative_to(catalog)).as_posix(), file)
                    )
        elif catalog.is_file() and not catalog.is_symlink():
            entries.append(("external-catalog.toml", catalog))
        else:
            raise SystemExit(f"selected catalog is not a regular file or directory: {catalog}")
    entries.sort(key=lambda entry: entry[0])
    return entries


def _entry_digest(local: Path) -> str:
    """Digest one payload file's content as it will exist on the box."""
    if local.suffix.lower() in config.TEXT_EXTS:
        return hashlib.sha256(_normalized_text_bytes(local)).hexdigest()
    digest = hashlib.sha256()
    with local.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_digest(entries: list[tuple[str, Path]]) -> str:
    """Identify a sync payload by CONTENT: sorted arcnames + per-file digests.

    Deliberately not a git revision: `sync` ships the uncommitted working tree,
    so two checkouts at the same commit can carry different code. Deliberately
    not the tar's own bytes either: gzip embeds mtimes, so an untouched tree
    would digest differently on every run. Text files are hashed after the
    CRLF->LF normalization `_add_to_tar` applies, so a Windows checkout and a
    Linux one with identical content agree.
    """
    digest = hashlib.sha256()
    for arcname, local in entries:
        digest.update(f"{arcname}\0{_entry_digest(local)}\0".encode())
    return digest.hexdigest()


def _local_revision() -> tuple[str, bool]:
    """Return ``(revision, dirty)`` evidence for the tree being synced.

    Evidence only -- identity is the payload digest. A checkout that is not a
    repository (or has no git) reports ``("unknown", True)``: unverifiable is
    recorded as dirty rather than as a clean revision nobody can reproduce.
    """

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=config.LOCAL_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except OSError, subprocess.SubprocessError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    if revision is None:
        return "unknown", True
    status = git("status", "--porcelain")
    return revision, status is None or bool(status)


@dataclass(frozen=True)
class SyncStamp:
    """Code identity recorded in ``<REMOTE_DIR>/.pyrite-sync`` after a sync."""

    digest: str
    revision: str
    dirty: bool
    synced_at: str
    source: str
    tables_digest: str = ""

    def render(self) -> str:
        """Render the stamp as ``key: value`` lines, matching job ``meta``.

        Job submission appends the ``code_`` lines verbatim into each job's
        ``meta``, so one parser serves both files.
        """
        return "".join(
            f"{key}: {value}\n"
            for key, value in (
                ("code_digest", self.digest),
                ("code_revision", self.revision),
                ("code_dirty", self.dirty),
                ("code_synced_at", self.synced_at),
                ("code_source", self.source),
                *((("code_tables_digest", self.tables_digest),) if self.tables_digest else ()),
            )
        )


def _sync_stamp(digest: str, tables_digest: str = "") -> SyncStamp:
    """Build the stamp for a payload about to be unpacked on the box."""
    revision, dirty = _local_revision()
    try:
        host = socket.gethostname() or "unknown"
    except OSError:
        host = "unknown"
    source = f"{re.sub(r'[^A-Za-z0-9._-]', '_', host)}:{config.LOCAL_ROOT.as_posix()}"
    return SyncStamp(
        digest=digest,
        revision=revision,
        dirty=dirty,
        synced_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=source,
        tables_digest=tables_digest,
    )


def _refuse_conflicting_live_jobs(digest: str, *, force: bool) -> None:
    """Refuse to overwrite the shared checkout under a job running other code.

    Content-aware, not blanket: a live job whose recorded digest equals the
    incoming one is running exactly this payload, so re-syncing it changes
    nothing and must stay silent -- starting a second material mid-run is a
    supported workflow (see ``cleanup._refuse_if_busy``). A live job that
    recorded no digest at all predates code stamping; it is reported but does
    not refuse, because refusing on absence would train every caller to pass
    ``--force``.
    """
    from . import state

    conflicts: list[tuple[str, str, str]] = []
    unstamped: list[str] = []
    for jobid, _quick, _materials, recorded, revision in state._live_job_code():
        if not recorded:
            unstamped.append(jobid)
        elif recorded != digest:
            conflicts.append((jobid, recorded, revision or "unknown"))
    if unstamped:
        print(
            "note: live job(s) "
            + ", ".join(sorted(unstamped))
            + " recorded no code digest (submitted before code stamping); "
            "this sync cannot verify their code.",
            flush=True,
            file=sys.stderr,
        )
    if not conflicts:
        return
    detail = "\n".join(
        f"  job {jobid} is running code {recorded[:12]} (revision {revision})"
        for jobid, recorded, revision in sorted(conflicts)
    )
    if force:
        print(
            f"WARNING: --force syncing payload {digest[:12]} over "
            f"{config.remote_dir()} while {len(conflicts)} live job(s) run other "
            "code; their remaining steps may import a different revision than "
            f"they started with, mixing revisions in one result set:\n{detail}",
            flush=True,
            file=sys.stderr,
        )
        return
    raise SystemExit(
        "refusing to sync: a live job is running different code, and overwriting "
        f"{config.remote_dir()} underneath it can mix two revisions in one set of "
        f"results.\n{detail}\n"
        f"  incoming payload digest {digest[:12]}\n"
        "wait for it (pyrite job attach <jobid>), stop it (pyrite job stop <jobid>), "
        "or override with pyrite remote sync --force."
    )


def sync_code(*, force: bool = False):
    """Tar SYNC_PATHS up (CRLF->LF normalized for text, via _add_to_tar) and
    extract them over the repo on the box. Stale Python sources are removed
    immediately before extraction so deleted or relocated modules cannot affect
    imports or source-keyed caches. Generated interpreter/tool caches are
    excluded: they are host-specific, unnecessary, and expensive to gzip.

    Normalizing line endings here keeps the edit-locally / run-remotely loop --
    it ships the current WORKING tree (no commit required) yet stays LF-clean, so
    the box's `git status` doesn't flag every synced .py as modified and a later
    `git pull` there isn't blocked. (The committed-state alternative -- `git push`
    from the laptop + `git fetch && git reset --hard origin/<branch>` on the box --
    would force a commit before every run, which this loop is built to avoid.)

    ``force`` skips the live-job code-conflict refusal, warning instead. It is a
    CLI escape hatch (`pyrite remote sync --force`); programmatic callers keep
    the refusing default."""
    entries = _sync_entries()
    digest = _payload_digest(entries)
    # Guard first: a refusal must cost no transfer at all.
    _refuse_conflicting_live_jobs(digest, force=force)
    local_artifacts = _local_energy_grid_artifacts()
    local_tables = _local_xsgen_tables()
    local_table_digests = {name: _local_sha256(path) or "" for name, path in local_tables.items()}
    local_datasets = _local_datasets()
    local_dataset_digests = {arc: _local_sha256(path) or "" for arc, path in local_datasets.items()}
    remote_artifacts, remote_tables, remote_datasets = _remote_inventories(
        artifacts=bool(local_artifacts), tables=bool(local_tables), datasets=bool(local_datasets)
    )
    table_files = _xsgen_to_ship(local_tables, local_table_digests, remote_tables)
    dataset_files = _datasets_to_ship(local_datasets, local_dataset_digests, remote_datasets)
    with tempfile.TemporaryDirectory() as td:
        tarpath = os.path.join(td, "pyrite_code.tgz")
        with tarfile.open(tarpath, "w:gz") as t:
            for arc, f in entries:
                artifact_digest = (
                    f.stem if Path(arc).is_relative_to(_ENERGY_GRID_ARTIFACT_ROOT) else None
                )
                if artifact_digest in local_artifacts and artifact_digest in remote_artifacts:
                    continue
                _add_to_tar(t, f, arc)
            for arc, f in table_files:
                t.add(f, arcname=arc)
            # Verbatim, never CRLF-normalized: the pins cover the CRLF bytes.
            for arc, f in dataset_files:
                t.add(f, arcname=arc)
        if local_datasets:
            shipped = sum(f.stat().st_size for _, f in dataset_files)
            print(
                f"datasets: shipping {len(dataset_files)} of {len(local_datasets)} "
                f"({shipped / 1e6:.1f} MB)",
                flush=True,
                file=sys.stderr,
            )
        if local_tables:
            shipped = sum(f.stat().st_size for _, f in table_files)
            print(
                f"xsgen tables: shipping {len(table_files) // 2} of "
                f"{len(local_tables) // 2} ({shipped / 1e6:.1f} MB)",
                flush=True,
                file=sys.stderr,
            )
        _run(
            config.scp_argv(tarpath, config.scp_remote_path("/tmp/pyrite_code.tgz")),
            label="Syncing code to remote box...",
        )
    # -n: redirect ssh's stdin from null. Without it, ssh.exe inherits the
    # interactive console stdin and its stdin-forwarding thread never sees EOF,
    # so the client hangs after the remote command (tar) has already exited.
    # The stamp is chained onto the same command with `&&`, so it is written
    # once, last, and only after the extraction it describes succeeded. A failed
    # tar leaves the previous stamp in place rather than claiming code that
    # never landed.
    guaranteed = {**local_table_digests, **local_dataset_digests}
    stamp = _sync_stamp(digest, _xsgen_tables_digest(guaranteed) if guaranteed else "")
    clear_catalog = (
        "rm -rf external-catalog external-catalog.toml && "
        if config.remote_catalog_path() is not None
        else ""
    )
    _run(
        config.ssh_argv(
            "-n",
            config.remote_host(),
            f"mkdir -p {config.shell_remote_dir()} && cd {config.shell_remote_dir()} "
            '&& for p in src/pyrite checks; do if [ -d "$p" ]; then '
            "find \"$p\" -type f -name '*.py' -delete; fi; done "
            f"&& {clear_catalog}tar xzf /tmp/pyrite_code.tgz && rm -f /tmp/pyrite_code.tgz "
            f"&& printf %s {config.shell_arg(stamp.render())} "
            f"> {config.shell_single_word(config.remote_sync_stamp_path())}",
        )
    )
