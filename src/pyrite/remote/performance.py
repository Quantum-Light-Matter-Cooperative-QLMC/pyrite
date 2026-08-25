"""Remote performance-artifact lifecycle."""

import uuid
from pathlib import Path

from ..cli import _core as _cli_core
from ..cli import dashboard as presentation
from . import config, state, transport


def pull_performance_profile(profile: str) -> list[Path]:
    """Pull performance NDJSON plus any Nsight report artifacts."""
    transport._check_shell_tokens([profile])
    jobs = config.remote_path(config.JOBS_SUBDIR)
    listing = transport._ssh_capture(
        f"JOBS={config.shell_word(jobs)}; "
        '[ -d "$JOBS" ] || exit 0; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        f'p="$d/performance/{profile}"; [ -d "$p" ] || continue; '
        'find "$p" -maxdepth 1 -type f '
        "\\( -name '*.ndjson' -o -name '*.nsys-rep' -o -name '*.sqlite' "
        "-o -name '*.nsys-stats.txt' -o -name '*.cpu.prof' -o -name '*.cpu.txt' \\) "
        "-printf '%f\\n' | while IFS= read -r f; do "
        'printf "%s\\t%s\\n" "$(basename "$d")" "$f"; done; done'
    )
    artifacts = []
    suffixes = (".nsys-stats.txt", ".cpu.prof", ".cpu.txt", ".nsys-rep", ".ndjson", ".sqlite")
    for line in listing.splitlines():
        jobid, separator, filename = line.partition("\t")
        suffix = next((item for item in suffixes if filename.endswith(item)), "")
        material = filename[: -len(suffix)] if suffix else ""
        if (
            not separator
            or not suffix
            or presentation._SHELL_TOKEN_RE.fullmatch(jobid) is None
            or presentation._SHELL_TOKEN_RE.fullmatch(material) is None
        ):
            continue
        artifacts.append((jobid, filename))
    if not artifacts:
        raise SystemExit(f"no remote performance artifacts found for profile {profile!r}")

    local_root = config.LOCAL_ROOT / "performance-profiles" / profile
    pulled = []
    for jobid, filename in artifacts:
        destination = local_root / jobid / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        remote = config.remote_path(
            config.JOBS_SUBDIR,
            jobid,
            "performance",
            profile,
            filename,
        )
        try:
            transport._ssh_download(f"cat {config.shell_arg(remote)}", temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        pulled.append(destination)
        print(f"pulled {jobid}/{filename} -> {destination}")
    return pulled


def remote_performance_inventory() -> list[tuple[str, str, int, int]]:
    """Return ``(job, profile, files, bytes)`` for remote performance directories."""
    jobs = config.remote_path(config.JOBS_SUBDIR)
    output = transport._ssh_capture(
        f"JOBS={config.shell_word(jobs)}; "
        '[ -d "$JOBS" ] || exit 0; '
        'for p in "$JOBS"/*/performance/*/; do [ -d "$p" ] || continue; '
        'job=$(basename "$(dirname "$(dirname "$p")")"); profile=$(basename "$p"); '
        'set -- $(find "$p" -maxdepth 1 -type f -printf "%s\\n" | '
        "awk '{n += 1; b += $1} END {print n+0, b+0}'); "
        'printf "%s\\t%s\\t%s\\t%s\\n" "$job" "$profile" "$1" "$2"; done'
    )
    inventory = []
    for line in output.splitlines():
        fields = line.split("\t")
        if (
            len(fields) != 4
            or presentation._SHELL_TOKEN_RE.fullmatch(fields[0]) is None
            or presentation._SHELL_TOKEN_RE.fullmatch(fields[1]) is None
        ):
            raise SystemExit("refusing performance operation: remote inventory was malformed")
        try:
            files, size = int(fields[2]), int(fields[3])
        except ValueError:
            raise SystemExit(
                "refusing performance operation: remote inventory was malformed"
            ) from None
        if files < 0 or size < 0:
            raise SystemExit("refusing performance operation: remote inventory was malformed")
        inventory.append((fields[0], fields[1], files, size))
    return sorted(inventory)


def list_remote_performance() -> list[tuple[str, str, int, int]]:
    """Print remote performance artifact inventory."""
    inventory = remote_performance_inventory()
    if not inventory:
        print("(no remote performance profiles)")
        return []
    for jobid, profile, files, size in inventory:
        print(f"{profile}/{jobid}: {files} artifact(s), {size} bytes")
    return inventory


def _performance_job_states(jobids: set[str]) -> dict[str, str]:
    """Read selected job states; only explicit terminal records are prune-safe."""
    return {jobid: state._job_state(jobid) for jobid in sorted(jobids)}


def prune_remote_performance(profiles=None, *, all_profiles=False, yes=False):
    """Preview/delete selected terminal-job performance directories."""
    profiles = list(profiles or [])
    transport._check_shell_tokens(profiles)
    before_live = {jobid for jobid, _quick, _materials in state._live_jobs()}
    inventory = remote_performance_inventory()
    selected_profiles = {profile for _job, profile, _files, _size in inventory}
    if not all_profiles:
        selected_profiles.intersection_update(profiles)
        missing = sorted(set(profiles) - selected_profiles)
        if missing:
            raise SystemExit(f"no remote performance profile: {', '.join(missing)}")
    selected = [row for row in inventory if row[1] in selected_profiles]
    selected_jobs = {jobid for jobid, _profile, _files, _size in selected}
    busy = sorted(selected_jobs & before_live)
    if busy:
        raise SystemExit(
            "refusing performance prune: selected artifacts belong to live job(s): "
            + ", ".join(busy)
        )
    before_states = _performance_job_states(selected_jobs)
    incomplete = sorted(
        jobid
        for jobid, job_state in before_states.items()
        if not job_state.startswith(("done", "FAILED", "cancelled"))
    )
    if incomplete:
        raise SystemExit(
            "refusing performance prune: selected artifacts belong to "
            "non-terminal job(s): " + ", ".join(incomplete)
        )
    if not selected:
        print("(nothing to prune)")
        return
    print("would delete remote performance profiles:")
    for jobid, profile, files, size in selected:
        print(f"  {profile}/{jobid} ({files} artifact(s), {size} bytes)")
    if not _cli_core.confirm_destructive(yes, "Delete these remote performance profiles?"):
        return

    after_live = {jobid for jobid, _quick, _materials in state._live_jobs()}
    after_states = _performance_job_states(selected_jobs)
    current = remote_performance_inventory()
    if after_live != before_live or after_states != before_states or current != inventory:
        raise SystemExit(
            "refusing performance prune: remote job state or artifact inventory changed"
        )
    targets = [
        config.remote_path(config.JOBS_SUBDIR, jobid, "performance", profile)
        for jobid, profile, _files, _size in selected
    ]
    command = " ".join(config.shell_arg(target) for target in targets)
    transport._ssh_capture(f'for target in {command}; do rm -rf -- "$target"; done')
    print(f"deleted {len(selected)} remote performance profile path(s)")
