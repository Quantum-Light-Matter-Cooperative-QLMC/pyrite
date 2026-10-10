"""Remote trajectory inventory, verified transfer, and bounded export.

The module entry point runs on the host; client helpers use the existing SSH
configuration. Complete HDF5 artifacts remain authoritative and unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

from . import config, transport


def capture_root(value=None):
    """Resolve a capture root inside the remote checkout, without local expansion."""
    base = PurePosixPath(config.remote_dir())
    path = (
        PurePosixPath(value)
        if value is not None
        else PurePosixPath(config.remote_output_path("trajectories"))
    )
    if not path.is_absolute():
        path = base / path
    if ".." in path.parts or not path.is_relative_to(base) or any(ord(c) < 32 for c in str(path)):
        raise ValueError(
            "trajectory root must be inside the remote checkout, without '..' or control characters"
        )
    return str(path)


def _host_paths(root, stem=None):
    """Discover complete-file candidates, refusing symlinks outside the root."""
    base = Path(root).resolve()
    directory = base
    if stem is not None:
        if not stem or PurePosixPath(stem).name != stem or stem in {".", ".."}:
            raise ValueError("STEM must be one directory name")
        directory = base / stem
    if not directory.resolve().is_relative_to(base):
        raise ValueError("trajectory directory escapes its capture root")
    for path in sorted(directory.rglob("*.h5")):
        if not path.resolve().is_relative_to(base) or path.is_symlink():
            raise ValueError(f"trajectory artifact escapes its capture root: {path}")
        yield path


def inventory(root, stem=None, cases=(), energies=(), *, hashes=False):
    """Read headers only; filter exact case names and incident energies in keV."""
    from ..montecarlo.trajectories import read_trajectory_header

    records = []
    for path in _host_paths(root, stem):
        header = read_trajectory_header(path)
        if cases and header["case_name"] not in cases:
            continue
        if energies and float(header["E0_keV"]) not in energies:
            continue
        stat = path.stat()
        record = {
            "path": str(path.relative_to(Path(root))),
            "bytes": stat.st_size,
            "case": header["case_name"],
            "energy_keV": header["E0_keV"],
            "segments": header["segment_count"],
            "schema_version": header["schema_version"],
            "case_sha256": header["case_sha256"],
            "complete": True,
        }
        if hashes:
            record["sha256"] = transport._local_sha256(path)
            if path.stat() != stat:
                raise ValueError(f"trajectory artifact changed during hashing: {path}")
        records.append(record)
    return records


def _host_command(action, request):
    return (
        f"cd {config.shell_remote_dir()} && {config.remote_runtime_env()} "
        f"PYRITE_REMOTE_DIR={config.shell_word(config.remote_dir())} "
        f"{config.shell_remote_uv()} run --no-sync python -m pyrite.remote.trajectories "
        f"{shlex.quote(action)} {shlex.quote(json.dumps(request))}"
    )


def remote_inventory(root=None, stem=None, cases=(), energies=(), *, hashes=False):
    """Fetch the exact selected remote inventory in one SSH round trip."""
    request = dict(
        root=capture_root(root), stem=stem, cases=cases, energies=energies, hashes=hashes
    )
    return json.loads(transport._ssh_capture(_host_command("inventory", request)))


def _destination(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"invalid remote artifact path: {relative}")
    destination = Path(root) / str(relative)
    if not destination.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"local destination escapes output directory: {destination}")
    return destination


def pull_files(records, remote_root, local_root, *, artifacts=True, overwrite=False):
    """Resume to digest-named staging files; install only after size/hash/header checks."""
    from ..montecarlo.trajectories import read_trajectory_header

    if shutil.which("rsync") is None:
        raise ValueError("trajectory pull requires rsync locally and on the remote host")
    targets = [(record, _destination(local_root, record["path"])) for record in records]
    for record, target in targets:
        if (
            target.exists()
            and transport._local_sha256(target) != record["sha256"]
            and not overwrite
        ):
            raise ValueError(f"local file differs: {target}; pass --overwrite to replace it")
    for record, target in targets:
        if target.exists() and transport._local_sha256(target) == record["sha256"]:
            if artifacts:
                read_trajectory_header(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.{record['sha256']}.partial")
        if staging.is_symlink():
            raise ValueError(f"refusing symlink transfer staging file: {staging}")
        source = str(PurePosixPath(remote_root) / record["path"])
        try:
            transport._run(
                [
                    "rsync",
                    "--protect-args",
                    "--partial",
                    "--checksum",
                    "-t",
                    "-e",
                    shlex.join(config.ssh_argv()),
                    "--",
                    f"{config.remote_host()}:{source}",
                    str(staging.resolve()),
                ],
                label=f"pulling {record['path']} ({record['bytes']} bytes)",
                stdout=sys.stderr,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"trajectory transfer failed; rerun to resume: {record['path']}"
            ) from error
        if (
            staging.stat().st_size != record["bytes"]
            or transport._local_sha256(staging) != record["sha256"]
        ):
            staging.unlink(missing_ok=True)
            raise ValueError(
                f"trajectory integrity check failed: {record['path']}; source may have changed"
            )
        if artifacts:
            read_trajectory_header(staging)
        os.replace(staging, target)
    return [str(target) for _, target in targets]


def remote_export(request, local_root, *, overwrite=False):
    """Export selected histories on the host; download only the resulting VTK files."""
    request = dict(request, root=capture_root(request.get("root")), export_id=uuid.uuid4().hex)
    response = json.loads(transport._ssh_capture(_host_command("export", request)))
    return pull_files(
        response["files"], response["root"], local_root, artifacts=False, overwrite=overwrite
    )


def submit_score(request):
    """Submit spectrum replay to SLURM, leaving both captures and checkpoints remote."""
    from . import jobs, scripts

    request = dict(request, root=capture_root(request.get("root")))
    records = remote_inventory(
        request["root"], request["stem"], request.get("cases", ()), request.get("energies", ())
    )
    if not records:
        raise ValueError("no matching complete trajectory artifacts")
    jobid = scripts._new_jobid()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    # The host computes identities from the artifacts, just like local scoring.
    # The exact stem is reserved before SLURM starts the replay.
    stems = [request["stem"]]
    payload = (
        f"JOBDIR={config.shell_word(jobdir)}\n"
        'echo "running trajectory score" > "$JOBDIR/state"\n'
        f'{_host_command("score", request)} >> "$JOBDIR/log" 2>&1\n'
        'rc=$?\nif [ "$rc" -eq 0 ]; then '
        f'echo {config.shell_word("completed: " + request["stem"])} >> "$JOBDIR/log"; '
        'echo "done" > "$JOBDIR/state"; '
        'else echo "FAILED trajectory score" > "$JOBDIR/state"; fi\nexit "$rc"\n'
    )
    script = scripts._slurm_batch_script(
        jobid, payload, job_name=f"pyrite-{jobid}", reservation_stems=stems
    )
    metadata = (
        f"job: {jobid}\nkind: trajectory-score\ntrajectories: {request['root']}\n"
        f"materials: {request['stem']}\nquick: False\nworkers: {config.SLURM_CPUS_PER_MATERIAL}\n"
        "parallel_materials: 1\nchunk_minutes: 0\n"
    )
    upload = scripts._write_job_script_command(jobdir, metadata)
    jobs._stage_job_script(jobid, stems, upload, script)
    jobs._submit_staged_job(jobid, stems)
    return jobid


def _selected_paths(request):
    records = inventory(
        request["root"], request.get("stem"), request.get("cases", ()), request.get("energies", ())
    )
    if not records:
        raise ValueError("no matching complete trajectory artifacts")
    return [Path(request["root"]) / record["path"] for record in records]


def _export(request):
    from ..montecarlo.trajectory_export import export_segments_vtp
    from ..montecarlo.trajectory_selection import select_trajectories

    paths = _selected_paths(request)
    selections = [select_trajectories(path, **request["selection"]) for path in paths]
    root = Path(config.remote_output_path("trajectories-exports", request["export_id"]))
    files = []
    for path, selection in zip(paths, selections, strict=True):
        relative = path.relative_to(request["root"]).with_suffix(".vtp")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        export_segments_vtp(path, target, selection=selection)
        files.append(
            dict(
                path=str(relative),
                bytes=target.stat().st_size,
                sha256=transport._local_sha256(target),
            )
        )
    return dict(root=str(root), files=files)


def _score(request):
    from ..checkpoints.trajectory_scoring import check_targets, plan_stems, score_stem
    from ..montecarlo.runner.artifacts import STREAM_MAX_SEGMENTS

    plans = plan_stems(_selected_paths(request))
    root = Path(config.remote_output_path("checkpoints"))
    if set(plans) != {request["stem"]}:
        raise ValueError("capture provenance must match the selected checkpoint stem")
    check_targets(plans.values(), root)
    for plan in plans.values():
        summary = score_stem(
            plan,
            root,
            max_segments=request.get("max_segments") or STREAM_MAX_SEGMENTS,
            overwrite=request.get("overwrite", False),
        )
        print(f"{summary.checkpoint_path}: scored {summary.scored}; kept {summary.kept}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inventory", "export", "score"))
    parser.add_argument("request")
    args = parser.parse_args()
    request = json.loads(args.request)
    from ..montecarlo.trajectories import TrajectoryArtifactError

    try:
        # Resolve on the host too, so direct invocation cannot bypass root checks.
        request["root"] = capture_root(request.get("root"))
        if not Path(request["root"]).resolve().is_relative_to(Path(config.remote_dir()).resolve()):
            raise ValueError("trajectory root escapes the remote checkout through a symlink")
        if args.action == "inventory":
            print(json.dumps(inventory(**request)))
        elif args.action == "export":
            print(json.dumps(_export(request)))
        else:
            _score(request)
    except (OSError, ValueError, TrajectoryArtifactError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
