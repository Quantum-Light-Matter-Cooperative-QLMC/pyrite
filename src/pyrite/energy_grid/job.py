#!/usr/bin/env python3
"""Submit and manage courteous sliced line-grid-bound diagnostics on qlmc.

This job uses pyrite.remote's existing staging, SLURM wrapper, state, attach,
and stop-sentinel contracts without pretending the diagnostic is a material
checkpoint job.
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import unicodedata
from datetime import date

from pyrite import remote

today = date.today()

DEFAULT_SLICE_MINUTES = 10.0
DEFAULT_JSON_OUT = f"line_grid_bounds_eval_{today}.json"
# All seven standard beam energies: bespoke per-material grids need a complete
# self-contained table per material, matching the profile's seven rows.
DEFAULT_ENERGIES = "30,50,100,150,200,250,300"
DEFAULT_GRID_STOP = 20_000.0
# Ceiling of the diagnostic brem coverage grid; mirrors analyze_line_grid_bounds
# WIDE_BREM_STOP_EV so per-material bespoke brem stops are not clipped.
DEFAULT_BREM_GRID_STOP = 40_000.0
# The line-grid eval targets the four standard-profile crystals (issue_notes.md
# #1). Overridable via --materials so other catalog keys can be scanned.
DEFAULT_MATERIALS = "hopg,diamond,wse2,mose2"
_REMOTE_OUTPUT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json")


def _reject_controls(field, value):
    if value is None:
        return
    if not isinstance(value, str):
        raise SystemExit(f"{field} must be text")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise SystemExit(f"{field} must not contain control characters")


def _validate_remote_output_name(value):
    _reject_controls("--json-out", value)
    if not _REMOTE_OUTPUT_RE.fullmatch(value):
        raise SystemExit(
            "--json-out must be a .json basename using letters, digits, '.', '_' or '-'"
        )


def _validate_remote_fields(
    *, json_out, energies, materials, tilts=None, azimuths=None, thickness=None
):
    _validate_remote_output_name(json_out)
    for field, value in (
        ("--energies", energies),
        ("--materials", materials),
        ("--tilts", tilts),
        ("--azimuths", azimuths),
        ("--thickness", thickness),
    ):
        _reject_controls(field, value)


def _slice_payload(
    jobid,
    *,
    slice_minutes,
    json_out,
    energies,
    grid_stop,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    brem_step=None,
    materials=DEFAULT_MATERIALS,
    tilts=None,
    azimuths=None,
    thickness=None,
    set_default=False,
):
    """Build one resumable slice payload using remote.py's chain contract."""
    _validate_remote_fields(
        json_out=json_out,
        energies=energies,
        materials=materials,
        tilts=tilts,
        azimuths=azimuths,
        thickness=thickness,
    )
    jobdir = remote.remote_path(remote.JOBS_SUBDIR, jobid)
    command_parts = [
        "PYRITE_MC_FREE_EVERY=40",
        "PYRITE_MC_FREE_WATERMARK_MB=15000",
        "PYRITE_MC_TIMING=1",
        remote.shell_remote_uv(),
        "run --no-sync python -m pyrite.energy_grid.derive",
        f"--grid-stop {grid_stop:g}",
        f"--brem-grid-stop {brem_grid_stop:g}",
        f"--energies {shlex.quote(energies)}",
        "--coarse-engine auto",
        f"--json-out {shlex.quote(json_out)}",
        f"--max-minutes {slice_minutes:g}",
        f"--materials {shlex.quote(materials)}",
    ]
    if brem_step is not None:
        command_parts.append(f"--brem-step {brem_step:g}")
    for flag, value in (
        ("tilts", tilts),
        ("azimuths", azimuths),
        ("thickness", thickness),
    ):
        if value:
            command_parts.append(f"--{flag} {shlex.quote(value)}")
    if set_default:
        command_parts.append("--set-default")
    command = " ".join(command_parts)
    return f"""JOBDIR={remote.shell_word(jobdir)}
cd {remote.shell_remote_dir()} || exit 1
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
echo "running line-grid bounds $(date -Is)" > "$JOBDIR/state"
rc=0
{command} >> "$JOBDIR/log" 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "done $(date -Is)" > "$JOBDIR/state"
  exit 0
fi
if [ "$rc" -ne 75 ]; then
  echo "FAILED (analysis exit $rc) $(date -Is)" > "$JOBDIR/state"
  exit "$rc"
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\n' "$SID" >> "$JOBDIR/meta"
"""


def _job_script(
    jobid,
    *,
    slice_minutes,
    json_out,
    energies,
    grid_stop,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    brem_step=None,
    materials=DEFAULT_MATERIALS,
    tilts=None,
    azimuths=None,
    thickness=None,
    set_default=False,
):
    payload = _slice_payload(
        jobid,
        slice_minutes=slice_minutes,
        json_out=json_out,
        energies=energies,
        grid_stop=grid_stop,
        brem_grid_stop=brem_grid_stop,
        brem_step=brem_step,
        materials=materials,
        tilts=tilts,
        azimuths=azimuths,
        thickness=thickness,
        set_default=set_default,
    )
    time_limit = str(max(1, math.ceil(slice_minutes * 3)))
    return remote._slurm_batch_script(
        jobid,
        payload,
        job_name="line-grid-bounds",
        time_limit=time_limit,
    )


def _metadata(
    jobid,
    *,
    slice_minutes,
    json_out,
    energies,
    grid_stop,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    brem_step=None,
    materials=DEFAULT_MATERIALS,
):
    _reject_controls("jobid", jobid)
    _validate_remote_fields(
        json_out=json_out,
        energies=energies,
        materials=materials,
    )
    fields = [
        f"job: {jobid}",
        "kind: line-grid-bounds",
        f"slice_minutes: {slice_minutes:g}",
        f"json_out: {json_out}",
        f"energies: {energies}",
        f"grid_stop: {grid_stop:g}",
        f"brem_grid_stop: {brem_grid_stop:g}",
        f"materials: {materials}",
    ]
    if brem_step is not None:
        fields.append(f"brem_step: {brem_step:g}")
    fields.extend(("progress_dashboard: False", ""))
    return "\n".join(fields)


def start(
    *,
    slice_minutes=DEFAULT_SLICE_MINUTES,
    json_out=DEFAULT_JSON_OUT,
    energies=DEFAULT_ENERGIES,
    grid_stop=DEFAULT_GRID_STOP,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    brem_step=None,
    materials=DEFAULT_MATERIALS,
    tilts=None,
    azimuths=None,
    thickness=None,
    set_default=False,
    no_sync=False,
    dry_run=False,
):
    """Sync, stage, and submit slice zero. Return remote job-directory ID."""
    if slice_minutes <= 0:
        raise SystemExit("--slice-minutes must be positive")
    jobid = remote._new_jobid()
    jobdir = remote.remote_path(remote.JOBS_SUBDIR, jobid)
    script = _job_script(
        jobid,
        slice_minutes=slice_minutes,
        json_out=json_out,
        energies=energies,
        grid_stop=grid_stop,
        brem_grid_stop=brem_grid_stop,
        brem_step=brem_step,
        materials=materials,
        tilts=tilts,
        azimuths=azimuths,
        thickness=thickness,
        set_default=set_default,
    )
    metadata = _metadata(
        jobid,
        slice_minutes=slice_minutes,
        json_out=json_out,
        energies=energies,
        grid_stop=grid_stop,
        brem_grid_stop=brem_grid_stop,
        brem_step=brem_step,
        materials=materials,
    )
    upload = remote._write_job_script_command(jobdir, metadata)
    if dry_run:
        print(script)
        return jobid
    if not no_sync:
        remote.sync_code()
    remote._stage_job_script(jobid, [], upload, script)
    scheduler_id = remote._submit_staged_job(jobid, [], nice=True)
    print(f"submitted SLURM job {scheduler_id} as {jobid}")
    print(f"status: pyrite energy-grid job status {jobid}")
    print(f"attach: pyrite energy-grid job attach {jobid}")
    return jobid


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--slice-minutes", type=float, default=DEFAULT_SLICE_MINUTES)
    start_parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    start_parser.add_argument("--energies", default=DEFAULT_ENERGIES)
    start_parser.add_argument("--grid-stop", type=float, default=DEFAULT_GRID_STOP)
    start_parser.add_argument("--brem-grid-stop", type=float, default=DEFAULT_BREM_GRID_STOP)
    start_parser.add_argument("--brem-step", type=float, default=None)
    start_parser.add_argument("--materials", default=DEFAULT_MATERIALS)
    start_parser.add_argument("--tilts", default=None)
    start_parser.add_argument("--azimuths", default=None)
    start_parser.add_argument("--thickness", default=None)
    start_parser.add_argument("--set-default", action="store_true")
    start_parser.add_argument("--no-sync", action="store_true")
    start_parser.add_argument("--dry-run", action="store_true")
    for command in ("status", "attach", "stop"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("jobid", nargs="?")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "start":
        start(
            slice_minutes=args.slice_minutes,
            json_out=args.json_out,
            energies=args.energies,
            grid_stop=args.grid_stop,
            brem_grid_stop=args.brem_grid_stop,
            brem_step=args.brem_step,
            materials=args.materials,
            tilts=args.tilts,
            azimuths=args.azimuths,
            thickness=args.thickness,
            set_default=args.set_default,
            no_sync=args.no_sync,
            dry_run=args.dry_run,
        )
    elif args.command == "status":
        remote.job_status(args.jobid)
    elif args.command == "attach":
        remote.attach(args.jobid)
    else:
        jobid = args.jobid or remote._latest_jobid()
        if not jobid:
            raise SystemExit("no jobs to stop")
        remote._stop_jobid(jobid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
