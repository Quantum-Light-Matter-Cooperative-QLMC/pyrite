#!/usr/bin/env python3
"""Submit and manage courteous sliced line-grid-bound diagnostics on qlmc.

This job uses cxr_mc.remote's existing staging, SLURM wrapper, state, attach,
and stop-sentinel contracts without pretending the diagnostic is a material
checkpoint job.
"""

from __future__ import annotations

import argparse
import math
import shlex
from datetime import date

from cxr_mc import remote

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


def _slice_payload(
    jobid,
    *,
    slice_minutes,
    json_out,
    energies,
    grid_stop,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    materials=DEFAULT_MATERIALS,
):
    """Build one resumable slice payload using remote.py's chain contract."""
    jobdir = f"{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}/{jobid}"
    command = " ".join(
        [
            "CXR_MC_FREE_EVERY=40",
            "CXR_MC_FREE_WATERMARK_MB=15000",
            "CXR_MC_TIMING=1",
            shlex.quote(remote.REMOTE_UV),
            "run --no-sync python -m cxr_mc.line_grid.derive",
            f"--grid-stop {grid_stop:g}",
            f"--brem-grid-stop {brem_grid_stop:g}",
            f"--energies {shlex.quote(energies)}",
            "--coarse-engine auto",
            f"--json-out {shlex.quote(json_out)}",
            f"--max-minutes {slice_minutes:g}",
            f"--materials {shlex.quote(materials)}",
        ]
    )
    return f'''JOBDIR="{jobdir}"
cd "{remote.REMOTE_DIR}" || exit 1
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
'''


def _job_script(
    jobid,
    *,
    slice_minutes,
    json_out,
    energies,
    grid_stop,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    materials=DEFAULT_MATERIALS,
):
    payload = _slice_payload(
        jobid,
        slice_minutes=slice_minutes,
        json_out=json_out,
        energies=energies,
        grid_stop=grid_stop,
        brem_grid_stop=brem_grid_stop,
        materials=materials,
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
    materials=DEFAULT_MATERIALS,
):
    return "\n".join(
        [
            f"job: {jobid}",
            "kind: line-grid-bounds",
            f"slice_minutes: {slice_minutes:g}",
            f"json_out: {json_out}",
            f"energies: {energies}",
            f"grid_stop: {grid_stop:g}",
            f"brem_grid_stop: {brem_grid_stop:g}",
            f"materials: {materials}",
            "progress_dashboard: False",
            "",
        ]
    )


def start(
    *,
    slice_minutes=DEFAULT_SLICE_MINUTES,
    json_out=DEFAULT_JSON_OUT,
    energies=DEFAULT_ENERGIES,
    grid_stop=DEFAULT_GRID_STOP,
    brem_grid_stop=DEFAULT_BREM_GRID_STOP,
    materials=DEFAULT_MATERIALS,
    no_sync=False,
    dry_run=False,
):
    """Sync, stage, and submit slice zero. Return remote job-directory ID."""
    if slice_minutes <= 0:
        raise SystemExit("--slice-minutes must be positive")
    jobid = remote._new_jobid()
    jobdir = f"{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}/{jobid}"
    script = _job_script(
        jobid,
        slice_minutes=slice_minutes,
        json_out=json_out,
        energies=energies,
        grid_stop=grid_stop,
        brem_grid_stop=brem_grid_stop,
        materials=materials,
    )
    metadata = _metadata(
        jobid,
        slice_minutes=slice_minutes,
        json_out=json_out,
        energies=energies,
        grid_stop=grid_stop,
        brem_grid_stop=brem_grid_stop,
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
    print(f"status: cxr line-grid status {jobid}")
    print(f"attach: cxr line-grid attach {jobid}")
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
    start_parser.add_argument("--materials", default=DEFAULT_MATERIALS)
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
            materials=args.materials,
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
