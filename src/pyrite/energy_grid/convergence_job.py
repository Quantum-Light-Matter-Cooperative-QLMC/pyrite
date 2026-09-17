"""Submit and manage the remote line-grid refinement ladder (issue #109).

Light on purpose, like :mod:`pyrite.energy_grid.job`: submitting, polling, and
pulling never import the transport/spectrum stack or probe a compute backend.
The slice body is ``python -m pyrite.energy_grid.convergence_case run``; each
slice checkpoints after every rung and exits 75 when its budget is spent, and
the chained-slice contract below resubmits it until the ladder completes.

    python -m pyrite.energy_grid.convergence_job start --dry-run
    python -m pyrite.energy_grid.convergence_job start --json-out NAME.json
    python -m pyrite.energy_grid.convergence_job status [JOBID]
    python -m pyrite.energy_grid.convergence_job pull --json-out NAME.json
    python -m pyrite.energy_grid.convergence_job start-precision --json-out NAME.json

``start-precision`` runs :mod:`pyrite.energy_grid.precision_ladder`: one FP64
transport, the line reduction under FP64 and under float32 on the same pickled
segments, and a deviation table against spacing/ulp. It is one short job, not a
chained ladder.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence
from datetime import date
from pathlib import Path

DEFAULT_SPACINGS = "12,6,3,1.5,0.75,0.375"
DEFAULT_MATERIALS = "hopg,wse2"
DEFAULT_ENERGIES = "30,100,300"
DEFAULT_TILTS = "5,85"
DEFAULT_NE = 2000
#: The 1 mm slab ``energy_grid.derive`` uses for its diagnostic geometry: the
#: longest flights, hence the narrowest sinc features, of any catalog thickness.
DEFAULT_THICKNESS_ANG = 1.0e7
DEFAULT_SLICE_MINUTES = 10.0
#: One fine rung at 300 keV and Ne=2000 can outlast a slice, so the SLURM
#: backstop is generous; slices still hand off between rungs.
DEFAULT_TIME_LIMIT_MINUTES = 120
REMOTE_JOB_KIND = "line-grid-convergence"
PRECISION_JOB_KIND = "line-grid-precision"
DEFAULT_PRECISION_RATIOS = "8,30,100,300,1000,3000"
DEFAULT_PRECISION_BANDS = "8192:16384,16384:32768"
DEFAULT_PRECISION_TIME_LIMIT_MINUTES = 60
_LIST_CHARS = frozenset("0123456789abcdefghijklmnopqrstuvwxyz,._-")

#: Window-refinement ladder (#101): samples per feature, coarsest first, on a
#: fixed backbone, judged against a dense uniform reference.
WINDOW_JOB_KIND = "line-grid-window-convergence"
DEFAULT_WINDOW_SAMPLES = "2,4,8,16,32"
DEFAULT_WINDOW_BACKBONE = 3.0
DEFAULT_WINDOW_PROVIDERS = "pxr-kinematic,absorption-edge,characteristic"
DEFAULT_WINDOW_TAIL_WIDTHS = 2.0
DEFAULT_REFERENCE_POINTS = 400_000


def add_ladder_arguments(parser: argparse.ArgumentParser) -> None:
    """Ladder selection flags shared by the local ``run`` and remote ``start``."""
    parser.add_argument("--materials", default=DEFAULT_MATERIALS)
    parser.add_argument("--energies", default=DEFAULT_ENERGIES)
    parser.add_argument("--tilts", default=DEFAULT_TILTS)
    parser.add_argument(
        "--azimuth", type=float, default=None, help="default: each material's first catalog azimuth"
    )
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS_ANG)
    parser.add_argument("--ne", type=int, default=DEFAULT_NE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spacings", default=DEFAULT_SPACINGS)
    parser.add_argument(
        "--json-out", default=f"line_grid_convergence_{date.today()}.json", help="checkpoint JSON"
    )


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Window-ladder flags shared by the local ``run-windows`` and remote ``start-windows``."""
    parser.add_argument("--materials", default=DEFAULT_MATERIALS)
    parser.add_argument("--energies", default=DEFAULT_ENERGIES)
    parser.add_argument("--tilts", default=DEFAULT_TILTS)
    parser.add_argument(
        "--azimuth", type=float, default=None, help="default: each material's first catalog azimuth"
    )
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS_ANG)
    parser.add_argument("--ne", type=int, default=DEFAULT_NE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", default=DEFAULT_WINDOW_SAMPLES)
    parser.add_argument("--backbone", type=float, default=DEFAULT_WINDOW_BACKBONE)
    parser.add_argument("--providers", default=DEFAULT_WINDOW_PROVIDERS)
    parser.add_argument("--tail-widths", type=float, default=DEFAULT_WINDOW_TAIL_WIDTHS)
    parser.add_argument("--reference-points", type=int, default=DEFAULT_REFERENCE_POINTS)
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_REFERENCE_POINTS,
        help="truncate the ladder at the finest rung within this many points",
    )
    parser.add_argument(
        "--json-out", default=f"line_window_convergence_{date.today()}.json", help="checkpoint JSON"
    )


def remote_windows_command(args: argparse.Namespace, uv: str) -> str:
    """The slice body: one budgeted ``convergence_case run-windows`` invocation."""
    parts = [
        "PYRITE_MC_FREE_EVERY=40",
        "PYRITE_MC_FREE_WATERMARK_MB=15000",
        uv,
        "run --no-sync python -m pyrite.energy_grid.convergence_case run-windows",
        f"--materials {shlex.quote(args.materials)}",
        f"--energies {shlex.quote(args.energies)}",
        f"--tilts {shlex.quote(args.tilts)}",
        f"--samples {shlex.quote(args.samples)}",
        f"--providers {shlex.quote(args.providers)}",
        f"--backbone {float(args.backbone):g}",
        f"--tail-widths {float(args.tail_widths):g}",
        f"--reference-points {int(args.reference_points)}",
        f"--max-points {int(args.max_points)}",
        f"--thickness {float(args.thickness):g}",
        f"--ne {int(args.ne)}",
        f"--seed {int(args.seed)}",
        f"--json-out {shlex.quote(args.json_out)}",
        f"--max-minutes {float(args.slice_minutes):g}",
    ]
    if args.azimuth is not None:
        parts.append(f"--azimuth {float(args.azimuth):g}")
    return " ".join(parts)


def remote_run_command(args: argparse.Namespace, uv: str) -> str:
    """The slice body: one budgeted ``convergence_case run`` invocation."""
    parts = [
        "PYRITE_MC_FREE_EVERY=40",
        "PYRITE_MC_FREE_WATERMARK_MB=15000",
        uv,
        "run --no-sync python -m pyrite.energy_grid.convergence_case run",
        f"--materials {shlex.quote(args.materials)}",
        f"--energies {shlex.quote(args.energies)}",
        f"--tilts {shlex.quote(args.tilts)}",
        f"--spacings {shlex.quote(args.spacings)}",
        f"--thickness {float(args.thickness):g}",
        f"--ne {int(args.ne)}",
        f"--seed {int(args.seed)}",
        f"--json-out {shlex.quote(args.json_out)}",
        f"--max-minutes {float(args.slice_minutes):g}",
    ]
    if args.azimuth is not None:
        parts.append(f"--azimuth {float(args.azimuth):g}")
    return " ".join(parts)


def _slice_payload(jobdir: str, command: str, remote, sync_block: str) -> str:
    """The derive job's chained-slice contract with this ladder as the body.

    ``sync_block`` is the standard remote dependency sync
    (``remote.scripts._uv_sync_block``); the body runs ``uv run --no-sync``, so
    without it a box whose venv predates the lockfile fails on import.
    """
    return f"""JOBDIR={remote.shell_word(jobdir)}
cd {remote.shell_remote_dir()} || exit 1
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
echo "running line-grid convergence $(date -Is)" > "$JOBDIR/state"
{sync_block}
rc=0
{command} >> "$JOBDIR/log" 2>&1 || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "done $(date -Is)" > "$JOBDIR/state"
  exit 0
fi
if [ "$rc" -ne 75 ]; then
  echo "FAILED (ladder exit $rc) $(date -Is)" > "$JOBDIR/state"
  exit "$rc"
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta"
"""


def _validate(args: argparse.Namespace) -> None:
    from .job import _validate_remote_output_name

    _validate_remote_output_name(args.json_out)
    for name in ("materials", "energies", "tilts", "spacings"):
        value = getattr(args, name)
        if not value or not set(value.lower()) <= _LIST_CHARS:
            raise SystemExit(f"--{name} may contain only letters, digits, ',', '.', '_', '-'")
    if args.slice_minutes <= 0:
        raise SystemExit("--slice-minutes must be positive")
    if args.time_limit_minutes <= args.slice_minutes:
        raise SystemExit("--time-limit-minutes must exceed --slice-minutes")


def _validate_windows(args: argparse.Namespace) -> None:
    from .job import _validate_remote_output_name

    _validate_remote_output_name(args.json_out)
    for name in ("materials", "energies", "tilts", "samples", "providers"):
        value = getattr(args, name)
        if not value or not set(value.lower()) <= _LIST_CHARS:
            raise SystemExit(f"--{name} may contain only letters, digits, ',', '.', '_', '-'")
    if args.backbone <= 0.0:
        raise SystemExit("--backbone must be positive")
    if args.tail_widths < 0.0:
        raise SystemExit("--tail-widths must be non-negative")
    if args.reference_points < 2:
        raise SystemExit("--reference-points must be at least 2")
    if args.slice_minutes <= 0:
        raise SystemExit("--slice-minutes must be positive")
    if args.time_limit_minutes <= args.slice_minutes:
        raise SystemExit("--time-limit-minutes must exceed --slice-minutes")


def start(args: argparse.Namespace) -> str:
    """Sync, stage, and submit slice zero; return the remote job id."""
    from .. import remote
    from ..remote.scripts import _uv_sync_block

    _validate(args)
    jobid = remote._new_jobid()
    jobdir = remote.remote_path(remote.JOBS_SUBDIR, jobid)
    command = remote_run_command(args, remote.shell_remote_uv())
    script = remote._slurm_batch_script(
        jobid,
        _slice_payload(jobdir, command, remote, _uv_sync_block(once=True)),
        job_name=REMOTE_JOB_KIND,
        time_limit=str(int(args.time_limit_minutes)),
    )
    metadata = "\n".join(
        [
            f"job: {jobid}",
            f"kind: {REMOTE_JOB_KIND}",
            f"slice_minutes: {float(args.slice_minutes):g}",
            f"json_out: {args.json_out}",
            f"materials: {args.materials}",
            f"energies: {args.energies}",
            f"tilts: {args.tilts}",
            f"spacings: {args.spacings}",
            f"ne: {int(args.ne)}",
            "progress_dashboard: False",
            "",
        ]
    )
    if args.dry_run:
        print(script)
        return jobid
    if not args.no_sync:
        remote.sync_code()
    remote._stage_job_script(jobid, [], remote._write_job_script_command(jobdir, metadata), script)
    scheduler_id = remote._submit_staged_job(jobid, [], nice=True)
    print(f"submitted SLURM job {scheduler_id} as {jobid}")
    print(f"status: python -m pyrite.energy_grid.convergence_job status {jobid}")
    print(f"pull:   python -m pyrite.energy_grid.convergence_job pull --json-out {args.json_out}")
    return jobid


def start_windows(args: argparse.Namespace) -> str:
    """Sync, stage, and submit slice zero of the window ladder; return the job id."""
    from .. import remote
    from ..remote.scripts import _uv_sync_block

    _validate_windows(args)
    jobid = remote._new_jobid()
    jobdir = remote.remote_path(remote.JOBS_SUBDIR, jobid)
    command = remote_windows_command(args, remote.shell_remote_uv())
    script = remote._slurm_batch_script(
        jobid,
        _slice_payload(jobdir, command, remote, _uv_sync_block(once=True)),
        job_name=WINDOW_JOB_KIND,
        time_limit=str(int(args.time_limit_minutes)),
    )
    metadata = "\n".join(
        [
            f"job: {jobid}",
            f"kind: {WINDOW_JOB_KIND}",
            f"slice_minutes: {float(args.slice_minutes):g}",
            f"json_out: {args.json_out}",
            f"materials: {args.materials}",
            f"energies: {args.energies}",
            f"tilts: {args.tilts}",
            f"samples: {args.samples}",
            f"providers: {args.providers}",
            f"backbone_eV: {float(args.backbone):g}",
            f"ne: {int(args.ne)}",
            "progress_dashboard: False",
            "",
        ]
    )
    if args.dry_run:
        print(script)
        return jobid
    if not args.no_sync:
        remote.sync_code()
    remote._stage_job_script(jobid, [], remote._write_job_script_command(jobdir, metadata), script)
    scheduler_id = remote._submit_staged_job(jobid, [], nice=True)
    print(f"submitted SLURM job {scheduler_id} as {jobid}")
    print(f"status: python -m pyrite.energy_grid.convergence_job status {jobid}")
    print(f"pull:   python -m pyrite.energy_grid.convergence_job pull --json-out {args.json_out}")
    return jobid


def remote_precision_commands(args: argparse.Namespace, uv: str) -> list[str]:
    """Transport once (FP64), evaluate under FP64 then float32, compare."""
    stem = args.json_out[: -len(".json")]
    module = "run --no-sync python -m pyrite.energy_grid.precision_ladder"
    payload, fp64, fp32 = (f"{stem}.segments.pkl", f"{stem}.fp64.npz", f"{stem}.fp32.npz")
    transport = " ".join(
        [
            f"PYRITE_FP64=1 PYRITE_MC_BACKEND=cuda {uv} {module} transport",
            f"--material {shlex.quote(args.material)}",
            f"--energy {float(args.energy):g}",
            f"--tilt {float(args.tilt):g}",
            f"--azimuth {float(args.azimuth):g}",
            f"--thickness {float(args.thickness):g}",
            f"--ne {int(args.ne)}",
            f"--seed {int(args.seed)}",
            f"--bands {shlex.quote(args.bands)}",
            f"--window {float(args.window):g}",
            f"--out {shlex.quote(payload)}",
        ]
    )
    evaluate = (
        f"{module} evaluate --payload {shlex.quote(payload)} --ratios {shlex.quote(args.ratios)}"
    )
    return [
        transport,
        f"PYRITE_FP64=1 PYRITE_MC_BACKEND=cuda {uv} {evaluate} --expect-dtype float64 "
        f"--out {shlex.quote(fp64)}",
        f"env -u PYRITE_FP64 PYRITE_MC_BACKEND=cuda {uv} {evaluate} --expect-dtype float32 "
        f"--out {shlex.quote(fp32)}",
        f"PYRITE_MC_BACKEND=cpu {uv} {module} compare --reference {shlex.quote(fp64)} "
        f"--candidate {shlex.quote(fp32)} --json-out {shlex.quote(args.json_out)}",
    ]


def _precision_payload(jobdir: str, commands: Sequence[str], remote, sync_block: str) -> str:
    steps = "\n".join(
        f'{command} >> "$JOBDIR/log" 2>&1 || {{ rc=$?; echo "FAILED (precision step {index} '
        f'exit $rc) $(date -Is)" > "$JOBDIR/state"; exit "$rc"; }}'
        for index, command in enumerate(commands, start=1)
    )
    return f"""JOBDIR={remote.shell_word(jobdir)}
cd {remote.shell_remote_dir()} || exit 1
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
echo "running line-grid precision $(date -Is)" > "$JOBDIR/state"
{sync_block}
{steps}
echo "done $(date -Is)" > "$JOBDIR/state"
"""


def start_precision(args: argparse.Namespace) -> str:
    """Submit the float32-versus-FP64 lineshape measurement as one job."""
    from .. import remote
    from ..remote.scripts import _uv_sync_block
    from .job import _validate_remote_output_name

    _validate_remote_output_name(args.json_out)
    if not args.material.isalnum():
        raise SystemExit("--material must be a catalog key of letters and digits")
    if not args.ratios or not set(args.ratios) <= set("0123456789.,"):
        raise SystemExit("--ratios may contain only digits, '.', ','")
    if not args.bands or not set(args.bands) <= set("0123456789.,:"):
        raise SystemExit("--bands may contain only digits, '.', ',', ':'")
    jobid = remote._new_jobid()
    jobdir = remote.remote_path(remote.JOBS_SUBDIR, jobid)
    commands = remote_precision_commands(args, remote.shell_remote_uv())
    script = remote._slurm_batch_script(
        jobid,
        _precision_payload(jobdir, commands, remote, _uv_sync_block()),
        job_name=PRECISION_JOB_KIND,
        time_limit=str(int(args.time_limit_minutes)),
    )
    metadata = "\n".join(
        [
            f"job: {jobid}",
            f"kind: {PRECISION_JOB_KIND}",
            f"json_out: {args.json_out}",
            f"material: {args.material}",
            f"energy_keV: {float(args.energy):g}",
            f"ratios: {args.ratios}",
            f"bands: {args.bands}",
            "progress_dashboard: False",
            "",
        ]
    )
    if args.dry_run:
        print(script)
        return jobid
    if not args.no_sync:
        remote.sync_code()
    remote._stage_job_script(jobid, [], remote._write_job_script_command(jobdir, metadata), script)
    scheduler_id = remote._submit_staged_job(jobid, [], nice=True)
    print(f"submitted SLURM job {scheduler_id} as {jobid}")
    print(f"status: python -m pyrite.energy_grid.convergence_job status {jobid}")
    print(f"pull:   python -m pyrite.energy_grid.convergence_job pull --json-out {args.json_out}")
    return jobid


def pull(args: argparse.Namespace) -> Path:
    """Copy the remote checkpoint JSON into ``--dest``."""
    from .. import remote
    from .job import _validate_remote_output_name

    _validate_remote_output_name(args.json_out)
    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    local = destination / args.json_out
    remote._run(["scp", remote.scp_remote_path(remote.remote_path(args.json_out)), str(local)])
    print(local)
    return local


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("start", help="submit a chained remote ladder job")
    add_ladder_arguments(submit)
    submit.add_argument("--slice-minutes", type=float, default=DEFAULT_SLICE_MINUTES)
    submit.add_argument("--time-limit-minutes", type=int, default=DEFAULT_TIME_LIMIT_MINUTES)
    submit.add_argument("--no-sync", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    windows = commands.add_parser(
        "start-windows", help="submit a chained remote window-refinement ladder job"
    )
    add_window_arguments(windows)
    windows.add_argument("--slice-minutes", type=float, default=DEFAULT_SLICE_MINUTES)
    windows.add_argument("--time-limit-minutes", type=int, default=DEFAULT_TIME_LIMIT_MINUTES)
    windows.add_argument("--no-sync", action="store_true")
    windows.add_argument("--dry-run", action="store_true")
    precision = commands.add_parser(
        "start-precision", help="submit the float32-versus-FP64 lineshape measurement"
    )
    precision.add_argument("--material", default="wse2")
    precision.add_argument("--energy", type=float, default=300.0)
    precision.add_argument("--tilt", type=float, default=5.0)
    precision.add_argument("--azimuth", type=float, default=95.0)
    precision.add_argument("--thickness", type=float, default=1.0e6)
    precision.add_argument("--ne", type=int, default=200)
    precision.add_argument("--seed", type=int, default=0)
    precision.add_argument("--ratios", default=DEFAULT_PRECISION_RATIOS)
    precision.add_argument("--bands", default=DEFAULT_PRECISION_BANDS)
    precision.add_argument("--window", type=float, default=200.0)
    precision.add_argument(
        "--json-out", default=f"line_grid_precision_{date.today()}.json", help="report JSON"
    )
    precision.add_argument(
        "--time-limit-minutes", type=int, default=DEFAULT_PRECISION_TIME_LIMIT_MINUTES
    )
    precision.add_argument("--no-sync", action="store_true")
    precision.add_argument("--dry-run", action="store_true")
    fetch = commands.add_parser("pull", help="copy a remote checkpoint JSON locally")
    fetch.add_argument("--json-out", required=True)
    fetch.add_argument("--dest", default=".")
    for name in ("status", "attach", "stop"):
        commands.add_parser(name).add_argument("jobid", nargs="?")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        start(args)
        return 0
    if args.command == "start-windows":
        start_windows(args)
        return 0
    if args.command == "start-precision":
        start_precision(args)
        return 0
    if args.command == "pull":
        pull(args)
        return 0
    from .. import remote

    if args.command == "status":
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
