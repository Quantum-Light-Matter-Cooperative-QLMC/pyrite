"""Remote ladder/precision job payloads (issue #109).

Regression: the first submissions ran ``uv run --no-sync`` on a box whose venv
predated the lockfile and died on ``import endf_parserpy``. Every payload must
run the standard remote dependency sync before its first work command.
"""

import shlex
from types import SimpleNamespace

import pytest

from pyrite.energy_grid import convergence_job as job

REMOTE = SimpleNamespace(shell_word=shlex.quote, shell_remote_dir=lambda: "/box/pyrite")
SYNC = "uv sync --package pyrite-xray --no-dev --extra nvidia  # SYNC-MARKER"


def _args(**overrides):
    parser = job.build_parser()
    args = parser.parse_args(["start-precision", "--json-out", "precision.json"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_ladder_slice_syncs_dependencies_before_running():
    payload = job._slice_payload("/box/jobs/j1", "LADDER-COMMAND", REMOTE, SYNC)

    assert payload.index("SYNC-MARKER") < payload.index("LADDER-COMMAND")
    assert payload.index("cd /box/pyrite") < payload.index("SYNC-MARKER")


def test_precision_job_syncs_dependencies_before_every_step():
    commands = job.remote_precision_commands(_args(), "uv")
    payload = job._precision_payload("/box/jobs/j2", commands, REMOTE, SYNC)

    first_step = min(payload.index(command) for command in commands)
    assert payload.index("SYNC-MARKER") < first_step


def test_precision_steps_pin_their_precision_and_share_one_payload():
    transport, fp64, fp32, compare = job.remote_precision_commands(_args(), "uv")

    assert transport.startswith("PYRITE_FP64=1 PYRITE_MC_BACKEND=cuda")
    assert "--expect-dtype float64" in fp64 and fp64.startswith("PYRITE_FP64=1")
    assert "--expect-dtype float32" in fp32 and fp32.startswith("env -u PYRITE_FP64")
    for step in (fp64, fp32):
        assert "--payload precision.segments.pkl" in step
    assert "--reference precision.fp64.npz --candidate precision.fp32.npz" in compare


def test_bandwidth_job_forwards_local_resolution_to_remote_steps():
    args = job.build_parser().parse_args(
        [
            "start-bandwidth",
            "--resolution",
            "local",
            "--compare-resolution",
            "uniform",
            "--json-out",
            "bandwidth.json",
        ]
    )
    reference, candidate = job.remote_bandwidth_commands(args, "uv")
    assert "--resolution local" in reference
    assert "--compare-resolution uniform" in reference
    assert "--payload bandwidth.segments.pkl" in reference
    assert "--payload bandwidth.segments.pkl" in candidate


@pytest.mark.parametrize("options", [["--production"], ["--resolution", "uniform"]])
def test_bandwidth_comparison_requires_a_distinct_reference_grid(options):
    args = job.build_parser().parse_args(
        ["start-bandwidth", "--compare-resolution", "uniform", *options]
    )
    with pytest.raises(SystemExit, match="--compare-resolution"):
        job.start_bandwidth(args)


@pytest.mark.parametrize("value", ["", "5000", "M", "5.5G", "-1M", "5000MB"])
def test_bandwidth_job_rejects_malformed_memory_requests(value):
    args = job.build_parser().parse_args(["start-bandwidth", "--mem-per-cpu", value])
    with pytest.raises(SystemExit, match="--mem-per-cpu"):
        job.start_bandwidth(args)


def test_bandwidth_attribution_runs_fp32_then_fp64_reports():
    args = job.build_parser().parse_args(
        ["start-bandwidth", "--attribute", "--top", "50", "--json-out", "attr.json"]
    )
    fp32, fp64 = job.remote_bandwidth_commands(args, "uv")
    assert fp32.startswith("env -u PYRITE_FP64") and " attribute " in fp32
    assert fp64.startswith("PYRITE_FP64=1") and " attribute " in fp64
    assert "--top 50" in fp32 and "--json-out attr.json" in fp32
    assert "--json-out attr.fp64.json" in fp64


@pytest.mark.parametrize("options", [["--production"], ["--compare-resolution", "local"]])
def test_bandwidth_attribution_runs_alone(options):
    args = job.build_parser().parse_args(["start-bandwidth", "--attribute", *options])
    with pytest.raises(SystemExit, match="--attribute"):
        job.start_bandwidth(args)
