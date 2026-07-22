"""Environment + configuration constants for the remote job subsystem."""

import os
from pathlib import Path

HOST = os.environ.get("CXR_REMOTE_HOST", "qlmc")
REMOTE_DIR = os.environ.get("CXR_REMOTE_DIR", "/home/aamador/dev/cxr-mc")
REMOTE_UV = os.environ.get("CXR_REMOTE_UV", "/home/aamador/.local/bin/uv")
SLURM_PARTITION = "gpu"
SLURM_GPUS = 1
SLURM_TIME = "UNLIMITED"
DEFAULT_PARALLEL_MATERIALS = 2
MAX_PARALLEL_MATERIALS = 4
# repo root = three levels up from src/cxr_mc/_remote/config.py. remote.py orchestrates
# the *checkout* (it tars the working tree up to the box), so it resolves paths
# against the repo root, not its own package dir.
LOCAL_ROOT = Path(__file__).resolve().parents[3]
MATS_FILE = LOCAL_ROOT / "mats_to_sim.toml"

# detached-job bookkeeping lives under <REMOTE_DIR>/jobs/<jobid>/ on the box
# (gitignored there): run.sh, meta, state, log. One subdir per `start`.
JOBS_SUBDIR = "jobs"
RESERVATIONS_SUBDIR = "reservations"

# The Zhai reproduction job has no crystal key of its own, but reusing the
# existing material-stem bookkeeping (_refuse_if_busy / _live_jobs /
# stop_jobs) needs one to key off of -- this is that synthetic token.
ZHAI_STEM = "zhai"

# what `sync` ships up: the code that changes (the src/ package now also carries
# data/, so it travels too), plus the root scan.py shim the box invokes and
# pyproject.toml. Not checkpoints/ (the output we pull back the other way).
SYNC_PATHS = [
    "src",
    "scan.py",
    "reproduce_zhai.py",
    "checks",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "mats_to_sim.toml",
]

# text extensions whose CRLF is normalized to LF before tarring (see _add_to_tar):
# the laptop is Windows so its working files are CRLF, and shipping those over the
# box's LF checkout dirties `git status` there even though content is identical.
TEXT_EXTS = {".py", ".toml", ".cfg", ".ini", ".txt", ".md", ".csv"}
