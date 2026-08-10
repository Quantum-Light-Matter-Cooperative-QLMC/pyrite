"""``pyrite remote`` -- schedule heavy CXR scans on the lab GPU box, keep data-vis local.

The split this enables: the laptop holds the project and does all interactive
analysis and static-HTML export, while
the lab box (an RTX 5080, ssh host 'qlmc') only does the GPU-heavy Monte-Carlo
sweep. Every compute-producing subcommand ships the current code up, submits a
one-GPU SLURM batch script there (see :mod:`pyrite.runs.scan`), and pulls results into
./checkpoints
-- so you never hand-ssh in or copy files, and the lab box needs no visualization
toolchain.

Optional, dev-only tool: it is only useful if you have an ssh host configured
(default 'qlmc', override via PYRITE_REMOTE_HOST) to run sweeps on. Every other
``pyrite`` command works without it.

Run, wait for SLURM, then pull:

    pyrite run standard -m mose2 --remote    # sync, submit, track, pull
    pyrite run standard -m mose2 --remote --quick
    pyrite run standard -m mose2 --remote --no-sync
    pyrite run sub_100keV --remote --detach
    pyrite remote pull mose2 wse2          # fetch existing checkpoints (grid-filtered)
    pyrite remote pull mose2 --full        # fetch the full, un-filtered checkpoint
    pyrite remote rebrem mose2 --ne-brem 1000   # brem-only recompute of the box's
                                             # checkpoints (GPU), follow, pull back
    pyrite remote sync                     # only push the current code
    pyrite run --preset zhai --remote [--ne N] [--ne-brem N] [--ne-supp N]
                                      # run the Zhai + supplementary MC
    pyrite remote pull --preset zhai      # pull an existing Zhai cache

Remote jobs survive SSH disconnects:

pyrite remote status [JOBID] --attach  # (re)connect + track live (default: latest)
    pyrite remote jobs                     # list jobs on the box + their state
    pyrite remote status [JOBID] [-v|-vv]  # job summary; SLURM details; case progress
    pyrite remote logs [JOBID] --follow    # tail the remote log (live)
    pyrite remote stop mose2 wse2          # cancel live SLURM job(s) by material
    pyrite remote stop --all               # cancel every live SLURM job
    pyrite remote pull mose2 wse2 mos2     # fetch the finished checkpoints (grid-filtered)

`run --remote --detach` returns after shipping code, writing a batch script
under <remote>/jobs/<jobid>/, and submitting it to SLURM. The batch job
processes profile members with bounded concurrency, writing meta/state/log and
its scheduler ID into the job dir.

The SLURM job is independent of the submission SSH connection, so the SSH link
is only ever a VIEWER. `attach` renders one local case-progress bar per material;
`logs --follow` remains the raw shared diagnostic stream. Both exit when the job
finishes. To DISCONNECT, just Ctrl-C (or close the terminal / drop the link) --
that tears down the viewer only, and the job runs to completion. Reconnect any
time with `attach`/`status`/`logs`, then `pull` once state is `done`.

Then locally: run ``pyrite app analysis <material>`` or ``pyrite app analysis export [stem]``.

Transport is ssh/scp only (uses the 'qlmc' host in ~/.ssh/config, cloudflared
ProxyCommand and all) -- no rsync dependency, so it works from Windows Git Bash.
Override the box via env: PYRITE_REMOTE_HOST / PYRITE_REMOTE_DIR / PYRITE_REMOTE_UV.
The corresponding CXR_* spellings remain compatibility aliases.
"""

# ---------------------------------------------------------------------------
# Facade module. The remote-job subsystem is split into focused submodules
# under ``pyrite.remote`` (config, transport, scripts, state, lifecycle,
# viewer, cli). This module re-exports their public and internal names so every
# existing ``remote.<name>`` reference keeps working unchanged.
#
# CAVEAT (monkeypatching): the names below are *snapshots* bound at import time
# for call convenience. Internal cross-module calls resolve through the OWNING
# submodule (e.g. ``lifecycle.py`` calls ``transport._ssh_capture``), so a test
# that patches ``remote._ssh_capture`` will NOT intercept those internal calls.
# Patch the owning module instead: ``monkeypatch.setattr(transport, "_ssh_capture", ...)``.
# Patching a facade name only works when the caller itself resolves through
# ``remote.<name>`` at call time (as ``energy_grid/job.py`` does).
# ---------------------------------------------------------------------------

from .._compat import env_value
from ..cli import dashboard as _presentation
from . import (
    cli,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)

# --- from config ------------------------------------------------------
# Compatibility snapshot only; subsystem calls resolve the effective host
# dynamically through ``config.remote_host()``.
HOST = env_value("CXR_REMOTE_HOST", "qlmc")
REMOTE_DIR = config.REMOTE_DIR
REMOTE_UV = config.REMOTE_UV
SLURM_PARTITION = config.SLURM_PARTITION
SLURM_GPUS = config.SLURM_GPUS
SLURM_TIME = config.SLURM_TIME
DEFAULT_PARALLEL_MATERIALS = config.DEFAULT_PARALLEL_MATERIALS
MAX_PARALLEL_MATERIALS = config.MAX_PARALLEL_MATERIALS
LOCAL_ROOT = config.LOCAL_ROOT
JOBS_SUBDIR = config.JOBS_SUBDIR
RESERVATIONS_SUBDIR = config.RESERVATIONS_SUBDIR
ZHAI_STEM = config.ZHAI_STEM
SYNC_PATHS = config.SYNC_PATHS
TEXT_EXTS = config.TEXT_EXTS
remote_host = config.remote_host
remote_dir = config.remote_dir
remote_uv = config.remote_uv
remote_path = config.remote_path
shell_word = config.shell_word
shell_remote_dir = config.shell_remote_dir
shell_remote_uv = config.shell_remote_uv
scp_remote_path = config.scp_remote_path

# --- from transport ---------------------------------------------------
_run = transport._run
_ssh_capture = transport._ssh_capture
_local_sha256 = transport._local_sha256
_remote_sha256 = transport._remote_sha256
_check_shell_tokens = transport._check_shell_tokens
_check_materials = transport._check_materials
_add_to_tar = transport._add_to_tar
sync_code = transport.sync_code

# --- from scripts -----------------------------------------------------
_stems = scripts._stems
_new_jobid = scripts._new_jobid
_validate_parallel_materials = scripts._validate_parallel_materials
_queue_script = scripts._queue_script
_chunked_queue_script = scripts._chunked_queue_script
_zhai_flags = scripts._zhai_flags
_zhai_queue_script = scripts._zhai_queue_script
_rebrem_flags = scripts._rebrem_flags
_rebrem_queue_script = scripts._rebrem_queue_script
_rebrem_chunked_queue_script = scripts._rebrem_chunked_queue_script
_reservation_root = scripts._reservation_root
_release_checkpoint_stems_command = scripts._release_checkpoint_stems_command
_release_job_reservations_command = scripts._release_job_reservations_command
_reserve_checkpoint_stems_command = scripts._reserve_checkpoint_stems_command
_slurm_batch_script = scripts._slurm_batch_script
_submit_slurm_command = scripts._submit_slurm_command
_queue_metadata = scripts._queue_metadata
_zhai_queue_metadata = scripts._zhai_queue_metadata
_rebrem_queue_metadata = scripts._rebrem_queue_metadata
_write_job_script_command = scripts._write_job_script_command
_squeue_state_command = scripts._squeue_state_command
_clear_checkpoint_stems_command = scripts._clear_checkpoint_stems_command
_reap_job_command = scripts._reap_job_command
_recorded_job_dirs_command = scripts._recorded_job_dirs_command
_job_assign = scripts._job_assign

# --- from state -------------------------------------------------------
_completed_materials = state._completed_materials
_live_jobs = state._live_jobs
_reservation_holders = state._reservation_holders
_slurm_job_id = state._slurm_job_id
_slurm_state = state._slurm_state
_job_state = state._job_state
_job_metadata = state._job_metadata
_job_succeeded = state._job_succeeded
_reservation_ledger = state._reservation_ledger
_is_reapable = state._is_reapable
_orphaned_reservation_jobs = state._orphaned_reservation_jobs
_latest_jobid = state._latest_jobid

# --- from lifecycle ---------------------------------------------------
pull = lifecycle.pull
pull_zhai_cache = lifecycle.pull_zhai_cache
_stage_job_script = lifecycle._stage_job_script
_submission_outcome = lifecycle._submission_outcome
_release_if_submission_definitely_failed = lifecycle._release_if_submission_definitely_failed
_submit_staged_job = lifecycle._submit_staged_job
_refuse_if_busy = lifecycle._refuse_if_busy
clear_remote = lifecycle.clear_remote
clear_all_remote = lifecycle.clear_all_remote
start_queue = lifecycle.start_queue
start_zhai_queue = lifecycle.start_zhai_queue
start_rebrem_queue = lifecycle.start_rebrem_queue
_stop_jobid = lifecycle._stop_jobid
stop_jobs = lifecycle.stop_jobs
reap_reservations = lifecycle.reap_reservations

# --- from viewer ------------------------------------------------------
list_jobs = viewer.list_jobs
_status_remote_command = viewer._status_remote_command
job_status = viewer.job_status
tail_logs = viewer.tail_logs
_disconnect_hint = viewer._disconnect_hint
_POLL_GRACE_POLLS = viewer._POLL_GRACE_POLLS
_is_terminal_state = viewer._is_terminal_state
_render_frame = viewer._render_frame
_attach_header = viewer._attach_header
_live_status = viewer._live_status
attach = viewer.attach

# --- from cli ---------------------------------------------------------
command = cli.command
remote_scan = cli.remote_scan
remote_check = cli.remote_check
_ensure_utf8_stdio = cli._ensure_utf8_stdio
_dispatch = cli._dispatch
_selected_materials = cli._selected_materials
_cli_rebrem = cli._cli_rebrem
_cli_pull = cli._cli_pull
_cli_start = cli._cli_start
_cli_jobs = cli._cli_jobs
_cli_status = cli._cli_status
_cli_logs = cli._cli_logs
_cli_stop = cli._cli_stop
_cli_reap = cli._cli_reap
_cli_clear = cli._cli_clear
_cli_sync = cli._cli_sync
_cli_check = cli._cli_check
main = cli.main

# --- from presentation ------------------------------------------------
_SHELL_TOKEN_RE = _presentation._SHELL_TOKEN_RE
_STATE_COLORS = _presentation._STATE_COLORS
_TQDM_FRAME_RE = _presentation._TQDM_FRAME_RE
_active_work_label = _presentation._active_work_label
_aggregate_progress = _presentation._aggregate_progress
_overall_progress_line = _presentation._overall_progress_line
_clean_recent_log = _presentation._clean_recent_log
_color_enabled = _presentation._color_enabled
_format_case_progress = _presentation._format_case_progress
_format_fields = _presentation._format_fields
_format_job_status = _presentation._format_job_status
_format_material_roster = _presentation._format_material_roster
_format_now_testing = _presentation._format_now_testing
_format_table = _presentation._format_table
_legacy_progress = _presentation._legacy_progress
_marked_sections = _presentation._marked_sections
_material_label = _presentation._material_label
_metadata_fields = _presentation._metadata_fields
_metadata_value = _presentation._metadata_value
_mode_summary = _presentation._mode_summary
_paint = _presentation._paint
_parse_progress_records = _presentation._parse_progress_records
_progress_group = _presentation._progress_group
_progress_track = _presentation._progress_track
_scheduler_fields = _presentation._scheduler_fields
_style_states = _presentation._style_states
