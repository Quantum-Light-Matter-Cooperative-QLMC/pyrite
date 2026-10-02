"""Compatibility facade for the split remote job lifecycle."""

from . import cleanup, jobs, performance, queue, recompute
from . import config as config  # noqa: F401 - compatibility module attribute
from . import pull as pulling
from . import transport as transport  # noqa: F401 - compatibility module attribute

archive = pulling.archive

_refuse_if_busy = cleanup._refuse_if_busy
clear_remote = cleanup.clear_remote
_clear_exact_remote_stems = cleanup._clear_exact_remote_stems
_profile_checkpoint_stems = cleanup._profile_checkpoint_stems
clear_all_remote = cleanup.clear_all_remote
prune_remote = cleanup.prune_remote
_run_remote_prune = cleanup._run_remote_prune
prune_job_dirs = cleanup.prune_job_dirs

_refuse_if_profile_live = queue._refuse_if_profile_live
_profile_jobid = queue._profile_jobid
start_queue = queue.start_queue
start_zhai_queue = queue.start_zhai_queue

pull_performance_profile = performance.pull_performance_profile
remote_performance_inventory = performance.remote_performance_inventory
list_remote_performance = performance.list_remote_performance
_performance_job_states = performance._performance_job_states
prune_remote_performance = performance.prune_remote_performance

start_rebrem_queue = recompute.start_rebrem_queue
start_reline_queue = recompute.start_reline_queue

_stage_job_script = jobs._stage_job_script
_submission_outcome = jobs._submission_outcome
_release_if_submission_definitely_failed = jobs._release_if_submission_definitely_failed
_submit_staged_job = jobs._submit_staged_job
_stop_jobid = jobs._stop_jobid
_stop_jobids = jobs._stop_jobids
stop_jobs = jobs.stop_jobs
reap_reservations = jobs.reap_reservations

_resolve_survey_stems = pulling._resolve_survey_stems
_split_profile_selector = pulling._split_profile_selector
_remote_meta_json = pulling._remote_meta_json
_profile_pull_candidates = pulling._profile_pull_candidates
resolve_profile_stem = pulling.resolve_profile_stem
resolve_profile_stems = pulling.resolve_profile_stems
pull = pulling.pull
pull_zhai_cache = pulling.pull_zhai_cache
