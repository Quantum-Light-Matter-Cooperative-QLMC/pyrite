# Remote per-material progress dashboard

## Goal

Make `cxr remote attach` show accurate, independent case progress for every
material in newly submitted concurrent SLURM jobs. Keep `cxr remote logs` as
the raw diagnostic stream.

## Scope

This applies only to jobs submitted after the change. Existing jobs retain
their existing log-only behavior because they do not create progress records.

## Data flow

1. The SLURM queue starts each material scan with an internal progress-file
   argument and disables that scan's terminal `tqdm` output.
2. `scan.py` creates and atomically replaces its own progress record when it
   starts, after each completed case, and when it finishes or fails.
3. `run_sweep` supplies the per-case completion event to the scan-level writer.
   The record includes material, total cases, cached cases, completed new cases,
   and state.
4. `cxr remote attach` polls all records in the job directory and renders one
   local progress bar per known material. It also continues to poll SLURM and
   exits once the allocation has finished.

## Behavior and failures

- Before a scan creates its record, attach shows that material as pending.
- A malformed or temporarily unreadable record is ignored for that poll; a
  later atomic replacement supplies a complete record.
- Ctrl-C disconnects only the local viewer. A terminal SLURM state closes the
  bars and prints the job's final persisted state.
- The shared job log no longer contains terminal progress-bar control output
  from remote queue scans.

## Tests

- Scan/run tests verify initial, per-case, and terminal records using a
  temporary path.
- Remote tests verify queue-script progress-file wiring and progress snapshot
  parsing/rendering without SSH or SLURM.
- Existing remote lifecycle tests remain the regression guard for disconnect,
  scheduler failure, and final-state handling.
