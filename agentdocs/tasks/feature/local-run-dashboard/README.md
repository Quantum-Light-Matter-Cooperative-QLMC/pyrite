# Local run progress dashboard

Branch: `feature/local-run-dashboard`
TODO scope: new P2 item, "Local run progress dashboard."

## Problem

`cxr remote status -a -vv` renders a rich, multi-panel dashboard (metadata /
mode summary, SLURM `SQUEUE`, compute usage GPU/CPU/RAM/VRAM, overall progress,
per-material CASE PROGRESS, now-testing, performance profiles, recent log).
Local `cxr run <profile>` renders only the bare `run_cases` **tqdm** bar
(`src/cxr_mc/montecarlo/runner.py:1277`). Same run, far poorer local
observability.

Local can only ever show a subset: no SLURM queue unless the run happens to sit
on a SLURM node, and not every machine has a GPU. The dashboard must degrade
gracefully — render whichever panels have data, drop the rest — never error or
show empty scaffolding.

## Key finding — most plumbing already exists

The remote/local split is a *rendering* seam, not a telemetry gap:

- **Renderer is already section-driven and partial-safe.**
  `presentation._format_job_status(sections, detail)`
  (`src/cxr_mc/_remote/presentation.py:729`) reads sections via
  `sections.get(...)` — a missing `SQUEUE`/`PERFORMANCE`/`PROGRESS` already
  yields an omitted panel, not an error. `detail` is the `-vv` verbosity level.
- **Section wire format is reusable in-process.** `_encode_sections` /
  `_marked_sections` (`presentation.py:58,629`) round-trip a
  `{name: payload}` map; known names: `META STATE SQUEUE PERFORMANCE PROGRESS
  QUEUE JOB` (see `_encode_sections` allow-list ~`presentation.py:31`).
- **Local run already emits the PROGRESS data.** `scan.py:command` has hidden
  `--progress-file` + `_write_progress_record` (`src/cxr_mc/scan.py:1094`),
  writing the exact record format `_parse_progress_records` consumes. Callbacks
  `_record_progress/_record_cost/_record_runtime/_record_timing/_record_activity`
  are already wired through `run_sweep`/`run_cases`. `cxr rebrem
  --progress-file` already feeds the remote dashboard from these records.
- **Remote emits PERFORMANCE (GPU/CPU) via an nvidia-smi + host shell probe**
  in `viewer._status_remote_command` (`_remote/viewer.py:82,94`). Local needs an
  in-process equivalent that no-ops when `nvidia-smi` is absent.

Gap = a local in-process **section emitter + refresh loop** that reuses
`_format_job_status`, replacing tqdm when interactive.

## Implementation path & likely owners

- `src/cxr_mc/_remote/presentation.py` — the renderer lives under `_remote/`
  but is transport-agnostic. Extract the render surface (`_format_job_status`,
  `_encode_sections`/`_marked_sections`, `_format_compute_usage`,
  `_overall_progress_line`, `_parse_progress_records`, `_style_states`) into
  `cli/_dashboard.py` (decided — see Decisions) that both `_remote` and local
  `scan.py` import. Keep `_remote/presentation.py` re-exporting to avoid churn.
  **No visual/format change to the remote view.**
- `src/cxr_mc/scan.py` — build sections in-process from the existing callback
  state (`latest_progress`, `latest_case`, cost/timing snapshots): `META` from
  run config/mode, `STATE` from run lifecycle, `PROGRESS` from records,
  `PERFORMANCE` from a local GPU/host probe when available, `SQUEUE` only when
  on a SLURM node (`$SLURM_JOB_ID` / `squeue` present). Render via
  `_format_job_status(sections, detail)` on a timed refresh.
- `src/cxr_mc/montecarlo/runner.py` — gate/replace the tqdm bar
  (`runner.py:1277`) when the local dashboard is active; keep tqdm as fallback.
- Local GPU/host probe: small helper (reuse backend detection from the
  backend-autodetect work if landed) shelling `nvidia-smi` guarded by
  `shutil.which`; emit nothing on absence.

## Degradation matrix (acceptance-relevant)

| Condition | Behavior |
|---|---|
| No SLURM (`$SLURM_JOB_ID` unset, no `squeue`) | omit `SQUEUE` panel |
| No GPU (`nvidia-smi` absent) | `PERFORMANCE` shows CPU/host only, or omit GPU rows |
| Non-interactive / piped / `NO_COLOR` / `--no-progress` | fall back to current tqdm/plain output |
| Verbosity | map a local `-v/-vv` to `detail` like remote |

## Stepwise checklist

- [ ] Extract shared renderer into `cli/_dashboard.py`; `_remote/presentation.py`
      re-exports; **snapshot-verify remote view byte-identical**.
- [ ] Local section emitter in `scan.py` from existing callback state; PROGRESS
      via in-memory records → `json.dumps` → `_parse_progress_records`, with
      sanitization factored into a shared helper.
- [ ] Local GPU/host probe helper; guarded, silent on absence.
- [ ] Refresh loop reusing `_render_frame` (`_remote/viewer.py:264`) + a thin
      local timer tick — not `_live_status` wholesale — replacing tqdm when
      interactive; tqdm fallback when non-tty.
- [ ] Wire `-v/-vv` verbosity → `detail`; honor `--no-progress`, `NO_COLOR`,
      non-tty fallback.
- [ ] Tests: emitter produces valid sections; partial-section rendering (no
      SLURM, no GPU); non-tty falls back to tqdm; remote render snapshot
      unchanged.
- [ ] Regenerate `docs/cli-reference.md` if any `cxr run` help/flags change.

## Decisions (closed)

- **Renderer: reuse `_render_frame` + `_format_job_status`; no `rich`.** `rich`
  is absent from `pyproject.toml` and unimported in `src/`; do not add it.
  `_remote/viewer._render_frame` (`viewer.py:264`) already does a zero-dep ANSI
  redraw (`\x1b[H\x1b[2J\x1b[3J` + frame), tty-gated, over exactly
  `_style_states(_format_job_status(sections, detail))`. Extract **only** the
  frame-render + a thin local timer-tick loop. Do **not** reuse `_live_status`
  wholesale — it is SSH/SLURM-welded (`_KeyListener`, `scancel` on x+y, chain-hop
  scheduler-ID watchdog, `_status_stream`; `viewer.py:408-482`), none of which
  the local case wants. tqdm stays the non-tty fallback, gated on
  `presentation._color_enabled()` exactly as the viewer does.
- **Shared-renderer home: extract into `cli/_dashboard.py`; `_remote` re-exports.**
  Move the transport-agnostic surface (`_format_job_status`,
  `_encode_sections`/`_marked_sections`, `_format_compute_usage`,
  `_overall_progress_line`, `_parse_progress_records`, `_style_states`,
  glyph/color tables). No cycle: `presentation.py` imports only stdlib +
  `cli._core` (`presentation.py:11`) — no transport/ssh — and `cli/_core.py`
  imports neither `scan` nor `presentation` nor `_remote` at module scope, so
  `scan → cli._dashboard → cli._core` has no path back to `scan`.
  `_remote/presentation.py` re-exports the moved names for back-compat. Guard
  the move with a snapshot test asserting `cxr remote status -a -vv` output is
  byte-identical before/after.
- **PROGRESS source: in-memory records → `json.dumps` → `_parse_progress_records`.**
  `scan.py` already holds the structured state before `_write_progress_record`
  serializes it, and `_parse_progress_records` (`presentation.py:888`) derives
  nothing new — it only validates/sanitizes and keys by material. So build the
  record dicts in memory, `json.dumps` them to a string, and feed
  `_parse_progress_records(string)`: same validation as remote, **no disk I/O**,
  no divergence. Factor the per-record sanitization
  (`_sanitize_cost_fields`/`_sanitize_timing_fields`) into a helper both the
  parser and the in-memory builder call, so validation lives in one place. A
  first slice may poll the existing `--progress-file` scan already writes, but
  the in-memory path is the clean end state — do not couple the local dashboard
  to a temp file long-term.
- **SLURM-on-node detection.** `$SLURM_JOB_ID` present ⇒ emit `SQUEUE` for own
  job only; do not shell `squeue` for the whole queue locally.

## Delegation / required skills

- `cli-ui-ux` — dashboard layout, verbosity, tty/NO_COLOR/`--no-progress`
  fallback contract.
- `run-cxr-mc` — drive real `cxr run <profile>` on GPU and CPU-only boxes to
  confirm partial rendering.
- `performance` — ensure refresh loop / probe adds no measurable per-case
  overhead vs tqdm.
- `regression-testing` — section-emitter + partial-render + snapshot tests.
- `documentation-maintenance` — `docs/cli-reference.md` regen if flags change.

## Acceptance checks

- `cxr run <profile>` on a GPU box shows META + PROGRESS + PERFORMANCE(GPU) +
  CASE PROGRESS panels live; matches remote look for shared panels.
- Same command on a CPU-only, non-SLURM box shows META + PROGRESS + CASE
  PROGRESS, **no** SQUEUE, **no** GPU rows — no errors, no empty scaffolding.
- Piped / `--no-progress` / non-tty falls back to tqdm/plain.
- Remote `cxr remote status -a -vv` render is byte-for-byte unchanged.
- `cxr-dev verify` green; `docs/cli-reference.md` regenerated if flags changed.
