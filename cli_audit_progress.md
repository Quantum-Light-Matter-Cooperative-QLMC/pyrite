## Current findings

  ### Confirmed defect

  cxr line-grid submit accepts --tilts, --azimuths, --thickness, and --set-default,
  then silently ignores them.

  Parser defines flags in src/cxr_mc/line_grid/__init__.py:155. _cli_submit() forwards
  only materials, energies, slice duration, sync, dry-run. Safe dry-run confirmed
  generated remote command omitted all four flags.

  Follow-up trace:

  - Local derive path is correct: _derive_argv() forwards all five shared value
    options plus --set-default to derive.main().
  - Remote submit path drops exactly --tilts, --azimuths, --thickness, and
    --set-default at _cli_submit().
  - job.start() and its generated slice payload have no parameters for those four
    values, so fixing only _cli_submit() is insufficient; remote job construction
    must carry them into python -m cxr_mc.line_grid.derive.
  - tests/test_line_grid_cli.py covers status, apply, stop, and regen-golden
    dispatch, but has no derive or submit dispatch regression.

  ### Additional confirmed defects

  1. cxr line-grid apply performs semantic validation too late. apply_file() parses
     proposed TOML, writes src/cxr_mc/data/materials.toml, then calls
     load_material_catalog(). A semantic validation failure therefore leaves the
     live catalog invalid on disk. Validation must run against proposed temporary
     content/path before atomic replacement.

  2. cxr line-grid set and set-brem validate TOML syntax only. They accept invalid
     numeric domains (negative/zero stop, step, num, energy, or start relationships),
     atomically write them into materials.toml, and update provenance. They neither
     run load_material_catalog() nor regenerate/warn about the now-stale golden.

  3. cxr line-grid defaults silently ignores supplied value options unless --set is
     also present. Example: defaults --tilts 5 prints existing defaults and exits 0.

  4. The standalone python -m cxr_mc.line_grid.derive parser accepts --brem-step,
     but the value never affects derived JSON. _brem_grid_for_rows() always emits
     WIDE_BREM_STEP_EV (25 eV). Without --set-default the option is fully ignored;
     with --set-default it changes only the persistent default later consumed by
     set-brem. The cxr line-grid derive wrapper does not expose this or the other
     advanced derive controls.

  5. cxr remote logs --follow and cxr line-grid logs --follow do not check the
     ssh subprocess return code. A missing/inaccessible job can therefore make ssh
     fail while the local cxr process exits 0. Both follow paths also catch
     KeyboardInterrupt, print a disconnect message, and return success.

  6. cxr remote pull masks every per-checkpoint OSError, CalledProcessError, and
     SystemExit. It prints "warning: could not pull checkpoint ...; continuing" to
     stdout, returns None, and exits 0 even when every requested transfer fails.
     Existing tests codify continuation but expose no aggregate failure result.
     Controlled probe with _run raising OSError confirmed return None, warning-only
     stdout, and empty stderr.

  7. The wheel ships `cxr line-grid regen-golden`, but not its target golden
     snapshot. `GOLDEN_PATH` is source-tree-relative
     (`Path(__file__).resolve().parents[3] / "tests/data/..."`), so an installed
     wheel resolves it under the environment's `lib/pythonX.Y/tests/data`.
     `regen-golden --check` therefore compares against a nonexistent file and
     always exits 1 with a full add-only diff. Mutation mode would target a
     nonexistent, distribution-external tests directory. Either make this command
     explicitly source-checkout-only and fail clearly, or move its reference
     artifact to packaged data and separate packaged validation from developer
     snapshot regeneration.

  8. Remote configuration values are trusted as shell syntax rather than treated
     as data. `CXR_REMOTE_DIR` and `CXR_REMOTE_UV` are interpolated raw into SSH
     command strings and generated Bash/SBATCH scripts. `CXR_REMOTE_HOST` is
     passed to `ssh`/`scp` without an option terminator and is also interpolated
     raw into at least the follow-log remote command. A crafted environment can
     therefore execute local SSH options/ProxyCommand behavior and arbitrary
     commands on the selected remote account. Pure builder probes confirmed:
     - `REMOTE_DIR='/safe"; SENTINEL_REMOTE_DIR; #'` produces
       `JOBDIR="/safe"; SENTINEL_REMOTE_DIR; #/jobs/..."` and injected SBATCH
       output/error directives.
     - `REMOTE_UV='uv; SENTINEL_REMOTE_UV #'` produces executable
       `uv; SENTINEL_REMOTE_UV # sync ...` and scan command lines.
     - `HOST='-oProxyCommand=SENTINEL_OPTION'` reaches argv as
       `['ssh', '-n', '-oProxyCommand=SENTINEL_OPTION', ':']`.
     - A quote/semicolon-bearing HOST breaks out of the follow-log `printf`
       command.
     Fix centrally before constructing any command: validate HOST as a host alias
     rather than an option, validate remote paths as absolute POSIX paths without
     control/newline characters, shell-quote every use, and build SBATCH directive
     values under a stricter newline/whitespace policy. Add hostile-config builder
     tests for every transport/script family.

  ### Shell/SSH security audit

  Scope: local argv/environment -> subprocess argv -> SSH remote shell -> generated
  SLURM Bash; remote metadata/output -> later commands and terminal rendering.

  Confirmed strengths:

  - No audited path uses local `shell=True`; local `ssh`/`scp` calls use argv
    lists.
  - Material keys are checked against `[A-Za-z0-9_-]+` and the catalog before
    shell interpolation. Checkpoint stems and explicit job IDs use the same token
    restriction.
  - Generated job IDs are timestamp plus hex UUID suffix. Scheduler IDs are
    accepted for later interpolation only when decimal digits.
  - Numeric CLI values reach generated scripts through argparse int/float parsing
    and numeric formatting. Boolean flags are fixed literals.
  - Line-grid `json_out`, energies, and material CSV values use `shlex.quote` in
    the derive command. Whole metadata content uses `shlex.quote` before
    `printf`; no shell escape was found there.
  - Remote-derived directory basenames and metadata are generally kept in quoted
    shell variables. Values later reused as explicit job IDs or reservation
    owners pass token validation first.
  - Reservation/delete commands constrain stems, quote variable-derived paths,
    and preserve ownership checks. No ordinary material/job CLI injection path
    was found.

  Confirmed config trust-boundary defect:

  - `config.py` reads HOST, REMOTE_DIR, and REMOTE_UV directly from environment
    with no validation.
  - REMOTE_DIR is embedded across sync, clear, pull, state, viewer, reservation,
    upload, submission, generated payload, and SBATCH output/error paths.
    Quoting is inconsistent: unquoted `cd`, double-quoted assignment, and
    single-quoted command fragments all occur. Applying `shlex.quote` at one
    call site cannot fix this; normalize/validate once, then use dedicated shell
    word and SBATCH-path renderers.
  - REMOTE_UV is raw executable shell text in scan, Zhai, rebrem, and reline
    payloads. Line-grid alone quotes it.
  - HOST is an SSH/SCP destination without `--`/validation. A leading dash is
    parsed as an OpenSSH option. HOST is also included in remote/local display
    strings without control-character filtering.
  - `scp HOST:PATH` adds a second parser boundary: even argv-safe local strings
    can be interpreted as remote-shell path syntax. Current fixed `/tmp` paths
    and validated stems reduce exposure, but REMOTE_DIR remains unsafe.

  Lower-severity hardening findings:

  - Standalone `cxr_mc.line_grid.job start --json-out` permits absolute and
    parent-traversal remote output paths. Shell quoting prevents command
    injection, but job can overwrite any file writable by the remote account.
    Restrict to a basename or a designated output directory.
  - Line-grid metadata accepts newline/control-bearing CSV/path strings. Shell
    quoting is correct, but embedded newlines can forge extra metadata fields or
    corrupt status presentation. Reject C0 controls/newlines before persistence.
  - Remote metadata, state, progress JSON, scheduler reason text, and log tails
    are rendered to local terminal without general control-sequence stripping.
    `_clean_recent_log` removes tqdm frames only. A compromised remote account or
    malicious child output can emit terminal escape sequences. Raw `logs` and
    structured table/status/attach views currently share no explicit trust policy.
  - Internal marker framing uses lines beginning `@@`. Child log/progress content
    sharing those prefixes can create false sections. Use length-delimited JSON
    or escape payload lines instead of sentinel text.
  - Public Python `pull(..., dataset=...)` interpolates `dataset` into remote
    filenames/options without validation, although assembled CLI callers supply
    only fixed `"line"`/`"brem"` values. Validate at public-function boundary.

  ### Architecture findings

  Strengths:

  - Clean argparse ownership: each module registers subparser and handler.
  - Required subcommands, selective mutually-exclusive groups, testable
    main(argv=None).

  - check-config has deliberate lightweight startup path.
  - Dangerous remote operations use previews, --yes, reservations, fail-closed checks.
  - Exit 75 deliberately represents resumable work.
  - Remote behavior receives substantial focused testing.

  Problems:

  1. Eager imports slow every ordinary invocation. src/cxr_mc/cli.py:37 imports nearly
     entire command stack before parsing. Direct .venv/bin/cxr --help and --version:
     ~1.45 s; check-config --help: ~0.08 s. Import profile dominated by blaze/config,
     Monte Carlo, xraydb, SciPy, pandas.

  2. Usage errors inconsistent. Unknown command exits 2 with standard cxr: error:. cxr
     scan hopg --all raises string SystemExit, exits 1, gives no usage. Same pattern
     appears across scan/rebrem/reline/remote. Runtime failures and bad arguments lack
     stable exit contract.

  3. Numeric validation weak. Many scientific parameters accept any int/float:
     negative energies, spacing, electron counts, families, workers, and durations.
     Only scattered options use choices or explicit positivity checks. Example parser
     surface: src/cxr_mc/scan.py:84.

  4. Parser duplication risks drift. Remote scan, start, rebrem, and reline repeat
     queue/material/chunk/sync flags in src/cxr_mc/_remote/cli.py:284. Local rebrem/
     reline repeat another near-identical set.

  5. Nested help incomplete. cxr line-grid --help lists 12 verbs without descriptions
     because every nested parser is created without help=. The derive/submit wrapper
     also exposes six shared options with no help text, while the underlying
     line_grid.derive parser documents them. Top level has 15 commands; remote has 14.
     Full audit: all 44 user-defined line-grid arguments lack help text; all 12
     verbs lack parent-list summaries and leaf descriptions. Remote has summaries
     for all 14 verbs, but only stop and reap retain descriptions in leaf help.

  6. No shell completion or machine-readable output. Material keys, checkpoint stems,
     and job IDs are strong completion candidates. remote jobs/status would benefit
     from --json.

  7. Assembled CLI testing thin. Most tests exercise module handlers/parsers. Only
     check-config and startup-error tests call top-level cli.main. Missing full
     parser-tree tests for --help, --version, dispatch, exit codes, and every nested
     command.

  8. Metadata/documentation drift. Version duplicated in pyproject.toml:3 and
     src/cxr_mc/__init__.py:16. pyproject.toml correctly wires
     cxr = "cxr_mc.cli:main", but cli.py's command inventory omits newer commands and
     says _grooved where live help says _blazed.

  ### Comparison

  - Click: best migration target. Better standardized errors, reusable options,
    completion, testing, command composition, lazy groups. Migration broad; Click
    documentation warns lazy loading needs help-path tests.

  - Typer: strongest automatic typed/Rich help and completion. Poor cost-benefit here:
    large nested CLI, many custom dispatch rules, namespace-based handlers, and
    scientific validation still need callbacks.

  - Python 3.13 argparse lacks newer 3.14 colored help and typo suggestions. Could
    adopt those after raising minimum Python version.

  Implementation plan moved to `cli_implementation_plan.md`.

  Verification completed before this continuation: focused CLI suite passed,
  63 passed, 237 deselected.

  Continuation evidence:

  - Source trace completed for shared line-grid derive/submit options.
  - Remaining line-grid verbs traced through downstream validation and side
    effects:
    - status, attach, and logs preserve exact optional job IDs, validate shell
      tokens in remote._job_assign(), and otherwise use latest recorded job.
    - stop resolves latest job only when no ID is supplied, validates exact ID,
      requires a live SLURM allocation, writes STOP before scancel, waits for
      retirement, releases reservations, and stamps terminal state.
    - apply optionally pulls the default combined JSON, rewrites catalog grids,
      stamps provenance, and optionally regenerates golden. --pull wins silently
      when a positional JSON path is also supplied.
    - set and set-brem directly mutate materials.toml plus provenance.
    - defaults atomically rewrites line_grid_defaults.toml only under --set.
    - show is read-only, but an unknown material prints an empty heading and exits
      0 rather than reporting invalid input.
    - regen-golden --check is read-only and returns 1 on drift; mutation mode writes
      the snapshot directly rather than atomically.
  - Assembled parser inventory found 46 typed numeric option actions. Only three
    have argparse choices: slim --compresslevel and the two remote
    --parallel-materials actions. Eleven additional line-grid options carry
    numeric comma-separated strings (--energies/--tilts/--azimuths/--thickness
    across derive/submit, plus three defaults geometry options): 57 numeric-bearing
    cxr option surfaces total. Source-wide standalone module parsers add 12 typed
    declarations (line_grid.derive and line_grid.job), for 58 typed declarations
    overall.
  - Required numeric domains:
    - positive counts: check/remote-check electron counts, rebrem/reline electron
      counts, line-grid top-k/coarse-ne/refine-ne, scan n-families, line-grid num.
    - nonnegative worker counts with 0=serial: scan/blaze/remote workers and
      line-grid max-workers. Negative values currently collapse to serial or one
      worker downstream.
    - positive durations: local max-minutes and line-grid slice-minutes.
      Remote chunk-minutes is nonnegative with 0=monolithic. Negative max-minutes
      currently produces immediate resumable exit 75; negative chunk-minutes
      silently selects monolithic mode.
    - positive energy/length/grid values: blaze energy/spacing, rebrem/reline
      steps, line-grid energy/stop/step/grid ceilings/grid spacing/thickness, and
      defaults brem step.
    - bounded angles: polar tilt 0 < tilt < 90 for emission; azimuth 0..360 with
      the existing 90-degree emission ban. TMD azimuth may remain any finite real
      if periodic normalization is intentional.
    - sentinel/bounded values: save-every should be nonnegative with 0 disabling
      intermediate saves; remote min-age-minutes should be nonnegative (currently
      clamped to 0); beam-uvw permits signed integers but must reject (0,0,0).
  - Downstream validation remains inconsistent: line-grid slice-minutes, groove
    spacing/angles, parallel-materials, and compression level have guards;
    most counts, budgets, steps, energies, workers, and CSV values do not.
  - Fresh serial subprocess parser matrix covered all 39 assembled leaf commands
    (78 runs total):
    - every leaf --help returned 0, rendered usage to stdout, and kept stderr empty;
    - every guaranteed-invalid option returned 2, rendered usage/error to stderr,
      and kept stdout empty;
    - no parser-level stream/exit anomalies found;
    - serial help startup ranged 0.155-1.584 s, mean 1.361 s. These timings include
      one process per leaf and confirm eager-import cost broadly, but are not a
      controlled cold/warm benchmark.
  - Root dispatch has no central KeyboardInterrupt or runtime-error policy.
    argparse parse failures are consistently 2, while handler validation commonly
    raises string SystemExit (1, no usage). Resumable scan/blaze/rebrem/reline and
    standalone line-grid derive use 75; regen-golden --check uses 1 on drift.
    Interactive check, remote attach, and remote logs catch KeyboardInterrupt and
    return success instead of 130.
  - Handler-level exit/error matrix is complete for all 39 assembled leaves.
    Shared process contract:
    - handler None/0 -> exit 0;
    - argparse ArgumentParser.error -> exit 2, usage + error on stderr;
    - string SystemExit -> exit 1, message on stderr, no usage;
    - uncaught ordinary exception -> exit 1, traceback on stderr;
    - uncaught KeyboardInterrupt -> exit 130, traceback/interruption on stderr;
    - SystemExit(75) -> exit 75 with no automatic message;
    - root main catches only MaterialConfigError and converts it to string
      SystemExit (exit 1, message stderr, no usage).
  - Local leaf matrix:
    - scan: success 0 with progress on stdout; selection/material errors 1;
      incomplete budget 75; other runtime errors 1; KeyboardInterrupt 130.
    - blaze: success 0 with progress on stdout; material or energy/spacing-count
      errors 1; incomplete budget 75; other runtime errors 1; KeyboardInterrupt 130.
    - export: success 0 and child output inherited; child failure is uncaught
      CalledProcessError (1 + traceback); KeyboardInterrupt 130.
    - analyze: success 0 and child output inherited; missing --default material or
      unknown material 1; child failure 1 + traceback; KeyboardInterrupt 130.
    - slim: success 0 with summary stdout; --grid stem/material guards 1; file,
      pickle, or write failures 1 + traceback; KeyboardInterrupt 130.
    - rebrem and reline: success 0 (including --all with no checkpoints, which
      prints a notice); invalid material/--all selection or multi-material hidden
      progress-file use 1; incomplete budget 75; other runtime errors 1;
      KeyboardInterrupt 130.
    - archive and restore: success 0 with stdout; missing source or overwrite guard
      1; other I/O failures 1 + traceback; KeyboardInterrupt 130.
    - archives: success 0 with stdout; unreadable individual archives are reported
      with record count "?" rather than failing; unexpected I/O errors 1;
      KeyboardInterrupt 130.
    - union: success 0 with stdout; missing input, backup collision, material
      mismatch, or overwrite guard 1; pickle/I/O failures 1 + traceback;
      KeyboardInterrupt 130.
    - check: success 0; interactive _launch catches KeyboardInterrupt and returns 0;
      --export KeyboardInterrupt remains 130; child/export failures 1 + traceback.
    - check-config: success 0 with stdout; MaterialConfigError 1 with message
      stderr/no usage; unexpected failures 1 + traceback; KeyboardInterrupt 130.
  - Remote leaf matrix:
    - scan, rebrem, and reline: success/dry-run/disconnected-viewer paths return 0
      and print stdout; selection, catalog, cross-option, busy, and captured-ssh
      guards return 1; KeyboardInterrupt during attach is caught and becomes 0,
      while interruption elsewhere is 130; uncaught runtime/child failures are 1.
    - start: success/dry-run 0; selection/catalog/cross-option/busy/captured-ssh
      guards 1; --follow KeyboardInterrupt is caught and becomes 0; interruption
      outside attach is 130; uncaught child/runtime failures are 1.
    - attach: terminal/disconnect/KeyboardInterrupt all return 0; no available job
      or validation/captured-ssh failure returns 1; unexpected runtime failure 1.
    - jobs and status: success 0 with stdout; captured-ssh failure returns 1 and
      writes remote stderr plus generic message to stderr; unexpected runtime
      failure 1; KeyboardInterrupt 130.
    - logs: non-follow success 0; captured-ssh failure 1; non-follow
      KeyboardInterrupt 130. Follow mode catches KeyboardInterrupt and returns 0,
      and ignores ssh nonzero status, also returning 0.
    - stop: success/no-live --all path 0 with stdout; selection, missing-live-job,
      invalid-token, or inactive-job guards 1; subprocess/runtime failure 1;
      KeyboardInterrupt 130.
    - reap: success/preview 0 with stdout; negative min-age silently clamps to zero;
      subprocess/runtime failure 1; KeyboardInterrupt 130.
    - pull: success 0 with stdout; selection/token and failures before the per-stem
      loop return 1. Inside the loop, OSError, CalledProcessError, and SystemExit are
      caught per stem, warning stdout, continuation, and final exit 0 even if every
      requested pull fails. Other runtime errors and KeyboardInterrupt remain 1/130;
      partial transfer/local mutation can precede either masked or hard failure.
    - clear: handler cross-option/missing-target errors use parser.error (2 + usage
      stderr); catalog, token, live-job, and reservation guards use string
      SystemExit (1); success/preview 0 with stdout; other failures 1;
      KeyboardInterrupt 130.
    - sync: success 0 with command output; child/runtime failure 1 + traceback;
      KeyboardInterrupt 130.
    - check: --follow without --detached uses parser.error (2 + usage stderr);
      success, pull, detached, and disconnected-viewer paths return 0; captured-ssh
      and unsuccessful-job guards return 1; attach KeyboardInterrupt becomes 0;
      interruption elsewhere is 130; other runtime failures are 1.
  - Line-grid leaf matrix:
    - submit: explicit success 0 with stdout; nonpositive slice is string
      SystemExit 1; remote/captured-ssh guards 1; other runtime failures 1;
      KeyboardInterrupt 130.
    - derive: success 0; incomplete slice 75 after stdout notice; malformed CSV,
      unknown material, bad numeric domain, and compute/I/O failures are mostly
      uncaught runtime errors (1 + traceback), not usage errors; KeyboardInterrupt
      130. --set-default can mutate defaults before later derive failure/75.
    - status: explicit success 0 with stdout; captured-ssh guard 1; runtime failure
      1; KeyboardInterrupt 130.
    - attach: terminal/disconnect/KeyboardInterrupt all become explicit 0; missing
      job or captured-ssh validation guard 1; unexpected runtime failure 1.
    - logs: non-follow success 0/captured-ssh failure 1/KeyboardInterrupt 130;
      follow catches KeyboardInterrupt and ignores ssh nonzero, returning 0.
    - stop: success 0; no job or inactive/invalid job guard 1; subprocess/runtime
      failure 1; KeyboardInterrupt 130.
    - apply: success/dry-run 0 with stdout; missing JSON selection 1; JSON/TOML,
      shape, path, and I/O failures generally 1 + traceback; post-write
      MaterialConfigError becomes message-only exit 1 and can leave live catalog
      invalid; KeyboardInterrupt 130.
    - set and set-brem: success is silent 0; unknown material/shape/I/O failures
      are uncaught runtime errors (1 + traceback); invalid numeric domains often
      succeed and mutate; KeyboardInterrupt 130.
    - defaults: success 0 and prints all defaults; malformed CSV is ValueError
      (1 + traceback); value flags without --set are ignored; KeyboardInterrupt
      130.
    - show: success 0 with stdout; unknown material also exits 0 with empty
      heading; TOML/I/O failures 1 + traceback; KeyboardInterrupt 130.
    - regen-golden: mutation success 0 with stdout; --check clean 0, drift 1 with
      diff on stdout and stderr empty; MaterialConfigError becomes message-only 1;
      other failures 1 + traceback; KeyboardInterrupt 130.
  - Controlled serial startup benchmark completed with .venv/bin/cxr. Each command
    ran seven times sequentially; first-of-series is reported separately from six
    warm samples. All 42 runs exited 0:
    - --version: first 1.475 s; warm median 1.461 s (1.393-1.665).
    - root --help: first 1.416 s; warm median 1.406 s (1.350-1.495).
    - check-config --help: first 0.146 s; warm median 0.147 s (0.130-0.158).
    - scan --help: first 1.420 s; warm median 1.416 s (1.394-1.459).
    - remote status --help: first 1.474 s; warm median 1.457 s (1.387-1.489).
    - line-grid derive --help: first 1.476 s; warm median 1.407 s (1.369-1.428).
    This confirms the special check-config dispatch is roughly 9.6x faster than
    ordinary warm help/version startup; nested parser depth adds negligible cost
    after eager imports. First-of-series is not a true cold-cache measurement
    because OS caches were not flushed.
  - Full remote/line-grid help audit completed:
    - line-grid has zero help text on 44 user-defined arguments across 12 leaves;
      no descriptions, examples, units, defaults, or cross-option contracts.
      Missing semantics include CSV formats; keV/eV/angstrom units; persistent
      mutation by --set-default/--set; --pull precedence over positional JSON;
      manual override/stale-golden behavior; optional job ID meaning latest job.
    - remote option help is generally useful, but 12 leaf parsers lose their
      parent-list summary. Missing help remains on scan material, scan/start
      --quick and --workers, and attach/status/logs jobid.
    - remote documents chunk-minute units/default/zero mode, step eV units,
      parallel-material constraint, destructive previews, follow disconnect
      behavior, and most pull dependencies.
    - remote omits scan --quick/--grid incompatibility, exact-one-of explicit
      materials versus --all, worker default/0 semantics, nonnegative numeric
      domains, TMD azimuth degrees, and remote-check numeric defaults.
    - neither nested group provides worked examples.
  - Stdout/stderr and automation audit completed:
    - Ordinary print-based status, summaries, warnings, previews, tables, and
      paths all go to stdout. No general diagnostic-output helper or quiet mode.
      Only explicit print(..., file=sys.stderr) in audited execution path is the
      optional CXR_MC_TIMING report. Python warnings and tracebacks also use stderr.
    - scan and blaze enable tqdm by default. tqdm writes to stderr and does not
      disable itself for a pipe. A real two-case run captured empty stdout and
      stderr containing a leading carriage return, initial 0% frame, UTF-8
      block-glyph 100% frame, and final newline. Their hidden --no-progress flag
      suppresses the bar. Remote queue scripts pass --no-progress and redirect both
      streams to the job log; current remote progress comes from JSON sidecars.
    - rebrem and reline use newline-only stdout progress every 50 records plus
      elapsed seconds; no carriage returns or ANSI. scan/blaze also print
      summaries, cache counts, elapsed time, and final paths. Human-readable,
      unstable for field-based parsing.
    - remote status/jobs use variable-width aligned columns, Unicode state glyphs
      and progress tracks, degree/micro symbols, and dynamic metadata. ANSI colors
      are correctly gated by stdout TTY, TERM != dumb, and NO_COLOR absence. Piped
      output is ANSI-free but remains Unicode and layout-dependent.
    - remote attach uses ANSI clear-screen sequences only on a color-capable TTY.
      When piped, it appends a complete report every two seconds separated by a
      Unicode rule: no escape pollution, but an unbounded repeated-frame stream.
    - non-follow logs buffers remote stdout and prints it only after ssh succeeds;
      failure discards partial remote stdout while remote stderr plus the generic
      failure message reach local stderr. Follow logs inherits child stdout/stderr,
      ignores return code, and prints its Ctrl-C notice to stdout.
    - export, analyze, check, remote transport._run, and follow logs inherit child
      streams. Parent preambles/traces go to stdout. Child output can interleave,
      and failure can leave partial stdout/stderr before a CalledProcessError
      traceback. No framing distinguishes parent from child data.
    - remote transport._run prints "+ <command>" to stdout before execution.
      sync, submission, pull, stop, reap, and line-grid remote operations can emit
      successful-looking trace lines before failure. Arguments truncate at 100
      characters, so trace is diagnostic, not round-trippable.
    - Many handlers emit partial stdout before hard failure or exit 75: scan/blaze
      setup and progress; rebrem/reline work from earlier records/materials;
      multi-material line-grid derive reports; export/analyze launch targets;
      union's successful pre-archive before later merge/write failure; remote pull
      output from earlier stems. Stdout is an event log, never an atomic result.
    - Warning/failure-like outcomes using stdout plus exit 0 include remote
      scan/rebrem/reline viewer disconnect, no successful remote checkpoints,
      remote pull per-stem failures, follow-log ssh failure/interrupt, unreadable
      archive record counts ("?"), and line-grid show unknown material. Largest
      current barrier to reliable automation.
    - line-grid regen-golden --check deliberately emits diff stdout with exit 1;
      line-grid apply --dry-run emits diff stdout with exit 0. Scripts must branch
      on command plus exit semantics, not assume diagnostics use stderr.
    - Stable machine data exists only in internal files/streams: progress JSON
      sidecars, remote metadata/state files, and line-grid derive --json-out.
      User-facing status/jobs/defaults/show/archives output has no versioned schema.
  - README/docs command-claim diff completed:
    - README command examples inspected are parser-valid: scan single material and
      --all modes, analyze material, export, and check-config optional catalog all
      match live behavior. Its mats_to_sim.toml claim correctly scopes --all to
      local scan and remote commands.
    - README is a workflow overview, not a command reference. It presents scan,
      analyze, export, check-config, and a brief remote pointer but omits current
      top-level blaze, slim details, rebrem, reline, archive/restore/archives/union,
      line-grid, check options, and all nested leaf inventories. No live 39-leaf
      reference exists.
    - docs/running-on-a-cluster.md describes obsolete remote defaults. It says the
      fixed allocation has unlimited wall time and multi-material runs start two
      scans concurrently by default, with --parallel-materials controlling
      concurrency. Live default is --chunk-minutes 10: one material at a time
      inside bounded self-resubmitting slices, each with a finite hard backstop.
      Parallel-materials is accepted only with --chunk-minutes 0; that monolithic
      mode defaults concurrency to two and uses UNLIMITED.
    - The same guide accurately describes remote scan as sync/submit/follow/pull,
      start as detached submission, status/logs/attach monitoring, stop via
      scancel, and detached remote check. It omits that viewer disconnect and
      masked pull failures can still yield local exit 0.
    - docs/repo_map.md calls itself canonical but its explicit cxr dispatch list
      and CLI dependency sketch omit blaze, rebrem, reline, and line-grid. Later
      module sections cover blaze but not a complete current command tree.
    - docs/superpowers/specs/2026-07-22-line-grid-cxr-commands-design.md promises
      cxr line-grid derive has the standalone derivation flags and stays identical
      to python -m cxr_mc.line_grid.derive. Live wrapper exposes only materials,
      energies, geometry, and set-default. It also promises submit geometry and
      set-default affect that run; live submit silently drops them. Its end-to-end
      example labels one-shot `line-grid status -vv` as "watch"; attach is the
      polling command.
    - Historical superpowers plans/specs contain many command snippets and
      pre-change problem statements. They are useful decision records, but lack a
      uniform current/historical banner and should not be indexed as current CLI
      documentation.
  - Wheel audit completed:
    - Temporary audit directory: /tmp/cxr-mc-wheel-audit.ckVcH7
    - First sandboxed build attempt failed only because isolated build resolution
      could not reach PyPI for hatchling (DNS/network restriction).
    - Approved retry succeeded:
      `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv build --wheel --out-dir
      /tmp/cxr-mc-wheel-audit.ckVcH7`
    - Built artifact:
      `/tmp/cxr-mc-wheel-audit.ckVcH7/cxr_mc-0.1.0-py3-none-any.whl`
    - Archive is a 533 KiB pure-Python `py3-none-any` wheel with 158 entries.
      Metadata says `cxr-mc 0.1.0`, Python >=3.13, and
      `[console_scripts] cxr = cxr_mc.cli:main`.
    - Wheel contains package modules plus `materials.toml`, 48 CIFs,
      line-grid defaults/provenance, Eagle XO QE, atomic-scattering provenance,
      and Mott transport tables. It does not contain
      `tests/data/material_catalog_golden.json`.
    - First Python 3.13 `--no-deps` install correctly installed the wheel but could
      not execute `cxr`: that interpreter's system site lacked NumPy. Offline
      dependency resolution also failed because Altair was absent from the uv
      cache. These are audit-environment limitations, not wheel failures.
    - Successful runtime isolation used a fresh Python 3.14 venv, installed only
      the wheel into that venv, and supplied dependencies from the existing project
      venv via `PYTHONPATH`. Because a `PYTHONPATH` site-packages entry does not
      process its editable `.pth` files, `cxr_mc` resolved from the fresh wheel
      install while dependencies such as NumPy resolved from the project venv.
      Runtime confirmed:
      `cxr_mc.__file__ = /tmp/cxr-mc-wheel-audit.ckVcH7/venv314/lib/python3.14/site-packages/cxr_mc/__init__.py`.
      This proves wheel code/data isolation, but not independently resolved
      dependencies.
    - From cwd `/tmp/cxr-mc-wheel-audit.ckVcH7`, installed `cxr --version`, root
      help, remote help, remote status help, line-grid help, line-grid derive help,
      and `check-config` all exited 0. Version was 0.1.0; bundled catalog validated
      49 materials, 48 crystals, and 1 medium.
    - Installed `DATA_DIR` contained materials, defaults, provenance, HOPG CIF, and
      Eagle XO QE. Golden snapshot was absent.
    - Installed `cxr line-grid regen-golden --check` resolved
      `/tmp/cxr-mc-wheel-audit.ckVcH7/venv314/lib/python3.14/tests/data/material_catalog_golden.json`,
      found no file, and exited 1 with an add-only diff. Confirmed packaging/API
      defect.
  - Shell/SSH interpolation audit completed:
    - Traced all subprocess, SSH/SCP, remote-command, generated-script, metadata,
      scheduler-ID, material/stem, and line-grid job paths.
    - Safe builder probes demonstrated environment-backed config injection without
      executing generated commands or contacting the remote host.
    - Ordinary assembled CLI identifiers remained constrained; no direct material,
      stem, explicit job ID, or typed numeric injection path was found.
    - No `shell=True` use found in audited remote/line-grid code.
  - Source metadata/entrypoint wiring reconfirmed.
  - TokenSave'd an additional ~21,608 tokens in this continuation
    (~37,111 cumulative before stream audit). Stream discovery saved ~7,979 more
    (~45,090 cumulative). Wheel/schema continuation saved ~4,830 more
    (~49,920 cumulative). Security trace saved ~5,002 more
    (~54,922 cumulative).

  ### Audit status

  General CLI audit complete. Implementation moved to
  `cli_implementation_plan.md`.
