# Live verbosity cycling in progress dashboards

Branch: `feature/dashboard-verbosity-keybind`
TODO scope: new P2/P3 item, "Dashboard live verbosity cycling."

## Problem

`detail` (0/1/2, i.e. plain/`-v`/`-vv`) is fixed for the lifetime of a
dashboard session today -- set once from CLI flags at invocation and never
changed thereafter, in both:

- Local: `_dashboard_loop` (`src/cxr_mc/scan.py:153`), driving `cxr run`.
- Remote: `_live_status` (`src/cxr_mc/_remote/viewer.py:398`), driving
  `cxr remote status -a` / `attach`.

User request: while watching either dashboard, press `v` to cycle
0 -> 1 -> 2 -> 0 live, no restart.

## Key finding -- asymmetric cost between the two loops

- **Local (`scan.py`) is cheap.** `_build_sections` (`scan.py:110`) does not
  gate anything on `detail` -- it always builds META/STATE/RESOURCES/PROGRESS/
  SQUEUE in-process from callback state. `detail` only affects client-side
  rendering in `_format_job_status`. So local cycling is just: track a mutable
  `detail`, poll a key each tick, update it, keep rendering -- no re-fetch.
- **Remote (`viewer.py`) is not.** `_status_remote_command` (`viewer.py:67`)
  bakes `detail` into the *remote shell command itself*: `RESOURCES` and the
  32 KiB `LOG` tail are only fetched server-side when `detail >= 2` (deliberate
  -- avoids paying `tail`/`nvidia-smi` cost on every poll at low verbosity, see
  comment at `_status_remote_command:67`). Crossing 0/1 <-> 2 live therefore
  needs the remote command rebuilt with the new `detail` and the SSH stream
  (`_status_stream`, `viewer.py:202`) torn down and reopened -- not just a
  render-side toggle.
- **Keypress plumbing already exists, remote-only.** `_KeyListener`
  (`viewer.py:277`) is a working nonblocking single-key reader (POSIX
  termios/cbreak + `select`, Windows `msvcrt`, no-op on non-tty) already
  driving the `x`/`y` cancel keybind in `_live_status`. It has no SSH/transport
  dependency -- purely terminal mechanics -- so it is reusable as-is for local.
  Free keys confirmed: `x`/`p`/`y` taken (`viewer.py:271-273`); `v` is open.

## Implementation path & likely owners

- `src/cxr_mc/cli/_dashboard.py` -- move `_KeyListener` here from
  `_remote/viewer.py` (transport-agnostic, same precedent as the existing
  `_render_frame` move -- `viewer.py:12` already imports that back from
  `cli._dashboard`). `_remote/viewer.py` imports it from there instead of
  defining it.
- `src/cxr_mc/scan.py` -- add a `_KeyListener` to `_dashboard_loop`; on `v`,
  cycle `detail = (detail + 1) % 3` and continue the same loop (no re-fetch
  needed, per finding above).
- `src/cxr_mc/_remote/viewer.py` -- in `_live_status`, add `v` handling
  alongside the existing arm/confirm key poll: cycle `detail`, rebuild
  `remote = _status_remote_command(scripts._job_assign(jobid), detail)`, close
  the current stream and reopen via `_status_stream(remote)`, resume the same
  `for output in stream` loop. Keep the existing `x`/`y` cancel-arm state
  machine untouched; `v` should not interact with the arm/confirm window (a
  `v` press while armed just disarms silently, same as any other key today).
- `_attach_header` (`viewer.py:355`) -- extend the hint string to mention `v`
  (alongside the existing cancel/pull hints) so the keybind is discoverable.

## Stepwise checklist

- [ ] Move `_KeyListener` to `cli/_dashboard.py`; `_remote/viewer.py` imports
      it; no behavior change (existing cancel/pull remote tests stay green).
- [ ] Local: wire `_KeyListener` into `_dashboard_loop`; `v` cycles `detail`
      0->1->2->0; verify a level change is visible on the next 1 Hz tick.
- [ ] Remote: wire `v` into `_live_status`'s existing key-poll block; rebuild
      remote command + restart stream on any `detail` change; confirm no
      interaction with the `x`-arm/`y`-confirm window.
- [ ] Discoverability: extend `_attach_header` hint text with the `v` keybind;
      decide (open question below) whether the local dashboard gets an
      equivalent footer/header hint line, since it currently has none.
- [ ] Tests: local key-cycle test (fake key listener, assert `detail` progresses
      and next render reflects it); remote key-cycle test extending the
      existing `_FakeKeyListener` pattern (`tests/test_remote.py:2126`),
      asserting stream restart only on an actual `detail` change and that the
      rebuilt remote command matches `_status_remote_command` for the new
      level.
- [ ] `docs/cli-reference.md` regen only if any help text changes (unlikely --
      this is a live-session keybind, not a new flag).

## Decisions / open questions

- **Cycle order fixed 0 -> 1 -> 2 -> 0**, matching existing `-v`/`-vv` step
  semantics; no reverse-cycle key planned unless requested.
- **Remote stream restart is unconditional on any `detail` change**, not
  gated to only the 0/1 <-> 2 boundary that actually changes the fetched
  sections. Simpler, and a human keypress is infrequent enough that one extra
  round-trip per press is not worth the added branching -- revisit only if a
  reviewer flags the extra SSH round-trip as user-visible lag.
- **Open:** should the local dashboard gain a header/hint line (it has none
  today, unlike the remote attach header)? Needed for `v` discoverability but
  is new UI surface beyond the ask -- implementer's call; a one-line footer
  reusing `_paint`/`_style_states` conventions is the minimal option if yes.
- **Open:** on remote stream restart, is there a brief blank/stale frame
  between teardown and the first new-detail output? If so, keep rendering the
  last frame until the new stream yields its first output rather than
  blanking.

## Delegation / required skills

- `cli-ui-ux` -- keybind discoverability, hint text placement, cross-platform
  (POSIX/Windows) key handling parity.
- `regression-testing` -- key-cycle tests for both loops; remote stream-restart
  assertions without a real SSH connection (reuse existing remote-viewer test
  fakes).
- `documentation-maintenance` -- only if CLI help text changes.

## Acceptance checks

- Local `cxr run <profile>` dashboard: pressing `v` repeatedly visibly cycles
  detail level on the next refresh tick, wrapping 2 -> 0.
- Remote `cxr remote status -a` / `attach`: pressing `v` cycles detail live;
  crossing into level 2 shows RESOURCES/LOG panels within one restarted poll;
  crossing back out drops them; `x`/`y` cancel-arm behavior unchanged.
- Non-tty / piped sessions: unaffected (existing `_KeyListener` no-op path).
- `cxr-dev verify` green; `cxr-dev test` full suite green.
