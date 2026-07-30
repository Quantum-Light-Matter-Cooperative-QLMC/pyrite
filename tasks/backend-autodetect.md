# Backend auto-detection + first-run CLI prompt

Branch: `feature/backend-autodetect`

TODO scope: new P2 item, "Backend auto-detection and first-run setup prompt."

## Problem

`CXR_MC_BACKEND` (`cuda|rocm|sycl|cpu|auto`) is currently hand-authored into a
gitignored repo-root `.env`; nothing detects hardware or vendor extras for the
user. This produced a real incident: a session's `.claude/settings.json`
SessionStart hook ran `uv sync` without `--extra nvidia`, silently uninstalling
`cupy` between sessions, surfacing as `ModuleNotFoundError: No module named
'cupy'` from `cxr run ... --nsys`. That specific hook gap is already fixed
(`.claude/hooks/sync_local_backend.sh` now reads `.env`'s `CXR_MC_BACKEND` and
passes the matching `--extra` to `uv sync`). This task is the CLI-level
counterpart: help a human first-time user pick and persist the right backend
without editing `.env` by hand or guessing extras.

## Goal

On first run (repo-root `.env` absent, or present without a `CXR_MC_BACKEND`
key), detect installed accelerator hardware and prompt the user to opt into
GPU acceleration; write the choice to `.env`; default to CPU when no GPU is
found or the user declines. Never re-prompt once `CXR_MC_BACKEND` is set.

## Scope

- **Hardware detection**, vendor python packages (`cupy`, `dpnp`/`dpctl`) are
  *not* installed pre-extra, so detection must use OS-level tooling, not
  `montecarlo._backend.select_backend`:
  - NVIDIA: `nvidia-smi` on `PATH` (and/or `/dev/nvidia*`).
  - AMD: `rocm-smi`/`rocminfo` (and/or `/dev/kfd`).
  - Intel: `clinfo`/`sycl-ls`, or `lspci` VGA entries naming Intel Arc/Xe.
  - Nothing found -> CPU, no prompt.
- **First-run gate**: only trigger when repo-root `.env` doesn't exist, or
  exists but has no `CXR_MC_BACKEND=` line. Never touch `.env` otherwise.
  Never prompt in a non-interactive/non-TTY context (CI, scripts, piped
  output) -- detect and fall back to CPU silently there.
- **Prompt + persist**: on a detected GPU, ask to enable acceleration
  (`click.confirm`-style). Accept -> append/create `.env` with the matching
  `CXR_MC_BACKEND` value (`cuda`/`rocm`/`sycl`) and, if the file is new, keep
  the existing header-comment style (see current `.env` for the pattern).
  Decline or nothing found -> write `CXR_MC_BACKEND=cpu` explicitly (avoids
  re-probing every invocation) or leave `.env` untouched and rely on the
  `auto` default -- **open question below**.
  Existing unrelated `.env` content (e.g. `MP_API_KEY_ENV`) must survive
  untouched; only add/append the one line.
- **Extra install**: writing `.env` alone doesn't install `cupy`/`dpnp`. Decide
  whether to also invoke `uv sync --extra <vendor>` inline (adds latency, may
  need network) or just print the follow-up command and rely on
  `.claude/hooks/sync_local_backend.sh` / a documented manual `uv sync
  --extra <vendor>` step -- **open question below**.
- **CLI surface**: touches `docs/cli-reference.md` contract if it adds a new
  subcommand/flag or changes startup output for existing commands -- decide
  entry point first (see open questions), then regenerate docs per
  `AGENTS.md`'s CLI contract rule.

## Decisions / open questions

1. **Trigger point.** An explicit opt-in command (e.g. `cxr setup` /
   `cxr backend init`) that onboarding docs point new users to, vs. an
   auto-fire check at the top of every `cxr` invocation. Auto-fire on *every*
   command risks breaking scripted/CI use (stdin prompt hangs) and the
   documented `--help`/output contract for unrelated commands. Leaning
   explicit command; needs a decision before implementation starts.
2. **Auto-sync or instruct.** Run `uv sync --extra <vendor>` inline after
   writing `.env`, or just persist the choice and print the follow-up command
   (leaning on the SessionStart hook for Claude Code users, a printed
   instruction for everyone else)?
3. **`CXR_MC_RESOURCE_POLICY` too?** User raised this but is unsure. Could
   default from detected VRAM size (mirrors the `<8 GiB -> conservative`
   heuristic in `resolve_resource_policy`,
   `src/cxr_mc/montecarlo/_resources.py:44`) and only ask when a GPU backend
   is actually chosen. Needs a vendor-specific VRAM query
   (`nvidia-smi --query-gpu=memory.total`, etc.) alongside the presence probe.
4. **Explicit `cpu` vs. untouched `.env`.** Writing `CXR_MC_BACKEND=cpu`
   avoids repeat probing on every future first-run check; leaving `.env`
   untouched relies on `select_backend`'s existing `auto` degrade path but
   means the first-run check re-triggers indefinitely without a `.env`. The
   gate as scoped above needs *some* durable "already asked" marker either
   way -- resolve alongside this.

## Delegation

Bounded single feature, but the four decisions above should be resolved
(by the assigned implementer, in dialogue with the user) before writing code.
Suggest `implement-task` (normal checklist) tier; not `-lite` given the CLI
contract and multi-branch detection logic, not `lead-task` given no
cross-module integration beyond one CLI entry point and `.env` I/O.

## Acceptance checks

- Detects NVIDIA/AMD/Intel presence via OS tooling with no vendor python
  package installed.
- First run with no `.env` / no `CXR_MC_BACKEND` prompts (interactively) or
  silently defaults to CPU (non-interactively); subsequent runs no-op.
- `.env` writes preserve any pre-existing unrelated lines.
- No GPU found or declined -> CPU-safe default; existing non-interactive/CI
  invocations of `cxr` are unaffected (no new prompt, no output-contract
  change for commands that don't opt into this flow).
- `docs/cli-reference.md` regenerated if the CLI surface changed.
- Tests cover each detection branch (mocked subprocess/tool output) and the
  `.env` create/append/no-op paths.
