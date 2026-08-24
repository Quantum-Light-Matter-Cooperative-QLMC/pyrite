---
name: cli-ui-ux
description: Use when designing, implementing, reviewing, or testing PyRITE CLI commands, options, help, prompts, completion, streams, exit codes, machine output, destructive actions, or migrations.
---

# CLI UI/UX

Treat CLI behavior as a public contract for humans and automation. Preserve
existing behavior unless a change is deliberate, documented, and tested.

## Design rules

- Inventory only affected commands/options/defaults/env/config precedence,
  streams, exit codes, completion, and side effects.
- Classify output as result, diagnostic, progress, or machine data before
  implementation.
- Use lowercase hyphenated names and one term per concept. Help should expose
  units, domains, defaults, precedence, incompatibilities, and side effects.
- Parse/validate at the boundary; name the invalid value, valid domain, and
  correction. Preserve meaningful zero; handle EOF/cancel without traceback.
- Keep root/nested help imports lazy enough to avoid hardware/network/optional
  service startup.
- Add completion only where it can be fast, side-effect-free, shell-safe, and
  useful.

## Streams, status, machine output

- Results: stdout. Diagnostics/warnings/prompts/progress: stderr.
- Default exits: `0` success, `1` runtime failure, `2` usage, `75` documented
  temporary failure, `130` interruption; preserve command-specific contracts.
- Prompt only on TTY; provide an explicit non-interactive policy. Never prompt
  in machine mode.
- Honor `NO_COLOR`; color must not carry meaning.
- Explicit JSON mode emits exactly one documented UTF-8 value plus newline on
  stdout, with no ANSI/progress/prose. Use NDJSON only for genuine streams.
- Represent partial failure structurally and exit nonzero; never swallow child,
  transport, follow-stream, or partial failures.

## Destructive actions

Preview exact targets, require explicit confirmation/force, revalidate before
mutation, and prefer atomic/recoverable writes. Never broaden the selected
scope during confirmation.

## Verify

Test the affected help/dispatch/default/validation/stream/exit paths plus the
edge cases actually relevant to the change (TTY/non-TTY, interruption,
machine schema, completion failure, hostile input). Use subprocess probes when
signals, streams, startup, or shell behavior matters.

When CLI surface changes, regenerate with `pyrite-dev cli-reference --write`
and verify with `pyrite-dev cli-reference --check`; list intentional
compatibility changes.
