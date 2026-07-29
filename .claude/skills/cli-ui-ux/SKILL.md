---
name: cli-ui-ux
description: Use when designing, implementing, reviewing, or testing cxr-mc commands, options, help, prompts, progress, completion, streams, exit codes, JSON/NDJSON, destructive actions, or migrations.
---

# CLI UI/UX

Treat CLI behavior as public contract for humans, scripts, pipes, and assistive
technology. Preserve behavior unless correction is deliberate, documented, and
tested.

## Design

1. Inventory affected paths, aliases, parameters, defaults, environment/config
   precedence, prompts, streams, exit codes, completion, and side effects.
2. Classify output: result, diagnostic, progress, or machine data.
3. Design help, validation, safety, compatibility, and automation before code.
4. Keep imports lazy so root/nested help avoid hardware, network, or optional
   services.

Use lowercase hyphenated names and one term per concept. Show units, domains,
defaults, repeatability, precedence, incompatibilities, and side effects in
help. Parse at boundary; name invalid value, expected domain, and correction.
Preserve meaningful zero. Handle EOF/cancel without traceback.

## Streams and status

- Results: stdout. Diagnostics, warnings, prompts, progress: stderr.
- `0`: success; `1`: runtime failure; `2`: usage; `75`: documented temporary
  failure; `130`: interruption. Preserve command-specific contracts.
- Never swallow child, transport, follow-stream, or partial failures.
- Prompt only on TTY; offer explicit non-interactive policy. Never prompt in
  machine mode.
- Honor `NO_COLOR`; keep redirected/narrow output readable. Color never carries
  meaning alone.

## Machine output

Machine mode must be explicit. Non-streaming JSON emits exactly one documented
UTF-8 value plus newline: no ANSI, progress, prompt, or prose on stdout. Keep
schema/units stable; represent partial failure structurally and exit nonzero.
Use NDJSON only for genuine streams with documented event order and terminal
events.

## Safety and completion

Preview exact destructive targets; require explicit confirmation/force without
broadening selection. Revalidate targets before mutation; prefer atomic,
recoverable writes. Completion must be fast, side-effect-free, shell-safe, and
silent on service failure.

## Verify

Test affected help/dispatch/default/validation/stream/exit paths plus
incompatible options, prompts, interruption, TTY/non-TTY, color/Unicode,
machine schemas, completion failure, and hostile inputs. Use framework tests
and real subprocess probes where signals, streams, startup, or shell behavior
matters. Regenerate `docs/cli-reference.md`; list each intentional compatibility
difference.
