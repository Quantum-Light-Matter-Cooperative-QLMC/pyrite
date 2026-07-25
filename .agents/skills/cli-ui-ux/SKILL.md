---
name: cli-ui-ux
description: Use when designing, implementing, reviewing, or testing any CLI command, option, argument, help, prompt, progress, completion, output stream, exit status, JSON/NDJSON, destructive action, or command migration; ensures clear discovery, safe interaction, stable automation, accessible presentation, and explicit compatibility.
---

# CLI UI/UX

Treat CLI behavior as an interface contract for humans, scripts, pipes, and
assistive technology. Preserve established behavior unless change is deliberate,
documented, and tested.

## Workflow

1. Inventory affected command paths, aliases, options, arguments, defaults,
   environment variables, config precedence, prompts, streams, exit codes, and
   completion behavior.
2. Classify output as human result, diagnostic, progress, or machine data.
3. Design names, help, validation, safety, and automation behavior before
   implementation.
4. Implement shared parameter types and output/error helpers where they prevent
   drift. Keep command imports lazy when startup cost matters.
5. Test success, usage errors, runtime failures, interruption, TTY and non-TTY
   output, hostile input, and compatibility.
6. Run real subprocess probes in addition to framework test runners when stream
   routing, signals, encoding, startup, or shell completion matters.

## Command Language

- Use short, lowercase, hyphenated command and option names.
- Prefer verb commands for actions and noun commands for resources only when
  subcommands supply the action.
- Reuse one term for one concept across commands, help, output, and docs.
- Avoid abbreviations unless established and unambiguous.
- Make required input positional only when order is obvious and stable.
- Prefer options for values that are optional, reorderable, or likely to grow.
- Give booleans explicit positive/negative pairs when disabling a default is
  useful. Avoid options whose meaning reverses under negation.
- Reserve `-h`/`--help` and `--version`. Use short options only when memorable
  and collision-free.

## Help and Discovery

- Give every command a one-line outcome summary. Put details after it.
- Show units, domains, defaults, repeatability, precedence, side effects, and
  incompatible options where users choose values.
- Distinguish required values from optional ones in usage and help.
- Include examples for non-obvious syntax, destructive actions, and common
  workflows. Use realistic shell-safe examples.
- Keep root and nested `--help` functional without optional runtime services.
- Keep help deterministic, fast, and free from hardware or network probes.
- Explain environment/config sources near related options, including precedence.

## Input, Validation, and Prompts

- Parse and validate at boundary. Report offending field, rejected value,
  expected domain, and correction when safe.
- Treat missing input and invalid syntax as usage errors; do not start work.
- Preserve meaningful zero values. Do not use truthiness where zero differs from
  omission.
- Reject controls and unsafe shell/path syntax before constructing external
  commands. Pass argument vectors; quote values at every shell boundary.
- Prompt only on interactive terminals and only when an informed choice is
  possible. Support non-interactive equivalents.
- State consequence and exact target in confirmation prompts. Require stronger
  confirmation for irreversible or broad actions.
- Never prompt during machine-output mode. Fail with actionable diagnostics
  unless an explicit non-interactive policy resolves the decision.
- Handle EOF and cancellation without traceback.

## Streams and Exit Status

- Write requested human results to stdout.
- Write diagnostics, warnings, prompts, and progress to stderr.
- Keep piped stdout stable and unpolluted.
- Return `0` only when requested operation satisfies its success contract.
- Return `2` for usage errors unless project contract says otherwise.
- Return nonzero for runtime or partial failures. Preserve documented resumable
  codes.
- Map user interruption to `130` and suppress framework tracebacks.
- For batch operations, retain successful results but report every failed item
  and return nonzero when any requested item fails.
- Do not swallow child-process, transport, or follow-stream failures.

## Progress, TTY, Unicode, and Accessibility

- Show progress only when useful. Disable it for piped stdout and machine output.
- Prefer bounded updates over noisy per-item lines. Never let progress obscure
  warnings or final status.
- Detect terminal capability; do not assume color, cursor motion, Unicode, or
  width.
- Honor `NO_COLOR` and explicit color policies. Never encode meaning by color
  alone.
- Provide ASCII-safe fallbacks for symbols and tables. Avoid emoji as status
  vocabulary.
- Use stable labels, plain language, adequate contrast, and readable ordering.
- Avoid rapidly updating output that screen readers cannot follow; provide a
  quiet or plain mode for dynamic interfaces.
- Handle narrow terminals and redirected streams without truncating essential
  identifiers or errors.

## JSON and Streaming Automation

- Make machine output explicit, such as `--json`; never infer it from piping.
- Emit exactly one documented UTF-8 JSON value plus newline for non-streaming
  JSON mode. Emit no ANSI, progress, prompt, child output, or warning on stdout.
- Use a versioned envelope with stable schema identifier, success flag, payload,
  and structured errors.
- Keep identifiers as strings when leading zeros or external formats matter.
- Encode units in field names, timestamps as RFC 3339 UTC, unknowns as `null`,
  and collections consistently as arrays or objects.
- Represent partial failure in payload and errors, set success false, and exit
  nonzero.
- Use NDJSON only for genuine streaming consumers. Emit one complete object per
  line; document event types, ordering, terminal events, and reconnect/resume
  semantics.
- Never mix human prose into JSON/NDJSON stdout.

## Completion

- Complete only syntactically valid, safe candidates.
- Complete dynamic values with bounded latency. Silence network/service failures
  and return no candidates.
- Avoid side effects, authentication prompts, broad scans, and expensive imports.
- Quote or escape candidates according to shell/framework rules.
- Do not complete dangerous targets for destructive commands unless project
  policy explicitly permits it.
- Test static choices, dynamic values, commas/repetition, spaces, Unicode, empty
  results, timeouts, and failures.

## Destructive Actions

- Resolve and display exact targets before mutation.
- Default to dry-run or preview when scope is broad or selection is indirect.
- Require explicit force/confirmation for overwrite, deletion, cancellation, or
  remote state changes. Never let `--yes` broaden target selection.
- Validate target identity again immediately before mutation when state can
  change between preview and action.
- Prefer atomic writes, backups, trash, or rollback where practical.
- Report completed, skipped, and failed targets. State recoverability.

## Tests

Cover every affected root and nested help path, dispatch path, option default,
validation boundary, incompatible combination, stream, and exit status. Add:

- framework runner tests for parsing and isolated invocation;
- subprocess tests for console entry point, stdout/stderr separation, encoding,
  signals, environment, and startup time;
- TTY/non-TTY and color/Unicode fallback tests;
- prompts, EOF, declined confirmation, force, and non-interactive behavior;
- machine-output snapshots or schema assertions, including partial failures;
- completion success, timeout, and failure tests;
- hostile control, shell, path, and metadata inputs at each boundary;
- regression tests for preserved command names, defaults, semantics, and output.

Prefer semantic assertions over full help snapshots when wrapping or framework
versions can vary. Snapshot deliberate public text and schemas.

## Compatibility Review

Before completion:

- Compare old and new command tree, aliases, options, defaults, precedence,
  prompts, stdout/stderr, exit codes, side effects, and machine schemas.
- Classify each difference as preserved, intentional correction, deprecation, or
  accidental regression.
- Keep aliases or deprecation warnings for affordable transitions. Put warnings
  on stderr and provide replacement plus removal policy.
- Document intentional breakage and migration steps in user-facing docs or
  release notes.
- Verify deprecated paths still obey safety and automation contracts.
- Report focused tests and any compatibility difference that remains.
