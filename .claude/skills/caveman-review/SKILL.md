---
name: caveman-review
description: Use when user requests code/PR/diff review or `/caveman-review`; return terse, actionable, location-first findings.
---

# Caveman Review

Review only; do not fix, approve, or run checks unless requested.

Format: `<file>:L<line>: <severity>: <problem>. <fix>.`

Severities: `bug`, `risk`, `nit`, `question`. Sort by severity, then file/line.
Keep exact symbols, concrete impact, and actionable fix. Omit praise, diff
restatement, hedging, and architecture essays. If no findings: `No issues.`

Use full explanation for security findings, architectural disputes, or
onboarding context. `normal mode` disables terse review style.
