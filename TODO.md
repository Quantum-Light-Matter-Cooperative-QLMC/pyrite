# TODO — feature/acp-server-autostart

This branch carries one backlog item; the full triaged backlog lives on `main`.

## Automated ACP server startups for marimo notebooks (P3)

Interacting with marimo notebooks over ACP (Agent Client Protocol) currently requires
manually running stdio-to-websocket bridges per agent:

```
npx stdio-to-ws "cmd /c npx @zed-industries/claude-code-acp" --port 3017
npx stdio-to-ws "cmd /c npx @zed-industries/codex-acp" --port 3021
```

**Implementation path.** Wrap these two invocations in a repo script (e.g.
`scripts/dev.py acp-up` or a small `.claude`/tooling shell script) that starts both bridges,
reports the assigned ports, and can be torn down cleanly (ctrl-C / a matching `acp-down`).
This is dev-environment tooling, not library code — no physics/validation obligations
apply. Confirm the two `npx` packages and ports still match whatever marimo-pairing setup
is current before wiring this up.
