---
name: caveman-stats
description: >
  Use when invoked with /caveman-stats. Output is injected by the mode-tracker
  hook; Show real token usage and estimated savings for the current session.
  Reads directly from the Claude Code session log — no AI estimation.
  the model itself does not compute the numbers.
---

This skill is delivered by `hooks/caveman-stats.js` (read by `hooks/caveman-mode-tracker.js` on `/caveman-stats`). The model does not need to do anything when this skill fires — the hook returns `decision: "block"` with the formatted stats as the reason. The user sees the numbers immediately.
