# PyRITE Claude adapter

@AGENTS.md

`.agents/skills` is canonical; `.claude/skills` is generated. After canonical
edits, run `pixi run --locked sync-skills` and `pixi run --locked check-skills`.
Claude commands are thin skill aliases; `physics-validator` follows
`docs/validation/methodology.md`. See `AGENTS.md` for focused environment,
workspace doctor and verification task selection.
