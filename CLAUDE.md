# PyRITE Claude adapter

@AGENTS.md

`.agents/skills` is canonical; `.claude/skills` is generated. After canonical
edits, run `uv run pyrite-dev sync-skills`. Claude commands are thin skill
aliases; `physics-validator` follows `docs/validation/methodology.md`.
