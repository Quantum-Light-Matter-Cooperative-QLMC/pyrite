# cxr-mc Claude adapter

@AGENTS.md

`.agents/skills` is canonical; `.claude/skills` is generated. After canonical
edits, run `scripts/dev.py sync-skills`. Claude commands are thin skill aliases;
`physics-validator` follows `docs/validation/README.md`.
