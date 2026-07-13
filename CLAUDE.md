# cxr-mc Claude notes

@AGENTS.md

`.claude/skills` is a generated mirror of the portable skills in
`.agents/skills`. Edit the canonical tree, then run
`uv run python scripts/dev.py sync-skills`.

Claude commands under `.claude/commands` are thin aliases to those shared
skills. The `physics-validator` agent is a restricted adapter to the independent
verification contract in `docs/validation/README.md`.
