# Conventions

- Make the smallest owning change and preserve unrelated dirty work.
- Add imports with first use. Ruff targets Python 3.13, 100 columns, and double-quote formatting.
- Treat CLI help/output/streams/exit behavior as a public contract; keep imports lazy and regenerate `docs/cli-reference.md` after visible CLI changes.
- New or edited physics requires a source equation, assumptions, limiting case, `Validation: <id>`, ledger row, and fresh-context validation.
- Keep CPU regression tests fast and deterministic; fix seeds for stochastic behavior.
- Stage explicit paths only. Do not push without authority.
