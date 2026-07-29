---
name: caveman-commit
description: Use when user requests a commit message, `/commit`, or `/caveman-commit`; produce terse Conventional Commits text without staging or committing.
---

# Caveman Commit

Return paste-ready commit message only.

## Contract

- Subject: `<type>(<scope>): <imperative summary>`; scope optional.
- Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `build`,
  `ci`, `style`, `revert`.
- Prefer ≤50 characters; hard cap 72. No trailing period.
- Match repository capitalization convention.
- Add 72-column body only for non-obvious why, breakage, migration, security,
  revert context, or issue links.
- Use `BREAKING CHANGE:` and `Closes #N`/`Refs #N` when applicable.
- Omit diff narration, filler, emoji, and AI attribution unless repository
  policy requires a trailer.

Never stage, commit, or amend. `normal mode` disables terse commit style.
