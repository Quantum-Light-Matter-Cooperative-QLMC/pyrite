#!/usr/bin/env bash
# Regenerate the generated .claude/skills mirror from canonical .agents/skills
# and stage the result, so canonical edits and their mirror always land in the
# same commit. Registered as a pre-commit hook in .pre-commit-config.yaml; it
# fires only when staged paths touch either skill tree.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

uv run cxr-dev sync-skills
git add -- .claude/skills
