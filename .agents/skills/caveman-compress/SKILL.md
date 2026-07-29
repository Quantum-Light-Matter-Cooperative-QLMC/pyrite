---
name: caveman-compress
description: Use when `/caveman-compress FILEPATH` or requests ask to compress natural-language memory/instruction files while preserving code, links, technical content, and Markdown structure.
---

# Caveman Compress

Compress prose; preserve substance and machine-sensitive regions.

## Run

Script sends file prose to configured Claude service. State that boundary and
obtain required approval before sending non-public or unverified repository
content.

From this skill directory:

```bash
python3 -m scripts <absolute_filepath>
```

CLI detects type, creates external backup under platform data directory,
compresses, validates structure/content, and attempts two targeted repairs.
Report failure without modifying source. If external execution is unavailable,
apply rules locally and disclose missing automated validation.

## Preserve exactly

- fenced/indented code and inline backticks
- URLs, links, paths, commands, environment variables
- technical terms, proper nouns, dates, versions, and numbers
- YAML frontmatter, headings, list/table hierarchy

## Compress

- Drop articles when clear, filler, pleasantries, hedging, and connective fluff.
- Use short words and fragments.
- Merge redundant prose; keep one equivalent example.
- Never edit uncertain code-like text.

## Scope

Allowed: `.md`, `.txt`, `.typ`, `.typst`, `.tex`, extensionless prose.
Forbidden: `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`, `.toml`, `.env`,
`.lock`, `.css`, `.html`, `.xml`, `.sql`, `.sh`, and `*.original.md`.
