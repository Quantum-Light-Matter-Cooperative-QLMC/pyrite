# Lightweight agent workflow skills

## Goal

Replace Superpowers' mandatory process layer with four small, task-triggered
skills shared by Claude Code and Codex. Preserve cxr-mc's existing domain
skills, repository rules, and verification commands.

## Scope

Create personal skills under `~/.agents/skills/`:

- `investigating-changes`
- `planning-changes`
- `implementing-changes`
- `verifying-changes`

Do not create a global dispatcher, repo-local copies, lifecycle hooks, scripts,
or mandatory artifacts. Do not modify unrelated `TODO.md` work.

Update only these cxr-mc skills when a small handoff improves composition:

- `repo-orientation` points investigation work to `investigating-changes`.
- `regression-testing` points implementation work to `implementing-changes`.
- `run-cxr-mc` points completion checks to `verifying-changes`.

Planning remains explicit-only through `AGENTS.md`; no repo planning adapter.

## Behavior

### Investigating changes

Trigger for diagnosis, uncertain ownership, regressions, failures, or requests
to explain observed behavior. Gather evidence before proposing a cause. Stop
after diagnosis unless user also requested a fix.

### Planning changes

Trigger only when user explicitly requests a plan, design, specification, or
implementation outline. Inspect relevant context, identify decisions and
risks, then produce a concise actionable plan. Do not create files or begin
implementation unless requested.

### Implementing changes

Trigger when user requests a build, fix, refactor, or behavior change. Preserve
unrelated work, choose proportionate tests, keep edits surgical, and continue
through verification. Tests-first is preferred when it materially proves new
behavior; it is not universal ceremony.

### Verifying changes

Trigger before claiming changed work complete or fixed. Run fresh checks
proportional to risk, inspect scoped diffs, and test real runtime behavior when
static checks cannot establish the outcome. Report exact checks and remaining
limits.

## Portability

Use the Agent Skills directory `~/.agents/skills/` as canonical personal
storage. Both Claude Code and Codex can discover this shared location. Each
skill contains only `SKILL.md`; no runtime-specific metadata is required.

Repo-local cxr-mc skills remain under `.agents/skills/`. Existing `.claude/skills`
copies continue providing Claude-specific project discovery; this change does
not add another mirrored copy mechanism.

## Skill shape

Each personal `SKILL.md`:

- uses lowercase hyphenated name and trigger-only description;
- stays below 250 words where practical;
- states one core principle;
- defines a short workflow and stop condition;
- includes a compact quick-reference table or checklist;
- avoids mandatory subagents, worktrees, commits, design docs, and TDD;
- defers project-specific commands and safety rules to applicable instruction
  files and domain skills.

## Validation

For each skill:

1. Run a baseline scenario without the skill and record whether generic agent
   behavior misses the intended boundary.
2. Create the minimal skill.
3. Validate frontmatter with `quick_validate.py`.
4. Run the same scenario with the skill and inspect compliance.
5. Check word count and remove redundant process.

After all skills:

- inspect scoped diffs;
- confirm unrelated `TODO.md` remains untouched;
- confirm repo handoffs name valid skills and introduce no circular mandatory
  chain;
- verify personal skills are discoverable from `~/.agents/skills/`.

## Success criteria

- Ordinary questions do not trigger workflow ceremony.
- Explicit planning remains planning-only.
- Requested changes proceed directly through implementation and proportionate
  verification.
- Diagnosis requests do not silently mutate code.
- Completion claims cite fresh evidence.
- cxr-mc domain skills remain authoritative for physics, notebooks, remote GPU
  work, performance, and runtime smoke testing.
