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

Planning remains governed by the personal skill plus `AGENTS.md`; no repo
planning adapter.

## Behavior

### Investigating changes

Trigger for diagnosis, uncertain ownership, regressions, failures, or requests
to explain observed behavior. Gather evidence before proposing a cause. Stop
after diagnosis unless user also requested a fix.

### Planning changes

Trigger when user explicitly requests a plan, design, specification, or
implementation outline. Also trigger at agent discretion for large or complex
changes spanning multiple steps, files, subsystems, or independently
delegatable tasks. Inspect relevant context, identify decisions and risks, then
produce a concise actionable plan. For discretionary planning, use the
lightest useful plan; do not require a separate specification or design file.

### Implementing changes

Trigger when user requests a build, fix, refactor, or behavior change. Preserve
unrelated work, choose proportionate tests, keep edits surgical, and continue
through verification. Tests-first is preferred when it materially proves new
behavior; it is not universal ceremony.

Use subagents when delegation would reduce primary-context use, parallelize
independent work, or provide valuable fresh-context review. Choose subagent
model and reasoning effort according to task scope, complexity, and risk.
Delegation remains discretionary, not a required phase.

### Verifying changes

Trigger before claiming changed work complete or fixed. Run fresh checks
proportional to risk and change type. Agent chooses among focused tests, lint,
type checks, runtime probes, full lint, and full verification. Documentation-only
or similarly inert changes may need no automated checks. Inspect scoped diffs,
test real runtime behavior when static checks cannot establish the outcome,
and report exact checks plus remaining limits.

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
- keeps subagents, worktrees, commits, design docs, and TDD conditional on
  task needs;
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
- Explicit plan requests and agent-identified complex changes receive
  proportionate planning.
- Requested changes proceed directly through implementation and proportionate
  verification.
- Subagents are used when their context savings, parallelism, or independence
  justify coordination cost; model and effort match delegated scope.
- Diagnosis requests do not silently mutate code.
- Completion claims cite fresh evidence.
- Verification depth matches change risk; inert documentation changes need no
  automatic lint or test ritual.
- cxr-mc domain skills remain authoritative for physics, notebooks, remote GPU
  work, performance, and runtime smoke testing.
