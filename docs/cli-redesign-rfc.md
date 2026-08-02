# RFC: `cxr` CLI structural redesign

- **Status:** Accepted — 2026-08-01
- **Author:** Alex Amador
- **Created:** 2026-08-01
- **Supersedes/relates:** `TODO_CLI.md` (punch-list — folded into `TODO.md` `## CLI backlog`, 2026-08-01), `docs/cli-energy-grid-sweep-rework-plan.md`, `docs/plans/scan-config-cli-plan.md`
- **Sub-RFC:** `docs/cli-artifact-model-rfc.md` (the content-addressed data-model change, split off — see D3)
- **Companion:** `docs/package-structure-rfc.md` (package/repo reorg; its
  command-home consolidation is a prerequisite for this RFC — see §4 phase 0)
- **Authoritative surface at time of writing:** `docs/cli-reference.md` (v0.1.0)

> An RFC is a design proposal circulated for critique *before* implementation, so
> the structure is torn apart on paper (cheap) instead of in code (expensive).
> Nothing here is implemented. It settles the noun/verb/flag layer so downstream
> work doesn't get built twice.

---

## 1. Why

The current surface is a **noun/verb hybrid** whose inconsistencies have already
produced concrete drift and duplication. The three load-bearing problems (fix
these or tear down later):

1. **Parallel local/remote command trees that have diverged.** The same
   operation is spelled differently by locality, and the spellings no longer
   match:

   | Operation | Local | Remote |
   |---|---|---|
   | recompute brem | `checkpoint recompute brem` | `remote rebrem` |
   | recompute line | `checkpoint recompute line` | `remote reline` |
   | run | `run` | `remote run` |
   | prune stale | `checkpoint prune` | `remote prune` |
   | perf list/prune | `performance list/prune` | `remote performance list/prune` |
   | derive grid | `energy-grid derive` | `energy-grid submit` |

2. **Four divergent async-job lifecycles.** `remote run` (+ `status/jobs/logs/
   stop/pull`), `energy-grid submit` (+ `energy-grid job attach/logs/status/
   stop`, no pull), `remote rebrem`/`reline` (self-contained auto-follow+pull, no
   status verb), and `remote validate` (own `-d/-f/--pull`). Four submit/attach/
   detach/pull UXs for what is one concept.

3. **Uncontrolled verb and flag vocabularies.** Five+ synonyms for "remove"
   (`clear`, `prune`, `slim`, `reap`, `delete`, `prune-jobs`); three names for
   crystal tilt (`--tilts`, `--polar`, `--angles`); two for beam energy
   (`--energies`, `--energy`); three for "persist as default" (`--default`,
   `--set-default`, `--set`).

Secondary issues (real, but mechanical once the above land) are catalogued in §4.

What is **already good and must be preserved**: the automation contract — exit
codes (2 usage / 1 runtime / 130 interrupt / 75 resumable), the versioned
`--json` envelope, stderr-for-diagnostics, `--dry-run` breadth,
preview-before-destructive. The contract layer is solid; only the
noun/verb/flag layer is being restructured.

---

## 2. Settled decisions

Approved for this RFC: **D1, D2** (the load-bearing surface changes) plus the
mechanical **D4–D7**, so the migration lands in one pass. **D3** (the
content-addressed data model) is approved in principle but split into its own
**`docs/cli-artifact-model-rfc.md`** — it is a data-model change on an
independent timeline, sequenced last (§4). Summarized below as **D3**; details
live in the sub-RFC.

### D1 — Canonical **noun → verb** ordering

Everything is `cxr NOUN VERB`: `cxr profile show`, `cxr material validate`,
`cxr energy-grid derive`. This is what `gh` (`gh pr create`), `gcloud`
(`compute instances create`), and DVC (`remote add`) do; held consistently, it
also *structurally* eliminates the "profile named `set`" ambiguity
(`TODO_CLI.md` Low-Pri #2) — a reserved-word blocklist is unnecessary when a
verb keyword is always required in a fixed position.

**Blessed exceptions** (documented, permanent — the single most common actions
stay verb-first as top-level shortcuts): `cxr run`, `cxr setup`. We do **not**
force `cxr profile run`. Everything else conforms.

Removes the `cxr app analysis [MATERIAL] [COMMAND]` foot-gun (optional
positional colliding with a subcommand) by making subcommands mandatory
keywords.

### D2 — Remoteness is a **modifier**, plus one **unified job noun**, plus a **context**

Three coupled changes that together kill the parallel tree (Problem 1) and the
four lifecycles (Problem 2).

**(a) Execution locality is a flag, not a namespace fork.** Every execution verb
takes `--remote[=TARGET]` (short `-R`). `cxr run PROFILE --remote` replaces
`cxr remote run`. `cxr checkpoint recompute brem --remote` replaces
`cxr remote rebrem`. There is exactly one spelling of each operation; locality
selects where it runs. This is the lesson of `rebrem` vs `recompute brem`
drifting apart.

**(b) One job lifecycle for every async submission.** A single top-level noun:

```
cxr job list [--kind run|grid|recompute|validate]
cxr job status [JOBID] [-v]
cxr job logs   [JOBID] [-f]
cxr job attach [JOBID]
cxr job stop   [JOBID|--all|--profile NAME] [--yes]
```

Every submitter (run, energy-grid derive, recompute, validate) feeds this one
lifecycle, tagged by `kind`. Submitters take uniform async modifiers
`--wait` / `--detach` (replacing today's `--headless`, `--no-pull`,
`-d/--detached`, `-f/--follow` variants). This is the SLURM/`kubectl` model and
resolves `TODO_CLI.md` High-Pri #3 ("should each type have its own submit/
pull?") — **no; one lifecycle, tagged jobs.**

**(c) A settable context for target + current profile.** Model on
`docker context use` / `kubectl config use-context` / `gcloud config`
([docs][k8s-ctx], [gcloud-cfg]). A single store holds the current remote
target, current profile, checkpoint dir, and backend:

```
cxr config set remote.target box-a
cxr config set profile.current sub_100keV
cxr config get / cxr config list
```

Routine commands read these defaults; any flag overrides per-call. This
collapses the dozens of implicit `--profile standard` and `CXR_REMOTE_*`
repetitions into set-and-forget. Following gcloud, context is
**environment-scoped, not identity-scoped**: switching profile never forces a
remote-target switch, and vice versa.

**Precedence chain (documented, single order everywhere).** Every context-backed
value resolves highest-wins:

```
per-call flag  >  environment (CXR_*)  >  config store  >  built-in default
```

This is the gcloud/kubectl contract. It must be stated in `docs/cli-reference.md`
and enforced by one shared resolver so no command invents its own order. The
`CXR_REMOTE_*` env vars keep working as the middle tier (they override the store
but yield to an explicit flag), which is what today's callers already assume.

**Naming caveat — `profile` is overloaded.** "Profile" already denotes the
physics campaign object (`standard`/`survey`/…). Reusing it for the context
pointer (`profile.current`) risks confusion with a future notion of a *context*
that bundles remote + backend + campaign the way a kubectl context bundles
cluster + user + namespace. Options: (a) keep `profile.current` (the pointer just
names which campaign profile is active — arguably fine); (b) rename the bundle to
`campaign.current` and reserve `context` for the whole environment set. See §6
Q3.

`cxr remote` survives **only** as the remote-as-*resource* namespace —
`remote fetch` / `remote pull` / `remote list` / `remote target ...` — mirroring
git/DVC `remote`. It no longer carries execution verbs.

### D3 — Content-hash **artifact** model for grids & materials → **sub-RFC**

**Split off into `docs/cli-artifact-model-rfc.md`.** In one line: derived
`material`/`energy-grid` data becomes an immutable, content-addressed artifact;
the profile becomes the sole mutable git-ref-like pointer; unreferenced
artifacts are reclaimed by `gc` (D4) instead of a bespoke `delete`. This
untangles today's knotted model (`energy-grid apply` mutating `standard`, the
dual grid store, asymmetric deletion) but is a data-model change, not a surface
one — so it is designed and shipped on its own timeline, sequenced last (§4
phase 5). The lifecycle verbs it needs (`add` / `verify` / `gc`) are the D4 set.
See the sub-RFC for the model, migration, and the campaign-lockfile question.

### D4 — Controlled **verb** vocabulary

One meaning per verb, drawn from a fixed set:

| Canonical verb | Meaning | Replaces |
|---|---|---|
| `show` | display one object | `show`, `status` (single) |
| `list` | display many | `list`, `jobs` |
| `set` / `add` / `remove` | mutate config fields | (as today) |
| `rm` | delete an explicit target | `delete`, `clear`, `prune-jobs` |
| `gc` | reclaim unreachable/obsolete | `prune`, `reap`, `clear --all` |
| `recompute` | overwrite derived data in place | `recompute`, `rebrem`, `reline` |
| `slim`/`export`/`archive` | serialize/transform (non-deleting) | keep, but `slim` never means delete |

Overloaded `validate` is split: `material validate` (static catalog check)
stays; `remote validate` ("run Zhai reproduction") is a **run**, not a validate —
becomes `cxr run --preset zhai` (or similar), feeding the D2 job lifecycle.

### D5 — Controlled **flag** vocabulary

One canonical name per physical quantity, everywhere:

| Quantity | Canonical (singular, repeatable + `START:STOP:STEP`) | Retired spellings |
|---|---|---|
| beam energy | `--energy` | `--energies` |
| crystal tilt (polar) | `--polar` | `--tilts`, `--angles` |
| azimuth | `--azimuth` | `--azimuths` |
| thickness | `--thickness` | — |
| material selector | `--material` (repeatable) | `--materials KEY,...` |

Other cross-cutting standardizations:
- **Persist-as-default:** one flag `--save-default` everywhere (retire
  `--default`, `--persist-default`, `--set-default`, `--set`).
- **Output:** replace boolean `--json` with `-o, --output [table|json|wide]`
  (gh/kubectl model) — extensible, and enforced as **universally present** on
  every non-interactive command (fixes today's patchy `--json` coverage).
  Contract: `table` is the human default; `json` is the machine-stable path and
  the **only** format bound by the versioned-envelope automation contract (§1).
  `wide` is a human convenience with no stability promise. `yaml` and
  `jsonpath=<expr>` are reserved extension points (kubectl parity) — add on
  demand, but every added format that isn't `json` is explicitly *not* a
  contract surface, so scripts must target `-o json`.
- Azimuth range corrected to the physical limit (`TODO_CLI.md` High-Pri #4 — see
  §5, it is a **bug**, tracked separately).

### D6 — Uniform destructive-op & async contract

- Every destructive command: **preview by default**, `-y/--yes` (both spellings)
  to execute, **and** an interactive `[y/N]` prompt when stdin is a TTY (so no
  "resubmit with `--yes`" dead-ends). Non-TTY/CI still requires `--yes`.
  (Resolves `TODO_CLI.md` High-Pri #6.)
- Every async submitter: `--wait` (default, attach) / `--detach` (return after
  submit), then the D2 `job` lifecycle. No per-command `--headless`/`--no-pull`/
  `--follow` variants.

### D7 — Explicit deprecation / alias policy

Model on Kubernetes' published policy ([k8s-deprecation]): a renamed/removed
command (a) keeps working for a **minimum support window** (propose: 2 minor
releases), (b) emits a **stderr deprecation warning naming the replacement**,
(c) is listed in a `docs/cli-deprecations.md` table with removal target. Applied
**retroactively** to today's already-hidden aliases (`slim`, `rebrem`, `check`,
flat energy-grid job verbs, `archive`/`restore`/`union`, …), which currently
have no documented window.

---

## 3. Before / after

```
# execution locality: flag, not fork
cxr remote run sub_100keV -m hopg          →  cxr run sub_100keV -m hopg --remote
cxr remote rebrem hopg                      →  cxr checkpoint recompute brem hopg --remote

# unified job lifecycle
cxr energy-grid job status                  →  cxr job status --kind grid
cxr remote jobs                             →  cxr job list
cxr remote status -vv                       →  cxr job status -vv

# context replaces repeated flags
cxr material set hopg --profile standard …   →  cxr config set profile.current standard
                                                cxr material set hopg …

# controlled vocab
cxr energy-grid derive --energies 30,60 --tilts 0
                                            →  cxr energy-grid derive --energy 30,60 --polar 0
cxr checkpoint prune / reap / clear         →  cxr checkpoint gc  /  cxr checkpoint rm
```

---

## 4. Migration plan

Phased so the automation contract never breaks:

0. **Command-home consolidation (prerequisite)** — before any surface reorder,
   unify where a CLI command *lives* in the tree. Today command implementations
   are split between loose `src/cxr_mc/*.py` modules and `src/cxr_mc/cli/*.py`,
   so D1's noun→verb reshuffle would edit two homes at once. This is a
   package-structure change, not a surface one; it is owned by
   **`docs/package-structure-rfc.md`** and lands first (pure refactor, zero
   surface change, guarded by the existing `docs/cli-reference.md` freeze test).
1. **Vocab (D4/D5) + deprecation harness (D7)** — additive: introduce canonical
   verbs/flags as primaries, wire old spellings as warning-emitting aliases.
   Zero behavior change. Ship first; lowest risk.
2. **Context (D2c) + `-o/--output` (D5)** — additive: new `cxr config`, defaults
   read where `--profile standard` / `CXR_REMOTE_*` are read today. Old flags
   still override.
3. **Unified `job` noun (D2b)** — new top-level `job`; existing job verbs become
   aliases dispatching into it.
4. **`--remote` modifier (D2a)** — teach execution verbs `--remote`; `cxr remote
   run/rebrem/reline/…` become aliases. `cxr remote` demoted to resource-only.
5. **Artifact model (D3)** — the deepest change; content-hash store + `gc`.
   Sequence last, behind the stabilized surface. Owned by its own sub-RFC,
   `docs/cli-artifact-model-rfc.md` (which carries its own phased migration).

Each phase regenerates `docs/cli-reference.md` (per repo contract) and lands with
its alias rows in `docs/cli-deprecations.md`.

---

## 5. `TODO_CLI.md` punch-list mapping

> `TODO_CLI.md` was folded into `TODO.md` `## CLI backlog` on 2026-08-01
> (package-structure RFC P5). The item numbers below refer to that former
> punch-list, preserved here as decision provenance.

Splitting bugs from ergonomics from structure:

| Item | Disposition |
|---|---|
| High #1 (singular flags) | Subsumed by **D5** |
| High #2 (`coherent\|incoherent\|both` on profile) | Independent ergonomics; ship anytime. Make the "added both ⇒ auto-both" switch explicit/logged, not implicit magic |
| High #3 (unify submit/pull) | Resolved by **D2b** |
| High #4 (azimuth 0–360 vs 90/270) | **BUG, not design** — misfiled. Pull out, fix + regression test independently of this RFC |
| High #5 (`energy-grid` reorg: `job`, material-scoping, `regen-golden`) | `regen-golden` → top-level/`material` (**D3/D4**); `job` folds into **D2b**; grid-as-material-artifact is **D3** |
| High #6 (`-y` + prompt) | Resolved by **D6** |
| High #7 (`run -p` requires single material AND profile) | Independent bug/ergonomics; contradicts `run`'s own optional `-m`. Fix alongside D2 |
| High #8 (`remote fetch`, git-style) | Lands as the resource-only `cxr remote` (**D2a**) + artifact hash-diff (**D3**) |
| Low #1 (`cxr` alone → help) | Independent; trivial |
| Low #2 (block command names as object names) | Structurally resolved by **D1** (no blocklist needed) |

**Bugs to file separately now** (don't wait for the RFC): azimuth range
(High #4); `run -p` dual requirement (High #7); stale `cxr profile members`
pointer in `cxr material` help; `app analysis` positional/subcommand collision.

---

## 6. Open questions

1. **`--remote` value semantics** — bare `--remote` uses `config remote.target`;
   `--remote=NAME` overrides. Confirm the ergonomics.
2. **`run` as blessed verb-first exception** — keep, or bite the bullet on
   `cxr profile run` for full consistency? (RFC recommends keep.)
3. **`profile` vs `context` naming** — keep `config set profile.current`, or
   rename the context bundle to `campaign.current` and reserve `context` for the
   whole environment set (remote + backend + campaign)? (D2c naming caveat.)

*Settled since the first draft:* `-o/--output` adopted over boolean `--json`
(D5); the artifact model split into `docs/cli-artifact-model-rfc.md`, which owns
the remaining open questions (lockfile, hash inputs, checkpoint reachability).

---

## References

- gh manual — noun→verb: <https://cli.github.com/manual/>
- Docker contexts: <https://docs.docker.com/engine/manage-resources/contexts/>
- [k8s-ctx] kubectl contexts: <https://kubernetes.io/docs/reference/kubectl/generated/kubectl_config/>
- [gcloud-cfg] gcloud configurations: <https://cloud.google.com/sdk/docs/configurations>
- [dvc-push] DVC push/pull/remote (content-addressed): <https://dvc.org/doc/command-reference/push>
- [k8s-deprecation] Kubernetes deprecation policy: <https://kubernetes.io/docs/reference/using-api/deprecation-policy/>
- clig.dev (CLI guidelines): <https://clig.dev/>
