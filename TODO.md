# TODO / Backlog

Items live on `feature/...` / `bugfix/...` / `explore/...` / `docs/...` branches, not `main`, till done.
Priority weigh value-to-goal (line-flux / enhancement predictions + publication validation story) vs effort and risk.

**`TODO.md` is identical on every branch and on `main`** — the full triaged
backlog, one summary line per item + a pointer. In-progress detail lives in
`tasks/<branch-leaf>.md` (see [`tasks/README.md`](tasks/README.md)), **never in
this file and never in `docs/`** (which is durable, science-facing repo
documentation). Keeping branch `TODO.md` == `main:TODO.md` is what stops a
fast-forward (branch→branch or `main`→branch) from silently clobbering the
backlog: an ff moves the ref with no merge, so any divergent per-branch
`TODO.md` gets overwritten. Reconciliation enforced by /todo-sync.

Item generation:
----------------

1. Create branch of relevant type, switch to it.
2. Write `tasks/<branch-leaf>.md`: 1-3 sentence problem summary + implementation path / checklist, scoped to the item only.
   1. If agent is *Opus/Sol/K3 tier or above*: determine if task is suitably complex/parallelizable for subagent delegation. If yes, write parallelized task lists into additional scoped task_subagent_<model>_<number>.md documents, with <model> chosen appropriately by task scope & complexity.
   2. Do **not** make branch `TODO.md`
   diverge from `main`. Publish to `origin`.
3. Add a 1-sentence item summary to `TODO.md` (branch and `main` stay identical),
   pointer `→ feature/<branch>; tasks/<branch-leaf>.md`; triage into existing
   items; push a tightly scoped `docs(todo)` commit.
4. **On landing the branch into `main` (branch to be dropped):** promote any
   durable design/physics from `tasks/<branch-leaf>.md` into a proper `docs/`
   note, then `git rm tasks/<branch-leaf>.md`, and slim the `TODO.md` item to
   reflect completion.

**NOTE:** User wrote item straight into `TODO.md` (denoted >user<)? Fold into
`tasks/<branch-leaf>.md` (steps 1-2), then drop >user< & slim to one-line
summary once the branch exists.

## P1 - top-priority / high-value

### Active

1. **>user< Profile-command CLI work.** Combine `cxr sweep` and `cxr profile` (drop sweep) unless there's another good reason
2. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
3.  **Debye-Waller provenance and anisotropy audit.** Continue replacing placeholder or reused `B_ang2` values with primary-source values and resolve per-site/tensor model needs. → `feature/debye-waller-audit`; audit: [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md).
   1. >user< Evaluate complexity/value of implementing full anisotropic/tensor-based Debye-Waller factors when available
   2. >user< Evaluate worth in both implementing the approximate scalar Debye-Waller formula (compare output to known values for various anisotropic materials we have in our DB), and in attempting to implement a fully-fledged DFPT system.
   3. >user< Take a crystal with a known DW factor, then manually change it up and down over a range of values that can reasonably be expected other crystals to have, and see how much it changes by -- if large, then its worth being careful here.
4.  **GPU-memory follow-up.** Benchmark remote `rebrem --all --ne-brem 500 --step 20` for bounded CuPy reserved-pool memory; assess analogous `reline` cleanup separately.

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3.  **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3); this item is the coherent-sum follow-on.

## P2 - medium-priority

1. **Remote-job UX.** Profile-based job names + same-profile submit block, quieter submit pull suggestion, explicit-material-only stop list, faster `stop --all`, attach-time cancel key + compute-aware progress bars. → `feature/remote-ux`.
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
4. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
5. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
6.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
7.  **>user< Fix cached pull-in of pre-rendered animations in trace_app (add more buttons or something) -- maybe due to regenerate's changing random seed?**
   1.  Progress bar here shows up at the top of the marimo notebook instead of near the button that is pressed to start the render -- confusing
   2.  clip off the extra figure background & legend in the render, it is ugly. We just want the black grid space, with the colorbar and mat/config title info overlayed, but no background color. Saved render is also a bit pixelated, especially when opened in an mp4 viewer outside of the marimo app.
   3.  add button to open render saving dialogue (so user can promptly move it from the cache)

## P3 - lower / exploratory / small bugfixes
