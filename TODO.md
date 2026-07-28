# TODO / Backlog

Items live on `feature/...` / `bugfix/...` / `explore/...` branches, not `main`, till done.
In-progress detail live on branch (or design doc);
`main` keep one-line summary + pointer, enforced by /todo-sync.
Priority weigh value-to-goal (line-flux / enhancement predictions + publication validation story) vs effort and risk.

Item generation:
----------------

1. Create branch of relevant type, switch to it
2. Overwrite branch TODO.md: 2-3 sentence problem summary + implementation path, scoped to item only. Publish to `origin`
3. Switch to `main`, add 1-sentence item summary, triage into existing TODO.md items, push tightly scoped `docs(todo)` commit to main

**NOTE:** User wrote item straight into `TODO.md` (denoted >user<)? Fold into branch (steps 1-2), then drop >user< & slim to one-line summary on `main` once branch exist.

## P1 - top-priority / high-value

### Active

1. **>user< general CLI fixes.**
   1. `cxr energy-grid` options are confused and need to be fixed. What does `defaults` apply to? `brem`, `line`, both, or is it just used for deriving the appropriate upper-bounds for each? What does `apply` do? I think this section needs to be reworked, or at a minimum, the help explanations made more detailed 
   2. I think once values are set for `cxr energy-grid defaults`, they cannot be removed (even though when I started, the fields for `tilts` and `azimtuhs` were both empty--not sure what empty would mean here)
   3. How does the `fidelity` field contained in the backend code relate to this? Where does `full` vs `survey` come into play?
2. **>user< `cxr remote attach` and related progress-tracking commands**
   1. Progress bars (both 'compute' and 'cases') still only show the number of cases for the materials current displayed. Example: sub_100keV has 21 mats, 1568 cases per mat, starts with HOPG, then hBN. I submit, then run `cxr remote attach sub_100keV`. the summary progress bar *labels* both then show `829/1568 cases`, though the actual bar fill seems correct.
      1. For the `cases` bar, this should be fixed to be `829/(total_cases)`
      2. For `compute`, it should just be a percentage of completed compute.
      3. We don't need the repeated `x/21 materials` fields; that can just be its own line reported once, which should be shown on all levels of verbosity.
      4. For all these scan-progression indicators on `attach` or otherwise, it would be good on high-verbosity modes to report (with more colored bars but different colorscheme) showing compute usage -- CPU utilization, GPU utilization, percent/absolute memory utilization for host & GPU VRAM, etc. Those can be only on the highest verbosity level, though.
      5. When `cxr remote attach` gets paused/queued, the job-wide pregress bar(s) go green with a checkmark as if they're done. They should turn orange and go to the paused state. Same with the 'State' and 'SLURM' texts--they go blue when 'PENDING/queued' rather than the paused color (also, do we really need both of those status indicators? Seems just one would do--the state one, and say 'PENDING' rather than 'queued')
      6. 'profile=sub_100keV' should go on its own line
   2. Add info on whether or not chi_g/U_g were pulled from cache or recomputed for each mat on the highest verbosity level
   3. `remote attach` often gets disconnected by remote host closing connection. Add error handling to try to reconnect a couple times before allowing disconnect with explanatory message
   4. Add some `squeue` information dipslay reports tracking SLURM progress
   5. `remote attach`, add `p -> y (confirm)` sequence to pull the (potentially partially completed) items from the current job/profile being tracked
   6.  `remote scan` and `remote submit` should be combined into one `remote submit` which defaults to `scan`'s behavior, with a `--headless` flag to do current `submit behavior` and a `--no-pull` flag (can't do both since `--headless` won't pull anyway) which will attach & track progress but won't auto-pull results
   7.  `remote clear` should be able to clear based on `--profile`
   8.  evaluate `cxr [remote] prune [--all OR --profile NAME]` to drop stale checkpoints locally or on remote. Prune shouldn't require confirmation
   9.  
3. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
4.  **Debye-Waller provenance and anisotropy audit.** Continue replacing placeholder or reused `B_ang2` values with primary-source values and resolve per-site/tensor model needs. → `feature/debye-waller-audit`; audit: [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md).
   1. >user< Evaluate complexity/value of implementing full anisotropic/tensor-based Debye-Waller factors when available
   2. >user< Evaluate worth in both implementing the approximate scalar Debye-Waller formula (compare output to known values for various anisotropic materials we have in our DB), and in attempting to implement a fully-fledged DFPT system.
   3. >user< Take a crystal with a known DW factor, then manually change it up and down over a range of values that can reasonably be expected other crystals to have, and see how much it changes by -- if large, then its worth being careful here.
5.  **GPU-memory follow-up.** Benchmark remote `rebrem --all --ne-brem 500 --step 20` for bounded CuPy reserved-pool memory; assess analogous `reline` cleanup separately.

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3.  **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)

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
