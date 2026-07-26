# TODO / Backlog

Items live on `feature/...` / `bugfix/...` / `docs/...` branches, not `main`, till done.
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

1. **>user< CLI rework: `line-grid` → `energy-grid` + new `cxr sweep`.** Disentangle line-grid / brem-grid / derivation-geometry / scan-parameter-sweep settings; add read-write `cxr sweep` (default `[profiles.*]` + per-material scan ranges). Design: [`docs/cli-energy-grid-sweep-rework-plan.md`](docs/cli-energy-grid-sweep-rework-plan.md). **Prior to direct implementation of the following subitems, !AUDIT/CRITIQUE THEM!. Give alternative suggestions if any are problematic, unless current implementation is truly preferred. Then, after conferring with user, write up a triaged plan**
   1. Deferred within that rework: rename catalog `[profiles.*]` TOML table (scan-grid defaults) to `[scan_defaults.*]` to end the name clash with `profiles.py` `SweepProfile` (full/survey). Schema change — see P3.
   2. >user< Add capability for optional `start` `step` `stop` args (default `endpoint=True`) instead of csv list for setting sweep profiles
   3. >user< Ensure that derived upper energy bounds are not removed from memory. Specifically, the custom-derived line-and-brem grid upper bounds that were set by long-running monte carlo sims cannot be deleted, they are expensive to rederive.
   4. >user< Existing profile name tabcomplete/listing for `cxr sweep set --profile PRE-EXISTING-NAME(S)`
   5. >user< fix this bug (150 keV has been derived, its in standard profile):
      1. `(cxr-mc) (base) alexa@Alex-XPS15:~/dev/cxr-mc$ cxr sweep set --profile sub_200keV --energy 150\nError: no derived line grid for energy 150 keV; derive and apply it with cxr energy-grid first`
   6. >user< (related to above) Capbility to force-set energy grid bounds per-material or per-profile.
   7. >user< `cxr remote submit -h` lists only two profiles, `[full|survey]`, and neither of them exist. make it `standard|survey` and we will generate a `survey` profile
   8. >user<Wwhen doing `sweep set`, we should be able to add single values to csv lists without having to just write the whole thing again, e.g., add `75` keV to the sub_100keV profile without running `cxr sweep set --profile sub_100keV --energy 30,40,50,60,75,100`. adding like this should not activate the overwrite prompt unless its on the standard profile
   9.  >user< Need capability to delete profiles. Of course, require -y/--yes. forbid deletion of standard profile
   10. `cxr sweep set --profile` is a bit cumbersome. changing this to `cxr profile set <NAME> [options]` and just `cxr profile [show] <NAME> [options]` to `show` would be simpler. `show` here optional alias since it seems like users might expect that functionality, but `cxr profile <NAME>` does same thing
   11. >user< Need way to change profile in use en-masse (e.g. `--all` to swap all materials to new profile). Should always be transient (current session only) -- permanent changes are to be made directly to standard profile
   12. >user< Add `-l/--ne-line` and `-b/--ne-brem` as adjustable profile settings (sweepable, though single-valued by default)
   13. >user< `cxr remote pull` is currently broken for `survey` profile items; they are created, but it doesn't reference their checkpoint filenames properly.
   14. >user< need to be able to pull remote checkpoints using profile names, e.g., `hopg@sub_100keV.pkl` or similar. Give suggestions on naming/metadata conventions
   15. **>user< IMPORTANT** it seems to me that, rather than having materials be tied to one profile at a time, profiles should reference materials, and those materials can be added or removed at will. Within each profile, the materials default to the general profile default settings unless manually configured otherwise *within that profile*. If no materials have been added, the profile defaults to picking up the full list of standard materials and giving them all the profile default values. **EXPAND UPON THIS IDEA AND/OR GIVE FEEDBACK/CRITIQUES IF WARRANTED**
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
3.  **Superradiant PXR/CBS** need bunch length knowledge, coherent emission *across segments* (also needed by channeling radiation as in long-term features #1, #3)

## P2 - medium-priority

1.  **>user< Add newly created compute-progress readout from ./notebooks/scan_app.py to the remote readout as well (using the same nice colored bars that are already used there)**
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

1. **Rename catalog `[profiles.*]` TOML table → `[scan_defaults.*]`.** The scan-grid
   defaults table clashes name-wise with `profiles.py` `SweepProfile` (full/survey),
   two unrelated "profile" concepts. Catalog-schema change (materials.toml,
   `_parse_profiles`/`_parse_materials`, golden, docs). Blocked-by / folds into P2.7
   (`cxr sweep` rework). Design: [`docs/cli-energy-grid-sweep-rework-plan.md`](docs/cli-energy-grid-sweep-rework-plan.md).