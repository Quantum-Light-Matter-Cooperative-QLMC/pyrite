;# TODO / Backlog

Shared, priority-ordered backlog. Keep identical on every branch. Branch detail
lives in `tasks/<branch-leaf>.md`; workflow and merge rules:
[`tasks/README.md`](tasks/README.md). `>user<` marks untriaged user text that
must remain until moved into a task file. Reconcile with `todo-sync`.

## P1 - top-priority / high-value

### Active

1. **Align local/remote run surfaces and profile selection.** Canonicalize
   `cxr run` / `cxr remote run`, move material-group selection to profiles,
   rename performance flags, add `--level9`, and drop deprecated command paths.
   → `feature/cli-run-alignment`; [`tasks/cli-run-alignment.md`](tasks/cli-run-alignment.md).
2. **Group interactive apps and exports.** Move analysis, viewer, and validation
   launch/export flows under `cxr app`.
   → `feature/cli-app-suite`; [`tasks/cli-app-suite.md`](tasks/cli-app-suite.md).
3. **Move prune under checkpoint.** Make `cxr checkpoint prune` canonical
   without changing prune behavior.
   → `feature/checkpoint-prune`; [`tasks/checkpoint-prune.md`](tasks/checkpoint-prune.md).
4. **Consolidate material command tree.** Fold catalog validation and blazed
   sweeps into `cxr material`.
   → `feature/material-command-tree`;
   [`tasks/material-command-tree.md`](tasks/material-command-tree.md).
1. **Physics validation ledger.** Continue fresh-context re-derivations, add missing in-code `Validation: <id>` markers. Design: [`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md); method: [`docs/validation/README.md`](docs/validation/README.md).
2. **Debye-Waller provenance and anisotropy audit.** Replace placeholder/reused `B_ang2`; evaluate scalar sensitivity, tensor factors, and DFPT value/complexity. → `feature/debye-waller-audit`; [`docs/debye-waller-audit.md`](docs/debye-waller-audit.md). >user<
3. **GPU-memory follow-up.** Benchmark remote `rebrem --all --ne-brem 500 --step 20` for bounded CuPy reserved-pool memory; assess `reline` cleanup separately.

### Gated

1. **Measured-data validation.** General experimental-simulation comparison & validation. Particularly: compare modeled broadened line widths vs measured HOPG rocking-curve / EDS dataset. Design: [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

### On-Hold

1. **High-energy electron/channeling support.** Evaluate `Geant4` or similar for REGAE@DESY-scale beams (3–5 MeV, 50 fs, 100 fC, 200–300 µm target diameter), JungFrau detector ~0.5–4.5 m from interaction point. USER QUESTION: What is rep rate?
2.  **Bent Crystals (After add channeling + relativistic electrons)**
3. **Superradiant PXR/CBS validation.** Optional phased segment/electron sum is implemented but unverified; resolve phase convention and bunch-form-factor limits before scientific use. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

## P2 - medium-priority

1. **Remote-job UX.** Profile-based job names + same-profile submit block, quieter submit pull suggestion, explicit-material-only stop list, faster `stop --all`, attach-time cancel key + compute-aware progress bars. → `feature/remote-ux`.
2. **Grazing grating — ALEX-s constants + hardware survey.** Research cited device constants and ~10 eV–4 keV CCD/grating landscape. → `docs/soft-xray-hardware-survey`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
3. **Parameter-space sampling review.** Design principled prioritization across high-dimensional sweep parameters. → `docs/parameter-space-sampling-review`.
4. **Grazing grating — groove efficiency.** Replace `Grating.groove_efficiency` placeholder scalar with groove-profile model. → `feature/grating-groove-efficiency`; Design: [`docs/grazing-grating.md`](docs/grazing-grating.md).
5. **pyelsepa / ELSEPA transport.** Maintain landed, validated adapter externally provisioned CI environment.
6.  **Material filters.** Model calibration-filter transmission between x-ray beam and detector. → `feature/material-filters`.
7. **Trace render UX.** Fix stale cached animation reuse/random seeds; place progress near render action; crop background/legend while keeping overlays; improve MP4 quality; add save dialog. >user<
8. **Coherence ON-vs-OFF comparison.** Pair otherwise-identical coherent/incoherent datasets; report peak and integrated-flux ratios. Design: [`docs/coherent-emission.md`](docs/coherent-emission.md).

## P3 - lower / exploratory / small bugfixes
