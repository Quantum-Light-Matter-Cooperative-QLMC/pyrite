# Beam phase space — activate the declared-inert fields

Branch: `feature/beam-phase-space`
Worktree: `worktrees/beam-phase-space`
TODO scope: direct `/triage` input (2026-08-05) plus absorbed Inbox item 1
("finite initial phase space ... into 2D and 3D trajectory plots").

## Problem

The user asked for a `beam` object integrating bunch charge, repetition rate,
energy, longitudinal shape, emittance/phase-space parameters and spatial
distribution, then wired into profiles.

**That object already exists and is already wired into profiles.** Do not build
a second one. Current state:

| Concern | Where it lives today | Status |
| --- | --- | --- |
| The object | `BeamSpec`, `src/cxr_mc/sweep.py:67-125` | exists, dataclass on `Sweep.beam` |
| Central energy | `BeamSpec.energy_keV` | wired, primary swept axis |
| Spatial distribution | `transverse_fwhm_x_mm` / `_y_mm` | wired (elliptical Gaussian spot) |
| Longitudinal shape | `bunch_length_fs`, `long_shape`, `long_offsets_fs`, `longitudinal` | wired (`LongitudinalDistribution`: gaussian / microtrain / compressed) |
| Bunch charge, rep rate | `bunch_charge_pc`, `rep_rate_hz` | wired as flux normalization |
| Divergence, energy spread | `divergence_mrad`, `energy_spread_frac` | **declared, decoded, hashed — but never sampled** |
| Emittance / Twiss | none as *input*; `beam_metrics.py` computes them as *output* | gap |
| Profile plumbing | `[profiles.NAME.beam]` in `materials.toml:147-195`; decode `materials/catalog.py:203` + `_BEAM_KEYS` at `:691-704`; apply `config.material_sweep` `:183-188` via `beam_replace`; hash `profiles.py:284-318` | already complete, including the two inert keys |
| Diagnostics | `beam_metrics.sampled_beam_metrics` — per-plane RMS, geometric + normalized emittance, Twiss, longitudinal emittance, peak/average current (`Validation: beam-phase-space-metrics`) | exists |

So the real task is **not** "add a beam object". It is: *make the transverse
and energy phase-space fields physically live, and give them a
parameterization a beam physicist would actually write down.*

Confirmed gaps:

1. Nothing under `src/cxr_mc/montecarlo/` reads `divergence_mrad` or
   `energy_spread_frac`. `simulate_trajectories`
   (`montecarlo/transport.py:1165-1190`) takes a scalar `E0_keV` and a single
   `beam_dir`; it fans them out at `transport.py:1511`
   (`dirs = np.tile(beam_dir, (Ne, 1))`). `geometry.py:344` states the
   assumption explicitly: "perfectly collimated lab beam (zero divergence)".
2. There is no correlation / waist parameter. RMS divergence alone cannot
   express `<x x'>`, so a converging or diverging beam at the crystal face is
   inexpressible.
3. Emittance is asymmetric: measurable on the way out (`beam_metrics.py`), not
   specifiable on the way in.

### Out of scope

Space-charge, source-to-crystal beamline transport, and the coherent form
factor. This task defines the beam **at the crystal entrance face** and samples
it; it does not evolve a beam to get there.

## Feedback / critique (read before writing code)

These are the substantive design risks. Items 1, 2 and 4 are the ones that
change the plan.

**Review status (user, 2026-08-05): 1-5 and 7 accepted as written and binding.
6 accepted after clarification (see its note). 8 ruled out — no chirp model.**

1. **Normalized, not geometric, emittance must be the stored input.**
   `energy_keV` is the primary swept axis and this repo spans 30 keV
   (`beta*gamma ~ 0.34`) to the REGAE-scale 3–5 MeV (`beta*gamma ~ 7`) case in
   the P1 backlog. Geometric emittance is not invariant across that sweep, so a
   single geometric value attached to a multi-energy `BeamSpec` is unphysical —
   it silently means a different beam at every energy. Store
   `normalized_emittance_mm_mrad` and derive `eps_geom = eps_n / (beta*gamma)`
   per case. This mirrors what `beam_metrics.py:94-95` already reports.
   Same argument applies to `divergence_mrad` as a raw input: keep it, but
   define it as the derived RMS at the case energy, not an energy-independent
   constant.

2. **Do not let FWHM, divergence and emittance all be settable at once.**
   Per plane there are three independent second moments
   (`<x^2>, <x x'>, <x'^2>`), equivalently `(eps, beta_twiss, alpha_twiss)`.
   Spot FWHM + divergence + emittance is three numbers that over-determine two
   and drop the correlation. Choose one canonical set and make the others
   mutually exclusive with a hard error — silent precedence here produces a
   plausible-looking beam that is not the one requested. Recommendation:
   canonical `(eps_n, beta_twiss_m, alpha_twiss)` per plane;
   `sigma_x = sqrt(eps_geom * beta)`, `sigma_x' = sqrt(eps_geom * gamma)`,
   `<x x'> = -eps_geom * alpha`. The existing FWHM fields stay as the legacy /
   convenience spelling and map to the zero-alpha waist.

3. **Follow the `longitudinal` precedent instead of growing the flat field
   list.** `BeamSpec.longitudinal: LongitudinalDistribution | None` is already
   the established pattern for "declarative sub-object, mutually exclusive with
   the flat legacy fields, resolved per case". Add
   `transverse: TransverseDistribution | None` the same way rather than adding
   six more flat floats. This gets mutual exclusion, per-case resolution and
   the divergence-only hashing rule for free, and keeps
   `profiles.py:284-318` structurally uniform.

4. **`_BEAM_POSITIVE_KEYS` is a trap for the new fields.**
   `materials/catalog.py:691-702` forces every listed beam key positive.
   `alpha_twiss` is legitimately negative (diverging beam past the waist), and
   any chirp / `<t delta>` term is signed. Adding them to that frozenset would
   reject valid profiles. They need a separate signed-key set.

5. **Divergence and energy spread are NOT inert on the spectrum — that is the
   point, but check where it lands.** `montecarlo/spectrum.py:571` consumes
   per-segment `v_hat` and per-segment `E_keV`, so once injection is
   randomized the emitted line spectrum picks it up with no further work. Two
   consequences:
   - The `E_grid_line` window is derived from the *nominal* case energy. An
     energy-spread-broadened line can fall outside it and be silently clipped.
     Cross-check against the per-material `E_grid_line_by_energy` bounds work
     before enabling `energy_spread_frac` in any shipped profile.
   - The analytic broadening helpers (`mosaic_fwhm_eV`, `aperture_fwhm_eV`,
     `mosaic_psi_rad` at `montecarlo/detector.py:117-153`) stay at the nominal
     `beam_dir` / `E0_keV` by design. They will not widen. That is acceptable
     as long as it is documented — they are diagnostics, not the spectrum — but
     it must not be mistaken for the sampled result being wrong.

6. **Charge and rep rate are normalization, not phase space.** They answer
   "how many electrons per second", not "where each electron is", so they are
   the wrong kind of number to feed the sampler. Today they reach only
   `beam_metrics.py` (diagnostics) and the CLI/profile plumbing — nothing under
   `montecarlo/` reads them.

   The risk is specific to this task: once emittance is a real input,
   `BeamSpec` starts to look like a complete physical beam, and the obvious
   "add realism" move is to derive MC statistics from charge — `Ne = Q/e`, or
   weighting the incoherent sum by `N_phys`. A 1 pC bunch is 6.24e6 electrons
   against ~300 macro-particles. Any such coupling changes the RNG draws (no
   run stays bit-for-bit against `main`), risks double-counting against
   downstream normalization, and wrecks runtime.

   Worth a test rather than a docstring because coherent/superradiant emission
   genuinely scales with `N_phys` (N^2 vs N), so a legitimate future path does
   let charge enter the physics. The test pins today's contract — coherent path
   off, `bunch_charge_pc` / `rep_rate_hz` bit-for-bit inert on every sampled
   array (`initial_r_ang`, `initial_v_hat`, `t0_ang`, segment arrays, same
   seed) — so when that changes it is a deliberate ledgered decision rather
   than drift.

   Noted separately: charge x rep-rate does **not** currently normalize the
   spectrum to absolute flux; there is no photons/second anywhere, only the
   diagnostics struct. Wiring an explicit absolute-flux multiplier is a real
   missing feature and a separate backlog item, not part of this task.

7. **State the reference plane.** At 1 pC in 200 fs at 30–100 keV, space charge
   is not negligible between a realistic source and the target. Since this task
   does not model it, the docstring must say the parameters describe the beam
   **at the crystal entrance**, so nobody plugs in gun-exit numbers and reads
   the output as physical.

8. **Chirp — RULED OUT (user, 2026-08-05).** The concern was that the
   `compressed` longitudinal policy implies `<t delta> != 0`, which would
   double-count against an independent `energy_spread_frac`. User ruling:
   `compressed` is terminology only, not a chirp model. So
   `energy_spread_frac` is sampled as an **independent, uncorrelated** draw
   regardless of `long_shape`, and `<t delta>` is zero by construction.
   `beam_metrics` will therefore report a longitudinal emittance of
   `sigma_t * sigma_delta` with no correlation term. Record this as an explicit
   assumption in the design note so a future chirp model is an additive change,
   not a silent reinterpretation.

## Implementation path

Likely owners, in dependency order:

- `src/cxr_mc/sweep.py` — `BeamSpec` fields + `TransverseDistribution`
  (mirroring `LongitudinalDistribution`), `build_cases` resolution per
  (material, energy) case.
- `src/cxr_mc/montecarlo/transport.py` — replace the `np.tile` fan-out at
  `:1511` with a sampler; add per-electron initial energy alongside the
  existing per-electron `E_cut_by_electrons` precedent; extend the
  `initial_*` diagnostic arrays already returned at `:1691`.
- `src/cxr_mc/montecarlo/geometry.py:344` — retire the zero-divergence
  assumption note; the projection helper must accept per-electron directions.
- `src/cxr_mc/materials/catalog.py:691-704` — signed vs positive key sets,
  nested `[profiles.NAME.beam.transverse]` decode.
- `src/cxr_mc/profiles.py:284-318` — divergence-only hashing for the new keys.
- `src/cxr_mc/cli/commands/profile.py` — `_beam_cli_options` (`:98-130`),
  `_collect_beam_updates` (`:183-211`), `_apply_beam_updates`, and the beam key
  tuple at `:342`. This is the only place `cxr` exposes a beam surface today:
  `--transverse-fwhm-mm`, `--rep-rate-hz`, `--bunch-charge-pc`,
  `--longitudinal`, `--envelope-rms-fs`. The transverse block needs the same
  treatment `--longitudinal` already got.
- `src/cxr_mc/cli/commands/scan.py:453-459` and `:485-491` — every `beam_*`
  kwarg is hardcoded `None`, so `cxr scan` reaches `scan.py`'s live
  `beam_overrides` dict (`scan.py:610-618`) with nothing. See open question 3.
- `src/cxr_mc/plots/trajectories.py:229` and `plots/plotly_trajectories.py` —
  the absorbed Inbox item; `initial_v_hat` is already returned and plumbed, it
  is simply degenerate today.
- `docs/sweep-profiles.md`, `docs/physics-validation-ledger.md`,
  `docs/validation/` — new `Validation: <id>` rows.

## Checklist

- [x] **A. Design note.** Write `docs/beam-phase-space.md`: canonical
      parameterization, mutual-exclusion table, reference plane, normalized-vs-
      geometric convention, chirp decision (critique 8). Get it reviewed before
      code.
- [x] **B. `TransverseDistribution`** in `sweep.py` + per-case resolution.
      Inert default is bit-for-bit legacy. Landed as its own module,
      `src/cxr_mc/transverse.py`, mirroring `longitudinal.py`; `sweep.py` holds
      the `BeamSpec` field, the mutual-exclusion error, and
      `BeamSpec.with_transverse`.
- [x] **C. Transport sampling** — per-electron `v_hat` and initial energy from
      the resolved distribution, on RNG children independent of the transport
      draws (follow `_sample_bunch_offsets` / `spawn(4)[3]` precedent so the
      zero-spread limit is bit-for-bit). Landed as `transverse_distribution` and
      `energy_spread_frac` kwargs on `simulate_trajectories`, forwarded by
      `runner._beam_kwargs`, on child streams `spawn(5)[4]` and `spawn(6)[5]`.
      Slopes are referred to the beam axis via the new
      `geometry.beam_frame_basis`, which is exactly `I` on axis; the
      zero-divergence note on `project_beam_entry` is retired.
- [x] **D. Limiting-case tests** — `eps_n -> 0` reproduces the collimated run
      exactly; charge/rep-rate invariance test (critique 6). Delivered as two
      tests rather than one: the *unset* policy is bit-for-bit with the
      pre-BeamSpec run, while `eps_n -> 0` is a convergence test — a strictly
      positive emittance cannot make `sqrt(eps*beta)` underflow to exactly zero,
      so the four geometry arrays are bounded by `< 1e-100` (which at their
      magnitudes of 0.3–4.8e4 means bit-for-bit anyway) and every other array,
      scalar and tally is asserted exactly equal.
- [x] **E. Round-trip test** — sample from a known `(eps_n, beta, alpha)` and
      recover it through `beam_metrics.sampled_beam_metrics` within MC error.
      This closes the input/output asymmetry and is the strongest single check.
- [ ] **F. Profile decode** — nested `beam.transverse` block, signed-key fix
      (critique 4), a demo profile in `materials.toml`, golden regen
      (`tests/data/material_catalog_golden.json`) via the `regen-golden` skill.
      *Partial:* decode and the `_TRANSVERSE_SIGNED_KEYS` split are done and the
      golden is unchanged (no shipped profile sets the block yet); the demo
      profile and its golden regen are still open.
- [x] **G. Identity hashing** — new keys join `parameter_sha256` only when they
      diverge from inert defaults; assert existing digests unchanged.
- [ ] **H. `E_grid_line` interaction** — verify a broadened line is not clipped;
      if it is, widen bounds or gate `energy_spread_frac` (critique 5).
- [ ] **I. Trajectory plots** (absorbed Inbox 1) — 2D/3D plots show the finite
      initial phase space; both longitudinal and transverse.
- [ ] **J. Physics ledger** — source equation, assumptions, limiting case,
      `Validation: <id>`, ledger row for the new sampling. Fresh-context
      verification via `physics-validator`; only the human signs off.
      *Partial:* row `beam-phase-space-injection` is in the ledger at status
      `filtered`, with `Validation:` markers on both public functions in
      `transverse.py`; fresh-context verification and its write-up are open.
- [x] **K. Docs** — `docs/sweep-profiles.md` beam block reference,
      `docs/repo_map.md` pointer, `docs/cli-reference.md` regen (with L).
      Also `docs/index.md` toctree entry for the design note.
- [x] **L. CLI surface** — extend `_beam_cli_options` with the transverse
      block, mirroring the `--longitudinal` / `--envelope-rms-fs` pair:
      `--transverse` (kind), the Twiss triplet per plane, and
      `--energy-spread`. Mutual exclusion with `--transverse-fwhm-mm` is a
      `click.UsageError` in `_collect_beam_updates` (decision 1: hard error, no
      silent precedence), matching the existing
      `"--envelope-rms-fs requires --longitudinal"` style. Negative
      `alpha_twiss` must pass the option type — do NOT use `FloatRange(min=0)`
      like its neighbours (critique 4, same trap as `_BEAM_POSITIVE_KEYS`).
      Add the new keys to the `:342` tuple and `_apply_beam_updates`. Follow the
      `cli-ui-ux` skill; regenerate `docs/cli-reference.md`; preserve the
      documented command/help/output/exit contracts.

## Decisions

Settled by user review, 2026-08-05. Binding — do not relitigate.

1. **Canonical parameterization: Twiss triplet per plane**,
   `(eps_n, beta_twiss_m, alpha_twiss)`. Normalized emittance is the stored
   input; geometric is derived per case as `eps_n / (beta*gamma)`. Spot FWHM
   remains the legacy convenience spelling, mapping to the zero-alpha waist,
   and is mutually exclusive with the triplet (hard error, no silent
   precedence).
2. **Shape:** new `TransverseDistribution` frozen sub-object on `BeamSpec`,
   mirroring `LongitudinalDistribution`. No further flat float fields.
3. **No chirp model.** `<t delta> = 0` by construction; `energy_spread_frac` is
   an independent uncorrelated draw for every `long_shape`, `compressed`
   included. (critique 8)
4. **Charge / rep-rate stay inert on sampling**, pinned by a bit-for-bit test.
   Absolute-flux normalization is a separate backlog item. (critique 6)
5. **Reference plane: crystal entrance face.** No space charge, no beamline
   transport. Must be stated in the `BeamSpec` docstring. (critique 7)

## Open questions

1. Do `divergence_mrad` / `energy_spread_frac` survive as public spellings, or
   become derived read-only properties of the transverse distribution? They are
   already decoded and hashed, so retiring them is a deprecation, not a delete.
   Lean: keep `energy_spread_frac` (it is not a transverse quantity and has no
   Twiss equivalent), demote `divergence_mrad` to a derived property.
2. Is the finite crystal footprint / groove entry path affected by per-electron
   directions? `entry_points` and the prism-exit helper assume the shared
   `beam_dir` in places.

3. Should `cxr scan` regain per-run beam overrides, or does the beam stay
   profile-only on that command? `scan.py:610-618` still builds a live
   `beam_overrides` dict, but `cli/commands/scan.py:453-459` / `:485-491` feed
   it `None` for all six keys, so the plumbing is reachable but unreached.
   Lean: leave `scan` profile-only and put the transverse surface on
   `cxr profile` alone (step L), so there is exactly one spelling — but that
   makes the `scan.py` dict dead code that should then be deleted or flagged.
   Decide before writing L; do not add flags to both.
   **Resolved 2026-08-07 (user):** the beam is profile-owned, with no per-run
   override path — "i don't want users manually set beam parameters every run".
   L shipped accordingly: transverse flags on `cxr profile create` /
   `cxr profile edit` only. The dead `beam_overrides` dict in `scan.py`, the six
   `None` kwargs at both `cli/commands/scan.py` call sites, and the orphaned
   `_beam_uvw` callback are deleted; `tests/test_scan_beam_options.py` already
   pinned the flags as rejected. (`cxr run` is the runner command now; `scan.py`
   survives as its backend.)

Closed during triage: nothing under `src/cxr_mc/montecarlo/` reads
`divergence_mrad` or `energy_spread_frac` today, and no shipped profile or
notebook sets them, so activating them cannot silently change an existing run.

## Delegation slices

| Slice | Steps | Skill |
| --- | --- | --- |
| Design note + decisions | A | `lead-task` + `physics-review` |
| Spec object + case resolution | B | `implement-task` |
| Transport sampling + limiting cases | C, D, E | `implement-task` + `monte-carlo` |
| Profile decode, golden, hashing | F, G | `implement-task` + `regen-golden` |
| Grid-bounds interaction | H | `implement-task` + `monte-carlo` |
| Trajectory plots | I | `implement-task-lite` + `notebook-workflow` |
| Ledger + docs | J, K | `physics-validation`, `documentation-maintenance` |
| CLI surface | L | `implement-task` + `cli-ui-ux` |

A is a blocking gate: C onward depends on the parameterization decision.
L depends on B only (it edits the option layer, not the sampler) and can run in
parallel with C–E.

## Acceptance checks

- `uv run cxr-dev verify` green.
- Zero-spread default run is bit-for-bit identical to `main` (segment arrays and
  spectrum), and every pre-existing `parameter_sha256` is unchanged.
- Round-trip test (E) recovers input Twiss within MC error at two energies that
  differ in `beta*gamma`, demonstrating the normalized-emittance convention.
- Profile TOML round-trips a `beam.transverse` block including a negative
  `alpha_twiss`.
- New physics carries a `Validation: <id>` marker and a ledger row;
  `physics-ledger-auditor` reports no orphans.
- CLI: a transverse block set purely from `cxr profile` flags produces the same
  `BeamSpec` as the equivalent TOML block; combining it with
  `--transverse-fwhm-mm` exits non-zero with a usage error naming both flags;
  a negative `alpha_twiss` is accepted; `docs/cli-reference.md` is regenerated
  and no pre-existing command help/exit contract changed.
