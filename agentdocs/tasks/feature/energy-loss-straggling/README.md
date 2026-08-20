# Energy-loss straggling

## Problem and scope

Transport is pure continuous slowing down. `_dEds_compound_scalar` returns a
*mean* loss rate and every core applies it deterministically, so the fluctuation
about that mean — Landau/Vavilov/Bohr straggling — is discarded by construction.
The repository already documents this as a known omission in four places
(`docs/physics/beam-transport/stopping-power.md` "Assumptions and limits",
`docs/physics/beam-transport/electron-transport.md`,
`docs/validation/beam-transport/energy-step-convergence.md`,
`docs/validation/beam-transport/substep-radiation-invariance.md`) and it is named
as a direction under TODO "Long-term plans → Broader physics scope".

The concrete driver is already measured. `energy-step-convergence` records that
by Jensen's inequality `<1/beta(E)> != 1/beta(<E>)`, so straggling biases the
**mean** arrival time and not only its variance — estimated at ~0.3 rad at
25 keV over 1 um at 1 keV photon energy, against the 0.1 rad numerical tolerance
the midpoint propagator was built to reach. Numerical precision has outrun the
transport model at that point: refining the propagator no longer improves phase
fidelity, because the limiting error is an unmodeled physical spread. Secondary
consequences are range straggling, the CSDA range/backscatter distributions, the
bremsstrahlung endpoint region, and a `exp(-sigma_phi^2/2)` Debye--Waller-type
suppression of the coherent line currently estimated at 1e-3--1e-5 of the
multiple-scattering term.

This task owns the **fluctuation** about the energy-loss rate: its model form,
its sampler, its RNG plumbing across the four transport cores, and its
observable consequences. It does **not** own the mean stopping power itself
(→ `feature/reference-electron-stopping-data`), elastic scattering
(→ `feature/reference-elastic-scattering-data`), radiative stopping, or
bremsstrahlung angular distributions.

### Sequencing: this is not the first transport fix

**Do the mean before the fluctuation.** Joy--Luo under-stops by 6% at 25 keV and
roughly a factor of two at 300 keV
(`stopping-power.md`, table `tbl-stopping-validity-ceiling`). That error is
first-order and **systematic** — same sign for every electron, accumulating
coherently across the ensemble, displacing the coherent line. Straggling's
contribution to the mean arrival time is a second-order Jensen bias off a
~300 eV spread, and its direct effect is random: it suppresses the line via
`exp(-sigma_phi^2/2)` rather than moving it. Sampling a fluctuation about a mean
that is itself 6--50% low is out of order.

`feature/relativistic-bethe-stopping` closes the systematic error with a
published closed-form expression and no licensing exposure. **This task should
be sequenced behind it**, and its acceptance numbers re-measured against the
corrected clock — the ~0.3 rad straggling estimate was computed against a CSDA
clock built on the biased stopping power, so the figure itself is provisional.

Nobody has yet computed the systematic phase error implied by the 6% stopping
bias at the same 25 keV / 1 keV operating point. That comparison is cheap and
belongs in slice A, because it is what decides whether straggling is the
limiting error at all once the mean is fixed.

### What the model-form question actually is

An earlier draft of this record claimed that unrestricted CSDA plus
full-distribution sampling double-counts the hard Moller tail. **That was
wrong** and is corrected here: replacing a deterministic `<dE/ds> * s` with a
random variable of the *same mean* has the right mean and the right variance and
is self-consistent. Double-counting would arise only if delta rays were also
produced as separate loss events on top of unrestricted stopping.

EGSnrc, Geant4, and PENELOPE use restricted stopping plus explicit hard
inelastic events because they want to *transport* the resulting delta rays.
PyRITE does not, and deposits that energy locally, so it does not inherit that
requirement. Restricted stopping is available as an option, not a prerequisite.

What does survive is the **regime** constraint. `energy-step-convergence`
measures `kappa = 0.015` for carbon at 25 keV over **1 um** — already strongly
skewed and outside the Gaussian (Bohr) limit — and records of order **one**
inelastic event per flight in carbon at 25 keV (~0.11 in tungsten). Flights are
elastic-mean-free-path scale, so per flight the process sits in the
few-collision limit where neither Bohr nor Vavilov applies.

That is a model-*selection* constraint with a known answer, not a blocker.
Geant4's Urban fluctuation model exists precisely for the regime where Landau
and Gaussian both fail, and is the leading candidate. Note also that the
observable in question — coherent phase — accumulates over the *whole
trajectory*, not one flight, so the many-collision limit is closer to applicable
at the level that matters than the per-flight statistics alone suggest. The
sampler still has to be correct per flight, which is what makes Urban-style
models the right family.

## Implementation path and likely owners

Current owners:

- `montecarlo/transport.py::_dEds_compound_scalar` / `::_dEds_packed_scalar` —
  the mean loss rate, applied identically by every core.
- The four ungrooved/grooved cores and their LUT variants:
  `::_transport_core_ungrooved`, `::_transport_core_ungrooved_lut`,
  `::_transport_core_grooved`, `::_transport_core_ungrooved_perelectron`,
  `::_transport_core_ungrooved_perelectron_lut`, plus the CUDA equivalents in
  `montecarlo/transport_jit_kernel.py`.
- `montecarlo/transport.py::stream_keys` and the `_splitmix64` /
  `_stream_uniform_scalar` counter-addressed stream machinery — the only
  mechanism by which a new in-core draw can stay reproducible and identical
  across the host and CUDA cores.
- `montecarlo/transport.py::simulate_trajectories` — the `energy_model` /
  `max_dE_frac` keyword surface and the cutoff-truncation solve
  (`eq-stopping-cutoff-distance`).
- `montecarlo/runner/__init__.py:646` — the production call site.
- `campaign/model.py::Numerics` — where a user-facing toggle would have to land.

Two structural facts shape the plan:

- **The production path never leaves `energy_model="frozen"`.** The runner call
  site passes neither `energy_model` nor `max_dE_frac`, and `Numerics` carries
  no field for either, so `midpoint` and substepping are today reachable only
  through the direct API. Whatever straggling attaches to inherits that
  reachability unless slice G plumbs it. This is a scope decision, not an
  oversight to fix in passing.
- **`energy_model="midpoint"` is implemented for the ungrooved lockstep core
  only** and raises on every other core. That fail-closed precedent is the model
  to copy for a staged rollout (slice F), not something to work around.

Interactions that need explicit design rather than incremental patching:

- **Cutoff truncation.** The midpoint rule solves the truncation distance for
  `E_end == E_cut` exactly. A straggled loss makes `E_end` a random variable, so
  that solve no longer defines the stopping point; the cutoff crossing has to be
  redefined (and the resulting `n_cutoff_stopped` bookkeeping re-checked).
- **Substep invariance.** `max_dE_frac` currently guarantees that subdividing a
  flight is algebraically invariant. With per-substep draws that becomes a
  *distributional* invariance at best. The infinite-divisibility property of the
  chosen distribution decides whether even that holds, and
  `substep-radiation-invariance` will need re-derivation rather than re-running.
- **RNG independence.** Existing independent draws (bunch offsets, beam spot,
  energy spread) are per-electron and pre-sampled outside the core loop via
  `SeedSequence` children `spawn(4)[3]`, `spawn(5)[4]`, `spawn(6)[5]`. A
  straggling draw is per-flight and cannot be pre-sampled, so it must be
  counter-addressed on `(electron, flight, substep)` to keep the
  straggling-off path bit-for-bit and keep the three host cores and CUDA
  addressing the same streams.

## Checklist

- [x] A — Regime audit **and priority check**. **Done. Gate outcome: PROCEED —
      the systematic term did dominate, and the task it gated on has already
      landed.** Instrument:
      `agentdocs/tasks/feature/energy-loss-straggling/slice_a_regime_audit.py`
      (`--part regime|mc|gate`), reusing the repository's own Browning elastic
      cross section, `spliced_stopping_keV_per_ang`, and the
      `_element_crossover_keV -> inf` seam that reproduces the retired
      pure-Joy--Luo model bit-for-bit.

      **Regime, 9 catalog materials (hopg, diamond, hbn, 4h_sic, silicon, mos2,
      ws2, wse2, ptbi2) plus bare `C(0.1136)` / `W(0.06305)`, at 1, 2, 5, 10,
      25, 50, 100, 200, 300 keV.**

      | quantity | range over the whole matrix |
      |---|---|
      | flight length `lam_el` | 4.3 Ang (W, 1 keV) to 4211 Ang (C, 300 keV) |
      | `kappa` per flight | 2.0e-5 (ptbi2, 300 keV) to 2.9e-2 (ptbi2, 1 keV) |
      | `kappa` over the CSDA range | **0.080 to 0.16, at every energy and material** |
      | hard (`eps^-2`) collisions per flight | 0.003 (ptbi2) to 0.16 (C, 300 keV) |
      | all inelastic per flight, `S lam / I` | 0.03 (W) to 2.6 (C, 300 keV) |

      Two results decide slice B. **(i)** Per flight the process is not merely
      sub-Gaussian, it is sub-*Landau*: `kappa <~ 6e-3` above 2 keV everywhere,
      with **well under one** close collision per flight (0.003--0.16). Landau
      theory assumes many collisions in the `eps^-2` tail; that assumption fails
      by two orders of magnitude. **(ii)** Integrating over the *entire* CSDA
      range does **not** rescue it: `kappa_R` is 0.080--0.16 across all 50-material
      chemistry and all 1--300 keV, essentially flat, and never approaches the
      `kappa >~ 10` Gaussian limit. So Bohr/Gaussian straggling is inadmissible
      at every point in the catalog, per flight *and* integrated, and the
      few-collision family (Urban-style) is the only admissible one.

      **Measured flight-length distribution** (production `mott` core, frozen
      energy model, 400 electrons x 4 seeds, slab = 2x `R_CSDA`, 1.3e5--4.0e6
      flights per cell). Normalized to the analytic `lam_el(E0)`: mean
      0.47--0.83, median 0.29--0.46, p10 0.06--0.09, p90 1.4--1.7, p99 3.0--3.6.
      Broadly exponential (median/mean ~0.62 against `ln 2` = 0.693) contracted
      by slowing-down, since the Browning hazard rises as `E` falls.

      **Gate, at `energy-step-convergence`'s own 25 keV / 1 um / 1 keV-photon
      point in graphite**, both clocks integrated exactly (RK4 on `dE/ds`,
      `t = int ds/beta`) rather than linearized:

      | term | value |
      |---|---|
      | systematic, retired Joy--Luo vs current splice | **24.6 rad** |
      | the same, linearized cross-check | 22.0 rad |
      | straggling Jensen bias, full Moller `T_max = E/2` (`sigma_E` = 1.54 keV) | 13.4 rad |
      | straggling Jensen bias, soft collisions only (`sigma_E` = 0.68 keV) | 2.6 rad |
      | `energy-step-convergence`'s stated figure (implies `sigma_E` ~ 0.3 keV) | ~0.3 rad |

      The systematic term dominated the straggling term by 1.8x against the
      loosest straggling estimate and by ~80x against the doc's own figure, so
      the gate fires: **this task waits on `feature/relativistic-bethe-stopping`.**
      That wait is already discharged — `2c51754`, `dcdb1cd`, `d15a8ef` are
      ancestors of both `main` and this branch, and that task's checklist A--G is
      complete. Nothing is blocked.

      **Acceptance numbers re-measured against the corrected clock.** With the
      splice in place the systematic channel is no longer 6%; what remains is the
      accuracy of the mean stopping power itself. On the corrected clock a
      residual stopping error of 0.1% / 0.5% / 1% / 2% costs **0.35 / 1.76 /
      3.51 / 7.02 rad** at the same operating point. The straggling bias
      (2.6 rad soft, 13.4 rad upper bound) is therefore no longer subdominant to
      anything that is *fixed*: it is comparable to a ~1% residual mean-stopping
      uncertainty and an order of magnitude above the 0.1 rad numerical
      tolerance. The `~0.3 rad` figure that motivates this task is a soft-collision
      underestimate by roughly 10--50x; the motivation is strengthened, not
      weakened.
- [ ] B — Distribution selection. Choose the fluctuation model for the measured
      few-collision regime — Urban-style (the leading candidate, built for
      exactly this regime), Vavilov, or Blunck--Leisegang-corrected Gaussian —
      justified against A's numbers. Decide whether to use unrestricted stopping
      with a full-distribution sampler (the simpler path, self-consistent while
      delta rays are untransported) or restricted stopping with explicit hard
      inelastic events (larger, and only warranted if delta-ray production
      becomes observable). Confirm whether delta rays remain a recorded
      non-goal.
- [ ] C — Sampler derivation and unit test. Derive the selected distribution's
      sampler from its source (Landau via Boersch-Supan/Koelbig--Schorr, Vavilov
      via Rotondi--Montagna or Chibani, Bohr/Blunck--Leisegang Gaussian —
      whichever B selects), check units, limits, and signs, and pin its first two
      moments against the analytic mean and variance. Include the limiting case
      that the sampler reduces to the current deterministic loss as its width
      parameter goes to zero.
- [ ] D — RNG plumbing. Add a counter-addressed straggling stream keyed on
      `(electron, flight, substep)`, reusing the `stream_keys` / `_splitmix64`
      machinery. Required properties, each with a test: straggling off is
      bit-for-bit identical to today on all four cores; straggling on never
      perturbs the free-path or scattering-angle draws; the same
      `(seed, electron, flight)` yields the same loss on lockstep, per-electron,
      and CUDA cores up to libm ulp.
- [ ] E — CPU integration on the ungrooved lockstep core, including the
      redefined cutoff crossing and the `max_dE_frac` substep interaction.
      Re-derive rather than re-run `substep-radiation-invariance` for the
      straggled path.
- [ ] F — Remaining cores: grooved, per-electron, per-electron LUT, CUDA. Any
      core not yet covered raises, matching the existing `energy_model`
      fail-closed precedent, rather than silently returning unstraggled results.
      Measure the per-step cost and device-register impact.
- [ ] G — Surface decision and plumbing. Decide whether straggling reaches
      production runs; if yes, thread it (and, unavoidably, the `energy_model`
      selector it depends on) through `Numerics` → case → runner, and extend
      checkpoint/case identity so straggled and unstraggled records cannot
      collide in the CAS.
- [ ] H — Observable measurement. Quantify the effect on backscatter and
      transmission fractions, CSDA range and range straggling, the
      bremsstrahlung spectral shape, and the coherent line. The load-bearing
      number is whether the ~0.3 rad Jensen mean-arrival-time bias at the
      25 keV/1 keV point closes, since that is the stated motivation; report the
      residual either way. Heavy/GPU matrices via `pyrite remote`.
- [ ] I — Docs, ledger, and goldens. Add `Validation: energy-loss-straggling`
      with a ledger row in `ledger-transport-background.md`; rewrite the
      "No straggling" block in `stopping-power.md`; and update the four
      write-ups that currently cite straggling as unmodeled so none of them
      still asserts an omission that has been closed.
- [ ] J — Computational/statistical techniques docs. Update
      `docs/computation/random-streams.md` for the new per-flight
      counter-addressed straggling stream: which counter tuple it is keyed on
      (`(electron, flight, substep)`), where it sits relative to the
      `SeedSequence` child tree and the existing counter-addressed streams, what
      the draw-order contract now guarantees, and the off-path bit-for-bit
      inertness claim — following that page's own "Adding a new random input"
      procedure rather than bolting on a section. Update
      `docs/computation/statistical-methods.md` where the added per-flight
      variance changes what the estimators see (error bars, seed replication,
      paired-seed shift, and the coherent-vs-incoherent split, since the
      `exp(-sigma_phi^2/2)` suppression lands on the coherent term). Check
      `docs/computation/precision-and-tolerances.md` for tolerance statements
      that assumed a deterministic loss, and correct or explicitly re-affirm
      each.
- [ ] K — Electron transport physics docs. Rewrite
      `docs/physics/beam-transport/electron-transport.md` for a stochastic loss:
      the "Model" and "Energy-controlled propagation" sections (the loss per
      flight is now a random variable of the stated mean), "Physical flights and
      numerical substeps" (substep invariance is distributional, not algebraic —
      carry E's re-derivation), the redefined cutoff crossing and its
      `n_cutoff_stopped` bookkeeping, and line 125's standing claim that
      straggling is unmodelled. State the source equation, assumptions, limiting
      case, and `Validation: energy-loss-straggling` marker per the physics-doc
      contract. Cross-link J so the RNG plumbing is documented once, not twice.

## Decisions and open questions

- **Closed by A — sequencing. Gate outcome: PROCEED.** Measured, not asserted.
  The systematic stopping error *did* dominate (24.6 rad against 0.3--13.4 rad
  at 25 keV / 1 um / 1 keV photon), so the gate condition fired and this task
  waits on `feature/relativistic-bethe-stopping` — but that task landed on
  `main` before slice A ran (`2c51754`, `dcdb1cd`, `d15a8ef`; checklist A--G
  complete). The dependency is satisfied, so nothing is blocked and slice B may
  be dispatched. The prior entries in this section that read "sequence behind
  `feature/relativistic-bethe-stopping`" and "this waits on" are superseded by
  this line; they were written before that branch merged.
- **Recorded by A — the motivating figure was an underestimate.** The `~0.3 rad`
  Jensen bias in `energy-step-convergence.md` corresponds to a `sigma_E` of
  ~0.3 keV over 1 um at 25 keV in graphite, which is a *soft-collision-only*
  spread. The Moller-cutoff variance the transport actually discards is
  `sigma_E` = 1.54 keV, and the Jensen bias 2.6 rad (soft) to 13.4 rad (full
  cutoff, an upper bound because the second-moment expansion is not valid on a
  heavy-tailed loss distribution). Slice H's acceptance target should be stated
  against this range, not against `~0.3 rad`.
- **Open, and NOT owned here — residual mean-stopping accuracy.** On the
  corrected clock a residual stopping error of 1% still costs 3.5 rad at the
  same operating point, the same order as the straggling bias. That residual
  belongs to `feature/reference-electron-stopping-data` (checklist entirely
  open). It does not re-block this task, because straggling owns a channel —
  the `exp(-sigma_phi^2/2)` suppression of the coherent line — that no
  mean-stopping work can supply. But slice H must not claim to have closed the
  phase budget while that term is outstanding.
- **Contradicted by A — the "whole trajectory is closer to many-collision"
  premise is false.** The scope section above argues that because coherent phase
  accumulates over the whole trajectory, "the many-collision limit is closer to
  applicable at the level that matters". Measured: `kappa` integrated over the
  *entire* CSDA range is 0.080--0.16 for every catalog material at every energy
  from 10 to 300 keV — flat, and three orders of magnitude below the `kappa >~ 10`
  Gaussian threshold. Integrating over the trajectory moves the process from
  sub-Landau to Vavilov, never to Gaussian. Slice B must select for the Vavilov/
  few-collision regime end to end and may not fall back on a whole-trajectory
  central-limit argument.
- **Contradicted by A — `energy-step-convergence.md`'s tungsten event count.**
  That page's "about 0.11 [inelastic events per flight] in tungsten at 25 keV"
  does not reproduce from the repository's own stopping power and mean excitation
  energy: `W(0.06305)` gives **0.03**, a factor of 3.7 low. The carbon figure
  reproduces exactly (measured 1.03 against the stated "about 1.0"), so the
  method is right and the tungsten number looks like an arithmetic slip. The
  page's conclusion — fewer than one inelastic event per flight in tungsten — is
  unaffected and in fact stronger. Fixing that sentence belongs to slice I, not
  here; slice A does not edit docs pages.
- **Open:** distribution choice for the few-collision regime (checklist B).
  Urban-style is the leading candidate; this is model selection against measured
  `kappa`, not an open-ended design question.
- **Corrected:** an earlier draft treated unrestricted-CSDA-plus-straggling as
  double-counting the Moller tail and therefore blocking. It is not — same mean,
  correct variance, self-consistent while delta rays are untransported. Slice B
  is correspondingly smaller than first scoped.
- **Open:** whether delta rays become transported particles. Currently a
  recorded non-goal ("No delta rays", `stopping-power.md`). Only the restricted-
  stopping branch of B forces the question; the unrestricted branch leaves the
  non-goal intact.
- **Open:** production reachability (G). Today `energy_model` is API-only, so
  straggling could land as a research capability with no `Numerics` field, or as
  a production toggle with checkpoint-identity consequences. Cheaper to decide
  before E than to retrofit after F.
- **Decided, and now satisfied:** sequence behind
  `feature/relativistic-bethe-stopping` rather than bounding the accepted range
  to ~<50 keV. That task is small, unblocked, and removes the constraint
  entirely; capping the range would leave the systematic error in place across
  most of the sweep. **It has since landed on `main`, so this is a discharged
  dependency, not a live one** — see the gate outcome at the top of this
  section.
- **Decided:** straggling defaults **off** and the off path is bit-for-bit
  identical to current transport on all four cores. Every existing ledger row,
  golden, and validation write-up must stay valid unchanged with the feature
  present.
- **Decided:** this task does not modify the mean stopping power. It consumes
  whatever model `feature/relativistic-bethe-stopping` (and later
  `feature/reference-electron-stopping-data`) establishes, rather than
  re-deriving one.

## Next slice

**Recommendation: dispatch B now.** Not "hold". The gate's own condition fired,
but the task it gated on is already merged, so the only thing that could have
held this task is gone. B is also the slice A most directly de-risks: it was
scoped as "model selection against measured `kappa`", and `kappa` is now
measured across the full catalog and energy sweep rather than at the single
carbon/25 keV/1 um point the scope section quotes.

B should be run knowing three things slice A established:

1. `kappa` per flight is 2e-5--3e-2 with **under one** close collision per
   flight. Landau is inadmissible per flight, not merely inaccurate.
2. `kappa` over the whole CSDA range is 0.080--0.16 and flat in `Z` and energy.
   Gaussian/Bohr is inadmissible everywhere too, and the whole-trajectory
   central-limit escape hatch in the scope section does not exist.
3. That leaves the few-collision family. Urban-style was already the leading
   candidate on prior reasoning; A converts that from a guess to the only
   option that covers the measured matrix, and B's real work is deriving and
   pinning the specific variant rather than re-litigating the family.

Slices J and K remain late doc slices as scoped; nothing in A changes their
dependency order, though K's rewrite of `electron-transport.md` should also
carry A's correction to the tungsten event count if slice I has not already.

## Delegation slices and required skills

- **A** — `lead-task` with `physics-review`, `monte-carlo`, and
  `repo-orientation`. This is the sequencing gate and should run before anything
  else in this task is dispatched; it may conclude the task waits.
- **B** — `physics-review` with `monte-carlo`. Model selection against A's
  measured `kappa`, not an open-ended design slice.
- **C** — `physics:implement` / `scientific-library` plus `monte-carlo`, then
  fresh-context `physics-validation`. `one-shot` once B names the distribution.
- **D--E** — `implement-task` with `monte-carlo` and `regression-testing`. The
  bit-for-bit and stream-independence tests in D are the gate for E.
- **F** — `implement-task` with `monte-carlo`, `performance`, and
  `remote-gpu-jobs`.
- **G** — `implement-task` with `cli-ui-ux` and `regen-golden`; blocked on the
  reachability decision above.
- **H** — `lead-task` with `remote-gpu-jobs` and fresh-context
  `physics-validation`.
- **I** — `documentation-maintenance` and `physics-review`; `one-shot` once H
  reports numbers.
- **J** — `documentation-maintenance`; depends on D (stream design settled) and
  H (numbers). `one-shot`.
- **K** — `documentation-maintenance` with `physics-review`; depends on B, E,
  and H. `one-shot` once H reports numbers. J and K touch disjoint files and may
  run in parallel, but both land after I so the ledger row exists to cite.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --numba
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
```

- The sequencing check in A is computed, not asserted: the systematic phase
  error from the stopping bias is compared against the straggling estimate at
  the same operating point, and the distribution choice in B is justified
  against measured `kappa`.
- Straggling off reproduces current transport bit-for-bit on lockstep, grooved,
  per-electron, and CUDA cores, pinned by test.
- Straggling on consumes only its own counter-addressed stream: free-path and
  scattering-angle draws are provably unperturbed, and the host and CUDA cores
  agree on the same `(seed, electron, flight)` up to libm ulp.
- The sampler's mean and variance match the analytic values, and it degenerates
  to the deterministic loss in its zero-width limit.
- Cutoff crossing and `max_dE_frac` substepping have documented, tested
  semantics under a random loss; `substep-radiation-invariance` is re-derived
  rather than assumed.
- Any core without an implementation raises rather than silently returning
  unstraggled results.
- The effect on backscatter/transmission, range straggling, bremsstrahlung
  shape, and the coherent-line phase is measured and reported, including the
  residual on the ~0.3 rad Jensen bias.
- `Validation: energy-loss-straggling` exists with a ledger row and
  fresh-context validation; every write-up asserting straggling is unmodeled is
  updated or explicitly still true.
- `docs/computation/random-streams.md` documents the straggling stream through
  that page's own "Adding a new random input" procedure, including the
  off-path inertness claim; `statistical-methods.md` and
  `precision-and-tolerances.md` no longer assume a deterministic per-flight
  loss.
- `docs/physics/beam-transport/electron-transport.md` describes a stochastic
  loss end to end — model, energy-controlled propagation, substep semantics,
  cutoff crossing — carries the `Validation: energy-loss-straggling` marker, and
  retains no claim that straggling is unmodelled.
- `uv run pyrite-dev docs` passes and cross-references between the computation
  and physics pages resolve.
