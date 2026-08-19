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

- [ ] A — Regime audit **and priority check**. Measure the distribution of
      per-flight path length, per-flight inelastic event count, and the Vavilov
      `kappa` per flight and per CSDA range, across low-/high-Z catalog
      materials over 1--300 keV. In the same slice, compute the systematic phase
      error implied by the Joy--Luo stopping bias at the 25 keV / 1 keV point
      and compare it against the ~0.3 rad straggling estimate. If the systematic
      term dominates, this task waits on
      `feature/relativistic-bethe-stopping` and its acceptance numbers are
      re-measured against the corrected clock.
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

## Decisions and open questions

- **Open/blocking:** sequencing (checklist A). The first-order systematic
  stopping error very likely dominates the second-order straggling bias at the
  operating point that motivates this task. If A confirms that, this waits on
  `feature/relativistic-bethe-stopping`. This is the gate, not the model form.
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
- **Decided:** sequence behind `feature/relativistic-bethe-stopping` rather
  than bounding the accepted range to ~<50 keV. That task is small, unblocked,
  and removes the constraint entirely; capping the range would leave the
  systematic error in place across most of the sweep.
- **Decided:** straggling defaults **off** and the off path is bit-for-bit
  identical to current transport on all four cores. Every existing ledger row,
  golden, and validation write-up must stay valid unchanged with the feature
  present.
- **Decided:** this task does not modify the mean stopping power. It consumes
  whatever model `feature/relativistic-bethe-stopping` (and later
  `feature/reference-electron-stopping-data`) establishes, rather than
  re-deriving one.

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
