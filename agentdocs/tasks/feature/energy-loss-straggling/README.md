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

### Why this is not a drop-in sampler

The obvious implementation — add a Gaussian Bohr fluctuation per flight — is
very likely wrong here, and the repository's own numbers say so. Two facts must
be reconciled before any code:

1. **The per-flight regime is not Gaussian.** `energy-step-convergence` measures
   `kappa = 0.015` for carbon at 25 keV over **1 um**, already strongly skewed
   and outside the Gaussian (Bohr) limit. Flights are elastic-mean-free-path
   scale, and the same write-up records of order **one** inelastic event per
   flight in carbon at 25 keV (~0.11 in tungsten). Per flight the process is
   therefore in the single-collision limit, where neither Bohr nor even Vavilov
   is the right distribution and the "many events per step" premise behind every
   condensed-history straggling sampler fails.
2. **Unrestricted CSDA plus full straggling double-counts.** Joy--Luo is an
   *unrestricted* stopping power: it already contains the mean of the hard
   Moller `1/T^2` tail. Sampling the full Landau distribution on top of it
   restores the tail's fluctuation while the tail's mean is applied twice. The
   standard resolution (EGSnrc, Geant4, PENELOPE) is a *restricted* collisional
   stopping power below a production threshold plus explicit hard inelastic
   events above it — which is a larger change than "add straggling", and touches
   the recorded non-goal that delta rays are not transported particles.

Slices A and B exist to close exactly this, before anything is implemented.

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

- [ ] A — Regime audit. Instrument existing transport runs to measure the
      distribution of per-flight path length, per-flight inelastic event count,
      and the Vavilov `kappa` per flight and per CSDA range, across
      representative low-/high-Z catalog materials over 1--300 keV. Produce the
      table that decides which straggling regime — if any — a per-flight
      sampler may legitimately assume.
- [ ] B — Model-form decision (**blocking**). Choose between (i) restricted
      collisional stopping power plus explicit hard inelastic events above a
      production threshold, and (ii) unrestricted CSDA plus a straggling
      distribution, with the double-counting argument written out and the choice
      justified against A's measured regime. Decide in the same slice whether
      delta rays become transported particles or remain a recorded non-goal with
      their energy deposited locally, and whether straggling is accumulated over
      a path longer than one flight if the per-flight regime forbids sampling.
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

- **Open/blocking:** model form (checklist B). Restricted stopping power plus
  explicit hard inelastic events, versus unrestricted CSDA plus a straggling
  distribution. Nothing downstream of B is dispatchable until this is decided,
  because it determines whether this task adds a sampler or adds an interaction
  channel.
- **Open:** whether the per-flight regime measured in A permits per-flight
  sampling at all. If flights carry of order one inelastic event, the honest
  options are explicit discrete events or accumulating the fluctuation over a
  longer path — both change the shape of E and F substantially.
- **Open:** whether delta rays become transported particles. Currently a
  recorded non-goal ("No delta rays", `stopping-power.md`); option (i) in B
  makes their production explicit even if their transport is not.
- **Open:** production reachability (G). Today `energy_model` is API-only, so
  straggling could land as a research capability with no `Numerics` field, or as
  a production toggle with checkpoint-identity consequences. Cheaper to decide
  before E than to retrofit after F.
- **Open:** energy ceiling for the acceptance claim. Joy--Luo under-stops by
  roughly a factor of two at 300 keV
  (`stopping-power.md`, table `tbl-stopping-validity-ceiling`), so sampling a
  fluctuation about a mean that is itself 50% low is not defensible at the top
  of the range. Either bound the accepted range to where the mean is trustworthy
  (~<50 keV), or sequence behind `feature/reference-electron-stopping-data`.
- **Decided:** straggling defaults **off** and the off path is bit-for-bit
  identical to current transport on all four cores. Every existing ledger row,
  golden, and validation write-up must stay valid unchanged with the feature
  present.
- **Decided:** this task does not modify the mean stopping power. If
  `feature/reference-electron-stopping-data` lands first, C and E consume its
  model rather than re-deriving one; if this lands first, that task inherits the
  fluctuation hook.

## Delegation slices and required skills

- **A--B** — `lead-task` with `physics-review`, `monte-carlo`, and
  `repo-orientation`. **Not `one-shot`**: B is the blocking model-form decision
  and A is the evidence it rests on.
- **C** — `physics:implement` / `scientific-library` plus `monte-carlo`, then
  fresh-context `physics-validation`. `one-shot` only after B closes.
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

- The model-form decision is written down with its double-counting argument and
  its supporting regime measurements, not asserted.
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
