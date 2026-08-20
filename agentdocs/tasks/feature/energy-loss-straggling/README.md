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
- [x] B — Distribution selection. **Done. Selected: the Geant4 Urban
      energy-loss fluctuation model, unrestricted (`T_up = T_max = E/2`),
      applied per element.** Instrument: `slice_a_regime_audit.py --part b0`
      (the stale-figure audit) and `--part urban` (the parameterisation and its
      validity boundaries), both extending slice A's harness rather than
      duplicating it.

      **Source.** Geant4 Physics Reference Manual, "Energy loss fluctuations",
      Urban model (`G4UniversalFluctuation`), after Bichsel, *Rev. Mod. Phys.*
      **60**, 663 (1988).

      **Parameterisation**, per element so it matches the Bragg-additive form
      the spliced mean stopping power already uses:

      | symbol | value |
      |---|---|
      | ionisation level `E_0` | 10 eV |
      | ionisation ceiling `T_up` | `T_max = E/2` (Moller; unrestricted) |
      | K-shell level `E_2` | `10 Z^2` eV |
      | K-shell strength `f_2` | `2/Z` (`Z >= 2`), `f_1 = 1 - f_2` |
      | loose level `E_1` | from `f_1 ln E_1 + f_2 ln E_2 = ln I`, `I` = transport's own `J_keV` |
      | normalisation `C` | **`|dE/dx|` itself** |
      | rate parameter `r` | 0.55 |

      `Sigma_i = C (f_i/E_i) [ln(2 mc^2 (beta gamma)^2 / E_i) - beta^2] /
      [ln(2 mc^2 (beta gamma)^2 / I) - beta^2] (1 - r)` for `i = 1, 2`;
      `Sigma_3 = C r (T_up - E_0) / (E_0 T_up ln(T_up/E_0))`;
      `dE = n_1 E_1 + n_2 E_2 + sum_k E_k` with `n_i ~ Poisson(s Sigma_i)` and
      `E_k = E_0 / (1 - u (T_up - E_0)/T_up)`, `u` uniform.

      **Why this and not the alternatives, against A's measured numbers.**

      1. *Landau is inadmissible per flight.* A measured 0.003--0.16 close
         (`eps^-2`) collisions per flight. The sharper form of the same
         statement is `xi/I`, which Landau requires to be `>> 1`: measured
         **0.087** for carbon at 25 keV (`xi` = 6.79 eV against `I` = 78 eV) and
         **0.004** for tungsten (2.9 eV against 727 eV).
      2. *Gaussian/Bohr is inadmissible everywhere.* `kappa` per flight
         2e-5--2.9e-2; `kappa` over the entire CSDA range 0.080--0.16. Never
         near the `kappa >~ 10` threshold, per flight or integrated.
      3. *Vavilov is inadmissible per flight* for the same reason, and
         separately because it assumes the free-electron `eps^-2` spectrum down
         to `eps -> 0`, which is exactly where atomic binding dominates once
         `xi <~ I`.
      4. **The decisive property is `C = dE/dx`.** Urban is the only candidate
         whose mean is *normalised to* the supplied stopping power rather than
         derived from its own `xi`. Measured: the closure ratio
         (model mean loss)/(`S s`) is **1.0000** in every cell where the
         parameterisation is intact. That is what makes it compatible with the
         *spliced* Joy--Luo / Berger--Seltzer mean — including the empirical
         Joy--Luo branch below each element's crossover, where no
         Bethe-consistent `xi` exists at all. Landau, Vavilov and Bohr each
         double-specify the mean and would disagree with the splice.
      5. *It resolves the regime rather than approximating it.* Measured Poisson
         means per flight: `n_1` = 0.015--1.80, `n_2` <= 0.070,
         `n_3` = 0.069--1.15 — under ~2.5 discrete events per flight and
         typically well under one, which is exactly the countable-collision
         picture A measured, sampled as such instead of replaced by a limit law.
      6. *Variance closes without tuning.* Urban's compound-Poisson variance
         `sum_i <n_i> <E^2>_i` (with `<E^2>_3 = E_0 T_up` exactly) against the
         analytic Moller second moment `xi T_max`: ratio **0.73--1.42** in
         variance (`sigma_E` within -14%/+19%) over the whole matrix, and within
         5% for carbon above 5 keV. High-`Z` runs ~26% low in variance, low
         energy runs high.
      7. *Infinite divisibility comes free.* A compound Poisson sum is
         infinitely divisible, so subdividing a flight at frozen energy is
         **exactly** distribution-preserving. That is the strongest available
         form of the `max_dE_frac` substep invariance E and K must re-derive;
         the only residual is the energy dependence of `Sigma_i`.

      **Domain of validity, measured, with two boundaries Geant4 never hits.**

      - `E_2 = 10 Z^2` eV sits below the Moller ceiling `T_max = E/2` only for
        `E > 20 Z^2` eV: C above **0.72 keV**, Si above **3.92 keV**, S above
        **5.12 keV**, W only above **109.5 keV**. So the K-shell channel is an
        unphysical level over most of the 1--300 keV sweep in high `Z`. Measured
        cost of dropping it: the `E_2` channel carries **0.0013--0.0047** of
        tungsten's mean loss.
      - Below `(beta gamma)^2 = E_2 / 2 mc^2` the `E_2` logarithm goes
        non-positive. A naive clamp to zero makes the mean **overshoot** —
        measured closure 1.0187 / 1.0098 / 1.0038 / 1.0010 for W at 1 / 2 / 5 /
        10 keV — because the clamp deletes a negative contribution.
        **Prescription for C:** whenever `E_2` is inadmissible on either
        boundary, re-solve the sum rules with `f_1 = 1`, `E_1 = I`. That
        restores closure to exactly 1.000 and preserves the mean bit-for-bit.
      - The PRM's own shape floor ("reliable distributions require the mean loss
        to be at least a few multiples of `I_exc`") is **violated in nearly
        every cell**: measured `dE/I` per flight is 0.017 (W in ws2 at 1 keV) to
        2.58 (C at 300 keV). Recorded honestly rather than argued away: below
        that floor the loss is a countable number of discrete events, so what is
        at stake is the two-level-plus-continuum *atomic parameterisation*, not
        a smooth spectral shape. The mean stays exact regardless and the
        variance is measured good to +-20%, so C must pin the first two moments
        against `xi` and `xi T_max` rather than trust the shape.

      **Unrestricted, not restricted.** `T_up = T_max = E/2`, no delta-ray
      production cut. A restricted sampler would require a *restricted*
      `dE/dx`, i.e. modifying the mean stopping power, which this task
      explicitly does not own. Both branches of the current splice are
      unrestricted, so unrestricted Urban is the self-consistent choice.
      **The "No delta rays" non-goal in `stopping-power.md` stays intact and
      unchanged**: replacing a deterministic `<dE/ds> s` with a random variable
      of the same mean creates no secondary particles. Slice I rewrites the
      "No straggling" bullet only; it must leave the delta-ray bullet alone.
      Consequence to carry into C and E: a single flight can now sample a loss
      up to `E/2`, so the cutoff-crossing solve (`eq-stopping-cutoff-distance`)
      and the `n_cutoff_stopped` bookkeeping must handle a flight that jumps
      straight past `E_cut`, and with `n_3` up to 1.15 per flight that is not a
      rare corner.
- [x] C — Sampler derivation and unit test. **Done. Implemented as a standalone,
      testable unit; nothing is wired into transport (D--F).** Owners:
      `montecarlo/transport.py` (`_urban_levels_scalar`,
      `_urban_channels_scalar`, `_urban_moments_element_scalar`,
      `_urban_poisson_scalar`, `_urban_ionisation_keV`,
      `_urban_sample_element_keV`, `_dEds_spliced_element_scalar`,
      `_urban_sample_compound_keV`, `urban_element_table`,
      `urban_loss_moments_keV`) and
      `tests/montecarlo/test_energy_loss_straggling.py` (134 cases).

      **Source.** Geant4 PRM, "Energy loss fluctuations", Urban model
      (`G4UniversalFluctuation`), after Bichsel, *Rev. Mod. Phys.* **60**, 663
      (1988). Parameterisation exactly as selected in B; `T_up = T_max = E/2`,
      per element, `C = |dE/dx|`.

      **Derivation, checked.** Units: `Sigma_i` is `C [keV/Ang]` over an energy
      `[keV]`, so `<n_i> = s Sigma_i` is dimensionless and `dE` is keV; the
      cores' `dE/dx` is negative and the sampler returns a positive loss, matching
      `E -= |dEds| * s`. Mean, from the two sum rules `f_1 + f_2 = 1` and
      `f_1 ln E_1 + f_2 ln E_2 = ln I`:

          sum_{i=1,2} Sigma_i E_i = C (1-r)/L_I [f_1 L(E_1) + f_2 L(E_2)]
                                  = C (1-r)/L_I [ln(2 mc^2 (bg)^2) - beta^2 - ln I]
                                  = C (1-r),
          Sigma_3 <E>_3           = C r,   <E>_3 = E_0 T_up ln(T_up/E_0)/(T_up-E_0),

      so `<dE> = C s` **identically**, for any `r`, `Z`, `I`. Variance, compound
      Poisson with no cross terms: `Var = s (Sigma_1 E_1^2 + Sigma_2 E_2^2 +
      Sigma_3 E_0 T_up)`, using `<E^2>_3 = E_0 T_up` exactly on the `1/E^2`
      spectrum. Continuum draw is that spectrum's exact inverse CDF, verified
      against `F(E)` on a 257-point grid: `u = 0 -> E_0`, `u -> 1 -> T_up`,
      monotone. **Limiting case:** every `<n_i>` is linear in `s`, so as `s -> 0`
      both moments vanish linearly and `P(dE = 0) -> 1` — tested at `s = 1e-12`
      Ang, where all 500 draws are identically 0.0, and the analytic mean equals
      `C s` at every `s`. The sampler degenerates to the present deterministic
      loss.

      **Closure, measured.** `(model mean)/(S s)` over the 9 catalog materials
      plus bare `C(0.1136)` / `W(0.06305)` at 1--300 keV: worst deviation from 1
      is **4.4e-16**, i.e. one ulp. Not merely 1.0000 — bit-for-bit in float64
      against the cores' own `_dEds_spliced_compound_scalar`. This is asserted by
      test per element (`rel=1e-14`) and per compound against the splice
      (`rel=1e-13`), plus a test that the new per-element split sums back to the
      compound law (which is why the split had to be a deliberate duplicate of
      that function's body rather than a factoring of it — re-associating the
      Joy--Luo/Berger--Seltzer sums would move the last bits of every existing
      transport result).

      **`E_2` re-solve, implemented and tested at the boundaries.** The channel
      switches off exactly at `E = 20 Z^2` eV (tested at +-0.1% of the boundary
      for C/Si/S/W: 0.72 / 3.92 / 5.12 / 109.5 keV) and wherever `L(E_2) <= 0`;
      the surviving level is then `f_1 = 1, E_1 = I`. B's clamp figures are
      reproduced verbatim by a reference clamp written into the test file —
      **1.0187 / 1.0098 / 1.0038 / 1.0010** for W at 1 / 2 / 5 / 10 keV — and the
      re-solve returns closure to 1 at `rel=1e-14`, including a 41-point scan
      straight through W's 109.5 keV boundary with no step in the mean.

      **Variance deficit, accepted and recorded, and it MOVED.** With the
      re-solve in place the ratio to the analytic Moller `xi T_max` over the same
      matrix is **0.70 (ptbi2, 100 keV) to 1.55 (ptbi2, 1 keV)** — `sigma_E`
      within **-16%/+24%** — against B's **0.73--1.42** measured with the naive
      clamp. The widening is a consequence of the re-solve, not a regression: the
      clamp leaves the surviving level at the sum-rule `E_1` (0.645 keV for W),
      the re-solve raises it to `I` (0.727 keV), and `Var` carries `E_1^2` while
      the mean carries `f_1 L(E_1)/L_I` and is invariant. B's number should be
      read as the clamped variant's. The PRM's width correction is **not**
      applied; the test pins the band `0.70 <= ratio <= 1.55` so any future
      change to it is visible. Sampled first two moments close on the analytic
      ones within 3% (mean) and 20% (variance) at 30k draws.

      **Three decisions C made that D--F inherit.**

      1. *Poisson variate advances the stream by an amount the caller can
         predict.* Inverse CDF by the recurrence `p_{k+1} = p_k lam/(k+1)`
         consumes exactly **one** uniform however large `n` comes out; Knuth's
         product method would consume `n + 1` and make the counter layout
         data-dependent. Above `lam = 100` it hands to the Gaussian limit
         (skewness <= 0.1) at **two** uniforms via Box--Muller. A channel with
         `lam = 0` consumes none. So a flight consumes `(#non-empty Poisson
         channels) + n_3` draws per element — variable, but a pure function of
         the inputs.
      2. *Inadmissible cells fall back to the deterministic loss and consume no
         stream.* `T_up <= E_0` (E below 20 eV) or `L(I) <= 0` (below ~0.18 keV
         in tungsten) leaves no admissible parameterisation at all; the sampler
         returns `C s` there rather than sampling a degenerate distribution. The
         mean stays exact and the fallback is silent in every moment test.
      3. *The sampler does not clamp `dE` to `E`.* `E_1 = I` can exceed
         `T_up = E/2` at low energy after the re-solve (W below 1.45 keV), and
         `n_3` can exceed 1, so a single flight can return a loss above the
         electron's kinetic energy. Clamping would break the mean, which is the
         one property the model was selected for. **E owns this**: the
         cutoff-crossing solve and `n_cutoff_stopped` must handle it, as B
         already flagged.

      **No ledger row, deliberately.** `Validation: energy-loss-straggling` would
      assert that PyRITE models straggling; after C nothing does — the sampler is
      unreachable from every transport core and from `simulate_trajectories`, so
      a row would describe behaviour no run has. The row belongs with the first
      slice that makes it reachable (E/F) and lands in
      `ledger-transport-background.md` under I. No `Validation:` marker was added
      to the code either, so there is no orphan for the ledger auditor to find.
      The derivation itself lives next to the code (a ~60-line block comment in
      `transport.py` carrying source, sum rules, both moments, units, signs,
      limiting case and the accepted limits) and in the test module docstring;
      `docs/` is untouched, per I/J/K.

      **Recommendation for D.** The sampler already takes `(key, counter)` and
      returns the advanced counter, so D's remaining choice is how to *address*
      it. Do **not** share the electron's existing counter: the per-flight draw
      count is variable (see decision 1), so a shared counter makes the free-path
      and scattering-angle draws depend on whether straggling is on, which
      violates D's own second requirement. Recommend a **separate key domain per
      flight**: `straggle_key = _splitmix64(stream_key ^ CONST)` once per
      electron, then `flight_key = _splitmix64(straggle_key + GOLDEN * (flight,
      substep index))`, drawing from counter 0 within it. That gives unbounded
      per-flight consumption with no collisions and no stride to bound, keeps the
      straggling-off path bit-for-bit by construction (the existing stream is
      never touched), and reuses `_splitmix64` / `_stream_uniform_scalar`
      unchanged so host and CUDA address the same streams. The host/CUDA parity
      claim should be stated as few-ulp, not bit-for-bit: `_urban_poisson_scalar`
      branches on a `log`/`exp` comparison, so a last-bit difference in `lam` can
      move `n` by one at a CDF boundary.
- [x] D — RNG plumbing. **Done on all five host cores; CUDA key derivation
      duplicated but the sampler itself is unverified (no cupy/GPU on the
      dispatching machine).** Owners: `montecarlo/transport.py`
      (`_urban_stream_key_scalar`, `_urban_flight_key_scalar`, the five
      `_transport_core_*` cores, `_run_per_electron_transport{,_lut}`,
      `simulate_trajectories`'s new `straggling=False` kwarg),
      `montecarlo/transport_jit_kernel.py` (`_urban_stream_key`,
      `_urban_flight_key`, `_urban_ionisation`, `_dEds_spliced_element`,
      `_transport_kernel`/`run_transport_kernel`), and
      `tests/montecarlo/test_straggling_rng_plumbing.py` (17 cases).

      **Key scheme, exactly as C recommended and confirming its salt
      constant.** `urban_key = _urban_stream_key_scalar(stream_key)` once per
      electron (`stream_key = stream_keys(seed, Ne)[e]`, the SAME array every
      core already uses or now receives), then
      `flight_key = _urban_flight_key_scalar(urban_key, flight, substep)`
      once per row, sampling from `(flight_key, counter=0)`. `flight`/
      `substep` are each core's own internal `flight_of[e]`/`substep_of[e]`
      (or per-electron `flight_id`/`substep_id`), tracked unconditionally
      regardless of `energy_model`, so no `energy_model="midpoint"` request is
      needed to address the stream correctly. The sampled loss accumulates
      into a new diagnostic `result["straggle_dE_keV"]` (Ne-sized, only
      present when `straggling=True`) and is **not** applied to
      `E_keV`/`E_end_keV` anywhere — integration is E/F's job, untouched here.

      **Host: all five cores.** `_transport_core_ungrooved`,
      `_transport_core_ungrooved_lut`, `_transport_core_grooved`,
      `_transport_core_ungrooved_perelectron`,
      `_transport_core_ungrooved_perelectron_lut`. The two LUT cores gained
      new `L_Js`/`L_Zs`/`L_ks`/`L_coeffs`/`L_E_cross` parameters (used only
      for straggling): the LUT bakes one interpolated `dE/ds` per layer and
      carries no per-element split, so straggling always evaluates the exact
      per-element stopping power via `_dEds_spliced_element_scalar` regardless
      of which core calls it — the LUT's own approximation error is
      irrelevant to the sampler's mean, only to the (unrelated) flight length
      that transport under that core produces.

      **Tests, one per "Done when" property, on every host core (including
      grooved) via `simulate_trajectories`, not core-internals:**

      1. `straggling=False` is bit-for-bit identical to a call that never
         passes the keyword — the default is a provable no-op.
      2. `straggling=True` never changes any pre-existing returned array
         (every field but the new diagnostic checked); `straggle_dE_keV` is
         finite, non-negative, and nonzero on at least one electron at 25 keV
         graphite (not a vacuous pass).
      3. `straggle_dE_keV` is reproduced bit-for-bit **offline**, on every
         core, from nothing but the run's own recorded `(E_start, L_ang)` per
         segment and each segment's position within its electron's own
         segment list (the flight index; substep is always 0 at the default
         `max_dE_frac=0`) — proving the core actually addressed the sampler
         with the documented key, not merely that the shared function is a
         pure function of its inputs.

      **CUDA.** Duplicated `_urban_stream_key`/`_urban_flight_key` (trivial,
      hand-verified integer ops) and the full sampler (channels, three
      Poisson draws, continuum quanta) inlined into `_transport_kernel`'s
      per-row body — device functions in this file return one value, so the
      multi-output channel computation and the mutable draw counter could not
      be factored out, matching the file's own established idiom (see the
      prism-exit inlining comment it already carries).
      `run_transport_kernel`'s new trailing `straggle_on`/`stragg_dE`
      parameters match `_transport_core_ungrooved_perelectron`'s positionally,
      which `test_launcher_signature_tracks_the_reference_core` enforces.
      **`_transport_lut_kernel` was NOT wired** (out of budget; no test
      enforces its parity); `simulate_trajectories` raises
      `NotImplementedError` for `straggling=True` + `transport_core="cuda"`
      with the LUT enabled (the default) rather than risk a silent
      unstraggled result or an opaque `TypeError`.

      **This machine has no cupy and no CUDA device** — `transport_jit_kernel`
      does not even import. The CUDA sampler code was cross-checked by hand
      against the tested host implementation and could not be compiled or
      executed. Slice D's own "up to libm ulp" CUDA claim is therefore
      asserted by construction (identical formulas, identical draw order,
      identical key scheme) but **not empirically verified**; whoever picks up
      F should verify it on hardware (`pyrite remote`) before relying on it,
      and should decide then whether to also wire `_transport_lut_kernel` or
      keep the LUT path CUDA-only-when-disabled.

      **Acceptance run:** `pyrite-dev test-suite core` (1761 passed),
      `pyrite-dev test --numba tests/montecarlo/` (681 passed), `lint`,
      `typecheck` — all clean. No existing test was weakened;
      `tests/montecarlo/test_groove.py`'s `_run_grooved` helper was updated
      for the grooved core's three new trailing parameters (mechanical, no
      behavior change since it passes `straggle_on=False`).
- [x] E — CPU integration on the ungrooved lockstep core, including the
      redefined cutoff crossing and the `max_dE_frac` substep interaction.
      **Done on `_transport_core_ungrooved` only** (the exact, non-LUT lockstep
      core); every other core stays diagnostic-only until F. Owners:
      `montecarlo/transport.py` (`_transport_core_ungrooved` and a ~215-line
      derivation block above it; `simulate_trajectories`'s `straggling`
      docstring) and `tests/montecarlo/test_straggling_transport_integration.py`
      (9 cases), plus a narrowing of slice D's property-2 test.

      **The loss is applied.** When `straggle_on`, `E_end = E_start - dE` with
      `dE` the slice-C draw over the row's own length, replacing
      `E_end = E + dEds*step` (frozen) and the midpoint predictor-corrector
      alike. `energy_model` still selects the clock's representative energy
      (`beta` at the realized `(E_start+E_end)/2` versus at `E_start`) and the
      `seg_E_end`/`seg_t_end` schema, but no longer the energy update: the
      corrector is a quadrature of a deterministic ODE that a random loss
      replaces, and the sampler's own mean is the left-endpoint `C(E_start) s`.
      The residual that leaves in the mean is exactly the O(s^2) term derived
      below, and `max_dE_frac` is the lever that controls it.

      **Cutoff crossing, redefined — and the indicator half is EXACT.** The
      Urban loss is a compound-Poisson *subordinator*, hence non-decreasing in
      the step length, so

          first passage lies inside the row  <=>  dE(row) >= E_start - E_cut,

      a function of the total loss alone, which is exactly what the sampler
      returns. **`n_cutoff_stopped` therefore carries no approximation**; it
      fires on precisely the rows whose true first passage is interior. Only
      the crossing *location* needs a rule, and the rule is the fluid
      interpolation at the row's own realized rate,
      `s_cut = s (E_start - E_cut) / dE`, with `E_end = E_cut` exactly.
      It reduces **algebraically**, term by term, to the deterministic
      `cutoff_distance = (E_cut - E_j)/dEds` in the zero-fluctuation limit; it
      is the same fluid picture the surrounding transport already uses for the
      clock and `seg_mid` (it changes *which* number is spread along the
      flight, not the spreading); it handles C's decision-3 overshoot with no
      special case (`dE >> Delta` sends `s_cut -> 0`, never a negative energy);
      and it consumes no extra random numbers, so D's stream layout and
      off-path bit-for-bit claim survive untouched. Cost, recorded honestly:
      the location is biased inside the crossing row — the true first passage
      is the crossing jump's position, uniform on `[0, s]` given one jump — so
      a large overshoot stops the track earlier than the truth, bounded by one
      flight length on the terminating row only.

      **Rejected, for the record.** (a) *Travel the full row, then stop* — does
      not degenerate to the deterministic solve at all and lengthens every
      terminated track by half a flight even with the fluctuation off.
      (b) *Uniform crossing position* — exact for a single-jump crossing but
      wrong for a multi-jump one, wrong in the deterministic limit, and
      outcome-dependent stream consumption. (c) *Clamp `dE` to `Delta`* — breaks
      `<dE> = C s`, the one property B selected Urban for. (d) *Sample the jump
      ladder for the exact first passage* — correct, but needs per-element
      counts and `n_i` extra position variates out of slice C's sampler, which E
      does not own. **(d) is the only one that is strictly better physics, and
      it is the recommended upgrade if range straggling near `E_cut` ever
      becomes load-bearing.**

      **Ordering.** With straggling on the `max_dE_frac` cap is applied
      **before** the sample and the cutoff test **after** it — the reverse of
      the deterministic order, which can afford to solve the cutoff first
      because the loss is a known function of distance. The step control itself
      stays deterministic (`max_dE_frac * E / |dEds|`, the mean rate): a
      substep grid chosen from the realized loss would be a *random* partition,
      the partition and the increments would be dependent, and the
      infinite-divisibility argument below — which holds for any *fixed*
      partition — would not apply.

      **`substep-radiation-invariance` re-derived for the straggled path**
      (in-code, for K to transcribe; `docs/` untouched). The invariance is
      distributional, not algebraic, and comes in two exact halves.

      1. *At frozen energy it is EXACT, not asymptotic.* `sum_m CP(s_m Sigma)
         =_d CP(s Sigma)` for **any** partition and any `N`, since
         `sum_m Poisson(s_m Sigma_i) = Poisson(s Sigma_i)` with i.i.d. marks.
         Distributional, not pathwise: each substep addresses its own
         `(flight, substep)` key, so the realized numbers differ.
      2. *The only substep dependence is the drift of `Sigma_i` with `E` inside
         the flight.* With `C(E) = |dE/dx|` and `V(E) = Sigma_1 E_1^2 +
         Sigma_2 E_2^2 + Sigma_3 E_0 T_up` per unit length,
         `<sum_m X_m> = sum_m s_m C(E^(m))` and likewise for `V`, so expanding
         about `E`,

             <sum_m X_m> - <X> = -C C' sum_m s_m sigma_{m-1} + O(s^3)
                               = -C C' s^2 (N-1)/(2N) + O(s^3),

         and identically `-V' C s^2 (N-1)/(2N)` for the variance. Positive,
         since `C' < 0`: substeps at the lower energies earlier substeps left
         behind lose more.
      3. *Its `N -> infinity` limit is the CORRECT moment.* The exactly
         integrated mean is `int_0^s C(E(s')) ds' = C s - (1/2) C C' s^2 +
         O(s^3)`, whose second-order term is the `N -> infinity` limit above.
         **So the entire substep dependence of the straggled loss is the
         pre-existing left-endpoint quadrature error of the frozen energy
         model; straggling introduces none of its own**, and refining
         `max_dE_frac` removes it at first order exactly as it does
         deterministically.
      4. *Bound, leading order:* `|Delta<dE>|/<dE> ~ (1/2)|dlnC/dlnE|
         (DeltaE/E)` and the same with `V` for the variance, `DeltaE = C s` the
         flight's own mean loss. An estimate, not a hard inequality — the
         dropped `O(s^3)` remainder is itself of relative size `DeltaE/E`.

      **Verified statistically, not asserted.** Graphite, 25 keV, `s = 1e4` Ang
      (`DeltaE/E = 0.09`), 20000 repetitions of a harness reproducing the
      core's substep recursion: frozen `N = 1` vs `N = 32` shift
      `-0.0014 +- 0.0150` keV on a mean of 2.243 keV (consistent with the exact
      invariance); drifting `N = 32` shift `+0.086 +- 0.015` keV against the
      leading-order prediction `+0.075`, ratio **1.15**, rising to 1.30 at
      `s = 1.5e4` (`DeltaE/E = 0.135`) as the `O(s^3)` remainder predicts. In
      transport, 600 electrons with `max_dE_frac` 0 vs 0.02: mean per-electron
      straggled loss 19.61 vs 19.70 keV, **0.5%**.

      **Tests, one per "Done when" property.** Off path: a cutoff- and
      substep-heavy run (>100 crossings; >2 rows per crossing at
      `max_dE_frac=0.02`) across frozen/midpoint/substepped, pinning
      `straggling=False` bit-for-bit against a call that never mentions the
      keyword, so both new orderings are covered as unreachable. On path:
      2000 electrons at 25 keV graphite — energies never below `E_cut`, all
      four exit channels sum to `Ne`, and the summed sampled loss closes on the
      transport's own mean stopping power to **0.07%** over 5.6e5 rows; row by
      row `E_end == E_start - dE` **exactly** against the sampler re-run
      offline from the documented key. Overshoot: 6 keV against a 5 keV cutoff
      (1 keV available, single quanta to `T_max = 3` keV) — median overshoot
      **2x**, max **>2000x**, the crossing loss reaches the available energy on
      **every** track without exception (the empirical form of the exact
      indicator), and inverting the truncation recovers the untruncated length,
      which re-drawn reproduces that very loss. Mutation-checked: travelling the
      full row, halving the crossing fraction, and ignoring the draw each fail.

      **Slice D's property-2 test was narrowed, deliberately.** "Straggling on
      never changes any pre-existing array" is now false on the one core that
      applies the loss, so that test runs on the four diagnostic-only cores and
      is restated for the lockstep exact core as "every electron's **first row**
      is bit-for-bit identical" — the lockstep core draws every electron's first
      free path before any second one, at the unperturbed start energy, so that
      still proves the straggling draw never touches the shared `Generator`.
      Slice D's property-1 (off is bit-for-bit) and property-3 (offline
      reproduction from `(seed, electron, flight, substep)`) pass **unmodified**;
      property 3 gained an explicit `n_cutoff_stopped == 0` assertion, because a
      crossing row is truncated *after* the sampler drew over the untruncated
      length, so its `L_ang` is no longer the length sampled over.

      **Acceptance run:** `test-suite core` (1770 passed, 69 skipped — the
      pre-existing 1761 plus this slice's 9), `test --numba` (3574 passed,
      69 skipped, 6m35s), `lint`, `typecheck` — all clean. No existing test was
      weakened or deleted.

      **Recommendation for F.** The crossing logic is core-agnostic: it needs
      only `E_start`, `E_cut`, the row's length, the sampled loss, and the
      geometry-event flag, and it is ~20 lines with no state. Both lockstep LUT
      and grooved cores can take it verbatim. Two caveats. (1) *LUT cores*: the
      cutoff distance there is solved on the interpolated `dE/ds`, but the
      straggling branch does not use a cutoff distance at all, so the LUT's
      interpolation error stops entering the crossing — F should decide whether
      that is desirable (it makes the straggled LUT crossing *more* accurate
      than the deterministic one) or whether the LUT rate should be used for the
      `max_dE_frac` cap only, for consistency. (2) *Per-electron and CUDA
      cores*: those carry an explicit `exit_code` (`EXIT_CUTOFF_STOPPED` /
      `EXIT_STEP_LIMITED`) rather than the lockstep core's booleans, so the flag
      clearing has to be re-expressed, and CUDA additionally needs the division
      guard (`stragg_loss > 0.0`) kept — a `0/0` there is a silent NaN
      propagating into `pos`. Grooved adds one genuinely new question E did not
      face: a row can be truncated by a *facet* crossing into vacuum, and the
      cutoff test must run against the material-side length only. Nothing here
      suggests a variant rule is needed; the geometry bookkeeping differs, the
      physics does not.
- [x] F — Remaining cores: grooved, per-electron, per-electron LUT, CUDA.
      **Done on all four host cores and the CUDA exact kernel; the CUDA LUT
      kernel stays deliberately unwired behind a fail-closed raise, and the
      CUDA leg as a whole is UNVERIFIED (still no cupy and no CUDA device).**
      Owners: `montecarlo/transport.py` (`_transport_core_ungrooved_lut`,
      `_transport_core_ungrooved_perelectron`,
      `_transport_core_ungrooved_perelectron_lut`, `_transport_core_grooved`,
      a ~90-line derivation block above the first of them, and
      `simulate_trajectories`'s CUDA-LUT raise),
      `montecarlo/transport_jit_kernel.py` (`_transport_kernel`,
      `run_transport_kernel`'s docstring), and
      `tests/montecarlo/test_straggling_remaining_cores.py` (29 cases),
      `tests/montecarlo/test_straggling_cuda.py` (7 cases, 5 hardware-gated),
      plus a narrowing of slice D's property-2 test to match slice E's.

      **E's rule is ported, not revised.** The crossing indicator stays the
      exact `dE >= E_start - E_cut` test, the location stays
      `s_cut = s (E_start - E_cut) / dE` with `E_end = E_cut`, the
      `max_dE_frac` cap stays deterministic and stays *before* the draw with
      the cutoff test after it, and `energy_model` still selects only the
      clock's representative energy. Not one line of E's derivation was
      changed; F re-checked it against each core and found nothing that needs a
      variant rule, exactly as E predicted. The three per-core questions and
      their answers live in a block comment above
      `_transport_core_ungrooved_lut` (for K to transcribe, per C's and E's
      precedent; `docs/` untouched).

      **(1) LUT cores — the crossing gets MORE accurate, and that is accepted.**
      E flagged it: the deterministic LUT path solves its truncation distance
      on the *interpolated* `dE/ds`, but the straggled crossing has no such
      solve, and slice D already draws the loss from the *exact* per-element
      stopping power (the LUT bakes one interpolated total per layer and
      carries no per-element split). So the straggled LUT crossing does not
      carry the LUT's interpolation error. **Decision: accept and document,
      do not reintroduce the error.** Reintroducing it would mean degrading an
      exact quantity to match an approximation that exists only for speed, and
      there is no LUT-consistent loss to degrade it *to* — the loss is a draw,
      not a function of an interpolated rate. E's alternative ("use the LUT rate
      for the `max_dE_frac` cap only, for consistency") is **adopted in full**
      and falls out naturally: under E's ordering the cap is a purely
      deterministic step control evaluated before the draw, so it uses the LUT's
      `dE/ds`, and the clock uses the LUT's `inv_beta` at whichever
      representative energy `energy_model` selects. The LUT therefore still
      governs the free-path rate, element selection, scattering angle, substep
      grid and timing; only the energy loss and the crossing come from the exact
      sampler — the same split slice D already shipped for the diagnostic.

      **(2) Per-electron and CUDA cores — `exit_code` was a smaller problem
      than it looked.** Those cores do carry an `exit_code` enum rather than
      shared counters, but they still hold the same local
      `cross_up_j`/`cross_dn_j`/`exit_top_j`/`exit_bot_j`/`exit_side_j`/
      `cutoff_j` booleans through the row and only *derive* `exit_code[i]` from
      them once at the end. E's flag clearing therefore transcribes literally,
      and the derivation chain needs no change at all: clearing the geometry
      booleans is precisely what makes the chain fall through to
      `EXIT_CUTOFF_STOPPED`. `EXIT_STEP_LIMITED` is the loop's initial value,
      overwritten only by a real exit, so a cutoff row that also cleared
      `limited_j` still classifies correctly. Verified by test rather than by
      inspection: the exit-channel sum is pinned to `Ne` on both cores.

      **(3) Grooved — the facet-crossing interaction, designed and tested.**
      The one genuinely new question. A grooved row can be cut short by a
      groove facet crossing into vacuum, after which the electron travels a
      vacuum leg to its re-entry point.
      - *Which length the sampler sees.* The material-side one, and by
        construction rather than by a new guard: step 2b truncates `step_j` to
        `s_surface` **before** the energy close, so the length handed to
        `_urban_sample_compound_keV` is already material-side. Sampling over the
        untruncated collision distance would attribute vacuum path length to
        material energy loss, which is simply wrong — vacuum has no stopping
        power.
      - *Precedence, and it is forced rather than chosen.* The loss is
        non-decreasing along the material path (E's P1), so if the
        material-side loss reaches `E_start - E_cut` the first passage lies
        strictly inside the **material** part of the row, i.e. before the facet.
        The electron therefore stops in material and never reaches the vacuum:
        the crossing wins, `surface_first` is cleared alongside the other
        geometry flags, and no vacuum segment is emitted. If it does not reach
        it, the electron leaves through the facet at `E_start - dE` and the
        vacuum leg proceeds losing nothing. Both branches are exactly what the
        deterministic core already does with `cutoff_distance` compared against
        the facet-truncated `step_j`, and `surface_first` joins
        `geometry_event` for the tie-break for the same reason it already does
        there. **Nothing in the sampler or the rule changes; only the flag list
        grows by one.**
      - *Recorded for H.* Because a facet crossing splits what would otherwise
        be one flight into a shorter material row plus a vacuum leg, a grooved
        geometry samples the straggling stream at a different
        `(flight, substep)` cadence than a flat one. That is not a bias — the
        per-row means still sum to `C` times the total material path, which is
        what the closure test measures — but grooved and ungrooved runs at the
        same seed address different straggling draws, exactly as they already
        address different free-path draws.

      **CUDA: exact kernel wired, LUT kernel deliberately not, both decisions
      recorded.** `_transport_kernel` now applies the loss under E's rule
      instead of only accumulating the diagnostic. Two mechanical changes were
      needed beyond transcription: the per-element accumulation
      (`stragg_dE[e] += dE_elem`) is folded into a thread-local row total
      before a single global update — which the crossing rule needs and which
      also matches the host's `_urban_sample_compound_keV`, so it is a parity
      *improvement* — and the `stragg_loss > 0.0` division guard E flagged is
      kept and commented as load-bearing, since `0/0` there is a silent NaN
      into `pos` rather than a crash. **`_transport_lut_kernel` was NOT wired.**
      It has no per-element split, no duplicated sampler, and
      `run_transport_lut_kernel` does not even accept the straggling
      parameters; adding a second ~150-line transcription of the sampler to a
      kernel that cannot be compiled or executed on the machine writing it was
      judged strictly worse than the existing fail-closed raise, whose message
      is sharpened and is now **pinned by a device-free test** rather than only
      documented. That test also pins that the guard does *not* fire with
      straggling off.

      **This machine still has no cupy and no CUDA device** —
      `transport_jit_kernel` does not import, so nothing in it can be compiled
      or run. **The CUDA straggling path is therefore transcribed-only and
      UNVERIFIED, now including the applied loss and not merely the
      diagnostic.** That is a real escalation of D's caveat: a transcription
      error used to corrupt a diagnostic and now corrupts physics.
      `tests/montecarlo/test_straggling_cuda.py` carries five
      `@pytest.mark.hardware` tests written to close it the moment a GPU is
      available (off-path bit-for-bit, determinism, energy conservation with an
      explicit finite-`r_mid` check for the NaN guard, first-row host parity at
      `rtol=1e-12`, and an ensemble closure at `rel=0.02`). **Whoever has
      hardware should run them via `pyrite remote` before any GPU straggling
      result is trusted.** Note also that the parity claim itself weakened by
      construction, and the docstring now says so: slice C's few-ulp caveat
      (`_urban_poisson_scalar` branches on a `log`/`exp` comparison, so a
      last-bit difference can move `n` by one at a CDF boundary) used to
      perturb only a diagnostic; with the loss applied it changes the energy and
      the trajectory diverges from that row on, so only the **first row** of
      each electron is host-comparable — the same scope
      `test_cuda_first_step_agrees_with_the_cpu_reference` already uses for the
      deterministic path.

      **Per-step cost, measured.** 13th Gen Intel i7-13620H (16 threads),
      Python 3.14.4, numba 0.66.0, numpy 2.4.6, single-threaded njit host
      cores, warm (compile and cache primed by a discarded run), 5 repeats,
      400 electrons, 25 keV graphite in a 6 um slab (60 keV / 200 um / blazed
      groove for the grooved core), `seed` fixed per core, mean +- sd:

      | core | rows | off [ms] | on [ms] | delta | per row [us] |
      |---|---|---|---|---|---|
      | lockstep exact | 1.12e5 | 27.3 +- 0.7 | 44.2 +- 0.7 | +62% | 0.15 |
      | lockstep LUT | 1.12e5 | 9.2 +- 0.3 | 31.2 +- 0.6 | +238% | 0.20 |
      | per-electron exact | 1.12e5 | 23.4 +- 0.6 | 45.2 +- 0.2 | +94% | 0.20 |
      | per-electron LUT | 1.12e5 | 13.3 +- 0.3 | 34.0 +- 0.3 | +157% | 0.19 |
      | grooved | 2.08e5 | 44.5 +- 0.4 | 75.3 +- 0.5 | +69% | 0.15 |

      The added cost is **0.15--0.20 us per row and essentially
      core-independent**, which is the right shape: it is the sampler's own
      cost, and the sampler is the same function everywhere. The *relative*
      figure is worst on the LUT cores precisely because the LUT made the
      deterministic row cheap — straggling re-evaluates the exact per-element
      stopping power that the LUT exists to avoid, so a LUT run pays the
      absolute cost against a smaller base. **Consequence for G:** enabling
      straggling roughly halves the LUT's speed advantage, so a production
      toggle should not assume the LUT still buys what it buys today.

      **Device-register impact, qualitative (no GPU to measure on).** Slice F
      adds only two live values per thread over slice D's already-inlined
      sampler: `stragg_loss` (one f64, live across the per-element loop) and
      `delta_cut` (one f64, short-lived) — about four 32-bit registers, against
      D's ~40-value inlined sampler body, most of which is short-lived inside
      the element loop and coalescible. The one new branch is the crossing
      test, whose bodies are register-light and which is taken at most once per
      electron over a whole track, so its divergence contribution is
      negligible. The real divergence source is D's variable-trip continuum
      quanta loop, whose trip count slice B measured at `n_3` = 0.069--1.15 per
      flight, i.e. `O(1)`. **All of this needs a real `nvcc`/`cuobjdump`
      register count to confirm; flagged for the same `pyrite remote` follow-up
      as the parity tests.**

      **Tests, one per "Done when" property.**
      1. *Off is inert.* `straggling=False` bit-for-bit against a call that
         never mentions the keyword, on all four newly wired cores, across
         frozen / midpoint / substepped (`max_dE_frac=0.02`), in a run with
         >100 cutoff crossings and >2 rows per crossing — so both new orderings
         would be taken if the gate leaked. Same for the grooved core in a run
         carrying facet crossings as well as cutoffs.
      2. *Energy conserved sensibly.* Per core: no row starts below `E_cut`,
         all exit channels sum to `Ne`, and the summed sampled loss closes on
         the transport's own mean stopping power to **1%** over ~1e5 rows.
         Grooved closes against the **material** path length only, which is the
         direct test that the vacuum leg was excluded.
      3. *Cross-core parity.* Stated where it holds and no wider: the cores
         realize different trajectories by construction, so whole-run equality
         between them is meaningless, but the **sampled loss for a given
         `(seed, electron, flight, substep, E_start, s)` and E's rule applied to
         it** are the same function everywhere. Both are reconstructed offline
         from each core's own recorded rows and demanded equal in **exact
         float64** — including slice E's own core as the reference, so a future
         change that moved the rule on one core but not another fails here. The
         only tolerance is on the crossing row's loss recovered as a residual
         of the accumulator (`rel=1e-9`, cancellation rounding), exactly as E's
         own round trip does and for the same reason.
      4. *Overshoot.* 6 keV against a 5 keV cutoff on every wired core: the
         terminal row ends at `E_cut` exactly, the crossing loss reaches the
         available energy on **every** track (the empirical form of the exact
         indicator), and the median overshoot is >1.5x, so the rule is
         genuinely exercised.
      5. *Grooved-specific.* A 59 keV cutoff against a 60 keV beam in a blazed
         groove, so cutoff crossings and facet crossings compete constantly:
         every cutoff-terminated track ends at `E_cut` exactly and **no**
         electron has a vacuum leg at or after its terminating row's clock —
         the assertion that fails if `surface_first` is not cleared on a
         crossing. Plus a direct check that every recorded `vacuum_E_keV`
         equals some material row's `E_end_keV` exactly, i.e. vacuum legs lose
         nothing.
      6. *Fail-closed.* `straggling=True` + `transport_core="cuda"` + LUT
         enabled raises `NotImplementedError` naming both escapes — asserted,
         not merely documented — and the guard provably does not fire with
         straggling off.
      7. *Replay and batching.* Applying the loss changes how many segments an
         electron needs and therefore which electrons overflow their slots, so
         capacity replay (`seg_capacity` 4 / 37 / 4096) and batch size
         (`scratch_budget_bytes` 2^18 / 2^27) are re-pinned on both
         per-electron cores for trajectories *and* `straggle_dE_keV`. A stale
         accumulator from a discarded attempt would double-count and is
         invisible to the deterministic replay tests.

      **Slice D's property-2 test was narrowed again, the same way E narrowed
      it.** "Straggling on never changes any pre-existing array" is now false on
      every core, so the whole-array form is gone and E's first-row form is
      parametrized over all four ungrooved cores plus grooved. The argument
      holds on each: every core takes an electron's first row at the
      unperturbed start energy, the lockstep and grooved cores draw every
      electron's first free path before any electron takes a second one, and
      the per-electron cores have fully disjoint counter streams. Slice D's
      property 1 (off is bit-for-bit) and property 3 (offline reproduction)
      pass **unmodified** on every core, grooved included.

      **Acceptance run:** `test-suite core` (1795 passed, 74 skipped — the
      pre-existing 1770 plus this slice's 25 host-visible cases; the 5
      hardware-gated CUDA cases account for the skip growth),
      `test --numba`, `lint`, `typecheck` — all clean. No existing test was
      weakened or deleted; `test_groove.py` needed no change this time.

      **Checkpoints:** `5484cc49` (remaining host cores), `43fb19ab` (CUDA
      exact kernel and fail-closed CUDA-LUT surface), and `3c5e2802` (capacity
      replay and batching).

      **Recommendation for G.** Three things F establishes that G inherits.
      1. *The reachability question is now sharper, not softer.* Straggling is
         live on every host core and on CUDA-exact, so `Numerics` is the only
         thing standing between it and production. But it is **not** live on
         CUDA-LUT, which is the *default* CUDA configuration — so a `Numerics`
         toggle that reaches the runner would make a production GPU run raise
         unless G also decides that question. **G must pick one of: wire
         `_transport_lut_kernel` (needs a GPU to verify), have the runner fall
         back to the exact CUDA kernel when straggling is on, or keep
         straggling off the CUDA path entirely.** Recommend the fall-back: it
         is the only one that needs no unverifiable new kernel code, and F's
         cost measurement shows the LUT's advantage is roughly halved under
         straggling anyway, so the fall-back costs less than it looks.
      2. *Checkpoint identity must carry the flag, and cannot infer it.* The
         off path is bit-for-bit identical to a pre-straggling run by
         construction, which is exactly what makes a straggled and an
         unstraggled record indistinguishable in the CAS unless `straggling` is
         part of the case identity explicitly. It cannot be recovered from the
         outputs.
      3. *`energy_model` comes along whether G wants it or not.* Straggling
         reads `energy_model` for the clock's representative energy and
         `max_dE_frac` for the substep grid, and `max_dE_frac > 0` already
         requires `energy_model="midpoint"`. So plumbing straggling through
         `Numerics` plumbs both, which is the coupling the task doc's
         "Implementation path" section flagged as a scope decision rather than
         an oversight. G should state it as a decision, not discover it.
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

- **Dependency noted 2026-08-20 (supervisor):** ELSEPA elastic-scattering data
  is being set up under `feature/reference-elastic-scattering-data`. Flight
  length is the elastic MFP, so every per-flight number slice A measured
  (`kappa`, collision counts, `xi/I`) is conditioned on the current Browning
  fit at `transport.py:255` and will shift when that cross section is replaced.
  Re-run `slice_a_regime_audit.py` rather than re-deriving. B's model selection
  is unaffected: readmitting Landau needs ~250x on tungsten's measured
  `xi/I` = 0.004 and ~100x on carbon's 0.087, which an elastic refinement
  cannot supply. Note also that ELSEPA redistribution is license-blocked per
  that task's `ALTERNATIVE_SOURCE_REPORT.md`; generated tables must not be
  committed without a written grant.

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
  against this range, not against `~0.3 rad`. **Sharpened by B0 and B.** B0
  identified the `~0.3 rad` figure as a plasmon-only Poisson estimate
  (`sigma_E = sqrt(dE eps_p)` = 0.237 keV at a 25 eV quantum), so it is a
  correctly computed number for a model that omits shell ionisation and the
  whole Moller tail. B then selected **unrestricted** Urban with
  `T_up = T_max = E/2`, so the sampler's own variance is the full-Moller one:
  **slice H's baseline is the 13.4 rad end**, scaled by Urban's measured
  0.73--1.42 variance ratio, i.e. **9.8--19 rad**, against the 0.1 rad
  numerical tolerance. H reports the *residual* against that baseline, and
  must carry A's caveat that the free-space clock spread is not the
  coherent-sum Debye--Waller exponent.
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
- **Closed by B0 — the contested `energy-step-convergence.md` figures, each
  classified by measurement.** The hypothesis under test was that the page's
  figures were measured against the retired pure-Joy--Luo model and never
  re-measured after the splice landed (`2c51754`, `dcdb1cd`, `d15a8ef`). The
  commit dates are consistent with it — the figures landed in `79f071b` on
  2026-08-11, the splice on 2026-08-19 — but the dates are not the test. Every
  figure was recomputed under **both** models through the
  `_element_crossover_keV -> inf` seam (`--part b0`). **The hypothesis is
  rejected for all four.** Verdicts, for slice I to execute without
  re-deriving:

  | doc figure | splice | retired Joy--Luo | verdict |
  |---|---|---|---|
  | "about 1.0" inelastic/flight, C at 25 keV | 1.026 | 0.962 | **reproduces; no change** |
  | "about 0.11" inelastic/flight, W at 25 keV | 0.029 | 0.028 | **real discrepancy** |
  | `kappa` = 0.015, C at 25 keV over 1 um | 0.01526 | 0.01526 | **reproduces; no change** |
  | ~0.3 rad Jensen bias, hopg 25 keV / 1 um / 1 keV | 2.596 (soft), 13.431 (full) | 2.564, 13.292 | **real, and explained** |

  - *Carbon event count and `kappa`: not contradicted at all.* Both reproduce.
    `kappa` is splice-independent **by construction** —
    `xi = 2 pi r_e^2 mc^2 n_e s / beta^2` contains no stopping power — so no
    stopping-model change could ever have moved it.
  - *Tungsten event count: a real discrepancy, and NOT a live transport bug.*
    The splice moves it by 3% (0.028 -> 0.029), nowhere near the 3.7x needed,
    and it cannot: the flight length is the *elastic* mean free path, which the
    splice does not touch. The repository's own `S`, `lambda_el` and `I` for
    tungsten are mutually consistent under both models; reaching 0.11 would
    require `I` = 193 eV (the transport table's `J_keV` for W is **727 eV**) or
    a flight length 3.76x longer. This is a documentation arithmetic error, not
    code. **Slice I:** restate as ~0.03. The page's conclusion — fewer than one
    inelastic event per flight in tungsten — is unaffected and strengthened.
  - *The ~0.3 rad Jensen bias: not a splice artifact, and its origin is now
    identified.* Joy--Luo and the splice agree to ~1%. The figure is an
    **under-scoped variance model**, not an error: it is a plasmon-only Poisson
    estimate. Measured, the mean loss over 1 um in hopg at 25 keV is
    **2.247 keV**; with the page's own 25 eV plasmon quantum that is `N` = 90
    events and `sigma_E = sqrt(dE eps_p)` = **0.237 keV -> 0.317 rad**, against
    the page's stated "order 300 eV of loss spread" and "most probable loss
    25--33 eV in graphite". At 33 eV: 0.272 keV -> 0.419 rad. The `sigma_E`
    that gives exactly 0.3 rad is 0.230 keV. So the doc's arithmetic is correct
    for a model that omits all shell ionisation and the entire Moller tail.
    This confirms A's "underestimate by 10--50x" and supplies the mechanism.
    **Slice I:** the fix is to state which spread the figure describes, not to
    change the number in place.
- **Closed by B — distribution choice.** The **Geant4 Urban model**,
  unrestricted (`T_up = T_max = E/2`), applied per element. Selected against
  measured numbers, not general practice: `xi/I` per flight is 0.004--0.087 so
  Landau fails; `kappa` never exceeds 0.16 even integrated so Bohr fails;
  Vavilov's free-electron spectrum is invalid where `xi <~ I`. The decisive
  property is `C = dE/dx` — Urban's mean is *normalised to* the supplied
  stopping power (measured closure 1.0000), which is the only way to stay
  consistent with a **spliced** mean whose low-energy branch is an empirical
  Joy--Luo fit with no Bethe-consistent `xi`. Full justification,
  parameterisation and validity boundaries in checklist item B.
- **Corrected:** an earlier draft treated unrestricted-CSDA-plus-straggling as
  double-counting the Moller tail and therefore blocking. It is not — same mean,
  correct variance, self-consistent while delta rays are untransported. Slice B
  is correspondingly smaller than first scoped.
- **Closed by B — delta rays stay a non-goal.** B selected the unrestricted
  branch, so the question is not forced. `stopping-power.md`'s "No delta rays —
  all inelastic loss is local and continuous, so knock-on electrons do not
  exist as transported particles" was re-read and is still true with straggling
  present: a random loss of the same mean creates no secondary particles.
  **Slice I rewrites the "No straggling" bullet only and must leave the
  delta-ray bullet untouched.** A restricted sampler was rejected because it
  would require a *restricted* `dE/dx`, i.e. modifying the mean stopping power,
  which this task explicitly does not own; both branches of the current splice
  are unrestricted.
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

**Next: decide G's production surface before implementation.** A--F are done.
The unresolved choice is whether straggling remains a direct transport-API
research capability or reaches production through `Numerics`. If it reaches
production, G must also choose the CUDA-LUT behavior. Slice F recommends an
automatic fallback to the exact CUDA kernel: it avoids shipping a second
unverified sampler transcription, and the measured LUT speed advantage is
already roughly halved when straggling is enabled.

If production reachability is selected, G owns all three coupled identity and
execution changes: add `straggling`, `energy_model`, and `max_dE_frac` to
`Numerics` and the lowered case; include them in checkpoint/CAS identity; and
make the runner's CUDA-LUT behavior explicit. Straggling stays default-off and
the existing production path remains bit-for-bit unchanged.

H remains blocked on G because the observable campaign needs a canonical way
to request straggled runs. I, J, and K remain late documentation slices as
scoped; I must preserve B0's two no-change verdicts and leave the "No delta
rays" bullet untouched.

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
