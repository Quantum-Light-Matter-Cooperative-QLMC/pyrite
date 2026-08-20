# Relativistic Bethe collision stopping

## Problem and scope

Joy--Luo is a low-energy fit and the repository already measures how badly it
degrades. From `docs/physics/beam-transport/stopping-power.md`
(`tbl-stopping-validity-ceiling`), the ratio to relativistic ICRU-37 Bethe
stopping is:

| Kinetic energy | Joy--Luo / ICRU-37 |
|---|---|
| 10 keV | 0.98 |
| 25 keV | 0.94 |
| 50 keV | 0.88 |
| 100 keV | 0.78 |
| 200 keV | 0.63 |
| 300 keV | 0.52 |

There is no guard on this and no automatic switch to a relativistic form. At the
300 keV model ceiling transport under-stops by roughly a factor of two.

This is a **first-order, systematic** error in the energy-versus-depth curve:
same sign for every electron, accumulating coherently across the ensemble. That
is the error class `energy-step-convergence` argues is worth bounding tightly —
it displaces the coherent line rather than merely suppressing it — and it is
currently 6% at 25 keV, the same operating point where the unmodeled straggling
bias was estimated at ~0.3 rad against a 0.1 rad numerical tolerance. The
propagator was refined to 0.1 rad against a clock built on a stopping power that
is 6% wrong.

### Why this is separable from the gated reference-data task

`feature/reference-electron-stopping-data` is blocked, and its own audit records
why: ESTAR is NIST SRD 124 with rights reserved and no located redistribution
grant; PENELOPE is behind controlled access with no public grant. That task went
directly to *packaging reference tables* and hit a licensing wall.

The relativistic form does not need a table. Berger--Seltzer/ICRU-37 collision
stopping for electrons is a published closed-form expression:

```
-(1/rho) dE/dx = (2 pi r_e^2 m c^2 / beta^2) (N_A Z / A)
                 [ ln( tau^2 (tau + 2) / (2 (I/mc^2)^2) ) + F^-(tau) - delta ]

F^-(tau) = 1 - beta^2 + [ tau^2/8 - (2 tau + 1) ln 2 ] / (tau + 1)^2
```

with `tau = E/mc^2` and `2 pi r_e^2 m c^2 N_A = 0.1535 MeV cm^2/mol`.

Every input is already in the tree. `materials/_transport_data.py` carries `Z`,
`A`, and `J_keV` per element for all 24 catalog elements, sourced from CIAAW
2024 and the PDG *Atomic and Nuclear Properties* tables; number densities come
from catalog composition. The density-effect coefficients `delta` would need
Sternheimer parameters — which the **same PDG tables** publish, so even that
path introduces no new source, no new provenance question, and no redistribution
gate.

This task therefore does **not** supersede
`feature/reference-electron-stopping-data`. That task wants provenance-
controlled reference data with an uncertainty budget and explicit
collisional/radiative separation. This one buys most of the accuracy for a
fraction of the cost with zero licensing exposure. If this lands, the reference
task's priority drops; its scope does not change.

Out of scope here: radiative stopping, straggling
(→ `feature/energy-loss-straggling`), elastic scattering, and delta rays.

### The splice looks well-conditioned already

Joy--Luo is not deleted. Berger--Seltzer omits shell corrections, which is why
NIST recommends restricting ESTAR collision stopping to >=10 keV, and the
default `E_cut_keV` is 5 keV — so the low-energy branch is load-bearing and
stays. The validity table shows the two forms agreeing to 2% at 10 keV, so a
crossover near there is well-conditioned rather than a discontinuity to be
smoothed away. That agreement is measured for one case and must be checked per
material rather than assumed to generalize from carbon.

## Implementation path and likely owners

- `montecarlo/transport.py::_dEds_compound_scalar` / `::_dEds_packed_scalar` —
  the scalar loss rate every core calls; the new form lands beside these with
  the same per-element additive coefficient rewrite
  (`eq-stopping-compound-coefficient`), which is an exact rewrite of `rho Z/A`
  and carries over unchanged.
- `::_dEds_compound` — the vectorized host path.
- `::build_transport_energy_lut` / `TransportEnergyLUT` — **the LUT bakes
  `dE/ds`**, so the builder needs the new model or the LUT cores silently keep
  the old physics. This is the easiest thing to miss.
- The four cores and their LUT variants, plus the CUDA equivalents in
  `transport_jit_kernel.py`.
- `campaign/sweep.py:831` — an independent copy of the Joy--Luo constants for
  cost estimation, explicitly documented as mirroring
  `transport._dEds_compound`. It drifts the moment the model changes.
- `materials/_transport_data.py` — where Sternheimer parameters would land if B
  says they are needed.
- `montecarlo/case.py` / `runner/__init__.py` — model selection and checkpoint
  identity.

Unlike straggling, this consumes **no random draws**: it replaces a
deterministic function with another deterministic function. There is no RNG
plumbing, no stream independence question, and no per-core sampling-order
concern. Every core can adopt it in one slice, and agreement between cores stays
exactly as tight as it is today.

Results *will* change for every run above ~10 keV. Goldens regenerate, and the
existing `Validation: electron-transport` row needs its scope narrowed to the
low-energy branch rather than being invalidated.

## Checklist

- [x] A — Implement Berger--Seltzer/ICRU-37 collision stopping as scalar and
      vectorized helpers beside the Joy--Luo pair, additive over elements by the
      existing coefficient rewrite, without the density-effect term. Unit-check
      the prefactor conversion to keV/Angstrom against the existing
      `7.85e-4` path, and pin the non-relativistic limit.
      **Done.** `_dEds_bs_keV_per_ang`, `_dEds_bs_compound_scalar`,
      `_dEds_bs_packed_scalar`, `_dEds_bs_compound` in `montecarlo/transport.py`.
      Prefactor `1.535e-6 keV/Ang` derived from `0.1535 MeV cm^2/mol` and pinned
      against Joy--Luo: `1.535e-6 * mc^2 = 7.844e-4` vs `7.85e-4`, 0.08% -- the
      rounding in the conventional constant. Non-relativistic limit converges
      monotonically with `O(tau)` residual. `delta` is carried as a per-layer
      scalar (it is a bulk property, so it factors out of the Bragg sum) and
      every call site passes `0.0`, so B lands as a data change, not a signature
      change.
- [~] B — Quantify the density-effect correction `delta` over 1--300 keV for the
      catalog materials. **Sized, not measured -- blocked on source data.** For electrons `beta gamma = 1` at ~212 keV, and typical
      solid `X_0` puts the onset near the top of the swept range, so this may be
      bounded and omitted with a stated error — but low-Z solids have low `X_0`
      and graphite is a primary material, so measure rather than assume. If it
      exceeds the accuracy target, add Sternheimer parameters from the PDG
      tables already cited for `J`.
- [x] C — Splice policy. **Decided: per element, not global.**
      Measured crossovers span 2.66 keV (B) to 10.46 keV (Bi), monotone in `I`;
      every catalog element crosses exactly once, well above the energy where the
      Berger--Seltzer bracket changes sign (below 0.71 keV for all 24). The task
      doc's premise that "the two forms agree to 2% at 10 keV" is a *carbon*
      number and does not generalize: at 10 keV the ratio runs 0.976 (B) to 1.003
      (Bi). The best single global splice energy is 8.0 keV and it still steps by
      1.84% in the worst element; no global choice does better than 1.5%.
      Because stopping is additive over elements, the splice does not have to be
      global -- switching each element at its own crossover makes every term
      continuous by construction, hence the compound too, for any material, with
      no per-material tuning and no fitted blend. Verified to `1e-12` relative
      for all 24 elements and all 50 catalog materials.
      Continuity of value only: the splice is C0, not C1. The log-slope steps by
      0.0145 (B, 2.0% of the local slope) to 0.0587 (Bi, 8.9%), worst at high `Z`.
      The midpoint solve needs only the value, so this is acceptable, but the
      cutoff bracket must be checked against the kink when D wires the cores in.
      Implemented as `_bs_joy_luo_crossover_keV`, `_dEds_spliced_compound_scalar`,
      `_dEds_spliced_compound`.
- [~] D — Integrate across all cores, both LUT variants, and the CUDA kernels.
      **Done on CPU; the CUDA kernel is written but unverified.**
      Three cores evaluate stopping directly (`_transport_core_ungrooved`,
      `_transport_core_grooved`, `_transport_core_ungrooved_perelectron`); the
      two `_lut` cores read `lut.dEds`, so `build_transport_energy_lut` was the
      only change they needed. A fourth per-element array `L_E_cross` is
      threaded beside `L_ks`/`L_coeffs` through the table builder, the LUT
      builder, `pack_layer_tables`, every core signature, the flight
      diagnostics, and the CUDA kernel + launcher; `delta` stays a scalar
      parameter passed `0.0` so B lands as a data change.
      The `campaign/sweep.py` and `spectrum/diagnostics.py` copies were
      **deleted, not re-synced** -- both now delegate to the new host helper
      `transport.spliced_stopping_keV_per_ang`. `spectrum/lines.py` keeps a
      device-array twin (`_spliced_stopping_magnitude_xp`) because its rows may
      live on the GPU; `test_stopping_mirrors_agree` pins it to the host helper.
      The `sweep.py` docstring's claim that it "imports only the leaf
      `TRANSPORT_ELEMENTS` table" was already stale -- it imports
      `montecarlo.case`, which pulls in the whole package -- so delegating cost
      nothing.
      **Not verified:** the CUDA kernel. There is no GPU here --
      `transport_jit_kernel` imports `cupy` at module scope, so it cannot even
      be imported locally and its tests skip. Two source-text pins were added
      (`test_cuda_stopping_constants_match_the_cpu_values`,
      `test_cuda_stopping_keeps_the_per_element_splice`) following the
      repository's existing precedent for this, and the pre-existing
      `test_cuda_source_uses_cpu_reference_cutoff_and_termination_rules` still
      passes. Those catch a constant or a branch edited on one side only. They
      are **not** a substitute for running the kernel: it needs a `pyrite
      remote` run before it can be trusted, and that is the first thing F should
      do.
- [ ] E — Model selection and identity: expose which branch/model produced a run
      in result metadata, and extend checkpoint/case identity so Joy--Luo and
      Berger--Seltzer records cannot collide in the CAS.
- [ ] F — Measure. CSDA range, backscatter and transmission fractions,
      bremsstrahlung spectral shape, and coherent-line phase, before and after.
      Validate CSDA ranges against ESTAR as a **user-run oracle** — comparing
      against a web service is not redistribution, so this is available now and
      needs no license determination. Heavy matrices via `pyrite remote`.
- [x] G — Docs, ledger, goldens. **Done.** The `Validation: relativistic-bethe-stopping`
      ledger row and the `electron-transport` narrowing were already in place
      from D; what remained was `stopping-power.md`, which still described only
      Joy–Luo. Added `## Berger–Seltzer relativistic branch` ({eq}`eq-stopping-bs`,
      `eq-stopping-bs-fminus`, prefactor derivation pinned against Joy–Luo,
      density-effect $\delta$ status) and `## Per-element splice` (crossover
      range, C0-not-C1 continuity, why the kink is harmless to the cutoff
      solve); rewrote "Validity ceiling" to state the table is a historical
      record superseded above each element's crossover, added the CSDA
      range-change table, and named the two items still open ($\delta$
      unmeasured; the 1–10 keV window unowned by either form). Updated the
      `Assumptions and limits` and `Validation` sections to reference both
      branches. Added `icru37` and `bergerseltzer1982` to `references.bib`.
      Catalog goldens: confirmed **not applicable**, per the "Not regenerated"
      note above — `data/materials.toml`/`materials/catalog.py` are untouched
      by this task and `test_material_catalog.py` passes unchanged.
      `pyrite-dev docs` (clean, offline, warnings-as-errors) and
      `pyrite-dev lint` pass; `test-suite core` (1589 passed) is unaffected, as
      expected for a docs-only change.

## Decisions and open questions

- **Decided:** analytic closed form, no packaged tables. This is the whole point
  — it sidesteps the ESTAR/PENELOPE redistribution gate that blocked
  `feature/reference-electron-stopping-data`.
- **Decided:** Joy--Luo is retained as the low-energy branch, not deleted.
  Berger--Seltzer omits shell corrections and the default cutoff is 5 keV.
- **Decided:** this does not close the reference-stopping task, which still owns
  provenance-controlled data, uncertainty budgets, and radiative stopping.
- **Decided:** splice is **per element**, each at its own Joy--Luo/Berger--Seltzer
  crossover, computed once at table-build time. Continuous by construction for
  every material; a global splice cannot be (best case 1.84% step). See C.
- **Open:** whether `delta` needs Sternheimer parameters over 1--300 keV.
  Bounded to the small-correction regime (`beta*gamma <= 1.24` across the whole
  swept range) but not closed -- needs `x_0, x_1, a, m` per material from PDG.
  See B.
- **Open:** the 1--10 keV region remains served by a fit neither form validates
  well. With `E_cut_keV` defaulting to 5 keV the exposure is bounded, but the
  accepted uncertainty there should be stated rather than inherited silently.
- **Open:** whether the model becomes user-selectable or the splice is
  unconditional. Unconditional is simpler and matches the `xray_dispersion`
  precedent (commit `500a1dc` made in-medium dispersion unconditional); a
  selector adds identity surface for a branch nobody should choose.

## Status

A, C, G, and the CPU half of D are implemented, tested, and green. B is sized
but blocked on PDG Sternheimer parameters. E and F are untouched.

**The model is now wired in and results have changed** above each element's
crossover (2.66--10.46 keV). Nineteen existing tests moved; every one was a
stale Joy--Luo oracle or a signature, and each is recorded below under
"Test fallout" rather than silently retuned.

Ledger row `relativistic-bethe-stopping` in
`docs/validation/ledger-transport-background.md` stays at `filtered` --
in-context units/limits/signs plus regression anchors only. Fresh-context
`physics-validation` has **not** run. It was the stated next step before D and
it still has not happened; D landed ahead of it on direct instruction, so the
verification debt is now larger, not smaller.

The `electron-transport` ledger row was narrowed in place: its claim now says
Joy--Luo is the low-energy branch only, and its "Validity ceiling" note is
labelled superseded above the crossover.

`docs/physics/beam-transport/stopping-power.md` now documents the
Berger--Seltzer branch and the per-element splice, and its "Validity ceiling"
section is rewritten to state what the transport does today rather than what
it used to do. `pyrite-dev docs` passes clean (offline, warnings-as-errors).
This closes the doc-vs-code gap; it does not substitute for the fresh-context
`physics-validation` pass still owed on the derivation (see above), nor for F's
measurement work.

### Test fallout from D

- Signature only (6 in `test_groove.py`, 2 in `test_transport_lut.py`): the
  synthetic layer tables needed `L_E_cross`.
- Stale Joy--Luo oracle, repointed at `spliced_stopping_keV_per_ang`
  (`test_transport_cutoff.py`, `test_transport_diagnostics.py`,
  `test_radiation_error_estimators.py`, `test_transport_energy_model.py`, and
  one assertion in `test_groove.py`).
- `test_midpoint_end_state_error_is_third_order_in_flight_length[t_end_ang]` --
  the clock's convergence order is unchanged but it reaches its asymptote at
  shorter flights than before, because the stronger stopping enlarges the
  O(s^4) term. Measured 600 -> 4.7 Ang: 12.3, 10.9, 9.8, 9.0, 8.5, 8.3, 8.07.
  The energy ratio is untouched at 8.04/8.02/8.01. Lengths are now
  per-field; the band is not widened.
- `test_transport_substeps_reach_the_grouped_reduction[lockstep]` -- asserted
  that *no* flight's substeps are ever adjacent rows. Electrons now stop
  sooner, so the tail where one electron runs alone got longer and its substeps
  are adjacent. Replaced with the assertion the test actually exists to make:
  adjacency-keyed grouping would report more groups than there are flights.

### Size of the change (CSDA range, `E_cut` 5 keV)

Analytic `int dE / |dE/ds|`, Joy--Luo versus the spliced model. This is the
magnitude every downstream measurement in F has to account for; it is not itself
F, which wants transported ranges validated against ESTAR.

| material | 25 keV | 100 keV | 300 keV |
|---|---|---|---|
| graphite | -4.2% | -15.1% | -35.4% |
| silicon | -4.0% | -14.9% | -35.2% |
| tungsten | -2.8% | -14.2% | -34.6% |

Ranges shorten, as they must: the old model under-stopped. At 300 keV graphite
goes 643 -> 416 um. The 25 keV figure is the one that matters most right now --
that is the operating point where the propagator was tuned to 0.1 rad against a
clock built on this stopping power.

### Not regenerated

The material-catalog golden is driven by `data/materials.toml` and
`materials/catalog.py`, neither of which this touches, and
`tests/materials/test_material_catalog.py` passes unchanged. Per `regen-golden`,
regenerating without a catalog-source change would mask drift. The task doc's
"goldens regenerate" expectation does not apply to this golden.

## Delegation slices and required skills

- **A--C** — `implement-task` with `physics:implement`, `scientific-library`,
  and `monte-carlo`; then fresh-context `physics-validation` on the derivation.
  A is `one-shot`: the formula is closed-form and every input is in the tree.
- **D** — `implement-task` with `monte-carlo`, `performance`, and
  `remote-gpu-jobs`. Not `one-shot`; the LUT builder and the `sweep.py` mirror
  are easy to miss.
- **E** — `implement-task` with `cli-ui-ux` and `regen-golden`.
- **F** — `lead-task` with `remote-gpu-jobs` and `physics-validation`.
- **G** — `documentation-maintenance`, `physics-review`, `regen-golden`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --numba
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
```

- The Berger--Seltzer implementation is derived from its published form with
  units, limits, and signs checked, and agrees with ESTAR CSDA ranges to a
  stated tolerance across low/high Z over the accepted range.
- The density-effect omission is bounded by measurement, not assumed.
- The splice is continuous to a stated tolerance per material, and the cutoff
  solve is verified across it.
- All four cores, both LUT variants, and the CUDA kernels evaluate the same
  model; the `sweep.py` cost mirror does not drift.
- Run metadata identifies the stopping branch, and checkpoint identity separates
  Joy--Luo from Berger--Seltzer records.
- The change in CSDA range, backscatter/transmission, bremsstrahlung shape, and
  coherent-line phase is measured and reported.
- `stopping-power.md`'s validity-ceiling warning is rewritten to describe the
  new behavior; `Validation: electron-transport` is narrowed rather than left
  claiming more than it covers; goldens regenerated.
