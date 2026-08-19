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

- [ ] A — Implement Berger--Seltzer/ICRU-37 collision stopping as scalar and
      vectorized helpers beside the Joy--Luo pair, additive over elements by the
      existing coefficient rewrite, without the density-effect term. Unit-check
      the prefactor conversion to keV/Angstrom against the existing
      `7.85e-4` path, and pin the non-relativistic limit.
- [ ] B — Quantify the density-effect correction `delta` over 1--300 keV for the
      catalog materials. For electrons `beta gamma = 1` at ~212 keV, and typical
      solid `X_0` puts the onset near the top of the swept range, so this may be
      bounded and omitted with a stated error — but low-Z solids have low `X_0`
      and graphite is a primary material, so measure rather than assume. If it
      exceeds the accuracy target, add Sternheimer parameters from the PDG
      tables already cited for `J`.
- [ ] C — Splice policy. Choose the crossover energy, verify per material that
      the two branches agree there to a stated tolerance, and decide whether the
      midpoint solve (`eq-stopping-cutoff-distance`) needs continuity of the
      derivative or only of the value. A kink inside the cutoff solve is the
      failure mode to check for.
- [ ] D — Integrate across all four cores, both LUT variants, and the CUDA
      kernels. Update `build_transport_energy_lut` in the same slice. Re-sync
      the `campaign/sweep.py` cost-estimation copy or make it delegate.
- [ ] E — Model selection and identity: expose which branch/model produced a run
      in result metadata, and extend checkpoint/case identity so Joy--Luo and
      Berger--Seltzer records cannot collide in the CAS.
- [ ] F — Measure. CSDA range, backscatter and transmission fractions,
      bremsstrahlung spectral shape, and coherent-line phase, before and after.
      Validate CSDA ranges against ESTAR as a **user-run oracle** — comparing
      against a web service is not redistribution, so this is available now and
      needs no license determination. Heavy matrices via `pyrite remote`.
- [ ] G — Docs, ledger, goldens. Add `Validation: relativistic-bethe-stopping`;
      rewrite the "Validity ceiling" section of `stopping-power.md`, which
      currently exists to warn about precisely the defect this closes; narrow
      `Validation: electron-transport` to the low-energy branch; regenerate
      catalog goldens.

## Decisions and open questions

- **Decided:** analytic closed form, no packaged tables. This is the whole point
  — it sidesteps the ESTAR/PENELOPE redistribution gate that blocked
  `feature/reference-electron-stopping-data`.
- **Decided:** Joy--Luo is retained as the low-energy branch, not deleted.
  Berger--Seltzer omits shell corrections and the default cutoff is 5 keV.
- **Decided:** this does not close the reference-stopping task, which still owns
  provenance-controlled data, uncertainty budgets, and radiative stopping.
- **Open:** splice energy, and whether it is global or per material.
- **Open:** whether `delta` needs Sternheimer parameters over 1--300 keV
  (checklist B decides).
- **Open:** the 1--10 keV region remains served by a fit neither form validates
  well. With `E_cut_keV` defaulting to 5 keV the exposure is bounded, but the
  accepted uncertainty there should be stated rather than inherited silently.
- **Open:** whether the model becomes user-selectable or the splice is
  unconditional. Unconditional is simpler and matches the `xray_dispersion`
  precedent (commit `500a1dc` made in-medium dispersion unconditional); a
  selector adds identity surface for a branch nobody should choose.

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
