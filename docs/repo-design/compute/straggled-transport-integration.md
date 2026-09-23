# Straggled transport integration

How the Urban energy-loss sampler is applied *inside* the transport cores: where the cutoff crossing is when the loss is random, what `max_dE_frac` substepping still guarantees, and what each core's own geometry forces on the bookkeeping.

The loss model itself is not derived here. Its source is the Geant4 Physics Reference Manual, "Energy loss fluctuations" (Urban model), and its derivation sits where it is implemented, above `_urban_levels_scalar` in `src/pyrite/montecarlo/transport/straggling.py` ([Validation: `energy-loss-straggling`](../../validation/beam-transport/energy-loss-straggling.md)). Nothing on this page adds physics to that model; this is the transport-side integration of it.

Everything below is gated behind `straggle_on`. With straggling off, every line it describes is unreachable and the deterministic code path is textually unchanged — the bit-for-bit guarantee stated at the top of `straggling.py`.

## 1. The loss over a flight is a subordinator, not just a random number

Urban's loss over a step of length `s` at frozen energy is the compound Poisson sum `dE(s) = sum_i sum_{k=1}^{n_i(s)} E_{i,k}` with `n_i(s) ~ Poisson(s Sigma_i)`. Read as a function of `s` it is a Lévy process with non-negative jumps: a subordinator. Two of its properties do all the work below.

**(P1) Monotone.** `dE(s)` is non-decreasing in `s`, so the electron's energy `E(s) = E_start - dE(s)` is non-increasing, exactly as in the deterministic model. Therefore

```text
inf{ s' <= s : E(s') <= E_cut }  exists  <=>  dE(s) >= E_start - E_cut.
```

The *indicator* of "this row crosses the cutoff" is a function of the total loss over the row alone — which is precisely what the sampler returns. Apart from an exact tie with a geometry event, the crossing decision is exact under this model. Geometry wins that tie by the transport's explicit row-end precedence convention.

**(P2) Infinitely divisible.** For any partition `s = sum_m s_m`,

```text
sum_m CP(s_m Sigma) =_d CP(s Sigma),
```

because `sum_m Poisson(s_m Sigma_i) = Poisson(s Sigma_i)` and the marks are i.i.d. from the same law. At frozen `Sigma` this is exact, not asymptotic. It is the substep invariance, derived in section 3.

## 2. Cutoff crossing: exact indicator, fluid-interpolated location

Let `Delta = E_start - E_cut > 0` (every alive electron satisfies this; a row that reaches `E_cut` is killed) and let `dE` be the sampled loss over the row's length `s`. By (P1) the row crosses iff `dE >= Delta`, and the crossing distance is the position of the jump that carries the running sum past `Delta`. The sampler returns the total, not the jump ladder, so the *location* needs a rule. Equality at a simultaneous geometry event belongs to geometry; equality without geometry belongs to the cutoff. The rule places a winning cutoff where the loss, accrued at the row's own realized average rate `dE/s`, reaches `Delta`:

```text
s_cut = s * Delta / dE,        E_end = E_cut,        cutoff_j = True.
```

### Why this rule

- **It degenerates algebraically, not merely in gate, to the deterministic solve.** Put `dE -> |dE/ds| s` (the zero-fluctuation limit): the crossing condition becomes `s > Delta/|dE/ds| = cutoff_distance` and `s_cut = s Delta / (|dE/ds| s) = Delta/|dE/ds| = cutoff_distance`, which is the frozen-model line `cutoff_distance = (E_cut - E_j) / dEds` verbatim.
- **It is the same approximation the surrounding transport already makes.** The deterministic core spreads a flight's loss uniformly along the flight even though the loss is physically a handful of discrete collisions; the clock (`s / beta`) and `seg_mid` are built on that fluid picture. Using the realized rate instead of the mean rate changes which number is spread, not the spreading.
- **It handles overshoot with no special case.** The sampler does not clamp `dE` to `E`, and with `n_3` up to 1.15 per flight a single row can sample a loss far above `Delta`. Then `dE >> Delta` gives `s_cut -> 0`: "the electron ran out of energy right at the start of this row". `E_end` is `E_cut` exactly, never negative, never below the cutoff.
- **It consumes no additional random numbers**, so the `(electron, flight, substep)` stream layout, the straggling-off bit-for-bit claim, and offline reproducibility of `straggle_dE_keV` from that key all survive unchanged.

### What it costs

The crossing *location* is biased inside the crossing row. The true first-passage distance is the position of the crossing jump, which given one jump is uniform on `[0, s]`; the rule returns the deterministic fraction `Delta/dE` of the row instead, so a large overshoot places the stop earlier than the truth. The bias is bounded by one row length and applies only to the row that terminates the track, so it perturbs the end of the range-straggling distribution by at most the final flight length — which at `E ~ E_cut` is the elastic mean free path at a few keV, Ångströms to tens of Ångströms.

### Alternatives considered and rejected

1. **Travel the full row, then stop if `E_end <= E_cut`.** It does not degenerate to the deterministic solve at all (in the zero-fluctuation limit it still overshoots by `s - cutoff_distance`), and it lengthens every terminated track by half a flight on average — a systematic range bias present even with the fluctuation switched off.
2. **Draw the crossing position uniformly on `[0, s]`.** Exact for a single-jump crossing, but wrong for a multi-jump one, wrong in the deterministic limit (it would randomize a stopping point that is not random), and it consumes a stream draw whose count depends on the outcome.
3. **Clamp the sampled loss to `Delta` and keep the analytic cutoff distance.** Clamping breaks `<dE> = C s`, the single property Urban was selected for, and makes `n_cutoff_stopped` blind to the fluctuation it is supposed to reflect.
4. **Sample the jump ladder (counts and uniform positions) for the exact first passage.** Correct, but it requires the sampler to return per-element counts and to draw `n_i` extra position variates — a change to the sampler, not to its application.

### `n_cutoff_stopped` bookkeeping

`cutoff_j` keeps its exact meaning ("this row ended because the electron reached `E_cut`"), so the increment, the `died_j` kill, and the geometry-flag clearing are unchanged; only the test that sets it is redefined. By (P1) the flag fires on exactly the rows on which the true first passage lies inside the row under the explicit tie convention, so the count is exact within that convention, not approximate.

### The energy model

Under straggling the loss over a row is the sampled `dE` and `E_end = E_start - dE` for **both** `energy_model` codes. The midpoint predictor-corrector is a second-order quadrature of the deterministic ODE `dE/ds = f(E)`; with a random loss there is no ODE to quadrature and the sampler's own mean is the left-endpoint one, `C(E_start) s`. `energy_model` therefore still selects the clock's representative energy — beta at the row's realized midpoint `(E_start + E_end)/2` versus at `E_start` — and the `seg_E_end`/`seg_t_end` schema, but no longer the energy update itself. The residual left-endpoint bias this leaves in the mean is exactly the `O(s^2)` term derived in section 3, and `max_dE_frac` is the lever that controls it.

## 3. Substep invariance under a stochastic loss

The `substep-radiation-invariance` claim is an **algebraic** invariance: subdividing a flight leaves the deterministic result unchanged. That claim does not survive a random loss and is re-derived here as a **distributional** one.

Setup: one physical flight of length `s` at start energy `E`, either taken whole (`N = 1` row) or split by `max_dE_frac` into `N` substeps of lengths `s_1..s_N` with `sum_m s_m = s`, substep `m` starting at energy `E^(m)`, `E^(1) = E`, `E^(m+1) = E^(m) - X_m`, and `X_m` the loss sampled over `s_m` at `E^(m)`.

**(i) At frozen energy the invariance is exact.** If every substep used the same rates `Sigma_i(E)`, then by (P2) `sum_m X_m =_d X`, the unsplit draw, for any partition and any `N`. Not a limit, not a tolerance: the same distribution. It is *distributional*, not pathwise — each substep addresses its own `(flight, substep)` key, so the realized numbers differ; only the law is preserved. This is the strongest form the invariance can take.

**(ii) Once energy evolves, the next jump kernel depends on the previous random loss.** Let `nu_E(d epsilon)` be the frozen-energy jump-intensity measure and `C(E) = integral epsilon nu_E(d epsilon)`. For two short rows `h_1`, `h_2`,

```text
<X_1 + X_2> - <X_frozen>
  = h_1 h_2 integral [C(E-epsilon) - C(E)] nu_E(d epsilon) + O(h^3)
  = h_1 h_2 [-C(E) C'(E) + R(E)] + O(h^3),

R(E) = integral [C(E-epsilon) - C(E) + epsilon C'(E)] nu_E(d epsilon).
```

Urban marks stay finite as `h -> 0`, so `R(E)` is generally nonzero at the same order as the deterministic linearization `-C C'`. For equal substeps, the generator coefficient is multiplied by `s^2 (N-1)/(2N)`.

**(iii) The substeps are conditionally, not unconditionally, independent.** For two rows,

```text
Cov(X_1, X_2) = h_2 Cov(X_1, C(E - X_1)),
```

and the law of total variance also contributes `E[h_2 V(E - X_1)]` and `Var(h_2 C(E - X_1))`. So the evolving-energy variance is not `sum_m s_m V(E^(m))` with cross terms discarded. `max_dE_frac` refines a state-dependent jump-process discretization; convergence cannot be reduced to deterministic stopping-power quadrature alone.

**(iv) Measured.** `tests/montecarlo/test_straggling_transport_integration.py`, graphite, `E = 25 keV`, `s = 1e4 Ang` (`DeltaE/E = 0.09`), 20000 repetitions:

| case | result |
| --- | --- |
| frozen, `N = 1` vs `N = 32` | mean shift `-0.0014 +- 0.0150 keV` on a mean of `2.243 keV` — consistent with the exact invariance of (i) |
| drifting, `N = 32` | shift `+0.086 +- 0.015 keV`; the regression evaluates the full generator coefficient from the three Urban channels and resolves its finite-jump remainder beyond the `-C C'` linearization |
| in transport, 600 electrons at 25 keV, `max_dE_frac` 0 vs 0.02 | mean per-electron straggled loss 19.61 vs 19.70 keV, 0.5% |

### Why the step-length control stays deterministic

`max_dE_frac * E_j / (-dEds)` uses the mean rate, not the sampled loss. A substep grid chosen from the realized loss would be a random partition, the partition and the increments would be dependent, and (P2) — which holds for any *fixed* partition — would no longer apply. `max_dE_frac` is a numerical control parameter and stays one.

### Ordering consequence

With straggling on, the row's length must be settled before the loss can be sampled over it, so the `max_dE_frac` cap is applied **before** the sample and the cutoff test **after** it — the reverse of the deterministic order, which can afford to solve the cutoff first because there the loss is a known function of distance. A substep cap that binds short of the crossing simply emits its row and lets the next substep cross: the same semantics at finer resolution.

## 4. Per-core integration

The crossing rule is core-agnostic: it needs only `E_start`, `E_cut`, the row's material path length, the sampled loss, and the geometry-event flag. No physics decision from sections 1–3 is revisited per core. What each core does change is the bookkeeping its own geometry and flag representation forces.

### 4.1 LUT cores: the crossing no longer carries the LUT's own error

The deterministic LUT cores solve the truncation distance on the *interpolated* `dE/ds`, so their crossing point inherits the LUT's interpolation error. The straggled crossing has no such solve: by (P1) the indicator is "does the row's total sampled loss reach `E_start - E_cut`", and the sampled loss on a LUT core is drawn from the *exact* per-element stopping power — the LUT bakes one interpolated total per layer and carries no per-element split, so `_urban_sample_compound_keV` is handed the exact tables regardless of which core calls it. Under straggling the LUT core's crossing is therefore strictly **more** accurate than its own deterministic path.

That is accepted, not reconciled. Reintroducing the LUT's interpolation error into the crossing would mean deliberately degrading an exact quantity to match an approximation whose only purpose is speed, and there is no LUT-consistent loss to degrade it *to* — the loss is a draw, not a function of an interpolated rate. What the LUT keeps is everything it exists to accelerate and everything still deterministic:

- the `max_dE_frac` step cap uses the LUT's `dE/ds` (the ordering above makes the cap a purely deterministic step control, evaluated before the draw), so the substep *grid* a LUT run produces is the LUT's own, not the exact core's;
- the clock uses the LUT's `inv_beta` at the representative energy `energy_model` selects, exactly as the deterministic LUT path does;
- the free-path rate, element selection and scattering angle are untouched.

The LUT therefore still governs step control, timing and geometry; only the energy loss and the crossing come from the exact sampler — the same split the straggling diagnostic already uses.

### 4.2 Per-electron and CUDA cores: `exit_code` instead of boolean flags

Those cores return an `exit_code` enum per electron rather than accumulating into shared counters. The re-expression is smaller than it looks: they still carry the same local `cross_up_j` / `cross_dn_j` / `exit_top_j` / `exit_bot_j` / `exit_side_j` / `cutoff_j` booleans through the row, and only *derive* `exit_code[i]` from them once at the end of the row. The flag clearing on a cutoff event therefore transcribes literally, and the derivation chain (`EXIT_BACKSCATTERED` / `EXIT_TRANSMITTED` / `EXIT_SIDE` / `EXIT_CUTOFF_STOPPED`, in that priority order) needs no change at all: clearing the geometry booleans is exactly what makes the chain fall through to `EXIT_CUTOFF_STOPPED`. `EXIT_STEP_LIMITED` is the loop's initial value, overwritten only by a real exit, so a cutoff row that also cleared `limited_j` still classifies correctly.

### 4.3 Grooved core: the cutoff test runs on the material-side length

The grooved core is the one place a row can be cut short by leaving the material entirely: a groove facet crossing into vacuum truncates the flight at `s_surface` and the electron then travels a vacuum leg to its re-entry point.

Straggling must not see that vacuum leg, and it does not, because of where the truncation already sits: `step_j` is truncated to `s_surface` in step 2b, *before* the energy close, so the length handed to the sampler is the material-side length by construction. Sampling over the untruncated collision distance would attribute vacuum path length to material energy loss — a straightforward physics error, since vacuum has no stopping power — and sampling over the full material+vacuum path would do the same.

The interaction with the crossing test is then a precedence question, and the answer is forced by (P1) rather than chosen. The loss is non-decreasing along the material path, so if the row's material-side loss reaches `E_start - E_cut`, the first passage lies inside the *material* part of the row, strictly before the facet. The electron stops in the material and never reaches the vacuum: the crossing wins, `surface_first` is cleared alongside the other geometry flags, and no vacuum segment is emitted. Conversely, if the material-side loss does not reach it, the electron leaves through the facet with `E_end = E_start - dE` and the vacuum leg proceeds at that energy, losing nothing.

Both branches are exactly what the deterministic core does with `cutoff_distance` compared against the already-facet-truncated `step_j`; `surface_first` joins `geometry_event` for the tie-break for the same reason it already does there, so a crossing landing exactly on the facet yields to the facet.

One consequence worth recording: because the loss is sampled per material row and a facet crossing splits what would otherwise be one flight into a shorter material row plus a vacuum leg, a grooved geometry samples the straggling stream at a different `(flight, substep)` cadence than a flat one. That is not a bias — the per-row means still sum to `C` times the total material path — but it does mean grooved and ungrooved runs at the same seed address different straggling draws, exactly as they already address different free-path draws.

## Invariants a future editor must not break

These are the statements the code depends on. Everything else on this page is the argument for them.

1. The cutoff crossing test is `dE >= E_start - E_cut` on the row's **sampled** loss, and the crossing location is `s * Delta / dE`. Do not reintroduce a solve against `dE/ds`, and do not clamp the sampled loss.
2. A geometry event at an exact tie beats the cutoff. `surface_first` counts as a geometry event for this purpose.
3. The loss is sampled over the **material-side** row length. In the grooved core that means after the step 2b facet truncation, never before.
4. The `max_dE_frac` cap is applied before the loss is sampled and the cutoff test after it. The cap itself is computed from the mean rate, never from a realized loss.
5. With `straggle_on` false, none of this executes and the result is bit-for-bit identical to the deterministic path.
6. The straggled path draws no extra random numbers, so `straggle_dE_keV` stays reproducible offline from `(electron, flight, substep)`.
