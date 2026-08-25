# `transport-midpoint-stopping`

Ledger row: [`transport-midpoint-stopping`](../physics-validation-ledger.md).
Code: `montecarlo/transport/api.py::simulate_trajectories` (`energy_model`);
host cores in `montecarlo/transport/cores.py`; exact CUDA core in
`montecarlo/transport/_jit_kernel.py`.

## Claim

A physical flight of length `s` starting at energy `E_start` uses explicit
midpoint RK2:

$$
E_{\rm pred}=E_{\rm start}+\frac{dE}{ds}(E_{\rm start})s,
\qquad
E_{\rm end}=E_{\rm start}
+\frac{dE}{ds}\!\left(\frac{E_{\rm start}+E_{\rm pred}}{2}\right)s.
$$

and advances the relative-age transport clock by

$$
\Delta t=\frac{s}{\beta((E_{\rm start}+E_{\rm end})/2)}
\quad [\text{\AA},\ c=1].
$$

replacing the left-endpoint ("frozen") rule `E_end = E_start + (dE/ds)(E_start)·s`,
`Δt = s/β(E_start)` that held both coefficients at the flight's start energy.

## Governing equations and where they come from

No new physical law enters. Current transport uses the per-element
Joy--Luo/Berger--Seltzer splice ledgered under
`relativistic-bethe-stopping`. The expression below is its low-energy Joy--Luo
branch and was the stopping function used by the historical numerical table:

$$
\frac{dE}{ds}(E)
=-\frac{7.85\times10^{-4}}{E}
\sum_i\frac{n_i}{0.602214076}Z_i
\ln\!\left(\frac{1.166(E+k_iJ_i)}{J_i}\right).
$$

in keV/Å, with `k_i = 0.731 + 0.0688 log10 Z_i`, and the clock is the elementary
kinematic integral

$$
t(s)=\int_0^s\frac{ds'}{\beta(E(s'))},
\qquad
\beta(E)=\sqrt{1-(1+E/511.0)^{-2}}.
$$

Both are initial-value problems in the path length `s`. The frozen rule is the
explicit Euler discretization of each with a single step of size `s`; this claim
replaces it with the midpoint rule for both.

## Discretization and its evaluation

The energy update is the standard explicit midpoint (RK2) scheme:

$$
E_{\rm pred}=E_{\rm start}+\frac{dE}{ds}(E_{\rm start})s,
\qquad
E_{\rm mid}=\frac{E_{\rm start}+E_{\rm pred}}{2},
\qquad
E_{\rm end}=E_{\rm start}+\frac{dE}{ds}(E_{\rm mid})s.
$$

The clock uses the midpoint quadrature of `∫ ds'/β` with the *corrected*
representative energy `(E_start + E_end)/2`, so a flight carries one
representative energy rather than two.

### Cutoff truncation

A flight truncated by the population energy floor `E_cut` has `E_end = E_cut` by
construction. This branch evaluates the stopping rate at
`(E_start + E_cut)/2` and defines the truncation distance as

$$
s_{\rm cut}
=\frac{E_{\rm cut}-E_{\rm start}}
{(dE/ds)((E_{\rm start}+E_{\rm cut})/2)}.
$$

This is a separate endpoint-average cutoff construction, not an inversion of
the explicit RK2 update. The frozen rule instead uses `(dE/ds)(E_start)`.
Because `|dE/ds|` grows as `E`
falls over the whole range above the cutoff, `|(dE/ds)(E_mid)| ≥ |(dE/ds)(E_start)|`
and therefore `s_cut(midpoint) ≤ s_cut(frozen)`: the frozen rule overshoots the
electron's true residual range.

That same inequality is what keeps the corrector well posed. For any accepted
step `s ≤ s_cut`,

$$
E_{\rm pred}
=E_{\rm start}+\frac{dE}{ds}(E_{\rm start})s
\ge E_{\rm start}
+\frac{(dE/ds)(E_{\rm start})}
{(dE/ds)((E_{\rm start}+E_{\rm cut})/2)}
(E_{\rm cut}-E_{\rm start})
\ge E_{\rm cut}.
$$

so the predictor never falls below the cutoff. The low-energy Joy--Luo branch
therefore stays inside its logarithm's supported region; the higher-energy
Berger--Seltzer branch is selected by the splice. No clamp is needed and none
is applied.

## Assumptions and limits of validity

- This row validates the deterministic `straggling=False` propagation rule.
  With straggling enabled, the Urban draw replaces the deterministic row loss;
  its mean is the same stopping-power integral, but its cutoff and substep
  semantics are validated separately by `energy-loss-straggling`.
- Elastic hazard remains frozen at `E_start`; the collision distance is still
  sampled from the start-energy mean free path. Only stopping and the clock are
  controlled here. Convergent hazard treatment is checklist step F of
  `feature/energy-controlled-electron-transport`.
- `|dE/ds|` monotone in `E` is used only for the cutoff-overshoot direction and
  the predictor bound above, and only over `E > E_cut`. The transport cutoff
  keeps the low-energy Joy--Luo branch away from its sign-changing logarithm;
  above the per-element crossover the Berger--Seltzer branch applies.
- The flight decomposition is untouched, so this is not yet the energy-limited
  substepping of checklist step F.

## Limiting cases

- `s → 0`: `E_mid → E_start`, so both the energy and clock updates reduce to the
  frozen rule. The two schemes' mutual difference vanishes as `s²`, not `s³`:
  it *is* the frozen rule's own local truncation term. `s³` is the midpoint
  rule's local error against the exact solution, which is the quantity the
  numerical evidence below measures.
- `dE/ds` constant in `E` (a hypothetical energy-independent stopping power):
  `(dE/ds)(E_mid) = (dE/ds)(E_start)` identically and the two schemes agree
  exactly, as does the clock when `β` is likewise constant.
- `energy_model="frozen"` (the default) is bit-for-bit identical to the
  pre-change core, pinned by
  `tests/montecarlo/test_transport_energy_model.py::test_frozen_energy_model_is_the_bit_for_bit_default`.

## Numerical evidence

One boundary-truncated carbon flight (`composition=[("C", 0.1136)]`,
`elastic_model="sr"`, `E0 = 25 keV`, `seed=11`), so that the row length is the
slab thickness exactly, compared against a 20 000-substep RK4 reference for both
`E(s)` and `t(s)`. Because one flight is one step, the quantity measured is the
*local* truncation error, which is one order higher than the global order.

| flight length | fractional loss | midpoint `E_end` err [keV] | frozen `E_end` err [keV] | midpoint clock err [Å] | frozen clock err [Å] |
|---|---|---|---|---|---|
| 600 Å | 0.51% | 5.785e-07 | 2.668e-04 | 1.161e-04 | 2.347e+00 |
| 300 Å | 0.25% | 7.199e-08 | 6.655e-05 | 1.511e-05 | 5.855e-01 |
| 150 Å | 0.13% | 8.978e-09 | 1.662e-05 | 1.925e-06 | 1.462e-01 |

Error ratios under halving: midpoint 8.04 / 8.02 (energy) and 7.69 / 7.85
(clock) against the expected 8 for an `O(s³)` local error; frozen 4.01 / 4.00
and 4.01 / 4.00 against the expected 4. At 600 Å the midpoint energy error is
0.22% of the frozen error and the midpoint clock error is 5e-05 of it.

Pinned by `tests/montecarlo/test_transport_energy_model.py`:
`::test_midpoint_end_state_error_is_third_order_in_flight_length`,
`::test_midpoint_end_energy_and_clock_beat_the_frozen_rule`,
`::test_midpoint_cutoff_flight_stops_exactly_on_the_energy_floor`,
`::test_midpoint_cutoff_distance_is_shorter_than_the_frozen_extrapolation`.

## What this row does not claim

- Nothing about emitted spectra. Line and bremsstrahlung kernels still read the
  flight-start energy; migrating them to an explicit representative energy and
  proving substep invariance are checklist steps D and G.
- Nothing measured here about the grooved, per-electron, or CUDA cores. The
  numerical evidence above is lockstep. Checklist step H has since ported the
  rule to those cores and dropped the fail-closed midpoint gates, so a midpoint
  request no longer raises on any of them.
- `_clip_segments_to_cutoff` now *reconstructs* `E_end_keV`/`E_repr_keV`/
  `t_end_ang` when it shortens a flight, by reapplying the core's own cutoff
  solve — which is exact, since that solve lands on `E_end == E_cut` by
  construction. It no longer drops them. That reconstruction, and the
  fact that the new fields are masked in step with the rows they belong to, are
  pinned by
  `::test_row_transforms_keep_the_new_fields_in_step_with_the_rows`.

## Independent verification

Fresh-context rederivation (2026-08-18, verifier context separate from the
implementation). Filters: units `pass` (`dE/ds` keV/Å, clock in Å with
$c=1$); signs/conventions `pass` ($dE/ds<0$, $E_{\rm end}<E_{\rm start}$);
limits `pass` ($s\to0$ and constant $dE/ds$ both collapse to the frozen rule).

The 2026-08-18 verification re-derived the scheme from the then-documented
Joy--Luo `dE/ds` and the clock integral before reading the implementation. The
explicit midpoint update
$E_{\rm pred}=E_{\rm start}+s\,f(E_{\rm start})$,
$E_{\rm mid}=(E_{\rm start}+E_{\rm pred})/2$,
$E_{\rm end}=E_{\rm start}+s\,f(E_{\rm mid})$
is algebraically the textbook explicit-midpoint (RK2) update
$E_{\rm end}=E_{\rm start}+s\,f\!\left(E_{\rm start}+\tfrac{s}{2}f(E_{\rm start})\right)$,
since $(E_{\rm start}+E_{\rm pred})/2=E_{\rm start}+\tfrac{s}{2}f(E_{\rm start})$.
Taylor expansion gives $O(s^3)$ local / $O(s^2)$ global truncation error in
$E_{\rm end}$ (frozen Euler: $O(s^2)$ local), and the clock rule
$\Delta t=s/\beta((E_{\rm start}+E_{\rm end})/2)$ independently expands to the
same $O(s^3)$ local order as the matching midpoint quadrature of
$\int ds'/\beta(E(s'))$. Solving $E_{\rm end}=E_{\rm cut}$ algebraically
reproduces
$s_{\rm cut}=(E_{\rm cut}-E_{\rm start})/(dE/ds)((E_{\rm start}+E_{\rm cut})/2)$
exactly, with no approximation — consistent with the code's cutoff branch
assigning `E_end_j = E_cut_e` directly rather than routing `s_cut` back
through the general predictor-corrector, which would *not* reproduce
`E_cut` exactly (the exactness is a construction property of the direct
assignment, not of the RK2 formula).

It then read `_transport_core_ungrooved` after the
derivation: `E_pred`/`E_mid`/`E_end`, `cutoff_distance`, and
`beta_j = beta_from_keV_scalar(0.5*(E_j+E_end_j))` match term for term. A
standalone reimplementation of `dE/ds` and $\beta$ from the governing
equations (independent of the transport implementation; $J=78$ eV,
`coeff=(0.1136/0.602214076)*6` for the ledger row's carbon case) against a
200000-step RK4 reference reproduced the ledger's numeric-evidence table to
3-4 significant figures without consulting it for the arithmetic beforehand
($E_{\rm end}$ errors 2.668e-4/6.655e-5/1.662e-5 keV frozen and
5.785e-7/7.199e-8/8.978e-9 keV midpoint at 600/300/150 Å; clock errors
2.347/0.586/0.146 Å frozen and 1.161e-4/1.511e-5/1.925e-6 Å midpoint; ratios
~4.0/4.0 and ~8.0/8.0), and confirmed
$s_{\rm cut}(\text{frozen})>s_{\rm cut}(\text{midpoint})$ from the same
monotone-$\lvert dE/ds\rvert$ trend the doc's predictor-boundedness argument
uses. `tests/montecarlo/test_transport_energy_model.py` (20 cases) passes.

One wording note, not a computed discrepancy: "Limiting cases" states that as
$s\to0$ "the difference between the two schemes vanishes as $s^3$." Read
literally this is imprecise — $E_{\rm end}(\text{midpoint})-E_{\rm end}(\text{frozen})$
is generically $O(s^2)$ (it reduces to the frozen rule's own local truncation
term), confirmed above by a ratio of 4.0 under halving, not 8.0. Each
scheme's own error against the true solution is $O(s^2)$/$O(s^3)$
respectively, which is what the numeric-evidence table and this verification
both correctly report; only the phrase describing the two schemes'
mutual difference is loose.

Verdict after the 2026-08-20 audit: `discrepancy`. The implementation and
numeric convergence evidence are consistent with explicit midpoint RK2, but
the prior ledger claim stated the implicit endpoint-average equation and the
write-up's independent numeric table used the retired pure Joy--Luo branch
rather than the current Joy--Luo/Berger--Seltzer splice. The corrected claim
requires fresh-context re-validation before returning to `rederived`;
`signed-off` remains a human decision.
