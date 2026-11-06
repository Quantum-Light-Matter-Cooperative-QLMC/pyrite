# `cr-axial-depth-budget`: source-relative budget arithmetic

**Status:** task-owner derivation (2026-11-05), awaiting fresh-context verification.
This claim covers how supplied bounds combine and when the guard refuses. It
contains no measured bound and no calibrated depth; those belong to
[#993](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/993).

## Relative source error

Fix one observable and a depth interval $(a,b]$. Let $Y_\mathrm{ref}(z)$ and
$Y_\mathrm{model}(z)$ be photon yields per incident electron. Suppose that with
joint confidence $c$, for every $z\in(a,b]$, $Y_\mathrm{ref}(z)\ge L>0$ and the
model error decomposes into channels with $|e_k(z)|\le B_k$. The triangle inequality
gives

$$
\frac{|Y_\mathrm{model}-Y_\mathrm{ref}|}{Y_\mathrm{ref}}
\le\frac{\sum_k|e_k|}{Y_\mathrm{ref}}
\le\frac{\sum_kB_k}{L}.
$$

The decomposition must be exhaustive for the first inequality to hold. Overlapping
channels only loosen the bound, but a missing channel breaks it. One-at-a-time probes
against one reference leave out interaction terms such as $e_1e_2$, so the supplier must
use chained probes, an all-on closure probe or an interaction allowance. The guard refuses
when the right side exceeds the tolerance.

## Paired probes

For bins $j$, reference means $r_j$ and a perturbed comparison $q_j$, with bounds
$s_j$ on the paired-difference estimate and $t_j$ on its drift across the interval,
each true bin difference satisfies $|\Delta_j|\le|q_j-r_j|+s_j+t_j$. Summing over
bins gives $B=\sum_j(|q_j-r_j|+s_j+t_j)$, which bounds the absolute error of the
integrated observable with no cancellation between bins. With one-sided allowances
$u_j,v_j$ on the reference, $L=\sum_j(r_j-u_j-v_j)$.

## Joint confidence

If interval $i$ holds with probability at least $c_i$, Boole's inequality bounds the
probability that any interval fails by $\sum_i(1-c_i)$, so all hold with probability at
least $1-\sum_i(1-c_i)$, with no independence assumption. Sampling risk $\delta_s$
and cap failure risk $\delta_c$ subtract from the source confidence the same way.

## Terminal-fate bound

Let $N$ independent unit-weight incidents each produce one terminal fate and depth.
For fate class $c$, $F_c(z)$ is the probability of a terminal event of class $c$ at
depth at most $z$, and $\hat F_c$ its empirical estimate. Each $\hat F_c(b)$ is a mean of
$N$ independent Bernoulli variables, so Hoeffding's one-sided bound gives

$$
P[F_c(b)-\hat F_c(b)\ge h]\le e^{-2Nh^2}.
$$

For $m$ interval stops and two classes, $2m$ events are bounded. Setting
$2m\,e^{-2Nh^2}=\delta$ yields

$$
h=\sqrt{\frac{\ln(2m/\delta)}{2N}}.
$$

$F_c$ is nondecreasing, so $F_c(z)\le F_c(b)$ for every $z\in(a,b]$, which makes the bound
uniform over the interval without interpolation. The union bound does not need
independence between depths, so reusing the same incidents at each stop is valid.
Multiplying $\min(1,\hat F_c(b)+h)$ by a per-fate cap $C_c$ bounds the incident-normalized
error when the cap holds for every fate of that class and a history with no class-$c$
event by $z$ has no error at $z$. That holds for yield at crystal thickness $z$, but not for
depth-resolved use where a later exit returns above $z$. A retained record must reach the
sampled maximum depth, or an incomplete history would understate $\hat F_c$.

## Limits

Zero channels give a zero bound. No observed events still leave $h\,C_c>0$. A requested
depth inside a gap refuses, because no bound covers it; a gap elsewhere does not affect a covered depth. The yield
of a cumulative observable vanishes at the entrance, so a useful first interval needs
$a>0$.

## Fresh-context verification (2026-11-05)

Scope: the ledger row only. The sections from "Independent derivation" through "Pre-code expectations" were written from the row before any implementation body, test body, task-owner derivation or research note of this claim was opened.

### Independent derivation

**Triangle bound.** Let $Y^{\rm ref}_b\ge 0$ be the reference photon source in spectral bin $b$ and $Y^{\rm app}_b$ the approximate one, both per incident over one depth interval. The error measure is

$$
\epsilon=\frac{\sum_b\lvert Y^{\rm app}_b-Y^{\rm ref}_b\rvert}{\sum_b Y^{\rm ref}_b}.
$$

The numerator step needs one premise: a telescoping identity $Y^{\rm app}_b-Y^{\rm ref}_b=\sum_k e_{k,b}$ with no missing term. Then $\lvert\sum_k e_{k,b}\rvert\le\sum_k\lvert e_{k,b}\rvert$ per bin, and if $\sum_b\lvert e_{k,b}\rvert\le B_k$ for every $k$,

$$
\epsilon\le\frac{\sum_k B_k}{L},\qquad 0<L\le\sum_b Y^{\rm ref}_b.
$$

*Exhaustiveness* (the identity) is necessary. *Disjointness* is not: the triangle inequality holds for any summands, and overlapping channels merely over-count. Disjointness matters only as a way to establish the identity, since an overlap correction would itself be a term of the identity. The direction of $L$ matters: it must bound the *reference* (the denominator) from below. A lower bound on the approximate yield would need the correction $L_{\rm app}-\sum_k B_k$.

**Per-bin paired bound.** A paired probe measures $\hat D_b$ with $\lvert D_b-\hat D_b\rvert\le u_b$ on its coverage event, so $\lvert D_b\rvert\le\lvert\hat D_b\rvert+u_b$ and

$$
\sum_b\lvert D_b\rvert\le\sum_b\bigl(\lvert\hat D_b\rvert+u_b\bigr).
$$

The modulus must be taken before the bin sum. $\lvert\sum_b\hat D_b\rvert$ would admit cancellation and is not an upper bound. For the reference, $\hat Y_b-u'_b\le Y^{\rm ref}_b$ and $Y^{\rm ref}_b\ge0$ give the valid lower bound $L=\sum_b\max(0,\hat Y_b-u'_b)$; without the clip the bound stays valid but is weaker. Each per-bin bound is an event: unless $u_b$ is a joint band, joint coverage over bins needs its own union term.

**Union-bound confidence.** With events $A_i$, $\Pr(\bigcap_iA_i)\ge1-\sum_i\Pr(A_i^c)$. So the confidence is $1-\sum_i(1-c_i)$ for **any** dependence between the events, floored at zero. Composition rule: the budget's risk is the sum over distinct events. If one event of risk $\delta$ supports $m$ intervals, it enters once with $\delta$. Charging $\delta$ per interval is conservative (double-counted) and charging less than $\delta$ in total is invalid (dropped). When a Hoeffding family with per-event risk $\delta/(2m)$ is split among $m$ intervals, each interval owns two events, risk $\delta/m$, and the intervals sum to exactly $\delta$.

**One-sided Hoeffding allowance.** For i.i.d. Bernoulli $X_j$ with mean $F$ and empirical mean $\hat F$, Hoeffding Theorem 1 gives $\Pr(F-\hat F\ge t)\le e^{-2Nt^2}$. Two fate classes at $m$ stops give $2m$ events. Setting $2m\,e^{-2Nt^2}=\delta$ gives

$$
t=\sqrt{\frac{\log(2m/\delta)}{2N}},
$$

which matches the row. It needs stops and $m$ fixed independently of the sample (data-chosen stops void the count), and no independence *between* fate classes, because Boole needs none. The upper bound clips to $\min(1,\hat F+t)$, which remains valid.

**Monotone uniformity.** With $F_f(z)=\Pr(\text{fate }f\text{ at depth}\le z)$, right-continuous and non-decreasing, $F_f(z)\le F_f(b)$ for every $z\in(a,b]$. So the stop bound at $b$ covers the whole interval with no extra event, including the cumulative value at $z\downarrow a$. The endpoint $b$ belongs to the interval, so an exit at exactly $z=b$ must count at stop $b$ ($z\le b$, not $z<b$). An exit at $z=0$ must count at every stop, since the surface outgoing fate has depth zero and lies in no half-open increment $(a,b]$ with $a\ge0$. A bound built from increments would drop it; a CDF bound retains it.

**Per-fate caps.** If every incident of class $f$ contributes at most $C_f\ge0$ to the interval's L1 error, conditional on terminating by the relevant depth, then

$$
\mathbb E[\text{L1 error from }f]\le C_f\,F_f(b)\le C_f\min\bigl(1,\hat F_f(b)+t\bigr).
$$

It is finite for $\hat F_f=0$ unless $C_f=0$, as the limiting case requires. A cap is a sup over the class, so the row's "hold uniformly per fate" is the required premise.

**Censoring.** For a record with terminal class $g$ and depth $z$, the indicator at stop $b$ is $\mathbb 1[g=f\wedge z\le b]$. A retained record has no terminal exit: it counts in $N$ and in neither exit CDF at any stop. If it stored depth $z_{\max}$ with an exit class it would be miscounted. All records must be observed through the stop, so $b\le z_{\max}$ for the common maximum depth; a stop above $z_{\max}$ leaves exits in $(z_{\max},b]$ unobserved and $\hat F(b)$ biased low, and must refuse. Dropping retained records from $N$ would inflate $\hat F$ conservatively but shrink the sample below its true count, which is an incorrect denominator. A record whose depth exceeds a stop is not an exit at that stop, and neither is one that exceeds $z_{\max}$; the latter contradicts the common-maximum premise and should refuse.

**Interval semantics.** Evidence on $(a_1,b_1],\dots,(a_m,b_m]$ with $a_{i+1}=b_i$ covers $(a_1,b_m]$. A requested depth $z$ is covered iff $a_1<z\le b_m$, so $z=a_1$ is not covered unless the evidence explicitly declares the entrance. A gap, an overlap or a non-increasing endpoint must refuse.

### Pre-code expectations

1. Refusal: tolerance exceeded, $L\le0$, non-finite values, a gap, wrong identity or under-confident joint evidence, all before any solver import or RNG use.
2. Counts use $z\le b$ with retained records in $N$ only.
3. Stops above the sampled maximum depth, or stops chosen after sampling, are the failure modes the premises must exclude.
4. Fate risk enters exactly once in total across intervals.

### Implementation comparison

Everything below was added after opening the implementation, at commit `1c1e001cd`. The owning code is `axial_depth_budget.py`, `axial_depth_measurements.py` and `axial_fate_source_bounds.py` in `src/pyrite/montecarlo/channeling/`.

| Item | Independent result | Code | Result |
| --- | --- | --- | --- |
| Relative bound | $\sum_kB_k/L$ | `relative_error_upper` | matches |
| Per-bin paired bound | $\sum_b(\lvert\hat D_b\rvert+s_b+t_b)$ | `_probe_upper` | matches |
| Reference lower bound | $\sum_b(\hat Y_b-u_b-v_b)$, unclipped form | `source_interval_bounds` | matches; valid, weaker than the clipped form |
| Interval confidence | $1-\sum_i(1-c_i)$ | `compile_axial_photon_depth_evidence` | matches |
| Fate composition | $c-\delta_s-\delta_c$, one charge in total | `add_physical_fate_source_bounds` | matches; nothing dropped or double-counted |
| Hoeffding allowance | $\sqrt{\ln(2m/\delta)/(2N)}$ | `physical_fate_source_uppers` | matches |
| CDF count | $z\le b$, cumulative, including $z=0$ | `outgoing_depths <= stop` | matches |
| Fate cap | $C_f\min(1,\hat F_f+h)$, $C_f$ the bin sum | `return_upper`, `unresolved_upper` | matches |
| Censoring | retained in $N$ only; must reach $z_{\max}$ | `_terminal_fate_depths` | matches |
| Stop above $z_{\max}$ | refuse | `stop > sampled_max_depth_ang` | matches |
| Interval membership | $a<z\le b$ | guard comparison | matches |
| Refusal order | before sampling | `sample_budgeted_physical_handoff` | matches |

No record is miscounted. The direct-count oracle used exits at $z=0$, at exactly a stop, $10^{-7}$ Å above a stop, retained records at a stop and an exit at $z_{\max}$. It gave outgoing counts 2, 3, 4 and unresolved counts 0, 1, 2 at the three stops, equal to the hand count.

**Correction to the pre-code text.** I wrote that data-chosen stops void the count. That holds for the cited Hoeffding-plus-Boole proof, but not for the value of the allowance. A Dvoretzky-Kiefer-Wolfowitz-Massart band gives ${\rm Pr}(\sup_zF-\hat F\ge h)\le e^{-2Nh^2}$ per class, so this $h$ remains valid with risk at most $\delta/m$ for any stops. I cite that bound from memory and did not check it against Massart (1990) in this pass. It is also why the implemented allowance is conservative by the factor $m$ inside the logarithm.

### Numerical checks

The oracles use exact binomial tails, analytic CDFs and direct counts, with no production helper. The scratch scripts were not retained, so the numbers here are the record.

| Check | Result |
| --- | --- |
| Exact binomial tail over the allowance for four $(N,m,\delta)$ settings | worst exact tail is $0.295$ of the per-event risk $\delta/(2m)$ |
| Production allowance, $N=200$, $m=3$, $\delta=0.2$, 1500 samples, dense depth grid | joint failure $0.0047$ |
| Same, $N=50$, $\delta=0.2$ | joint failure $0.0040$ |
| Same, $N=200$, $\delta=0.01$ | joint failure $0$ |
| Composition with five intervals, $c=0.97$, $\delta_s=0.01$, $\delta_c=0.002$ | $0.958$; bound terms equal the hand values |
| Real `sample_physical_handoff` records through `physical_fate_source_uppers` at $z_{\max}=0.05,\ 0.2,\ 1/3$ Å | accepted; retained depth equals $z_{\max}$ exactly; fractions equal direct counts; this fixture produced no unresolved fates |
| Guard refusal in a fresh process | `axial_outgoing_launch` and `axial_fiber_unraveling` stay unloaded |

Eager package initialization does load `axial_energy_mcwf`, which is independent of the guard. Importing `axial_fate_source_bounds` loads `axial_outgoing_launch` at module level, so "before solver import" holds only on the path that does not import the composer. "Before sampling" holds on every path.

### Findings

**High:** none. The arithmetic is correct and no record is miscounted.

**Medium 1: the gap clause is false as written (first divergence).** The row says the preflight refuses "gapped" evidence and that "a gap between intervals refuses". The code tests only $a_i<z\le b_i$ for the requested depth. With intervals $(100,200]$ and $(300,400]$, $z=350$ returns the second interval and only a request inside the gap refuses. `test_depth_intervals_are_uniform_and_are_not_interpolated_or_extrapolated` pins this acceptance through `require(ev, 900.0)`. In `test_gaps_and_independent_overbudget_intervals_remain_fail_closed` the request past the gap refuses only because that interval is over budget. The same wording appears in `axial-depth-budget-884.md` and `axial-depth-measurements-884.md`.

The behaviour is sound for a cumulative observable read at the requested depth, because that interval's bound covers $Y(z)$ directly. It gives no bound for a consumer scoring depth-resolved output inside the gap, although the sampler does emit records from those depths. A contiguity rule would also conflict with the required uncovered prefix $(0,a_1]$. I recommend correcting the wording rather than the code.

**Medium 2: the row names the wrong premise, and the producer does not satisfy the right one by itself.** The triangle step needs exhaustiveness, meaning the telescoping identity, and not disjointness. The row, the `AxialPhotonBoundInterval` docstring and the research note say disjoint, while `add_physical_fate_source_bounds` deliberately adds overlapping terms, which is valid.

The identity is also not delivered by `AxialPhotonPairedProbe`, whose contract is one perturbation against the SAME reference. For $Y=r(1+e_1)(1+e_2)$ the star of probes gives $r(e_1+e_2)$, and the omitted term is

$$
\boxed{r\,e_1e_2}.
$$

At $e_1=e_2=0.005$ the code reports $0.01000000$ while the true error is $0.01002500$, and the guard accepts at tolerance $0.01$. The row's supplier premise formally covers this, but neither it nor the task-owner derivation discloses it. #993 needs a chain of probes, an all-perturbations-on closure probe or an explicit interaction allowance.

**Medium 3: an unstated causal premise behind $F_c(b)$.** Multiplying by $F_c(b)$ requires zero error at $z$ from incidents with no class-$c$ event by $z$. The cap docstrings include "any return into the entrance zone" and allow return probability one, but an exit at $z'>b$ that returns to depth at most $b$ is not counted at stop $b$. The step holds for yield defined at crystal thickness $z$, and fails for depth-resolved yield in a thicker crystal unless the supplier excludes such returns or the cap covers them through the sampled maximum.

**Medium 4: test power.** 23 in-memory source mutants ran against the three anchor modules; the baseline was 57 passed.

| Surviving mutant | Consequence |
| --- | --- |
| `outgoing_depths <= stop` to `<` | the closed end of $(a,b]$ is unpinned in the count |
| `unresolved_depths <= stop` to `<` | same |
| `return_upper` replaced rather than added | `test_extra_bounds_are_additive_not_a_replacement_and_compose_confidence` asserts only `>` |
| solver import moved ahead of the guard | "before solver import" is unpinned; the tests assert call order only, and load the module first |
| guard `>` to `>=` | acceptance at exactly the tolerance is unpinned |

The other 18 are killed, including every allowance, risk, retained-record and lower-bound mutant.

The two coverage tests the task asked about are not discriminating oracles:

- `test_hoeffding_allowance_covers_true_cdfs_simultaneously_at_declared_rate` has an independent analytic truth but never calls production code, and it killed 0 mutants. Its assertion has little power: at its own settings the failure rate is about $0.005$ against the limit $0.2$, and even $h/2$ gives $0.142$, which still passes.
- `test_compiled_bound_contains_randomized_true_relative_error` killed 0 mutants, surviving removal of $s_b+t_b$ and of the reference allowance. It measures $\lvert\sum_be_b\rvert$, not the L1 quantity, and random signs leave large slack.

`test_code_allowance_matches_the_coverage_oracle_on_a_sampled_census` is meaningful: it uses an independent count and formula and killed 5 mutants. The deterministic hand-valued tests supply the rest of the pinning.

Suggested tests: an exit at exactly a stop; `return_upper` equal to the base plus the fate term; a fresh-process `sys.modules` check; a sign-aligned adversarial case that attains the bound with equality; and exact binomial tails evaluated at the allowance returned by production code.

**Low.**

- `_terminal_fate_depths` claims to validate "a complete incident census", but IDs `4..399` with all exits removed are accepted with $\hat F=0$. Only uniqueness is checked. The row's premise covers this; the docstring overclaims.
- No anchor passes real sampler records through `physical_fate_source_uppers` or `add_physical_fate_source_bounds`. They pass the former here, but only this review shows it, and neither route has seen a real unresolved record.
- The direct `AxialPhotonBoundInterval` constructor allows $a=0$, unlike both producers. The task-owner page notes a cumulative yield vanishes at the entrance, so a positive $L$ there is only a supplier assertion.
- The docstring of `test_channeling_axial_depth_budget.py` still says "unverified", which needs updating when the status changes.

### Verdict

`discrepancy`, limited to the gap clause; the arithmetic re-derives. Correcting the row needs no source change for the arithmetic, and the row then matches this derivation at `rederived`. I do not recommend `anchored` until the Medium 4 survivors are closed. Status is left for the task owner; human sign-off stays in #277.
