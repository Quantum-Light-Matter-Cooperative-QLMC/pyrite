# Transverse flat-term omission certificate

Validation: `coherent-transverse-flat-omission`.

## Claim and source

On a finite footprint with a recorded Gaussian spot, the reducer blends each
reflection/orientation row $r$ with

$$
F_r(\omega)=F_z(\omega)\,F_{\perp,r}(\omega),\qquad
F_z=e^{-(\omega\sigma_z)^2},\qquad
F_{\perp,r}=e^{-q_r(\omega)},\quad
q_r=\mathbf{K}_r^{\mathsf T}\Sigma_f\mathbf{K}_r ,
$$

with $\mathbf{K}_r=\omega\mathbf{a}+\mathbf{g}_{r,xy}$ and
$\mathbf{a}=(\hat{\mathbf n}-\mathbf b/\beta)_{xy}$. The factor law and its
eligibility belong to
[`transverse-bunch-form-factor`](../radiation-physics/transverse-bunch-form-factor.md)
and `finite-footprint-longitudinal-decoherence`. The omission inequality is
[`coherent-flat-term-omission`](coherent-flat-term-omission.md):
for any factor $F\in[0,1]$, replacing the physical estimator by the grouped
floor changes the row by at most $F(N-1)G/M$.

That inequality uses only $0\le F\le1$. Certifying with $F_z$ alone is
therefore valid but conservative by $F_\perp\le1$. This claim certifies with an
upper enclosure of the product the reducer actually uses. It changes no field,
factor, transport, RNG stream, population weight or omission share $\ell$.

## Derivation

Let $W\ge\alpha_*(M-1)$ be the existing outward pair weight. An evaluated
coordinate may be omitted for row $r$ when some $U_r\ge F_r(\omega)$ satisfies
$U_r W\le\ell$; a coordinate is omitted only when every row passes. The
row-sum inference for nonnegative row weights is unchanged.

**Pointwise enclosure.** Treat the binary64 inputs
$\omega,\ \Sigma_f,\ \mathbf a,\ \mathbf g_{xy}$ as exact, formed exactly as in
`transverse_form_factor` (its off-diagonal $\Sigma_{f,01}$ multiplies both
cross terms). Evaluate $q$ with 50-digit interval arithmetic, clamp its lower
endpoint at zero like the production PSD guard, and round
$e^{-q_{\rm lo}}$ upward. Positive underflow keeps the smallest positive
binary64 value, never a certified zero. For an interval
$[\omega_{\rm lo},\omega_{\rm hi}]$ the same evaluation encloses
$\sup F_\perp$ over the interval (interval dependency only loosens it).

**Monotone sides.** For these exact inputs $q$ is a real quadratic with
$q'(\omega)=2\,\mathbf a^{\mathsf T}\Sigma_f(\omega\mathbf a+\mathbf g_{xy})$,
affine in $\omega$. If interval evaluation certifies $q'>0$ at $\omega_i$ and
$\omega_j>\omega_i$, then $q'>0$ throughout $[\omega_i,\omega_j]$ for any affine
function, whatever the sign of its slope. There $F_\perp=e^{-\max(q,0)}$ is
nonincreasing. $F_z$ is nonincreasing on $\omega\ge0$, so the product is too:
a certified $U(\omega_k)W\le\ell$ at one coordinate covers every coordinate in
$[\omega_k,\omega_j]$. If instead $q'<0$ is certified at both ends of
$[\omega_i,\omega_j]$, $F_\perp$ is nondecreasing; with $F_z\le1$,
$F_r(\omega_m)\le F_\perp(\omega_m)\le F_\perp(\omega_k)$ for $m\le k$. Only
$F_\perp$ is used on that side.

On an ascending grid this gives bisection: each claim covers coordinates
beyond a coordinate whose own directed bound was evaluated and passed. The
floating minimizer $-\mathbf a^{\mathsf T}\Sigma_f\mathbf g_{xy}/
\mathbf a^{\mathsf T}\Sigma_f\mathbf a$ only seeds the split; certified slope
signs decide whether either side is used. Undecided signs certify nothing.

**Combination.** The longitudinal certificate is retained unchanged. The
transverse rules examine only coordinates it keeps. Each certifies a subset of
coordinates for which a valid upper bound on $F_r$ passes, so their union with
the longitudinal set satisfies the same per-row error bound.

## Units, limits and signs

$\omega$ and $\mathbf K$ are in Å$^{-1}$, $\Sigma_f$ in Å$^2$; $q$, $F$, $W$
and $\ell$ are dimensionless. Limits:

- $\Sigma_f\to0$: $F_\perp=1$; only the longitudinal certificate can omit.
- Phase matching $\mathbf K_r(\omega_0)=0$: $F_\perp(\omega_0)=1$; the slope
  interval contains zero and the resonance is kept.
- $\mathbf a=0$: constant $F_\perp=e^{-\mathbf g^{\mathsf T}\Sigma_f\mathbf g}$;
  slopes are undecided, and the whole-range enclosure alone certifies.
- $\mu$m–mm spots: $q\gg1$ away from phase matching; one whole-range
  enclosure certifies the entire row.
- Short bunch, $F_z\to1$: omission follows $F_\perp$, where the
  longitudinal certificate previously omitted nothing.

## Scope

- Encloses the real-valued production law at binary64 inputs. Backend casting
  of $F_\perp$ to the working precision, the $F_zF_\perp$ product and
  reduction roundoff require the separate working-precision allowance stated
  for `coherent-flat-term-omission`. $F_\perp$ is evaluated at the reducer's
  own `omega_grid` values; $F_z$ keeps its existing energy-grid enclosure.
- Requires the eligibility already enforced before the reducer: recorded
  spot, face-arrival delay, uncorrelated Twiss, no energy spread on a tilted
  face, every electron on the footprint, $6\sigma$ footprint support.
- Rows without a spot or row vector, non-ascending or negative $\omega$ grids,
  nonfinite inputs and helper refusals retain the longitudinal result.
- Certifies evaluated coordinates only, against the grouped per-electron
  floor. It retains every complete per-electron field: no intra-electron,
  flight or region boundary is introduced, so no cross-boundary term arises.
- Uses the existing $\ell$; it spends no extra budget and makes no
  continuous-bin, centroid or relative-error claim.

## Implementation map

`coherent_transverse.transverse_form_factor_upper` and `transverse_slope_sign`
evaluate the pointwise/interval enclosure and certified $q'$ sign.
`_per_hkl._flat_energy_keep` applies `_transverse_certified` to the
coordinates its longitudinal bisection keeps; `_row_transverse_certified`
performs the whole-range check, then one bisection per certified monotone side.
All routes reach it through `_flat_energy_keep` with their row vectors.

## Anchors

`tests/montecarlo/test_coherent_transverse_omission.py`:

- The directed bound encloses an independent 60-digit evaluation and the
  production value, stays tight up to rounding, and the interval form bounds
  the sampled supremum.
- Slope signs match the quadratic and abstain at exact phase matching.
- A phase-matched row omits on both sides and keeps the resonance. Every
  omitted node satisfies the exact certificate; kept nodes are uncertified up
  to threshold rounding. Without the spot, nothing is omitted.
- No row vector or a point spot retains the longitudinal mask; one failing row
  retains its coordinates for all rows; a constant row certifies by the whole
  range.
- On identical offset-free tilted tracks with a $10^{-12}$ fs bunch, exact and
  sinc-windowed batched spectra reproduce $G[1+(N-1)F_\perp]$. They omit only
  certified nodes, equal the full reducer elsewhere, and stay within
  $\ell G$. Disabled omission is bit-identical to the full reducer.

## Status

Owner checks: units, limits and signs pass. The fresh-context verification
below found one binary64 discrepancy, which the owner has resolved; the ledger
row records the current status.

## Fresh-context verification (2026-10-09)

A fresh context verified commit `1e94883a`. It wrote the derivation below from
the ledger row, the two source write-ups and the helper docstrings, before
reading any implementation body.

### Independent derivation

Fix a row $r$ and an evaluated coordinate $i$. The bound in
`coherent-flat-term-omission` holds for any blend factor $F\in[0,1]$. The
reducer uses $F_r(\omega_i)=F_z(\omega_i)F_{\perp,r}(\omega_i)$, so the bound
gives $\lvert\Delta_r\rvert\le F_rW\,G_r/M$. The coordinate is therefore safe
for row $r$ whenever some $U\ge F_r(\omega_i)$ satisfies $UW\le\ell$. Summing
over rows with nonnegative weights needs every row to pass. The longitudinal
certificate $U_z\ge F_z\ge F_zF_\perp$ does not depend on the row. The
omitted set is $\{i:\ \mathrm{long}(i)\ \lor\ \forall r\ \mathrm{trans}_r(i)\}$,
which is the same as $\forall r\,[\mathrm{long}(i)\lor\mathrm{trans}_r(i)]$.
Both conditions are sound.

With $\mathbf K=\omega\mathbf a+\mathbf g_{xy}$,

$$
q(\omega)=\omega^2\,\mathbf a^{\mathsf T}\Sigma_f\mathbf a
+2\omega\,\mathbf a^{\mathsf T}\Sigma_f\mathbf g_{xy}
+\mathbf g_{xy}^{\mathsf T}\Sigma_f\mathbf g_{xy},\qquad
\tfrac12 q'(\omega)=\omega\,\mathbf a^{\mathsf T}\Sigma_f\mathbf a
+\mathbf a^{\mathsf T}\Sigma_f\mathbf g_{xy}.
$$

The function $\tfrac12q'$ is affine. If it is positive (or negative) at
$\omega_1<\omega_2$, every convex combination keeps that sign. This holds for
either sign of the slope, so it also survives a non-PSD $\Sigma_f$ produced by
rounding. The map $q\mapsto e^{-\max(q,0)}$ is nonincreasing:

- If $q'>0$ on $[\omega_1,\omega_2]$, then $F_\perp$ is nonincreasing there.
  $F_z=e^{-(\omega\sigma)^2}$ is nonincreasing for $\omega\ge0$, so the
  product is nonincreasing. A passing directed bound of the product at
  $\omega_j$ covers every $\omega_k\in[\omega_j,\omega_2]$.
- If $q'<0$ on $[\omega_1,\omega_2]$, then $F_\perp$ is nondecreasing. Because
  $F_z\le1$, $F_r(\omega_k)\le F_\perp(\omega_k)\le F_\perp(\omega_j)$ for
  $\omega_k\le\omega_j$. A passing bound of $F_\perp$ alone at $\omega_j$ covers
  every $\omega_k\in[\omega_1,\omega_j]$.
- A whole-range lower bound $q_{\rm lo}\le\min q$ gives
  $e^{-\max(q_{\rm lo},0)}\ge F_\perp\ge F_r$ at every coordinate.

The predicate need not be monotone for bisection to be sound. The only
requirement is that every claimed coordinate lies on the monotone side of an
evaluated coordinate whose own directed bound passed.

Limits: $\Sigma_f\to0$ gives $q\equiv0$, so slopes are undecided and the
whole-range $U=1$ omits only when $W\le\ell$. Exact phase matching
$\mathbf K(\omega^*)=0$ gives $q'(\omega^*)=0$ and $F_\perp=1$, so the node is
kept. With $\mathbf a=0$, $q$ is constant, slopes are undecided, and only the
whole-range bound $e^{-\mathbf g^{\mathsf T}\Sigma_f\mathbf g}$ applies. For
large spots $q\gg1$ and the bound underflows to the smallest positive value.
For $F_z\to1$ the transverse rule alone decides. Units: $\mathbf K$ is in
Å$^{-1}$ and $\Sigma_f$ in Å$^2$, so $q$ is dimensionless.

### Comparison with the implementation

The derivation matches `coherent_transverse.py:179-248` and
`lines/_per_hkl.py:255-441` term for term, with one exception described under
the discrepancy below.

- `_interval_exponent` uses the same binary64 $\mathbf a$
  (`n_hat[:2] - beam_xy_over_beta`), $\mathbf g_{xy}$ and $\Sigma_{f,01}$ for
  both cross terms as `transverse_form_factor` (lines 99-105).
  `half_slope` equals $\tfrac12\,dq/d\omega$ exactly. mpmath interval `kx**2`
  is a true square: $[-1,2]^2=[0,4]$, whereas `x*x` gives $[-2,4]$. The lower
  endpoint is clamped at zero, matching the production `maximum(exponent, 0)`.
  The bound is rounded upward and moved outward at subnormals.
- $\omega$ is `st.omega_grid` cast to float64, the same values the reducer
  passes as `omega64` (line 164). The row vectors are the same
  `G[i]`/`g_vec_d` arrays given to `_row_decoherence_factor` on all four
  routes. Ascending, nonnegative and finite $\omega$ is checked (line 358).
- Bisection invariants hold. On the descending side (lines 419-426), `hi` is
  either $n$ or an index that was evaluated and passed, and the claim
  `out[lo:]` lies within indices `start` through $n-1$, where the certified signs at
  `start` and $n-1$ are both $+1$. On the ascending side (lines 433-440), a
  final `lo` $>0$ means `lo-1` passed, and the claim `out[:lo]` lies within indices $0$
  through `stop-1`, where the signs at $0$ and `stop-1` are both $-1$. The
  side-nudging loops (lines 409-411 and 429-431) move at most two steps. Each
  loop re-checks the end sign before any claim, so neither has an off-by-one
  that affects soundness. The two sides cannot overlap, since the
  descending `lo` is at least `start`, which is at least `split`, which is at least `stop`.
- Coordinates that reach this certificate have already passed the existing
  check on the actual production `F` stack (lines 287-296), followed by the
  `any(axis=0)` row union.

### Discrepancy: product underflow becomes a certified zero

The first divergent term is at `lines/_per_hkl.py:416-417` together with
`passes` (line 386):

```text
product = upper_z * perp(k)
return passes(product if product == 0.0 else np.nextafter(product, np.inf))
...
return bound == 0.0 or np.nextafter(bound * weight, np.inf) <= limit
```

Each factor is a positive directed bound, but their binary64 product can round
to $0$. A product of $0$ is not an upper bound on $F_zF_\perp>0$, which
contradicts the claim's $U_r\ge F_r$ and the helpers' rule of never returning a
certified zero. Counterexample: $E=1000,1001,1002$ eV; $\sigma$ chosen so that
$(E_0\sigma/\hbar c)^2=743.5$; $\Sigma_f=\mathrm{diag}(1.6/\omega_0^2,0)$;
$\mathbf a=(1,0)$; $\mathbf g=0$; $W=10^{10}$; $\ell=10^{-318}$.
`_row_transverse_certified` returns `[False, True, True]`. The true $FW$ at
the two certified nodes is $5.75\times10^{-315}$ and
$1.29\times10^{-315}$, both above $\ell$. The float64 production $F$ there is
exactly $0$, so the first-stage check also passes and the nodes are omitted.
A violation needs $\ell<2.5\times10^{-324}\,W$, which is a subnormal-scale
limit. `coherent_flat_omission_limit` sets no lower bound on $\ell$. The
default $\ell=10^{-4}$ and every practical limit are unaffected. A fix is for
the owner to decide: rounding the product outward unconditionally
(`np.nextafter(0, inf)` is the smallest positive binary64 value) would restore
$U\ge F$.

### Numerical checks (scratch, not committed)

- 400 random rows used spots of 1 nm to 100 µm (including degenerate PSD),
  random tilt $\mathbf b_{xy}/\beta$ and $\hat{\mathbf n}$, and grids of
  5–400 nodes. Rows were phase matched at a node, phase matched between
  nodes, constant ($\mathbf a=0$) or generic, with $\sigma\in\{0\}\cup[10^{-3},10^2]$
  Å, $W\in[1,10^{12}]$ and $\ell\in[10^{-8},10^{-2}]$. Of 83,084 coordinates,
  67,541 were certified. A separate 80-digit evaluation of
  $F_zF_\perp$ at the same binary64 inputs found 0 violations of $FW\le\ell$.
- 300 adversarial rows placed the minimizer anywhere inside or just outside a
  1–60-node grid, with the threshold $q$ at a random node; 1/7 of them used a
  slightly indefinite $\Sigma_f$. There were 0 violations, and 6,169
  coordinates were certified against 6,181 that truly pass.
- The anchors `test_coherent_transverse_omission.py` and
  `test_coherent_flat_omission.py` pass 95/95 on CPU.

### Residual scope

The enclosure covers the real-valued law at binary64 inputs. The first-stage
check on the actual production `F` bounds the realized factor. Casting to
REAL, the backend product and reduction roundoff stay under the existing
working-precision allowance. CUDA runtime was not exercised. Eligibility
(recorded spot, footprint support) belongs to `transverse-bunch-form-factor`.

### Verdict

Units, limits and signs pass. The mathematics matches. The verdict is
**discrepancy**, confined to the binary64 product-underflow corner above.
Human sign-off is not assigned here.

### Owner resolution (2026-10-09)

`passes` now always rounds `bound * weight` outward, and the descending
product always rounds `upper_z * perp` outward, with no zero shortcut. Every
directed factor is positive, so an underflowed product becomes the smallest
positive binary64 value, never a certified zero. A regression,
`test_underflowed_bound_product_is_never_a_certified_zero`, reproduces the
counterexample above: it fails on `1e94883a` and passes after the fix.
The fix was re-verified below.

## Fresh-context re-verification (2026-10-09, `619a8ab3`)

A second fresh context checked the fix. `git diff 1e94883a 619a8ab3 -- src`
touches only `passes` and `descending` in `_row_transverse_certified`.

- **Rounding.** For finite $x\ge0$, the successor of the round-to-nearest
  value of $x$ is at least $x$. This includes $x$ that underflows to $0$,
  whose successor is $2^{-1074}$. Every factor is a positive directed bound:
  `transverse_form_factor_upper` returns a value in $[2^{-1074},1]$ (lines
  224 and 230) and the Gaussian upper bound is positive. Hence
  `nextafter(upper_z*perp)` $\ge F_zF_\perp$ (line 418), and
  `nextafter(bound*weight)` $\ge UW\ge F_rW$ (line 388), for any $W>0$,
  including $W<1$ such as $N=1.5$, $M=6$ ($W=0.5000000000000001$). The
  ascending side and the whole-range check use `perp` directly (lines 397
  and 437). Both directed helpers keep the subnormal outward loop and the
  $q_{\rm lo}\ge1000$ shortcut to $2^{-1074}$. No path returns a certified
  zero.
- **Nonfinite and overflow.** The limit is finite and nonnegative
  (`_setup.py:558`), and the transverse path runs only for `weight > 0`.
  Since bounds are at most 1, an overflowing $UW$ gives `inf`, which fails.
  Nonfinite $\omega$, $\Sigma_f$, $\mathbf a$ or $\mathbf g$ raise
  `ValueError`, which `_transverse_certified` turns into "certify nothing"
  (line 368). A NaN curvature only reseeds the split.
- **Numerics.** The stored counterexample now returns
  `[False, False, False]`. In 1,500 randomized rows (38,112 coordinates),
  limits came from the true $FW$ at a node, a subnormal, or $2^{-1074}$;
  $q$ and $(E\sigma/\hbar c)^2$ were driven to 650–760; 35% of rows had
  $W<1$. Against an 80-digit reference, 10,810 certified coordinates gave 0
  violations, and 4,468 helper bounds gave 0 violations. The same harness
  with the `1e94883a` row function finds 18 violations on seed 1, so it can
  detect the defect.
- **Anchors.** `test_coherent_transverse_omission.py` and
  `test_coherent_flat_omission.py` pass 96/96 on CPU.

Verdict: **rederived**. Residual scope as above: working-precision casting
and CUDA runtime fall outside this certificate. Human sign-off is not
assigned here.
