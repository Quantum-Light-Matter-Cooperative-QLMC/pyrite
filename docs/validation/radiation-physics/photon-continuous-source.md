# Continuous weighted photon source

Validation: `photon-continuous-source`.

## Independence and scope

Fresh-context derivation on 2026-10-22, written before inspecting implementation
bodies. Inputs were the ledger claim, owner signatures/docstrings,
[ADR-0020](../../adr/0020-photon-transport-source-and-scoring.md), and the
[BremsLib angular convention](bremslib-angular-model.md). The claim concerns
integration of a prescribed straight-segment source, not physical emission
events, primary recoil, full-track accuracy, or the photon interaction queue.

## Independent derivation

Let segment $i$ have length $L_i$ in Å, fixed representative kinetic energy
$T_i$ in keV, and unit tangent $\hat{\mathbf v}_i$. Its supported photon-energy
width is $\Delta k_i=(\min(k_{\rm ceiling},1000T_i)-k_{\rm cutoff})_+$ in eV.
For arclength $s\in[0,L_i]$, the source measure is

$$
d\nu_i=j_i(k,\Omega)\,ds\,dk\,d\Omega.
$$

The intended source is constant in position within each representative segment.
For any linear response $R$, including an indicator of a source phase-space
region, its mean is $\sum_i\int j_i R\,ds\,dk\,d\Omega$.
Choose $A=\sum_i L_i\Delta k_i$ and $p_i=L_i\Delta k_i/A$ over active segments.
Draw segment $i$, uniform arclength, uniform energy in its own interval, and an
independent isotropic direction. The density with respect to the joint measure
is

$$
q(i,s,k,\Omega)=p_i\frac{1}{L_i}\frac{1}{\Delta k_i}\frac{1}{4\pi}
=\frac{1}{4\pi A}.
$$

The importance-sampling identity then requires each of $M$ trials to carry

$$
\boxed{w=\frac{j_i(k,\Omega)}{Mq(i,s,k,\Omega)}
=\frac{4\pi A}{M}j_i(k,\Omega).}
$$

Evaluation must retain joint energy/direction dependence. Sampling an angular
marginal followed by an unrelated energy marginal would generally fail for a
nonseparable DDCS. Uniform position has density per arclength, not per volume;
there is no transverse-volume Jacobian. No primary denominator appears here.

BremsLib's physical DDCS is $d^2\sigma_a/(dk\,d\Omega)$ in
cm$^2$/(eV sr), evaluated at
$\theta=\arccos(\hat{\mathbf v}_i\cdot\hat{\mathbf n})$.
Because $1\,{\rm cm}^2=10^{16}\,{\rm Å}^2$ and the composition density
$n_a$ is atoms/Å$^3$,

$$
j_i(k,\Omega)=\sum_a n_a10^{16}
\frac{d^2\sigma_a}{dk\,d\Omega}
\quad[\mathrm{photons}/(\mathrm{Å}\,\mathrm{eV}\,\mathrm{sr})].
$$

Thus $A$ has units Å eV and $w$ is a photon count. The BremsLib angular shape
already integrates to its parent SDCS; no additional physical $4\pi$ belongs
in $j_i$. The $4\pi$ in the weight comes only from the isotropic proposal.

## Cheap filters

Weights are nonnegative for a nonnegative source. Constant $j$ makes every
trial have weight $4\pi Aj/M$, so their sum is exactly $4\pi Aj$.
Zero length or zero energy width supplies zero measure. Zero-emissivity trials
may be discarded only while retaining the original $M$ in all surviving
weights. Full-sphere sampling preserves photons initially outside the aperture.
No attenuation factor or extra primary energy debit is part of this measure.

## Source-to-code comparison

After writing the derivation above, inspection of `sample_continuous_photons`
and `_segments` found exactly $L_i\Delta k_i$ proposal masses, uniform
arclength offsets about each midpoint, uniform supported energies, and
uniform cosine and azimuth over the full sphere. `weight` is exactly
`rate * (4*pi*total/n_samples)`. The denominator remains the original trial
count when zero-weight rows are omitted. Primary identifiers are retained;
there is no division by their population in the source.

`sample_bremslib_continuum` evaluates the dot product of each sampled direction
with its own segment tangent and passes its own sampled energy to
`evaluate_bremslib_rows`. Explicit segment material labels select the atomic
densities. `_CM2_TO_ANG2` is $10^{16}$, with no extra isotropic factor or
attenuation. The existing validated BremsLib DDCS convention is inherited;
the library model itself was not revalidated here. Unsupported active incident
energies fail closed. The source returns a separate weighted batch and does
not mutate or debit the charged track.

An independent constant-source probe used lengths 2 and 3 Å, respective energy
widths 2 and 3 eV, emissivity 0.25 photons/(Å eV sr), and 101 proposals.
Both the analytic integral and returned total were 40.840704496667314 photons.

## Verdict

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches; no divergent factor, joint-measure Jacobian,
  angular convention, unit conversion, or primary denominator found.
- **Verdict:** `rederived`.
- **Suggested ledger change:** set `photon-continuous-source` to `rederived`,
  replace the pending note with this fresh-context comparison and link this page.
- **Human sign-off:** pending.

## Verification 2026-10-24: bremsstrahlung defensive-mixture proposal

Fresh-context verification of commit `c2cb361a`, which added
`proposal="bremsstrahlung"` to `sample_continuous_photons` (helper
`_sample_bremsstrahlung_proposal`) and made it the default of
`sample_bremslib_continuum`. The uniform/isotropic path derived above is
unchanged. The target measure $d\nu_i=j_i(k,\Omega)\,ds\,dk\,d\Omega$ and the
BremsLib emissivity conversion are inherited from the derivation above; this
section derives and checks only the new proposal. The forward-density
normalization and inverse CDF were derived before the helper body was read.

### Independent derivation

**Forward direction density.** Take $\mu=\hat{\mathbf v}_i\cdot\hat{\mathbf n}$
and $d\Omega=d\mu\,d\phi$. Since

$$
\int_{-1}^{1}\frac{d\mu}{(1-\beta\mu)^2}
=\frac{1}{\beta}\left[\frac{1}{1-\beta\mu}\right]_{-1}^{1}
=\frac{1}{\beta}\left(\frac{1}{1-\beta}-\frac{1}{1+\beta}\right)
=\frac{2}{1-\beta^2},
$$

$f(\mu)=(1-\beta^2)/[2(1-\beta\mu)^2]$ is a normalized density on $[-1,1]$
for every $0<\beta<1$. With uniform azimuth, the per-steradian density is
$f(\mu)/(2\pi)=(1-\beta^2)/[4\pi(1-\beta\mu)^2]$, which is the stated forward
term. Its CDF is

$$
F(\mu)=\frac{1-\beta^2}{2\beta}\left(\frac{1}{1-\beta\mu}-\frac{1}{1+\beta}\right).
$$

Setting $F(\mu)=u$ and $X\equiv 1/(1-\beta\mu)$ gives

$$
X=\frac{2\beta u}{1-\beta^2}+\frac{1}{1+\beta},\qquad
\mu=\frac{1-1/X}{\beta}.
$$

The endpoints follow directly. For $u=0$, $X=1/(1+\beta)$ and $\mu=-1$. For
$u=1$, $X=(2\beta+1-\beta)/(1-\beta^2)=1/(1-\beta)$ and $\mu=1$. The median
is $\mu=\beta$. As $\beta\to0$, $f\to1/2$, which is isotropic.

**Mixtures.** Given segment $i$, the energy proposal is
$q_E=a\,q_{\log}+(1-a)\,q_{\rm uni}$. Here
$q_{\log}=1/(k\ln(k_{\max,i}/k_{\min}))$ is sampled by
$k=k_{\min}(k_{\max,i}/k_{\min})^u$, and
$q_{\rm uni}=1/\Delta k_i$. Each component integrates to 1 on
$[k_{\min},k_{\max,i}]$. The direction proposal is
$q_\Omega=(1-b)/(4\pi)+b f(\mu)/(2\pi)$. Choosing a component with a
Bernoulli draw and then sampling within it reproduces the mixture density.
The weight must then use the full mixture density, not the component density
(one-sample balance-heuristic form). Energy and direction are drawn
independently given $i$. Position is uniform on $[0,L_i]$. The joint proposal
density with respect to $ds\,dk\,d\Omega$ is therefore

$$
q(i,s,k,\Omega)=\frac{p_i}{L_i}\,q_E(k)\,q_\Omega(\mu),
$$

and the importance-sampling weight for $M$ trials is

$$
\boxed{w=\frac{j_i(k,\Omega)}{M\,q}=\frac{L_i\,j_i(k,\Omega)}{M\,p_i\,q_E(k)\,q_\Omega(\mu)}.}
$$

**Support.** $q_E\ge(1-a)/\Delta k_i>0$ on the whole supported energy
interval. $q_\Omega\ge(1-b)/(4\pi)>0$ on the whole sphere. $p_i>0$ exactly
when $L_i>0$ and $k_{\max,i}>k_{\min}$, which are the segments with nonzero
target measure. The proposal therefore dominates the target, so
$\mathbb E[\sum w]=\sum_i\int j_i\,ds\,dk\,d\Omega$ for any nonnegative
$j_i$, with no factorization assumption on $j_i$.

**Bounds.** Relative to the uniform proposal's energy factor $\Delta k_i$,
$1/q_E\le\Delta k_i/(1-a)$. Relative to its angular factor $4\pi$,
$1/q_\Omega\le4\pi/(1-b)$. These are the stated $1/(1-a)$ and $1/(1-b)$
bounds, giving at most $16$ for $a=b=0.75$. The segment factor
$p_i^{\rm uni}/p_i$, where
$p_i^{\rm uni}/p_i=(\Delta k_i/\ln(k_{\max,i}/k_{\min}))\cdot
(\sum_jL_j\ln(k_{\max,j}/k_{\min}))/(\sum_jL_j\Delta k_j)$, is a fixed
per-segment ratio and is not covered by these bounds. The contract claims
bounds only for the energy and angle factors, so this is not a discrepancy.

**Units and normalization.** $L_i$ [Å] $\times$ $j_i$
[photons/(Å eV sr)] $/(q_E$ [1/eV] $q_\Omega$ [1/sr]$)$ gives photons per
trial set, the same as the uniform path's $4\pi A j_i/M$. $p_i$ is
dimensionless. No $n_{\rm primaries}$ appears; per-primary normalization
stays deferred to the detection boundary.

### Source-to-code comparison

The helper body was read after the derivation.

- **Segment probability:** `selection = length * log_span`, with `log_span`
  set only where the uniform mass is positive. This gives $p_i$ as derived,
  and `probability = selection[choice]/total` enters the weight.
- **Energy:** a Bernoulli `from_log` draw with share $0.75$ and one uniform
  `u` feed either branch:
  `cutoff*exp(u*ratio)` or `cutoff + u*span`. Reusing `u` across the
  exclusive branches is valid. `np.clip(energy, cutoff, upper)` only removes
  floating-point overshoot. Since $u\in[0,1)$, both branches are already in
  range in exact arithmetic. `energy_density` is evaluated at the clipped
  energy and equals $q_E$.
- **Direction:** a Bernoulli `forward` draw with share $0.75$, then either the
  inverse CDF above or $2u-1$. The code uses
  `inv_gamma2` $=1/\gamma^2$ for $1-\beta^2$, which avoids cancellation, with
  $\gamma=1+T/m_ec^2$ and $m_ec^2=510.99895$ keV. $\beta$ comes from
  `E_repr_keV` when present, else `E_keV`, matching the energy used for the
  segment's support and BremsLib evaluation. The cosine clip to $[-1,1]$
  affects only rounding. `angle_density` equals $q_\Omega$.
- **Frame:** the helper vector is $\hat{\mathbf x}$ unless
  $\lvert v_x\rvert\ge0.9$, in which case it is $\hat{\mathbf y}$, with
  $\lvert v_y\rvert\le0.436$. Gram-Schmidt then gives $\mathbf e_1$, and
  $\mathbf e_2=\hat{\mathbf v}\times\mathbf e_1$ completes a right-handed
  orthonormal frame. Then
  $\hat{\mathbf n}=\mu\hat{\mathbf v}+\sqrt{1-\mu^2}(\cos\phi\,\mathbf e_1+\sin\phi\,\mathbf e_2)$
  has $\hat{\mathbf n}\cdot\hat{\mathbf v}=\mu$ and uniform azimuth. The
  final renormalization changes only rounding. The BremsLib `emissivity`
  closure computes its cosine from the same `v_hat`.
- **Weight:** `rate * length / (n_samples * probability * energy_density * angle_density)`
  matches the boxed expression. `_weighted_batch` drops zero-weight trials
  without changing $M$.
- **$\beta>0$:** any selected segment has $T_i\cdot10^3>k_{\min}>0$, so the
  division by $\beta$ cannot be singular.

### Numerical checks

Scratch scripts, outside the repository, call only the public
`sample_continuous_photons` and compare against independently written CDFs
and densities.

1. **Distribution tests.** These used one segment, $4\times10^5$ trials,
   $T\in\{3,10,10^2,10^3,10^4,10^5\}$ keV ($\beta=0.108$ to $0.999987$), and
   segment axes $(1,0,0)$, $(0.95,0.3,0.1)$ and $(0.1,-0.4,0.9)$. The axes
   exercise both helper-vector branches. A 60-bin $\chi^2$ test compared:
   - $\mu$ about $\hat{\mathbf v}$ against the analytic mixture CDF;
   - azimuth in an independently built frame against a uniform law;
   - energy against the analytic mixture CDF;
   - arclength position against a uniform law.

   All p-values were in $[0.23,0.98]$. The Spearman correlation between
   $\mu$ and $k$ was at most $6\times10^{-4}$, consistent with independence.
   Recomputing every returned weight from my own $L/(Mq_Eq_\Omega)$ formula,
   with $j=1$, agreed to $\le1.4\times10^{-13}$ up to 10 MeV and
   $\le4\times10^{-11}$ at 100 MeV. The worst single-segment weight
   relative to the uniform weight was $12.3$, below the bound $16$.
2. **Unbiasedness.** This used 200 seeds of $2\times10^4$ trials over five
   segments. The segments had different $T$ (0.5 to 2000 keV), lengths and
   directions. One segment was below cutoff and one had zero length.
   The checks used $k_{\min}=200$ eV, $k_{\rm ceiling}=1.5$ MeV and analytic
   integrals:

   | emissivity | z (mixture) | batch rel. SD, mixture / uniform |
   |---|---|---|
   | constant | $+1.15$ | $0.025$ / exact |
   | $(1-\beta'^2)/[(1-\beta'\mu)^2k]$, $\beta'=0.95$ | $+0.01$ | $0.016$ / $0.24$ |
   | $\mu^2k$, nonseparable energy and angle | $+1.58$ | $0.042$ / $0.0082$ |
   | $\mathbf 1[\mu<-1/2]/k$, backward only | $-0.19$ | $0.021$ / $0.14$ |
   | $\mathbf 1[1-\mu<10^{-4}]/k$, narrow cone | $-0.44$ | $0.61$ / $8.6$ |

   All are consistent with zero bias. The backward-only case confirms
   full-sphere support. The $\mu^2k$ case shows the expected variance cost of
   a mismatched proposal and remains unbiased.
3. **Inverse-CDF precision.** The check used
   $u\in\{0,10^{-12},\tfrac12,1-10^{-12},1\}$ and $T$ from 1 keV
   ($\beta=0.0625$) to 1 GeV. The endpoints map exactly to $\mp1$, and
   $u=\tfrac12$ maps to $\mu=\beta$. Recomputing the CDF gave
   $\lvert F(\mu)-u\rvert\le2.2\times10^{-16}$ at 1 keV,
   $\le4.6\times10^{-14}$ at 20 MeV ($\beta=0.99969$) and
   $\le5\times10^{-11}$ at 1 GeV. The only theoretical failure is
   $1-\beta$ rounding to 0, near $\gamma\gtrsim10^{8}$. That would make
   $q_\Omega$ infinite at $\mu=1$, but it is far outside BremsLib table
   coverage, which fails closed.
4. `uv run pyrite-dev test tests/montecarlo/test_photon_sources.py` passed
   all 16 tests.

### Verdict (mixture proposal)

- **Filters:** units pass; limits pass ($\beta\to0$ isotropic, endpoints,
  constant-$j$ mean); signs/conventions pass ($\mu$ measured about the same
  `v_hat` used by the BremsLib angle).
- **Re-derivation:** matches. The forward normalization, inverse CDF, mixture
  densities, frame, weight, support and bounds agree with the code. No
  divergent factor was found.
- **Observation (not a discrepancy):** the segment-selection factor is not
  bounded by $1/(1-a)$ or $1/(1-b)$. Weights are bounded relative to the
  uniform proposal only per segment, up to the fixed ratio
  $p_i^{\rm uni}/p_i$.
- **Verdict:** `rederived`.
- **Suggested ledger change:** set `photon-continuous-source` from
  `unverified` to `rederived`. Append to Notes: "Fresh-context verification
  2026-10-24 rederived the defensive-mixture proposal (forward density
  normalization and inverse CDF, mixture densities, frame, weight, support,
  1/(1-a) and 1/(1-b) energy/angle bounds); χ² and analytic-integral checks
  pass; no discrepancy."
- **Human sign-off:** pending.
