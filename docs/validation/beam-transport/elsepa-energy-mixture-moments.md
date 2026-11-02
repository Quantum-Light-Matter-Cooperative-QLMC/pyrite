# ELSEPA energy-node mixture: moment bounds

Validation: `elsepa-energy-mixture-moments`.

**Status: anchored.** Fresh-context re-derivation and analytic/release-wide
regressions verify the candidate moment identities and bounds in issue #895.
Production continues to interpolate quantiles. The study-only samplers below
do not change production defaults, random streams or cross-section interpolation.

## Source law and assumptions

The source is the normalized, piecewise-linear angular density in
[`elsepa-elastic-sampling`](elsepa-elastic-sampling.md), derived there from
the ELSEPA DCS and $d\Omega=4\pi\,d\mu$, where
$\mu=(1-\cos\theta)/2\in[0,1]$. The statements below follow from conditional
expectation, expansion of a square and inversion of that panel CDF; they are
derived here, rather than attributed to an external equation number.

Fix two adjacent energy nodes with CDFs $F_0,F_1$ and quantiles $Q_0,Q_1$.
Let $f=\ln(E/E_0)/\ln(E_1/E_0)\in[0,1]$. All angular variables, probabilities,
moments and bounds below are dimensionless. The common energy unit cancels.
The assumptions are nonnegative normalized densities on $[0,1]$, positive
panel masses, independent node selection and a uniform polar quantile
conditional on either selected node. This applies separately to each selected
element and to electron or positron tables. It does not validate those tables.

## Two different angular laws

Production draws $U\sim\mathrm{Uniform}(0,1)$ and sets

$$
\mu_Q=(1-f)Q_0(U)+fQ_1(U).
$$

The candidate draws $B\sim\mathrm{Bernoulli}(f)$ independently of $U$ and sets
$\mu_M=Q_B(U)$. Its CDF is

$$
F_M(x)=(1-f)F_0(x)+fF_1(x).
$$

Both laws reduce exactly to node 0 at $f=0$ and node 1 at $f=1$, including
the final table endpoint. At identical node distributions both reduce to that
distribution for every $f$. The mixture CDF is generally different from the
CDF obtained by interpolating quantiles. Reusing a random number without
preserving the conditional uniform law does not implement this mixture;
bit splitting or residual-uniform remapping requires a separate RNG check.

Write $m_i=\int_0^1 Q_i(u)\,du$, $a=\int_0^1Q_0^2\,du$,
$b=\int_0^1Q_1^2\,du$, $c=\int_0^1Q_0Q_1\,du$, and

$$
D=\int_0^1[Q_1(u)-Q_0(u)]^2\,du=a+b-2c\ge0.
$$

For $Y=1-\cos\theta=2\mu$, linearity gives the **same first moment**:

$$
\mathbb E[Y_M]=\mathbb E[Y_Q]=2[(1-f)m_0+fm_1].
$$

Expansion of the square gives

$$
\mathbb E[Y_M^2]-\mathbb E[Y_Q^2]
=4f(1-f)D\ge0.
$$

Consequently the maximum absolute second-moment difference over the entire
energy interval is $D$, attained at $f=1/2$. The variance increases by the
same amount. The difference vanishes at nodes and for identical distributions;
it cannot be a negative second-moment change. The ratio of second moments has
the exact maximum

$$
\max_{0\le f\le1}
\frac{\mathbb E[Y_M^2]-\mathbb E[Y_Q^2]}{\mathbb E[Y_Q^2]}
=\frac{D}{(\sqrt a+\sqrt b)^2-D},\qquad
f_* = \frac{\sqrt a}{\sqrt a+\sqrt b}.
$$

This assumes $a,b>0$, as in the released tables. Substituting an upper bound
for $D$ gives an upper relative bound when the denominator remains positive.
The ELSEPA second *transport* moment is different:
$\langle1-P_2(\cos\theta)\rangle=6(\langle\mu\rangle-\langle\mu^2\rangle)$,
so its mixture-minus-quantile change is $-6f(1-f)D$. Exact agreement of the
first moment does not imply agreement of arbitrary transport observables.

## Deterministic integral envelopes

{source}`src/pyrite/validation/_elsepa_interpolation.py::_pair_moment_bounds`
splits $u\in[0,1]$ at the union of both node CDF knots. Each cell lies in one
linear-density panel of each node. On a panel with density slope $s_i$,

$$
p_i(Q_i(u))^2=p_{i,j}^2+2s_i(u-F_{i,j}).
$$

The derivative of $Q_1-Q_0$ is $1/p_1(Q_1)-1/p_0(Q_0)$ wherever finite.
Its interior stationary points therefore solve equality of the two affine
squared-density expressions. There is at most one, unless they coincide
throughout the cell, when the quantile difference is constant. Evaluate this
candidate and the endpoints to bound $Q_1-Q_0$; square its range, including
zero if the range crosses zero. Multiplying by cell width and summing gives
lower and upper envelopes for $D$ valid for **every** $f$ through the formula
above. Subdivision tightens these envelopes without a stochastic error bar.

Node first and second moments are integrated analytically as local
polynomials. The calculation uses FP64 with a $64\epsilon_{64}$ absolute
padding on $D$; this is a numerical enclosure of the analytic bound, not a
directed-rounding interval-arithmetic certificate. It is not a bound on the
additional error of production FP32 interpolation or finite random grids.

## Evidence and reproducibility

{source}`tests/validation/test_elsepa_interpolation.py` anchors uniform and
triangular densities: $Q_0(u)=u$, $Q_1(u)=\sqrt u$, $D=1/30$, and the interior
extremum at $u=1/4$. It checks both node endpoints, the moment sign, the
relative bound at $f_*$, subdivision tightening, reversed node order,
identical nodes and normalization invariance. These are analytic checks,
independent of production inversion helpers.

{source}`checks/elsepa_interpolation_moments.py` audits every adjacent pair of
every table in the pinned electron and positron release indices, verifies
each stored manifest digest against its release pin, and includes the extra
muffin-tin/free-atom crossover pair used by every released elementary solid.
The latter pair is absent from either raw table alone. Missing tables fail
the check; no reduced subset is reported as complete.

```bash
export PYRITE_HOME=/tmp/elsepa-mixture-audit
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite tables fetch elsepa
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite tables fetch elsepa --projectile positron
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python checks/elsepa_interpolation_moments.py \
  --subdivisions 16 --output scratch/elsepa-mixture-moments.json
PYRITE_ELSEPA_AUDIT_HOME="$PYRITE_HOME" UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  --slow tests/validation/test_elsepa_interpolation.py --durations=0
```

Use an isolated workspace when installed manifests differ from the pins;
the audit refuses those manifests even when their request keys match.

Measured 2026-11-02 with 16 subdivisions per common-CDF panel: **55 tables,
3,040 adjacent raw-table pairs and 13 extra production crossover pairs per
species**. The electron release is `tables-elsepa-4`, archive SHA-256
`2155aaedb20797fcafe712386a24d37d5a40021c9d3481d62e27ae697d66b861`;
the positron release is `tables-elsepa-positron-4`, archive SHA-256
`cc0046c83660c8a6b4526725d9e875418f33a0ec52676d4ef27ce6b1eb261779`.
These pins are read from the packaged release indices, not inferred from
the files present in a cache. The complete deterministic audit took 29.3 s
on one CPU process; it runs no transport or Monte Carlo.

| species | global maximum absolute second-moment gap, enclosed | location | upper relative gap over all weights |
| --- | --- | --- | --- |
| electron | $[0.0099443150,\ 0.0099566622]$ | iodine, 100–125 eV | 1.391% at that pair |
| positron | $[0.0027147585,\ 0.0027169584]$ | hydrogen, 100–125 eV | 0.799% at that pair |

The maximum *relative* change occurs elsewhere. On the HOPG muffin-tin to
free-atom crossover at 1–1.25 MeV, the maximum relative second-moment
increase is enclosed by **10.2214–10.2367% for electrons** and
**10.2280–10.2434% for positrons**. The maximizing weights are approximately
0.63843 and 0.63890, respectively. Excluding production joins would miss
these maxima: the raw-table relative upper maxima are 6.831% for electrons
(Bi, 60–80 MeV) and 5.079% for positrons (Hf, 60–80 MeV).

All numbers are mixture-minus-quantile changes of the normalized angular
law, relative to the quantile second moment where labeled relative. The
first-moment difference is analytically zero across every pair and all
weights. The slow release-wide regression pins absolute ceilings of 0.01
(electron) and 0.003 (positron), a relative ceiling of 11%, and the HOPG
crossover's relative lower bound above 10% for both species. It skips when
the exact pinned tables are unavailable; the audit command itself fails
closed instead of accepting a subset.

The remote repeat on 2026-11-15, clean revision `2b7b3e4ea` (CPU job 12696),
passed all seven tests in the named regression module, including the full
release audit in 22.02 s. A separate audit took 15.89 s and reproduced the
upper gaps above: 0.009956662116 / 10.236637% for electrons and
0.002716958318 / 10.243395% for positrons. The latter percentages are the
global relative maxima, not the relative gaps at the absolute maxima.

## Study-only single-uniform sampler

{source}`checks/elsepa_statistical/samplers.py` supplies CPU and exact-CUDA
candidates for detailed elastic transport. Given the existing polar deviate
$x$, choose the upper node and set $u=x/f$ when $x<f$; otherwise choose the
lower node and set $u=(x-f)/(1-f)$. At $f=0$ or $f=1$, select the corresponding
node and preserve $x$. Each branch is uniform conditional on its selected
node in continuous arithmetic, so this produces the mixture law without
consuming an additional counter-addressed draw. It changes the draw-to-angle
map between nodes. It does not implement the hard-tail conditional law used
by mixed elastic transport or cone splitting.

For a regular $N$-point uniform input grid, each joint event “selected node
$i$ and conditional quantile $\le t$” is one interval of the original input.
In exact arithmetic its probability differs from the continuous target by
at most $1/N$. Node probabilities obey the same count bound. Conditional
quantile grids need not be canonical uniform grids or exactly independent
of the node choice; conditional discrepancies can scale as $1/(Np_i)$ for
branch probability $p_i$. The production input-grid scale is $2^{-53}$.
This counting statement excludes floating-point remapping and inversion error.

Node selection and conditional remapping stay FP64 in both CUDA builds.
The mixed build receives an already rounded energy weight; widening it does
not restore the mathematical log-energy fraction. Its existing FP32
complementary-CDF inversion follows the remapping.

{source}`tests/validation/test_elsepa_statistical.py` checks representative
joint CDF counts on 4096 midpoint inputs and the uniform/triangular mixture
moments. {source}`checks/elsepa_statistical/parity.py` tests analytic densities
with four angular knots at weights 0, 0.125, 0.5, 0.875 and 1. The remote
CUDA preflight (job 12695, revision `2b7b3e4ea`) passed both precisions; its
largest mixed absolute error was $1.081\times10^{-6}$ against the declared
$2\times10^{-6}$ tolerance. This checks the selected examples; it is not a
universal FP32 angular-law error bound or a hardware rounding certificate.
CPU full-transport integration is outside this study harness.

## Fresh-context verification (2026-11-15)

A read-only verifier derived the identities and extrema from the node law
before inspecting implementation bodies. With $c=\int Q_0Q_1$, it obtained
$D=a+b-2c$ and

$$
\mathbb E[Y_Q^2]=4[(1-f)a+fb-f(1-f)D],
$$

then independently maximized the absolute and relative gaps. Its local
polynomial integrals match `_moments`; inverse-function differentiation gives
the same stationary-point offset and sign as `_difference_envelopes`.
The independently derived lower-first uniform partition is equivalent to the
implementation's upper-first partition. No divergent factor, normalization,
sign or interpolation weight was found. Verdict: **rederived**, applied by
the task owner and advanced to **anchored** on the passing named regressions.

| Filter | Result |
| --- | --- |
| Units, normalization, limits, signs and node reversal | Pass |
| Moment identities and analytic deterministic envelopes | Pass; FP64 evaluation is not directed-rounding interval arithmetic |
| Single-uniform law and unchanged draw count | Pass in continuous arithmetic; finite-grid caveats above |
| CUDA precision rebinding | Pass by inspection and the bounded analytic preflight |
| Released-table numerical maxima | Reproduced by the task owner's pinned remote audit |
| Universal FP32 distribution accuracy | Not established by these analytic examples |

No headline-observable equivalence, speed gain, default adoption or human
sign-off follows from these moments.

## Bounded W 1 MeV pilot: no production adoption

[Retained study record](../check-records/elsepa-statistical-895.json).
GPU job 12695 ran the original #821 W 1 MeV / 100 µm scene on an RTX 5080,
revision `2b7b3e4ea`, with 4000 histories per arm and seeds 89501–89503.
Both precisions use t32, Q1024, device staging, the continuous point-cone
estimator, full-workload cold warmups and alternating arm order. CUDA events
surround every transport launch in both arms. Warm request time includes
transport, assembly and scoring; cold compilation and profiler replay are
excluded. The reducer at `673523e67` verifies paired controls, tables and
primary-score digests before applying the existing #838 interval rules.

| Precision | Quantile kernel ns/row, median | Mixture kernel ns/row, median | Candidate/baseline kernel ratio, median [range] | Warm request ratio, median [range] |
| --- | ---: | ---: | --- | --- |
| mixed | 8.7663 | 8.6270 | 0.9822 [0.9808, 0.9931] | 0.9675 [0.9571, 1.0844] |
| FP64 | 14.1693 | 13.7966 | 0.9716 [0.9703, 0.9881] | 0.9709 [0.9694, 0.9727] |

Ratios are computed within each seed before taking the median; they are not
ratios of the displayed median times. These are three-seed descriptive
spreads, not speedup confidence intervals. The measured kernel reduction is
small (1.8% / 2.8%); mixed warm-request timing includes one slower candidate
run. This one-scene pilot does not establish performance across all anchors.

The paired primary intervals use the #838 criterion at $\epsilon=2\%$:
each individual 95% interval must fit within a relative margin
$\epsilon/3=0.6667\%$. Of 26 headline tests across the two precisions, 25 are
unresolved and one fails. The mixed $\ge100$ keV photon yield has relative
change $-4.6742\%\pm3.5921\%$, giving an interval
$[-8.2663\%,-1.0821\%]$ outside the margin. The corresponding FP64 interval
is unresolved. These are per-observable intervals without familywise
adjustment; the failed cell is not a familywise discovery of a physics bias.

**Decision:** no production adoption from this pilot. It does not establish
the required equivalence. Other original #821 scenes, larger independent
confirmation populations and CPU full-transport wiring remain untested;
they would be needed to reconsider the candidate. The production model,
case identities and their digest pins are unchanged.
