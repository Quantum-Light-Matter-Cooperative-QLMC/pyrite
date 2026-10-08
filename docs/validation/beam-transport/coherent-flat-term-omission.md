# Coherent flat-term omission

Validation: `coherent-flat-term-omission`.

## Source and independence

Source: Cauchy–Schwarz in the finite-dimensional complex field space and
its application to the distinct-pair population estimator. This is an
algebraic approximation certificate; the ensemble model itself belongs to
`coherent-physical-bunch-population`. A fresh verifier derived the expressions
below from the intended quantity and reducer signatures/docstrings before
reading the implementation bodies. The older ledger text names the sampled
population alone; the revised claim distinguishes sampled count $M$ from
physical bunch population $N$.

The intended operation replaces weighted coherent row power by the grouped
floor at selected evaluated energies. `_flat_energy_keep(st, F)` selects the
retained coordinates; `_decoherence_blend(st, F, grouped, flat, keep)` applies
the full estimator there and the grouped floor elsewhere. Raw reducer powers
are divided by the incident sample count $M$ during final normalization.

## Independent derivation

For one reflection/orientation row and one evaluated energy, let
$\mathbf{S}_e$ be the complex vector amplitude of incident sample $e$,
including that sample's segment interference and sampled geometry. Missed
entries have zero field. Assume finite fields, equally weighted incident
samples, $M\ge2$, and physical population $N\ge1$. Define

$$
G=\sum_{e=1}^{M}\|\mathbf{S}_e\|^2,\qquad
P=\left\|\sum_{e=1}^{M}\mathbf{S}_e\right\|^2,\qquad
\alpha=\frac{N-1}{M-1}.
$$

The intended physical estimator per incident sample is

$$
Q_N=\frac{G+\alpha F(P-G)}{M}.
$$

The distinct-pair expansion removes self pairs: $P-G$ sums the ordered
cross terms. Its iid expectation is $M(M-1)\|\mathbb{E}\mathbf{S}\|^2$;
$G/M$ estimates the single-electron second moment. Thus the expected
per-electron population power is the self moment plus
$(N-1)F\|\mathbb{E}\mathbf{S}\|^2$. This interpretation requires the
separately specified ensemble assumptions; the following deterministic bound
requires only finite vector fields and nonnegative $\alpha F$.

Cauchy–Schwarz gives $0\le P\le MG$, hence

$$
-G\le P-G\le(M-1)G,\qquad
\lvert P-G\rvert\le(M-1)G.
$$

Replacing $Q_N$ by the floor $Q_0=G/M$ therefore incurs

$$
\Delta=Q_0-Q_N=-\frac{\alpha F(P-G)}{M},\qquad
\boxed{\lvert\Delta\rvert\le F(N-1)\frac{G}{M}.}
$$

The omission certificate is $F(N-1)\le\ell$, measured against $G/M$.
The factor depends on physical population, not computational sample count.
At fixed $N$, increasing $M$ changes the estimator and its sampling variance,
but does not relax this worst-case certificate.

When physical $N$ is absent, retain the historical sampled-population blend

$$
Q_{\rm hist}=\frac{(1-F)G+FP}{M},\qquad
\lvert Q_0-Q_{\rm hist}\rvert\le F(M-1)\frac{G}{M}.
$$

It is the special case $\alpha=1$. Physical $N=M$ recovers that algebra only
when both modes use the same $F$ and the same sampled fields.

For physical $N=1$, pair excess is exactly zero. The existing $N=0$
convention also sets pair excess to zero and returns the diagnostic
per-incident-sample floor $G/M$. Downstream physical zero-charge observations
apply zero multiplicity. The displayed $N-1$ formula is not extrapolated to
$N=0$; its implemented pair weight is piecewise zero for $N\le1$.
Omission changes neither zero-population convention nor its result. For sampled $M=1$, $P=G$ exactly; no distinct
pair estimator exists for physical $N>1$, which must be refused. Historical
$M=1$ and physical $N\le1$ need no flat reduction. Empty sample sets contain
no power and cannot support normalization by $M$.

Equal aligned fields saturate the bound: $P=MG$. Two opposite unit fields
have $P=0$, $G=2$ and the reverse error sign. The physical finite-sample
estimator can become negative when the estimated cross term is negative and
$N$ is large; that does not invalidate the absolute bound. It must not be
interpreted as a relative error against $Q_N$, which can vanish or be negative.

## Filters, conventions, and scope

$F$, $M$, $N$, $\alpha$, and $\ell$ are dimensionless. $G$, $P$, and
$M Q_N$ have field-squared units; normalization and any common nonnegative
spectral prefactor preserve the certificate. $F\to0$ gives the grouped
floor for $N\ge1$; $N\to1$ removes pair excess. Limit $\ell=0$ is an
explicit full-evaluation control even at a mathematical zero certificate.

Finite footprints average independent Gaussian arrival times analytically:
$F=\exp[-(\omega\sigma_t)^2]$, with $\omega\sigma_t$ dimensionless.
Sampled transverse geometry remains in $\mathbf{S}_e$. Physical infinite
slabs use $F=1$ because the complete sampled fields already contain their
longitudinal phases. Applying an additional empirical slab form factor
would suppress the same sampled phases twice. Historical slabs retain their
older empirical-factor convention. The production factor must be finite and
in $[0,1]$; invalid factors cannot authorize omission.

For incoherently summed rows with nonnegative weights $w_r$,

$$
\left\lvert\sum_r w_r\Delta_r\right\rvert
\le\sum_r w_r\lvert\Delta_r\rvert
\le\ell\sum_r w_r\frac{G_r}{M}.
$$

A common omitted coordinate must pass every row; retained indices form the
union of row failures. Positive discrete energy and detector weights preserve
the discrete bound. Neither inter-node values nor continuous bin integrals
are certified. Negative row weights invalidate the sum inference.

Floating-point certification must bound both the host pair-scale ratio
$\widehat{\alpha}_{\rm host}$ and its conversion to the backend real
precision $\widehat{\alpha}_{\rm REAL}$. Define

$$
\alpha_* = \max(\widehat{\alpha}_{\rm host},
                 \widehat{\alpha}_{\rm REAL}),\qquad
W\ge\alpha_*(M-1).
$$

Form $W$ and then $FW$ with outward binary64 products. The maximum covers
an upward backend scalar conversion; retaining the host value also covers a
downward conversion or positive underflow. A conversion to infinity retains
evaluation. Using only
an independently rounded $N-1$ is insufficient if the implemented ratio
rounded upward. For finite footprints, additionally use a directed upper
enclosure of the positive analytic Gaussian, including subnormal tails;
it must also bound the factor actually used by the reducer. Zero population
or exactly zero pair weight has zero certificate. Overflow or invalid
certificate intermediates must retain evaluation. The omission allowance
covers algebraic term removal for the enclosed scalar weight. Backend
rounding of the scalar-factor product, subsequent multiplication/subtraction,
and reduction reassociation require their own working-precision allowance.

The default omission share $\ell=10^{-4}$ remains separate from the
previously documented decoherence-grid charges. Combining the stated
$4\times10^{-4}$ charges within a $5\times10^{-4}$ budget requires the
same reference quantity and scope; this claim does not establish that match.

Independent algebraic points, using no implementation helpers: three aligned
unit fields give $M=3$, $G=3$, $P=9$. With $N=101$, $F=10^{-6}$,
$Q_0=1$, $Q_N=1.0001$, and the error $10^{-4}$ saturates the bound.
For $M=2$, opposite unit fields, $N=5$, $F=0.01$, $Q_N=0.96$ and the
floor overestimates by $0.04$, again saturating $F(N-1)G/M$. Historical
three-field power at $F=0.01$ differs from the floor by $0.02$.

## Implementation comparison

`pair_scale` returns $(N-1)/(M-1)$ for $N>1$, zero for
$N\le1$, and rejects physical $N>1$ with fewer than two incident
samples. `mixed_row_power` uses the exact historical arithmetic when $N$
is absent and otherwise forms $G+\widehat{\alpha}F(P-G)$ before final
division by $M$. This matches the positive-population derivation. The task
owner confirmed that the existing $N=0$ convention retains $G/M$ at this
per-incident-sample boundary; zero physical multiplicity is downstream.

`_flat_omission_weight` uses the same host-rounded `pair_scale` as the
reducer and takes its maximum with the backend `REAL` conversion before
multiplying by $M-1$ for $M\ge2$. It moves a positive product one
binary64 representable value toward infinity. This matches $\alpha_*$
above and encloses upward backend scalar conversion, including CUDA fp32.
Historical mode uses $M-1$.
`_flat_energy_keep` checks factor finiteness and range, multiplies by this
outward weight, rounds positive products outward again, and retains every
coordinate whose certificate fails. Stacked row failures are united with
`any(axis=0)`. A zero policy limit or inactive factor keeps the full reducer.
This is conservative for both host and backend rounded pair scales, as well
as the exact rational formula. Overflow and invalid products retain evaluation.
The argument assumes practicable integer sample counts exactly representable
in binary64; reduction roundoff is not charged to this certificate.

For finite footprints, the helper additionally tests the analytic Gaussian
upper bound at the first remaining omission candidate and uses its monotonic
decrease along the supported ascending nonnegative energy axis. Its
length-to-time conversion uses the same speed-of-light convention. The
production factor is checked independently at every coordinate. The directed
Gaussian helper itself was not revalidated in this task; this comparison
relies on its separately established upper-enclosure contract.

`_decoherence_blend` starts with a copy of the full grouped floor, then calls
`mixed_row_power` only for retained coordinates. With no mask it directly
calls that same reducer. Thus physical population weighting is present on
retained coordinates, historical arithmetic is preserved in historical
mode, and omitted coordinates retain the same grouped floor. At $N\le1$
both full and omitted raw row powers are $G$. No divergent population factor,
sign, or normalization was found in the five reviewed helper bodies.

Physical infinite-slab $F=1$ and finite-footprint Gaussian conventions are
part of the supplied reducer contract and the inspected docstrings; this
verification did not re-audit setup, route dispatch, CUDA execution, or
coefficient-capture policy. The preserved benchmarks below do not test the
new physical-population weighting.

## Adjudication

Units, limits, and signs/conventions pass. Independent re-derivation matches
the reviewed implementation. Verdict: **rederived**. Suggested ledger update:
replace sampled-only $F(M-1)$ certification with physical $F(N-1)$ for
$N\ge1$, keep historical $F(M-1)$ separately, and record zero pair excess
for physical $N\le1$. State the $G/M$ reference, evaluated-node scope,
host/backend rounded pair-scale enclosure, complete-field physical slab $F=1$,
and historical scope of earlier benchmarks. The owner applies the ledger
change and any regression status; this verifier does not assign human sign-off.

Current revision documentation tests, HTML rendering checks, and physical
population runtime tests are delegated to the task owner. No current pass
claim is made here; the evidence below belongs to the historical revision.

## Historical CPU evidence (sampled population)

The following pre-port measurements and checks exercised historical
sampled-population semantics without physical bunch $N$. They are retained
for provenance; they do not establish runtime or accuracy for physical
population weighting. They were not rerun by the current verifier.

The same-trajectory CPU anchor suite passed all 24 cases, covering empirical
slab, per-reflection exact and sinc-windowed routes, finite footprint,
single-electron limits, full-evaluation control, invalid-factor refusal,
outward thresholds, analytic-Gaussian protection and shared row-mask union.
CUDA runtime parity was not exercised by this verifier; its routing was
inspected structurally. Reassociation and roundoff in field reductions are
separate from the mathematical omission bound; tests allow the stated
working-precision comparison tolerance.

Supplemental task-owner timing report (not independently rerun by this
verifier): the CPU float64 fixture used 12 primaries, 19 segments, seed 362,
1000 angstrom thickness, 60 keV beam energy, 1 attosecond longitudinal RMS,
and 5750 nodes spanning 500–12000 eV with 2 eV spacing. Its trajectory hash
was `7786720ee56da4d4af96292cf1ac6f85735ad466ab924e00341f2e898d8b199a`.
After one warmup per arm and three alternating repeats, full-reducer times
were 0.096018, 0.060195, and 0.058987 seconds; omission times were 0.078615,
0.044173, and 0.095395 seconds. The median speed ratio was 0.7657, so this
small fixture establishes no local performance improvement. Reported yield
and centroid relative changes were respectively $1.236094449\times10^{-9}$
and $8.84038176\times10^{-10}$. These are local implementation measurements,
not remote performance evidence or a grid-convergence certificate.

## Historical task-owner paired GPU measurements (sampled population)

The task owner subsequently ran `checks/coherent_flat_omission.py` under
SLURM job 1136 on `qlmc`, in the isolated `~/pyrite-issue362` checkout.
Each full/masked comparison reused the same in-memory trajectories. The
three route processes also produced the same segment fingerprint:
`847faf003b1fbd9a04cf6d2aa79eee2cdf414020f48572c0e812bdec9cf8d71a`.
The benchmark's grouped-floor comparison passed the omission bound plus
its declared 100-epsilon peak-scaled reduction-roundoff allowance on every
route. These are runtime measurements by the task owner, separate from the
fresh-context algebraic verification above.

Workload: HOPG, 60 keV, 200 primaries, 50000 angstrom thickness (a thick
member of the `hopg_short` thickness class), finite 1 mm square footprint,
1 attosecond Gaussian longitudinal RMS, seed 362, 13232 segments, two
basal reflections, observation angle 119 degrees. Transport was the
screened-Rutherford/continuous-stopping lockstep reference model; this is
a reducer benchmark, not a full production-profile sweep. Both arms used
5750 identical coordinates from 500 to below 12000 eV at 2 eV spacing.
The axis is held fixed to isolate omission; no quadrature-convergence claim
is made for these samples.

Environment: NVIDIA GeForce RTX 5080 (16303 MiB), driver 610.47, CuPy,
float32, Python 3.14.6, four allocated CPU threads. One warmup per arm,
three alternating-order repeats, GPU synchronization at both timing
boundaries. Peak process RSS was 740–1023 MiB; GPU pool usage was queried
at process end, not sampled as a peak.

| Route | Full wall time, three repeats (s) | Masked wall time, three repeats (s) | Median speed ratio | Relative yield change | Relative centroid change |
| --- | --- | --- | --- | --- | --- |
| Default streaming | 0.038559, 0.022759, 0.026459 | 0.027561, 0.032946, 0.025291 | 0.9600 | 2.977e-11 | 2.826e-11 |
| CUDA JIT | 0.046605, 0.057112, 0.046842 | 0.047342, 0.054562, 0.072304 | 0.8585 | 2.977e-11 | 2.826e-11 |
| Eager CuPy | 0.133725, 0.133853, 0.133669 | 0.106294, 0.118124, 0.106930 | 1.2506 | 3.259e-08 | 7.193e-10 |

The maximum relative change across measured yield and centroid was
$3.259\times10^{-8}$. The eager route shows a speedup on this workload;
the default streaming and JIT medians do not. Their repeat ranges overlap,
so these short timings support no general speedup claim. CUDA route
differences remain outside this omission comparison (#298).

Reproduce after `pyrite remote sync` into an isolated lab directory, using
a bounded GPU allocation (10 minutes for the recorded run):

```bash
uv sync --extra nvidia --locked --no-dev
PYRITE_HOME=. PYRITE_MC_BACKEND=cuda PYRITE_FP64=0 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --no-sync python checks/coherent_flat_omission.py \
  --electrons 200 --thickness-ang 50000 --energy-kev 60 --repeats 3 --route auto
```

Repeat with `--route jit` and `--route eager`. The benchmark runs the full
and masked spectrum phases internally on one transport realization; it
never compares two independent Monte Carlo runs.


## Port verification on current physical-population model

The port retains the existing physical-pair estimator and changes only the
energy mask and restricted Flat evaluation. The focused omission module has
39 passing CPU checks. New aligned-field fixtures attain the Cauchy–Schwarz
upper bound with six sampled fields and physical populations 1.5, 12, and
120, on batched, per-reflection, and sinc-windowed routes. They compare against
the independent reference $(1+F_z(N-1))G/M$, test kept and omitted energies,
and verify bit-identical output with omission disabled. Further checks cover
zero pair population, physical slabs using complete fields rather than an
empirical factor, and preservation of the refusal for unresolved signed
physical-pair power.

The same physical fixtures select streaming, eager CuPy, and reduction-JIT
routes when run with the CUDA test backend. That current CUDA check remains
pending lab GPU availability; no revised physical-population GPU result or
speedup is claimed here. The paired timing harness accepts
`--physical-electrons N` for future physical-charge measurements; omitting
that argument reproduces historical sampled-population measurements.
