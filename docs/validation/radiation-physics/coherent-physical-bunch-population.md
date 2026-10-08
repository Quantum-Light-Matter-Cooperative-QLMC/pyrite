# `coherent-physical-bunch-population`

## Claim and source

The production coherent line spectrum uses the physical bunch population
$N=Q/e$, where $Q$ is `bunch_charge_pc` converted to coulombs and $e$ is the
SI elementary charge. `--ne-line` supplies $M$ equally weighted incident Monte
Carlo samples, including missed entries. It controls estimation and normalization,
not the physical enhancement.

Source: expansion of the independent-electron radiation field,
$|\sum_e S_e|^2=\sum_e|S_e|^2+\sum_{e\ne f}S_e S_f^*$; the identical-emitter
form is $N[1+(N-1)F]$ times the one-electron intensity, as in
[Su et al., Nuclear Science and Techniques 29, 30 (2018), Eq. (2)](https://www.nst.sinap.ac.cn/article/id/3343).
The distinct-pair estimator below follows directly from this expansion.

This is an implementation-context derivation. The fresh-context review below verifies this operator; it does not certify the
full window-envelope or production sampling claim.

## Derivation

For iid complete electron fields $S$, let $\mu=\mathbb E S$ and
$v=\mathbb E|S|^2$. The power per physical electron is

$$
Y=v+(N-1)|\mu|^2.
$$

Polarizations add after squaring. Reflections and mosaic orientations remain
incoherent and retain their production weights. For a finite footprint, independent
Gaussian arrival times can be averaged exactly: retain each electron's transverse
geometry and attenuation in $S_e$, remove its arrival offset, and replace the cross
term by $F_z|\mu|^2$, where $F_z=\exp[-(E c\sigma_t/(\hbar c))^2]$.

With $G=\sum_{e=1}^M|S_e|^2$ and $P=|\sum_{e=1}^M S_e|^2$,

$$
\widehat{Y}=\frac{1}{M}\left[G+\frac{N-1}{M-1}F(P-G)\right],\qquad M\ge2.
$$

$G/M$ estimates $v$. $P-G$ contains only ordered distinct pairs; its expectation
is $M(M-1)|\mu|^2$. Thus the estimate is unbiased, provided the incident samples
are iid and equally weighted and the analytic $F$ is independent of those fields.
Infinite slabs use complete fields, including sampled offsets, with $F=1$.
They use no empirical characteristic-function multiplier, so correlated offsets
and trajectory fields do not require a phase/field independence assumption.
Finite footprints use only the supported independent Gaussian longitudinal
average, with sampled transverse offsets retained in the fields. This does not
prove iid/equal weighting for an arbitrary imported or structured source.

The positive alternative $G/M+(N-1)F|\overline S|^2$ has bias
$(N-1)F(v-|\mu|^2)/M$. It is not used. The historical empirical
$|\overline{\exp(i\varphi)}|^2$ also contains self pairs; production physical
infinite-slab evaluation avoids that second estimator entirely.

## Limits, units, and refusal

- Identical aligned fields: $G=M|S|^2$, $P=M^2|S|^2$, so $\widehat Y=N|S|^2$;
  multiplying by $N$ gives raw bunch power $N^2|S|^2$, with excess
  $N(N-1)|S|^2$. This holds for every $M\ge2$ at fixed physical charge.
- $N=1$ or $F=0$: only $G/M$ remains. Zero charge uses this per-incident-electron
  self-term convention but has zero detected flux. A nonzero population below
  one physical electron is refused; fractional multi-electron charge is treated
  as an effective population.
- At $N=M$, the finite-footprint estimate recovers $(1-F)G/M+FP/M$.
- Missed incident electrons contribute zero fields and remain in both $M$ and
  $M(M-1)$. The number of emitting electrons cannot replace $M$.
- $N$, $M$, and $F$ are dimensionless. The output remains photons/eV/sr/incident
  electron. Detected counting multiplies it once by $Q/e$ times pulse count;
  fully aligned bunches therefore have quadratic charge dependence, not cubic.

A multi-electron physical bunch requires at least two incident samples, even if
its Gaussian factor underflows on the requested axis. Negative or nonfinite final
spectral or temporal estimates raise `CoherentSamplingError`. They are neither
clipped nor silently resampled. Negative row contributions may cancel before the
final check. Nonnegativity is a necessary check, **not** a statistical error bound:
a positive estimate can still have large Monte Carlo uncertainty. Accepted outputs
conditioned on passing this check are not themselves an unbiased ensemble.

Low-level `mc_spectrum(physical_electrons=None)` retains the historical
sampled-bunch semantics. Every production runner coherent call supplies physical
charge, including dual spectra, recompute and coefficient capture.

## Window bounds, audits, temporal output, and identity

If $H$ incident electrons emit in a row, Cauchy--Schwarz gives
$0\le P\le HG$, so $|P-G|\le(H-1)G$ for $H\ge2$ (zero for $H=1$).
The conservative absolute tail multiplier is

$$
1+\frac{N-1}{M-1}F_{\max}(H-1).
$$

Captured windows use that effective population and the same complete fields as
the reducer. Their reported metadata distinguish physical, incident and emitting
populations. The existing conservative grouped/all-electron step rule uses this
multiplier; it is not a new production envelope or relative-error certificate.

The existing full-axis audit encloses convex mixtures. It uses the actual
$F_{\rm eff}=(N-1)F/(M-1)$ when the entire axis has $0\le F_{\rm eff}\le1$,
with outward product rounding. It refuses larger weights, which need a signed
composition extension. Existing stored-input audit evidence remains scoped to
its original operator; it does not validate this new estimator automatically.
Temporal output scales its cross-electron difference by the same pair weight.

Coherent grid cache revision is 7 and includes the physical population. Dataset
and content identities carry `physical-distinct-pairs-v1` only for coherent
emission, preventing reuse of sampled-population spectra. Previous remote ladders
remain evidence for their recorded operator; charge-weighted default-policy
reference/thick runs must be repeated before production acceptance.

## Checks and status

Status: **rederived**, following the independent review below.
`tests/montecarlo/test_coherent_physical_population.py` checks the exact aligned
limit at fixed charge, unequal geometric and offset phases, the exact expectation
of iid zero-mean fields by enumerating all small samples, missed entries,
one-electron and insufficient-sample limits, refusal of negative power, detector
charge scaling, and coherent content-cache invalidation. Related window, temporal
and identity checks cover their integration; CUDA hardware verification remains
separate.


Implementation checks (2026-10-08): complete window-grid module **62 passed**;
coherent/temporal/chunk/identity core group **77 passed** (before the final added
dataset-generation case); complete physical-population module **18 passed**;
docs/ledger **24 passed**; lint, typecheck, diff check and strict Sphinx build
passed. The rendered derivation contains 52 math elements and no leftover dollar
delimiters. Four neighboring profile/scan tests fail identically on unchanged
`main`: three depend on unavailable BremsLib tables/canonical identity, and one
expects MoSe2's high-energy range to omit the catalog's 5 MeV entry. They are
outside this slice. Existing dispersion warnings remain; no CUDA or new remote
production certificate is claimed.


## Independent verification (2026-10-08)

A fresh verifier derived the expression before inspecting implementation bodies
or the owner derivation. Review target: `4507c4b5`. Primary-source indexed text
confirmed the cited Eq. (2); direct page retrieval timed out.

For an iid complete incident-electron field $S$, including zero for a missed
entry, independent expansion gives

$$
\mathbb E\left|\sum_{e=1}^N S_e\right|^2
=N\mathbb E|S|^2+N(N-1)|\mathbb E S|^2.
$$

For $M$ equally weighted incident histories, $P-G$ contains $M(M-1)$ ordered
distinct pairs and has expectation $M(M-1)|\mathbb E S|^2$. Thus the independently
derived per-physical-electron estimate is

$$
\widehat Y=\frac{G+(N-1)(P-G)/(M-1)}{M},\qquad M\ge2.
$$

Independent Gaussian arrival times multiply only the cross term by
$F_z=\exp[-\omega^2\sigma_t^2]$, with amplitude characteristic function
$\chi=\exp[-\omega^2\sigma_t^2/2]$. Infinite slabs retain complete sampled
fields, avoiding a second empirical form-factor multiplier. The limits are
$N|S|^2$ for identical aligned fields independently of $M$, $G/M$ for $N=1$,
and no mean excess for zero-mean iid fields. Factors are dimensionless.

The verifier reproduced $|P-G|\le(H-1)G$ for $H\ge2$ and zero difference
for $H=1$. The absolute tail multiplier agrees. The full-axis audit's convex
composition is supported only when $0\le (N-1)F/(M-1)\le1$ across the axis;
its refusal outside that scope agrees. The temporal distinct-pair coefficient
and Fourier normalization obey discrete Parseval. Finite-footprint temporal
self terms retain sampled realized arrival shifts; analytic filtering applies
to the cross term only. This preserves the spectral self/distinct-pair integral.

Source-to-code comparison found no divergent normalization, factor, sign or
exponent in charge conversion, pair weighting, complete offsets, Gaussian
averaging, final incident normalization, negative/nonfinite refusal, temporal
output, window support bounds, convex audit gating or charge-sensitive cache
identity. CUDA branches were inspected but not executed by the verifier.

Verifier checks: **37 passed** across population, temporal and docs modules;
**2 passed** for physical-window widening and charge-sensitive grid caching.
Rendered math had no leftover dollar delimiters. Verdict: **rederived** for the
population operator and inspected CPU integrations. This does not establish
Monte Carlo convergence, the full production window-envelope claim, CUDA
execution correctness, or human sign-off.


## Charge-weighted remote checks (2026-10-08)

The CUDA float64 harness at clean revision `d8a1c07e4ca9d35d46d540a0cedaa7be4fb41f32`
uses 1 pC, or 6,241,509.074460763 physical electrons, and 40 incident Monte Carlo
histories for each thick HOPG ladder. Geometry is 1 mm thickness, 5° tilt,
0° azimuth, 100 fs bunch duration, 1 mm beam FWHM, 5 mm footprint and seed 7.
One persisted transport, checked by its segment fingerprint, serves every grid
and scheduler slice. The automatic grid has eight nodes per frozen-carrier
Nyquist step; the finer grid doubles that resolution. Uniform reference axes
include the automatic axis's two endpoints. These comparisons concern yield,
centroid and global FWHM, each with a relative tolerance of $10^{-3}$.

The CUDA aligned-field checks pass at fixed physical population 12 with
2, 3 and 7 samples; the negative-estimate refusal also passes. These are scoped
execution checks, not a general CUDA equivalence proof.

The [30 keV record](../check-records/coherent-physical-population/charge30.json)
from SLURM 1148 (`20261008-101206-268396aa`) passes against the initial
24-node uniform reference:

| grid | coordinates | wall [s] | yield rel. | centroid rel. | FWHM rel. |
|---|---|---|---|---|---|
| automatic | 2,184,166 | 42.97 | 7.43e-9 | 3.67e-10 | 1.72e-6 |
| finer | 4,368,330 | 91.57 | 1.12e-9 | 5.38e-11 | 3.15e-8 |
| uniform reference | 6,552,489 | 142.25 | — | — | — |

The exported harness used `ceil(span/target_step)` points for an
endpoint-inclusive reference. The resulting 30 keV spacing is
0.0005631448695518405 eV, versus the requested 0.0005631448171156663 eV
(relative excess 9.31e-8). Actual counts and spacing in the records identify
the grids used for these comparisons. Subsequent commit `fe2dc9b6` uses
`ceil(span/target_step) + 1` and budgets that count; a spacing-bound regression
fails before and passes after the correction. The immutable remote runs continue
with the exported harness. No observable-error bound is inferred from the
small spacing difference.

Peak host RSS was 2,293,412 KiB. The stronger nominal 32-node reference
and the smaller-chunk 60 keV retry have completed (below). No sampling or
production-window certificate follows from these runs. The full windowed-resolution
claim remains `discrepancy`.


### Stronger 30 keV reference

The [stronger reference record](../check-records/coherent-physical-population/charge30-reference32.json)
from SLURM 1151 (`20261008-105411-ee57f43f`) uses the original 30 keV
transport fingerprint `589c5d8fb186dc76f891bc31415fb3d3`, importing its
completed automatic/finer evaluations without regenerating trajectories.
Only the uniform-reference spacing selector changes. Actual reference spacing
is 0.0004223586360494427 eV; peak host RSS for the reference-only process
is 1,939,480 KiB. Both window grids pass all three $10^{-3}$ gates:

| grid | coordinates | wall [s] | yield rel. | centroid rel. | FWHM rel. |
|---|---|---|---|---|---|---|
| automatic, reused | 2,184,166 | 42.97 | 7.84e-9 | 3.87e-10 | 1.86e-6 |
| finer, reused | 4,368,330 | 91.57 | 1.54e-9 | 7.36e-11 | 1.64e-7 |
| stronger uniform reference | 8,736,652 | 203.03 | — | — | — |

### Completed 60 keV ladder

The [60 keV record](../check-records/coherent-physical-population/charge60.json)
from job `20261008-101206-e4a577f6` uses the original transport fingerprint
`d73442df808c54affafeed86a1ca2c18`. The initial allocation (SLURM 1149)
timed out; the retry (SLURM 1152) completed at 13:26 Pacific using one-row
spectrum chunks and the same saved trajectories and axes. Both window grids
pass all three $10^{-3}$ gates against the 29,115,499-coordinate reference:

| grid | coordinates | wall [s] | yield rel. | centroid rel. | FWHM rel. |
|---|---|---|---|---|---|
| automatic | 9,705,171 | 460.13 | 4.02e-8 | 1.51e-8 | 2.00e-4 |
| finer | 19,410,336 | 875.35 | 6.55e-9 | 2.32e-9 | 1.57e-5 |
| uniform reference | 29,115,499 | 1319.64 | — | — | — |

Actual reference spacing is 0.00020573235601190816 eV. Peak host RSS was
3,614,760 KiB. Timings cover the completed spectrum evaluations, excluding
the timed-out attempt. The record identifies clean revision `d8a1c07e` and
code digest `775d7674ba6d77c9211c6dbd3dfe285e671a8a7f70d14fab75aee87a5d355a17`.
The reference uses the exported harness's endpoint-count convention described
above. This is same-trajectory grid-convergence evidence for the physical
population operator, not a sampling or production-envelope certificate.

### Reduced short-bunch refusal

The [short-bunch record](../check-records/coherent-physical-population/short-bunch.json)
from SLURM 1150 (`20261008-101206-eca01929`) records a failed full-spectrum
check. Inputs are HOPG, 60 keV, 10 µm thickness, 45° tilt, 135° azimuth,
0.1 mm beam FWHM, 5 mm footprint, Gaussian RMS duration 0.001 fs, seed 1,
1 pC and 200 incident histories. The fixed transport contains 39,187 segments
(fingerprint `f9d5bae2d8e6254ae1912c56ebe2bfd2`). Its 11,936,606-coordinate
axis from 10 to 6000 eV is within the explicit 20,000,000-point budget.

The final coherent-spectrum guard raises `CoherentSamplingError` for negative
or nonfinite pair power. The exported diagnostic combines those conditions;
it does not record their separate counts. No accepted coherent spectrum,
yield or FWHM is produced; the sample, charge and operator are unchanged,
and no clipping or resampling is applied. Allocation elapsed time was
48 min 37 s, not a successful spectrum-evaluation timing.

The record also retains the unresolved material-dispersion envelope warning:
the finite-axis excluded-power upper bound is 7.444e4 times the frozen
reference for row 2, with a material phase-slope step of 1.537e-5 eV.
This upper bound is not a measured grid error. This run does not resolve
sampling or production-window certification, and does not establish the
full 81-case profile's acceptance.
