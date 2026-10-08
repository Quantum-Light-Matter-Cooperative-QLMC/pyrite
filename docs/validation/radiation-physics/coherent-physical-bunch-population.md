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

This is an implementation-context derivation. Fresh-context verification is
pending; it does not certify the full window-envelope or production sampling claim.

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

Status: **filtered**, pending fresh-context derivation and source-to-code review.
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
