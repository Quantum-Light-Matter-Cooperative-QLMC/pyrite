# Physical scales for a dechanneling boundary

Investigation for #522. These are diagnostic boundary hypotheses; the production
model retains the declared 300 eV margin. Validation claim:
`cr-dechanneling-collapse`. No new validation status is asserted here.

## What the reference constrains

Kephart et al., [Phys. Rev. B **40**, 4249–4263 (1989)](https://doi.org/10.1103/PhysRevB.40.4249),
Secs. III, V B, V D and VI, measure forward radiation versus thickness.
Their Eq. (5) convolves band populations and along-plane angles with the
photon spectrum. Eq. (6) simplifies it when angular broadening is negligible.
The integrated-exponential fit constrains the decay of the radiation-weighted
band population, not a first-passage escape probability.

For 54 MeV electrons the fitted windows cover 1→0 and 2→1 in (100), and 1→0,
2→1 and 3→2 in (110). At 17 MeV the n=2 length includes the photon-absorption
parallel-sum correction: $L^{-1}=(20\,\mu\mathrm m)^{-1}+(120\,\mu\mathrm m)^{-1}$,
giving about 17 µm. The slower apparent decay beyond 1 mm is ambiguous:
the paper also identifies normalization, background and beam-condition
uncertainties. Beam divergence is inferred, not independently measured;
Table II's averages remain assumptions of the reproduction harness.

The harness retains the previously chosen bound-weighted forward-detector
observable. Comparisons must use `detector_ratio.bound`, not the unweighted
`ratio`. Both are stored, together with fit residuals and tolerance results.

## A continuum is not an absorbing boundary

Sen & Lynn, [FERMILAB-PUB-14-243-APC](https://lss.fnal.gov/archive/2014/pub/fermilab-pub-14-243-apc.pdf),
Eqs. (1), (5), (21)–(23) and Sec. 2.3, distinguish reversible interband
transitions from a phenomenological irreversible sink. Their sink index is
fitted to yields and varies with thickness. This source supplies no
parameter-free energy at which return becomes impossible.

For our real, one-dimensional periodic Hamiltonian, making the Bloch gaps
small leaves a near-free transverse electron in the crystal. The scattering
kernel can still return it to a bound band. Spectral continuity alone does
not remove those matrix elements. Consequently a gap-based absorbing rule
is a testable approximation, not a consequence of the eigenproblem.

## Diagnostic definitions

Write $U_l=\min U$, $U_b=\max U$, $D=U_b-U_l$, with both extrema measured
relative to the same potential zero. The paraxial transverse energy measured
from the well bottom is $pv\psi^2/2$, where $pv=\gamma m_ec^2\beta^2$.
An angle multiple $a\geq1$ therefore defines

$$
\psi_c=\sqrt{\frac{2D}{pv}},\qquad
E_{\rm cut}=U_l+a^2D,\qquad
\text{margin}=(a^2-1)D.
$$

This mapping is invariant to shifting the potential zero. At $a=1$ it
reduces to barrier absorption; at fixed $a$ its angle scales as
$\gamma^{-1/2}$ in the relativistic limit. Choosing $a>1$ still introduces
a free parameter. The reconstructed handoff angle uses kinetic energy
relative to the cell mean; it must not be confused with the angle referenced
to the well bottom used to define this diagnostic.

The gap candidate evaluates eigenvalues at $k=0,g/2$ to obtain band extrema
$E_j^-,E_j^+$. Define

$$
w_j=E_j^+-E_j^-,\qquad
r_j=\frac{\max(0,E_{j+1}^--E_j^+)}{(w_j+w_{j+1})/2}.
$$

Choose the first three consecutive gaps with $E_j^+>U_b$ and $r_j\leq f$,
and set $E_{\rm cut}=E_j^+$. A zero denominator does not qualify. Only bands
entirely below $U_l+(\hbar cHg/2)^2/(2\gamma m_ec^2)$ qualify, avoiding
the outer half of the momentum window. This guard and $f$ are explicit
diagnostic choices. Repeat in $H$; lack of a qualifying triplet is an error,
not evidence for irreversible escape. The free-particle limit has touching
bands and no physical escape boundary, illustrating the limitation.

At 293.15 K the computed margins are:

| Plane, energy | Well depth [eV] | $a=4$ margin [eV] | $f=0.1$ margin [eV] | $f=0.01$ margin [eV] |
|---|---:|---:|---:|---:|
| Si(110), 16.92 MeV | 21.040 | 315.601 | 1.984 | 4.997 |
| Si(100), 16.92 MeV | 11.657 | 174.860 | 1.927 | 5.958 |
| Si(110), 54.5 MeV | 21.040 | 315.601 | 2.197 | 2.197 |
| Si(100), 54.5 MeV | 11.657 | 174.860 | 1.407 | 3.325 |

The gap margins agree at $H=32$ and $H=48$ to the six decimal places checked
(harness revision `27670b6e`). This establishes convergence of the spectral
threshold only. It says nothing yet about depth-evolution convergence or
agreement with occupation lengths. An approximately continuous free band
is reached far below the calibrated 300 eV margin.

## Screening and partial convergence evidence

Remote screening arrays 1422 and 1454 used $H=32$, 16 Bloch cells and
$\Delta z=125$ Å. The entries below are the bound-weighted fitted length
divided by the corresponding Kephart length. They compare boundary
hypotheses on one grid; they are not converged validation anchors.

| Case | 300 eV | No cutoff | $a=1$ | $a=2$ | $a=3$ | $a=4$ | $f=0.1$ | $f=0.01$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Si(110), 17 MeV, n=1 | 0.779 | 1.319 | 0.090 | 0.351 | 0.582 | 0.796 | 0.105 | 0.127 |
| Si(110), 17 MeV, n=2 | 0.910 | 1.512 | 0.115 | 0.422 | 0.687 | 0.929 | 0.136 | 0.163 |
| Si(110), 17 MeV, n=3 | 0.906 | 1.481 | 0.113 | 0.431 | 0.690 | 0.925 | 0.142 | 0.173 |
| Si(100), 17 MeV | 0.968 | 1.779 | 0.100 | 0.339 | 0.545 | 0.752 | 0.130 | 0.172 |
| Si(100), 54 MeV | 1.233 | 1.718 | 0.101 | 0.408 | 0.700 | 0.964 | 0.120 | 0.147 |
| Si(110), 54 MeV | 0.796 | 0.873 | 0.078 | 0.366 | 0.614 | 0.812 | 0.096 | 0.096 |

Absorption at the free-band onset severely underpredicts all six lengths
on this grid. This weighs against interpreting spectral continuity as
irreversible escape. The no-cutoff results cannot yet establish the
large-window limit: final overflow is 0.64–0.82 of incident flux and the
absolute final balance defect reaches 0.26. Increasing the tracked window
and reducing the depth step remain necessary for that hypothesis.

Arrays 1458 and 1497 refine angle candidates at revision `54b29d7c`.
Completed $H=48$ versus $H=64$ pairs for $a=3,4,5$ change fitted lengths
by less than 0.3% at $\Delta z=125$ Å, including both 54 MeV cases. For
$a=4$, doubling Bloch cells from 16 to 32 changes the
lengths by less than 0.5%. Overflow itself remains more basis-sensitive;
convergence of these fitted lengths does not establish convergence of the
handoff distribution.

For example, at $a=4$ and 125 Å the final overflow fractions for $H=48$
versus $H=64$ are 0.0407/0.0231 for Si(110), 17 MeV; 0.0134/0.00855 for
Si(100), 17 MeV; 0.0334/0.0211 for Si(100), 54 MeV; and 0.1085/0.0557 for
Si(110), 54 MeV. The almost unchanged fitted lengths coexist with a strong
change in the partition between resolved escape and overflow. A length-only
convergence gate would therefore miss the handoff uncertainty.

Depth-step refinement at $a=4$, $H=48$, 16 Bloch cells gives:

| Case | Ratio, dz=125 Å | Ratio, dz=62.5 Å | Ratio, dz=31.25 Å |
|---|---:|---:|---:|
| Si(110), 17 MeV, n=1 | 0.796 | 0.828 | 0.838 |
| Si(110), 17 MeV, n=2 | 0.930 | 0.966 | 0.978 |
| Si(110), 17 MeV, n=3 | 0.925 | 0.961 | 0.973 |
| Si(100), 17 MeV | 0.752 | 0.773 | 0.778 |
| Si(100), 54 MeV | 0.964 | 1.027 | 1.044 |
| Si(110), 54 MeV | 0.822 | 0.912 | 0.939 |

The final balance defect (`trace + total_flux - 1`) for Si(110) at 17 MeV
is −0.057, −0.020 and −0.0051 across these steps; for Si(100), −0.039,
−0.011 and −0.0028. The last halving reduces the defect approximately
fourfold, consistent with the integrator's second-order convergence.
Si(100) at 17 MeV remains outside the ±20% acceptance interval for $a=4$:
its finest result is 0.778. At 54 MeV the balance defects for (100) and
(110) are −0.096/−0.16, −0.026/−0.048 and −0.0068/−0.013 at 125, 62.5 and
31.25 Å, respectively. Thus the coarse-step lengths cannot decide the
model, while the 31.25 Å results put both 54 MeV cases within the length
criterion.

The $a=4$, dz=125 Å plane-length ratio at 54 MeV is about 0.78 versus the
measured $24/36=0.67$. At $a=5$, dz=62.5 Å, the individual ratios are
1.306 for Si(100) and 1.129 for Si(110), with defects −0.039 and −0.069.
Thus raising $a$ can move the 54 MeV Si(100) case outside tolerance even
while improving the 17 MeV cases. A shared $a$ is still a fitted parameter;
agreement alone would not derive an irreversible boundary from physics.
No production replacement or new validation status follows from these
partial results.

At 17 MeV, $a=4.5$, dz=62.5 Å gives ratios 0.930, 1.082, 1.073 for
Si(110) n=1,2,3 and 0.875 for Si(100); at 54 MeV it gives 1.169 for
Si(100) and 1.026 for Si(110). All six pass the length criterion on that
grid. Its balance defects remain −0.024, −0.013, −0.032 and −0.059 in the
same case order. The completed refinements below test depth step, $H$ and
Bloch cells separately.

The screening outputs are `scratch/boundary-screen/run-{0..31}.json` in
the isolated remote checkout `/home/user/pyrite-522`; the four timed-out
screening tasks were rerun by array 1454. Each task uses case index
`id % 4` in `(si110_17, si100_17, si100_54, si110_54)` and candidate
index `id // 4` in `(300 eV, inf, a=1, a=2, a=3, a=4, f=0.1, f=0.01)`.
Convergence and depth-refinement outputs are in `scratch/boundary-conv/`
and `scratch/boundary-dz/`, with the exact task mapping in each
`submit.sh`. Every expected JSON retrieved on 2026-10-17 parses and its log
reports exit status zero. Inspect `detector_ratio.bound`, `balance_defect`
and `final_overflow_flux` together.

Array 1513 tests the promising $a=4.5$ candidate at the same source revision.
Tasks 0--3 use $H=48$, 16 cells and 31.25 Å; tasks 4--7 use $H=64$, 16 cells
and 62.5 Å; tasks 8--11 use $H=48$, 32 cells and 62.5 Å. Within each group,
task index modulo four selects `(si110_17, si100_17, si100_54, si110_54)`.
Its outputs and logs are `scratch/boundary-a45-refine/` in the isolated remote
checkout. All twelve results were retrieved on 2026-10-17; each JSON parses,
records revision `54b29d7c`, and has a log reporting exit status zero.

| Case | H48, 16 cells, dz31.25 Å | H64, 16 cells, dz62.5 Å | H48, 32 cells, dz62.5 Å |
|---|---:|---:|---:|
| Si(110), 17 MeV, n=1 | 0.944 | 0.930 | 0.931 |
| Si(110), 17 MeV, n=2 | 1.097 | 1.082 | 1.082 |
| Si(110), 17 MeV, n=3 | 1.088 | 1.073 | 1.073 |
| Si(100), 17 MeV | 0.883 | 0.875 | 0.874 |
| Si(100), 54 MeV | 1.193 | 1.169 | 1.174 |
| Si(110), 54 MeV | 1.062 | 1.027 | 1.025 |

Relative to the matching H48/16-cell/dz62.5 Å baseline, increasing H changes
lengths by at most 0.12%, doubling cells by 0.48%, and halving the depth step
by 3.54%. All six pass on each grid, but the finest Si(100) 54 MeV ratio is
only 0.007 below the upper acceptance boundary. Independent refinements do
not bound their combined error; a further depth halving and combined finer
grid are needed before declaring tolerance robust.

At dz31.25 Å the final balance defects in case order are −0.00624,
−0.00345, −0.00839 and −0.01546; overflow fractions are 0.0532, 0.0173,
0.0458 and 0.1684. At dz62.5 Å increasing H48 to H64 reduces the 54 MeV
Si(110) overflow from 0.1615 to 0.0793 while its length changes by only
0.12%. Thus stable lengths still do not establish a stable handoff
distribution. These are convergence tests, not a model recommendation.

Array 1525 tests the no-cutoff hypothesis on larger windows with 16 Bloch
cells throughout. Tasks 0--3 use $H=48$ and 62.5 Å, tasks 4--7 use $H=48$
and 31.25 Å, and tasks 8--11 use $H=64$ and 31.25 Å. The case order within
each group is the same as array 1513. Outputs, logs and the submission
script are in `scratch/boundary-nocut/` in the isolated remote checkout;
the source stamp remains `54b29d7c`. At most four allocations run concurrently.
Comparing the first two groups isolates the depth-step error; comparing the
last two isolates window dependence at the finer step. Any persistent
overflow or balance defect must accompany the fitted lengths when judging
the large-window limit.

All twelve no-cutoff results were also retrieved on 2026-10-17, parsed,
checked against revision `54b29d7c`, and matched to exit-status-zero logs.

| Case | H48, dz62.5 Å | H48, dz31.25 Å | H64, dz31.25 Å |
|---|---:|---:|---:|
| Si(110), 17 MeV, n=1 | 1.566 | 1.596 | 1.640 |
| Si(110), 17 MeV, n=2 | 1.784 | 1.818 | 1.865 |
| Si(110), 17 MeV, n=3 | 1.739 | 1.772 | 1.816 |
| Si(100), 17 MeV | 1.910 | 1.945 | 1.953 |
| Si(100), 54 MeV | 2.531 | 2.713 | 2.984 |
| Si(110), 54 MeV | 1.385 | 1.465 | 1.772 |

No case passes. At the finer step, H48 to H64 increases the 54 MeV lengths
by 10.0% for Si(100) and 21.0% for Si(110). H64 final overflow fractions
are 0.3702, 0.1798, 0.7280 and 0.8926, with balance defects −0.0111,
−0.00795, −0.0556 and −0.0427. The no-cutoff window limit remains
unresolved, and these finite-window results provide no successful replacement
for the calibrated boundary. The a=4.5 candidate improves plane dependence
but still supplies an adjustable sink, not a derived irreversible boundary.
Production behavior and validation statuses remain unchanged.

Array 1560 follows up the marginal $a=4.5$ result. Tasks 0–3 use H48,
16 cells and dz15.625 Å; tasks 4–7 use H64, 32 cells and dz31.25 Å.
Both groups use `(si110_17, si100_17, si100_54, si110_54)` in that order.
The first group isolates another depth halving; the second combines the
previously separate window and Bloch refinements. Outputs, logs and
`submit.sh` are in `scratch/boundary-a45-combined/` in the isolated remote
checkout. Each task checks the harness SHA-256 before running and records
source revision `54b29d7c`. The array permits four concurrent single-CPU
allocations, 4 GiB each, with a 24-hour time limit. Submission is not
completion evidence; compare parsed outputs and successful logs before
using these runs.

On 2026-10-18, tasks 0–5 were retrieved to
`docs/validation/check-records/kephart_occupation_runs/a45-combined-1560/`.
Their JSON parses, records revision `54b29d7c`, and each paired log reports
exit status zero. The depth-halved H48 results give bound-weighted ratios
0.948, 1.101, 1.092, 0.885, 1.199 and 1.072 in the stated case order;
Si(100), 54 MeV remains only 0.001 below the upper tolerance boundary.
The completed H64/32-cell results cover only the two 17 MeV cases, with
ratios 0.944, 1.098 and 1.089 for Si(110) $n=1,2,3$, and 0.882 for Si(100).
Tasks 6–7 were subsequently retrieved on 2026-10-18. All eight JSON outputs
parse, record revision `54b29d7c`, and pair with exit-status-zero logs; the
four newly retrieved files also match the remote SHA-256 hashes. The combined
H64/32-cell/dz31.25 Å grid gives 1.198623 for Si(100), 54 MeV and 1.062592
for Si(110), 54 MeV. Thus all six targets pass on both completed grids,
but Si(100), 54 MeV remains marginal: only 0.001377 below the upper bound
on the combined grid. These grids do not combine all three refinements;
H64/32 cells at dz15.625 Å remains untested.

For the newly retrieved Si(100)/Si(110) 54 MeV configurations respectively,
final overflow is 0.024204/0.082854 and balance defect is
−0.008441/−0.015466. On the depth-halved H48 grid, the corresponding
overflow is 0.046086/0.170181 and balance defect is
−0.002123/−0.003933. These simultaneous window/Bloch/depth differences
do not isolate a single cause, and passing length ratios do not establish
handoff convergence. Array 1560 is fully retrieved; this is evidence for
the adjustable candidate, not a fresh-context validation or a physically
derived boundary recommendation. Production behavior and validation
statuses remain unchanged.

Array 1577 tests the remaining joint refinement: H64, 32 cells and
dz15.625 Å at $a=4.5$ for `(si110_17, si100_17, si100_54, si110_54)`
(tasks 0–3). It retains source revision `54b29d7c` and the same harness
SHA-256 check. The script is retained under
`docs/validation/check-records/kephart_occupation_runs/a45-final-1577/submit.sh`;
remote outputs go to `scratch/boundary-a45-final/`. Each allocation requests
one CPU and 4 GiB, with a 36-hour limit and at most two simultaneous tasks.
Tasks 0–1 completed and were retrieved on 2026-10-19. The remaining local
54 MeV outputs were reviewed on 2026-10-20. All four JSON files parse, record
revision `54b29d7c`, match the remote SHA-256 hashes, and pair with remote
logs reporting `Exit status: 0`. Their stored bound-weighted lengths, ratios,
tolerance flags and final flux balances are internally consistent.

| Case | Bound-weighted length [µm] | Length / Kephart | Within ±20% |
|---|---:|---:|---|
| Si(110), 17 MeV, n=1 | 18.956149 | 0.947807 | Yes |
| Si(110), 17 MeV, n=2 | 18.728251 | 1.101662 | Yes |
| Si(110), 17 MeV, n=3 | 18.573398 | 1.092553 | Yes |
| Si(100), 17 MeV | 14.144110 | 0.884007 | Yes |
| Si(100), 54 MeV | 28.918240 | 1.204927 | **No** |
| Si(110), 54 MeV | 38.600834 | 1.072245 | Yes |

The joint refinement therefore breaks the apparent six-case agreement:
Si(100), 54 MeV exceeds the upper accepted length of 28.8 µm by 0.118240 µm.
At fixed H64/32 cells, halving dz31.25→15.625 Å raises its fitted length by
0.526%, from ratio 1.198623 to 1.204927. Si(110), 54 MeV increases by
0.909%, from 1.062592 to 1.072245. The same depth halving reduces their
absolute final balance defects by factors 3.951 and 3.930, respectively.
A conditional second-order extrapolation of these two ratios gives 1.207028
and 1.075463; this is a numerical diagnostic, not another computed grid or
an error bound. It offers no evidence that further depth refinement would
restore the Si(100) pass.

| Configuration | Final overflow, Si(100) 54 MeV | Final overflow, Si(110) 54 MeV | Balance defect, Si(100) | Balance defect, Si(110) |
|---|---:|---:|---:|---:|
| H64, 32 cells, dz31.25 Å | 0.024204 | 0.082854 | −0.008441 | −0.015466 |
| H64, 32 cells, dz15.625 Å | 0.024348 | 0.083738 | −0.002136 | −0.003935 |
| H48, 16 cells, dz15.625 Å | 0.046086 | 0.170181 | −0.002123 | −0.003933 |

At the finest step, the combined H48/16→H64/32 refinement reduces overflow
by 47.2% and 50.8% for the two 54 MeV planes, despite almost unchanged
Si(110) length. This comparison combines window and Bloch changes; it does
not isolate either. It does establish that length stability alone cannot
justify replacing the overflow-angle prescription. At the joint refinement,
17 MeV Si(110)/Si(100) overflow is 0.030686/0.009950, with balance defects
−0.001578/−0.000869.

### Bounded conclusion for approaches 1–2

The tested free-band-onset sink fails the six-case comparison, the finite
no-cutoff windows overpredict all six lengths without a resolved window
limit, and the adjustable a=4.5 sink fails Si(100), 54 MeV on the jointly
refined grid. Thus these prototypes supply no demonstrated replacement for
the declared 300 eV model. This is a negative result for the tested choices,
not a proof that every critical-angle multiple fails: no converged scan
between a=4 and a=4.5 has been performed. Selecting another multiple from
these measurements would remain calibration rather than a derivation of
irreversible escape.

The research recommendation is to retain 300 eV for reproducibility and
carry the unresolved handoff distribution into the conditional overflow-tail
investigation below. Tail sampling must first reproduce the existing overflow
weight; changing only its outgoing angles would not repair the failed
occupation-length comparison, because the current zone evolution already
removes that flux. A claim that it repairs occupation lengths would require
feedback/re-entry or a changed evolution operator and a separate comparison.
No production model or model marker changes follow from this review. The
fresh-context reviews and conditional tail prototype are recorded below;
they do not establish a physical replacement boundary or validate the size
of the rechanneling omission.


## Experimental differential overflow instrument

The opt-in `montecarlo/channeling/overflow.py` prototype contracts a supplied
plane density with an expanded differential transfer kernel without wrapping
outgoing momentum. An optional solver observer accumulates the exact discrete
jump-source quadrature used to book overflow, permitting full-depth marginals
without storing a density history. Fixed-energy measurements are driven by
`checks/channeling_overflow.py`. No production handoff selects this path.

Fresh-context validation re-derived the coherent contraction, Bloch residue
carry, closure identity and observer quadrature. The new diagnostic claim
`cr-planar-overflow-spectrum` is anchored on analytic tests and observer parity;
this does not validate a physical escape boundary or a converged material tail.
The same independent review confirms the negative a=4.5 comparison above and
the declared handoff accounting within its approximations.

Initial remote array 1614 uses an isolated checkout
`/home/user/pyrite-522-overflow`, with task mapping and source hashes in
`docs/validation/check-records/channeling_overflow_runs/initial/submit.sh`.
The first two retrieved results (Si(110), 5 MeV, H16, 16 cells, 20 µm,
300 eV margin) give:

| dz [Å] | Reference overflow | Resolved outside, T64 | Unresolved weight, T64 | In-band gain shift, T64 | Conditional mean absolute angle / band-edge assignment |
|---|---:|---:|---:|---:|---:|
| 125 | 0.165639 | 0.138827 | 0.009655 | 0.017158 | 1.540759 |
| 62.5 | 0.165880 | 0.139074 | 0.009687 | 0.017119 | 1.540991 |

The reference overflow equals resolved outside weight plus unresolved weight
plus changed in-band gain. Neither run passes the 1% closure gate, so sampling
is refused. These finite-window conditional means are not converged replacement
handoff distributions. The final solver balance defects are −0.010285 and
−0.007032, respectively; these depth steps do not establish convergence.

An omitted transfer can land inside the original particle band. Consequently
the current band-edge angle is a declared assignment, not a proven lower bound
for the entire legacy overflow residual. Even resolved tail support depends on
the Bloch residue. Production behavior remains unchanged.

All four tasks completed. The retrieved 54.5 MeV records use H48, 16 cells,
$\Delta z=31.25$ Å, a 20 µm path and the declared 300 eV margin. Their
largest tested transfer window gives:

| Plane | Transfer order | Reference overflow | Resolved outside | Unresolved weight | In-band gain shift | Conditional mean absolute angle / band-edge assignment |
|---|---:|---:|---:|---:|---:|---:|
| Si(100) | 96 | 0.019860 | 0.011327 | 0.007262 | 0.001271 | 1.329262 |
| Si(110) | 96 | 0.038881 | 0.026567 | 0.008730 | 0.003584 | 1.323744 |

The unresolved weights are 36.57% and 22.45% of reference overflow; the
in-band gain shifts are another 6.40% and 9.22%. Both independently exceed
the 1% closure tolerance. Increasing transfer order from 64 to 96 raises
the conditional mean absolute angles by 14.03% and 11.74%, respectively.
Thus neither plane has a closed or converged replacement tail on these
grids. Solver balance defects are −0.003698 and −0.004264. The conditional
angle ratios describe only the resolved finite-window measures; no accepted
tail samples were drawn.

Both tasks wrote valid JSON and ended with diagnostic exit status 1, as
expected when closure is refused. The [artifact record](../../validation/check-records/channeling_overflow_runs/initial/README.md)
links the parameters, source hashes and retrieved outputs, and distinguishes
schema-1 field names from schema-2 reporting vocabulary. Integrated full
loss is an expected jump count, not a probability bounded by one.

### Recommendation after boundary and tail prototypes

Retain the declared 300 eV model for reproducibility. The tested boundary
prototypes do not support a replacement, and the differential instrument
refuses every tested tail window in array 1614. This is a bounded negative
result: it rejects promotion of these computed candidates, not every possible
boundary or differential-kernel model. Sampling an unclosed tail after
renormalization would conceal unresolved weight and changed in-band gain.
Larger transfer windows alone also do not demonstrate compatibility with
the legacy overflow residual; that compatibility must be measured anew.

The production band-edge angle remains a declared assignment. Its earlier
description as a lower bound is withdrawn for the whole legacy overflow
channel, following the independent instrument review. No production output
changes, so `CHANNELING_MODEL` is unchanged. A different angle assignment
alone would leave the zone's occupation lengths unchanged; a replacement
escape model must separately pass the detector-weighted six-case comparison.

The existing fresh-context verdicts cover projected handoff accounting,
the negative a=4.5 comparison and the finite-tail instrument. They do not
anchor a new escape law or bound the error from omitted classical returns.
The latter remains a validity limit as specified below; no quantitative
thickness or energy threshold for negligible rechanneling has been derived.

## Rechanneling integration assessment

The angle-only hook in candidate 3 is a no-go for production in this slice.
This is an implementation/design assessment against ADR-0018 and the existing
`cr-dechanneling-collapse`, `cr-entry-state-projection` and
`cr-zone-handoff-accounting` contracts, not a new validated capture model.
The decision does not assert that rechanneling is negligible. In particular,
the one-way model supplies no validated treatment of repeated capture in
millimetre-thick targets; the ambiguous experimental tail cannot establish
that omission's size. The current results remain diagnostics of the stated
Kephart fitting windows, not an extrapolation to that regime.

An angle crossing does not specify a quantum entry density. A return rule
also needs the position relative to the planes, local kinetic energy,
remaining path and a declared mixed-state or wavepacket prescription.
The existing `planar_entry_density(..., impact="fixed")` can retain a
chosen state's position relative to the planes; its averaged-impact mode
would erase that information. Neither mode derives the returning state
from a classical position and direction. Choosing a packet width would
introduce another model assumption requiring independent validation.

The available solver seam is explicit: construct a positive plane-wave
density on the solver's Bloch grid at the return energy, normalized together
with its outside-band weight to one;
`DensityMatrixSolver.entry_from_plane` separates it into retained density,
in-basis rejected weight and outside-band weight. Pass the retained density
and the sum of rejected weights to `propagate(initial=(rho_S, entry_flux))`
over the remaining path. These are weighted branches: renormalizing the
retained density to one while also retaining the rejected branch would
double-count returning electrons. Bound/free occupation inside the retained
subspace must remain governed by the existing jump operator, rather than
be replaced by an angle-based assignment of unit bound probability.

The runner currently performs one ensemble zone solve, samples one handoff,
and calls native continuation for the remainder of the trajectory. Repeated
returns therefore need a new stop/resume exchange between native transport
and the zone solver, preserving position, direction, energy, time, electron
identity, RNG progress and cumulative flux across each exchange. The
deterministic radiation source would also need the returning population;
changing only transported trajectories would leave source and transport
describing different ensembles. These requirements rule out treating a
post-processing angle filter as a rechanneling implementation.

## Overflow information needed for candidate 4

The stored `ScatteringKernel.momentum_kernel` covers the retained transfer
window; `diagonal_coefficients` contains the full-transfer loss diagonal.
`DensityMatrixSolver.jump` returns the retained jump, the above-cutoff
in-basis diagonal, and the total in-band trace. The residual loss therefore
measures overflow but does not supply its outgoing momentum distribution.
An accessor cannot recover a unique tail distribution from that residual
alone. The prototype above retains differential transfer information from
the TDS and electronic kernel builders and conditions it on the incoming
state and on landing outside the tracked window. Its closure tests require
agreement with the existing overflow weight before sampled angles can
replace the declared band-edge assignment. The measured failures leave
that requirement unmet. The strong H dependence measured above makes
normalization and conditioning essential; fitted occupation lengths alone
cannot validate the tail.

The boundary-sweep source is pinned to the pre-axial revision `54b29d7c`, rather than
the rebased issue-branch tip. SHA-256 checks on the harness and the planar
potential, thermal, transverse, entry, density-matrix and scattering-operator
owners match that recorded revision. Reproduction should retain this source
stamp; testing a later production revision is a separate comparison.
The differential overflow instrument uses the separate revision and owner
hashes recorded in its array-1614 `submit.sh`, not the boundary-sweep stamp.

## Reproduction

Run on the lab through an isolated `pyrite remote sync` checkout and SLURM;
do not run this sweep locally:

```bash
PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 uv run --no-default-groups \
  python checks/kephart_occupation.py \
  --cases si110_17 si100_17 si100_54 si110_54 \
  --margins 300 inf --critical-angle-multiples 1 2 3 4 \
  --gap-fractions 0.1 0.01 --orders 32 --cells 16 --dz 125 --out boundaries.json
```

The output records the resolved margin, barrier, well depth, critical angle,
gap criterion and endpoint bands. Screening at this grid does not establish
convergence. Refine $H$, $\Delta z$ and the Bloch grid separately before
recommending a production model. `--margins` with no values suppresses fixed
margin runs, allowing one candidate per scheduler allocation. The older
`--refit` mode continues to select only the declared 300 eV margin.
