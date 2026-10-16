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
same case order. Its 31.25 Å, $H=64$ and full 32-cell checks are not yet
complete. Those refinements remain necessary before assessing one shared angle
multiple.

The screening outputs are `scratch/boundary-screen/run-{0..31}.json` in
the isolated remote checkout `/home/aamador/pyrite-522`; the four timed-out
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
checkout. The completed 32-cell checks at 17 MeV give ratios 0.931, 1.082,
1.073 for Si(110) n=1,2,3 and 0.874 for Si(100), changing the 16-cell lengths
by less than 0.08%. Their logs report exit status zero. The remaining ten
tasks were still running at retrieval; no result is inferred from that state.
These are convergence tests, not a model recommendation.

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

The remote source is pinned to the pre-axial revision `54b29d7c`, rather than
the rebased issue-branch tip. SHA-256 checks on the harness and the planar
potential, thermal, transverse, entry, density-matrix and scattering-operator
owners match that recorded revision. Reproduction should retain this source
stamp; testing a later production revision is a separate comparison.

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
