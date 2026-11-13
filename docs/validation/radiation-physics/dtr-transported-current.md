# Transported-current Maxwell field: independent derivation

## Scope and independence

Claim: `dtr-transported-current`. This evidence page concerns charge currents,
not the separately ledgered CBS force source or absolute TR normalization.
The independent calculation below was recorded on 2026-11-13 before reading
implementation bodies, author implementation derivations, or earlier independent
DTR write-ups. Inputs were the ledger row and AST-extracted signatures,
class fields and docstrings of `dtr_history.py` and `dtr_emission.py`.

The source equations are Maxwell's curl laws, with local scalar permittivity
and unit magnetic permeability. Their underlying field equations appear in
[Feynman Lectures II, 20-1, Eq. (20.1)](https://www.feynmanlectures.caltech.edu/II_20.html).
The Gaussian-unit conversion and the material/two-harmonic projection below
are derived here; no numbered Caticha equation or published spectrum is being
reproduced. The ledger's author equations are comparison material, not an
independent source for this calculation.

Intended output: coupled and uncoupled outgoing Cartesian amplitudes per
energy and history, in the existing TR convention of inverse eV. Positions,
path-clock time and durations use Å with $c=1$; velocities are fractions of
$c$, charges are signed multiples of $+e$. Each history is a continuous,
piecewise straight conserved point-current path, supplemented by its exterior
continuations at the faces. Fourier frequencies have units of inverse Å.
The model assumes a planar laterally infinite finite stack, vacuum outside,
one perfect two-harmonic crystal layer, classical prescribed trajectories,
no recoil, and no inter-electron reduction inside the field solver.

## Filters before implementation inspection

The finite current must be integrated in time, not divided by $v_z$:
a horizontal segment has a finite nonzero source. Integrals must be additive
under constant-velocity subdivision. Charge reversal changes the field sign
and leaves its intensity unchanged. Zero crystal coupling makes coupled and
uncoupled charge fields identical, hence their correction vanishes even for
an accelerated finite history. A globally straight current, with both exterior
tails, must reduce to the straight-passage boundary problem. A common space/time
translation contributes only a unit-modulus phase for real observation direction.
These dimensional, limiting and sign filters are consistent at the equation
level; implementation results are recorded separately below.

## Curl reduction and point-current source

Choose inverse Fourier factors $\exp(-i\omega t+i\mathbf K_\parallel\cdot \mathbf r_\parallel)$ and order tangential fields as
$\mathbf f=(E_x,E_y,H_x,H_y)^T$. In Gaussian units with $c=1$,

$$
\nabla\times\mathbf E=i\omega\mathbf H,\qquad
\nabla\times\mathbf H=-i\omega\varepsilon\mathbf E+4\pi\mathbf J.
$$

The longitudinal equations give

$$
H_z=(K_xE_y-K_yE_x)/\omega,\qquad
E_z=(K_yH_x-K_xH_y)/(\omega\varepsilon)
       -4\pi iJ_z/(\omega\varepsilon).
$$

Consequently $\partial_z\mathbf f=iM\mathbf f+\mathbf b$ with

$$
\mathbf b=4\pi
\begin{pmatrix}
K_xJ_z/(\omega\varepsilon)\\
K_yJ_z/(\omega\varepsilon)\\
J_y\\-J_x
\end{pmatrix}.
$$

The positive longitudinal-current terms are essential: dropping them loses
charge's longitudinal contact field after eliminating $E_z$. For two harmonics,
$\varepsilon$ is the two-by-two convolution matrix, each $K_x,K_y$ is diagonal
in harmonic index, and the longitudinal terms use $K_x\varepsilon^{-1}\mathbf J_z/\omega$ and $K_y\varepsilon^{-1}\mathbf J_z/\omega$. Matrix order matters.
The remaining curl terms define the eight-dimensional homogeneous generator;
this claim uses that boundary system without recertifying its eigenmode solver.

For a point charge $q$ with path $\mathbf r(t)$, its Fourier current is
$q\int dt\,\mathbf v(t)\delta(z-z(t))\exp(i\omega t-i\mathbf K_\parallel \cdot\mathbf r_\parallel(t))$. In harmonic gauge $h$, include
$\exp(-ihg_z z(t))$, with $\mathbf K_{h\parallel}=\mathbf k_\parallel+ h\mathbf g_\parallel$. Thus integrating the spatial source against $dz$
reduces exactly to a time integral with drive proportional to $q\mathbf v$.
There is no residual $1/\lvert v_z\rvert$ in a finite-segment Green integral.

## Modal finite-segment Green function

Let $M V=V\Lambda$, modal source $\mathbf d=V^{-1}\mathbf b$, and a segment
$\mathbf r(t_0+u)=\mathbf r_0+\mathbf v u$, $0\le u\le T$. At face $z_f$,
mode $a$ receives

$$
I_a(z_f)=d_a\exp\!\left[i\omega t_0-i\mathbf K_{h\parallel}\cdot
\mathbf r_{0\parallel}-ihg_z z_0+i\lambda_a(z_f-z_0)\right]
F(A_a,T),
$$

$$
A_a=\omega-\mathbf K_{h\parallel}\cdot\mathbf v_\parallel
 -(hg_z+\lambda_a)v_z,\qquad
F(A,T)=\frac{\exp(iAT)-1}{iA},\qquad F(0,T)=T.
$$

The outgoing Green field is $+V_+ I_+(z_{\rm high})$ at the upper face and
$-V_- I_-(z_{\rm low})$ at the lower face. The lower sign follows from the
unit jump of the Green function: the difference of upper and lower modal
projectors is the identity. These signs also apply when $v_z<0$ or $v_z=0$;
mode direction, rather than particle direction, selects the projector.
Numerically, evaluation should avoid subtracting enormous intermediate
exponentials for passive opaque media. Coincident endpoint phases give $T$.

$\mathbf b$ is linear in dimensionless velocity and charge; $F$ has length.
The integrated source has the same dimensions as a straight-current particular
field, since a longitudinal Fourier mismatch supplies one inverse length.
Conversion to inverse eV occurs once in the outgoing-field normalization,
using $k=E/(\hbar c)$; it does not modify the modal phase or relative sign.

## Exterior tails and interface matching

For an exterior straight ray through a boundary event $(\mathbf r_b,t_b)$,
parameterize the ray by $u=t-t_b$. Put
$A_a=\omega-\mathbf k_\parallel\cdot\mathbf v_\parallel-\lambda_a v_z$.
The adiabatically switched integrals for prehistory and posthistory are,
respectively,

$$
\int_{-\infty}^{0}du\,e^{iA_a u}=\frac1{iA_a},\qquad
\int_0^\infty du\,e^{iA_a u}=-\frac1{iA_a}.
$$

These are limiting-absorption expressions, not absolutely convergent real-axis
integrals. Equivalently the uniform-current particular field at the boundary
is obtained by substituting its longitudinal phase into the exterior curl
system. Only modes whose Green field reaches the stack contribute to its
boundary forcing: upper-going modes from the lower exterior and lower-going
modes from the upper exterior. The pre/post sign follows from whether the
ray approaches or leaves the stack. Tangential field continuity then solves
one linear boundary problem with the finite-segment face fields and exterior
particular fields as known inhomogeneous jumps. The observed outgoing field
is the upper vacuum homogeneous radiation, transverse to its wavevector.

## Subtraction, baseline, and history grouping

Linearity gives coupled field $\mathbf C_j$ and uncoupled charge field
$\mathbf U_j$ for the same complete history $j$. The medium-induced correction
is $\mathbf D_j=\mathbf C_j-\mathbf U_j$. With the separately modeled local
crossing-TR amplitude $\mathbf T_j$, the intended joint amplitude is

$$
\mathbf A_j=\mathbf T_j+\mathbf D_j,\qquad
L_j=\lVert\mathbf A_j\rVert^2-\lVert\mathbf T_j\rVert^2
=2\operatorname{Re}(\mathbf T_j^*\cdot\mathbf D_j)
 +\lVert\mathbf D_j\rVert^2.
$$

This is a signed line correction. Subtracting intensities of $\mathbf C_j$
and $\mathbf U_j$ would be a different expression and retain unwanted
interference with the ordinary finite-current field. Adding a separate
positive line intensity would lose the TR/line cross term. Summing segment
intensities would likewise lose signed intra-history interference.
For $N$ incoherent sampled histories, reduction is $N^{-1}\sum_jL_j$ times
the existing spectral normalization. For coherent populations, first define the quadratic estimator independently
of the stored baseline choice. The
physical-population distinct-pair estimator for $N>1$ sampled fields $\mathbf X_j$
is

$$
Q_P(X)=\frac1N\sum_j\lVert\mathbf X_j\rVert^2+
\frac{P-1}{N(N-1)}\left(\left\lVert\sum_j\mathbf X_j\right\rVert^2
-\sum_j\lVert\mathbf X_j\rVert^2\right).
$$

If both stored components used coherent baselines, their correction would
be the difference of this estimator for joint and TR fields. The actual
baseline convention is identified in the implementation comparison. Sampled
space/time offsets already occur in $\mathbf X_j$ and must not receive a
second analytic form factor. This construction establishes an accounting
model, not a theorem that locally straight crossing TR is exact for a
scattered trajectory; that approximation remains explicit.

## Implementation comparison and verdict

Inspection followed recording the derivation. The generic coherent-baseline
paragraph above was then clarified to distinguish the field estimator from
the repository's stored-component convention; the independent curl source,
finite-segment integral and subtraction expressions were unchanged.

| Term or convention | Implementation comparison | Result |
| --- | --- | --- |
| Charge drive and contact term | `_drive` uses $4\pi(K_x\varepsilon^{-1}J_z/\omega,K_y\varepsilon^{-1}J_z/\omega,J_y,-J_x)$ in the harmonic-major tangential ordering. Polarization/magnetization terms belong to the separate CBS claim. | Matches |
| Finite-segment slope and phase | `_segment_faces` has $\omega-\mathbf K_{h\parallel}\cdot\mathbf v_\parallel-hg_zv_z-\lambda_a v_z$ and the independently derived face phase. | Matches |
| Modal projectors | Upper field is positive; lower field is negative. Neither finite source nor time integral divides by $v_z$. | Matches |
| Equal-exponent and passive evaluation | `_time_integral` switches to the bounded endpoint and uses `expm1`; its small argument series is $1+\delta/2+\delta^2/6$. | Matches |
| Exterior field | `_exterior_rhs` uses the vacuum infinite-straight particular field, with bottom plus/top minus in the continuity right-hand side. This is equivalent to the tail projectors above after outgoing homogeneous coefficients are redefined. | Matches |
| Units | Code uses $\omega=E$ in eV, divides Å coordinates/clocks by $\hbar c$, multiplies reciprocal Å vectors by $\hbar c$, and sets current charge to $\sqrt{\alpha}\,q$. Result is inverse eV, as for the existing own field. | Matches |
| Trajectory grouping | `_transport_current` retains each row's path-clock duration and signed charge; `_segment_rhs` accumulates face fields into columns indexed by history before solving. Exterior events alone supply own fields. | Matches |
| Ordinary-current removal | `joint = tr + (coupled - bare) * flip`; charge-only uncoupled fields use the same histories, with zero harmonic coupling and no CBS potential. | Matches |
| Incoherent line slot | `case_dtr_spectrum` returns $a[Q_1(A)-Q_1(T)]$, $a=E n_z^2/(16\pi^4)$; adding stored TR returns $aQ_1(A)$. | Matches |
| Coherent line slot | `_power` gives $Q_P$ with pair weight $(P-1)/(N-1)$. The returned slot is $a[Q_P(A)-Q_1(T)]$, because stored TR remains history-only. Adding stored TR returns $aQ_P(A)$. Subtracting $Q_P(T)$ here would fail that component contract. | Matches |
| Backward observation | `_emitting_frame` makes a proper rotation with diagonal $(1,-1,-1)$, reverses the layer order and shifts both positions and lattice origin to the new slab bottom. `crossing_amplitudes` uses the same emitting-face phase convention; its crossing phase depends only on transverse position and time. | Matches |

The exterior own field can be checked directly, without its implementation
helper. For $\mathbf K=(\mathbf k_\parallel,q_z)$,
$q_z=(\omega-\mathbf k_\parallel\cdot\mathbf v_\parallel)/v_z$, define
$\mathbf j=\sqrt{\alpha}\,q\mathbf v/\lvert v_z\rvert$. Fourier curl-curl gives

$$
\left[(\mathbf K^2-\omega^2)I-\mathbf K\mathbf K^T\right]\mathbf E_p
=4\pi i\omega\mathbf j,\qquad
\mathbf H_p=\mathbf K\times\mathbf E_p/\omega.
$$

Using $\mathbf K\cdot\mathbf v=\omega$ gives exactly the vacuum
`_particular_field` expression. The absolute normal velocity belongs to
this spatially Fourier-transformed infinite-current own field, not to the
finite time-integrated drive. Both passage senses retain the same charge sign.

## Bounded independent numerical comparison

An independent scratch calculation formed the eight-component generator by
applying the four curl equations to each Cartesian basis field, eliminated
$E_z$ by a direct two-by-two solve, and used a full matrix exponential transfer
through one finite layer. It integrated the forced field by 96- and
128-point Gauss–Legendre quadrature in time. Analytic vacuum transverse
basis waves supplied the exterior outgoing subspaces; the six-component
curl-curl equation above supplied own fields. The interface solve was

$$
[-e^{iMD}V_-,\ V_+]\begin{pmatrix}\mathbf r\\\mathbf t\end{pmatrix}
=e^{iMD}\mathbf f_{p,0}+\int dt\,e^{iM(D-z(t))}\mathbf b(t)
-\mathbf f_{p,D}.
$$

No production `_drive`, `_segment_faces`, `_time_integral`, `_modes`,
`_maxwell`, `_boundary_response` or `_particular_field` was called by the
reference calculation. The production entry point was called only for the
field being compared. This tests a different transfer/forced-curl construction
rather than another modal-Green implementation.

All cases used $E=1000$ eV, $D=80$ Å,
$\varepsilon=0.995+0.0002i$,
$\mathbf n=(0.3,0.2,\sqrt{0.87})$,
$\mathbf g=(0.01,-0.005,0.04)$ inverse Å, and lattice origin $(1,2,3)$ Å.
There were three successive velocities $(0.1,0.03,0.8s)$,
$(0.7,0.1,0)$ and $(-0.08,0.04,0.7s)$, with durations
$40$, $25$ and $48/0.7$ Å. Start point was $(2,-1,0)$ Å for $s=1$
and $(2,-1,80)$ Å for $s=-1$, starting clock $7$ Å, charge $-1$,
and consistent pre/post exterior continuations. CBS potentials were zero.

| Passage sense | Transfer/reverse susceptibilities | Relative full-field error | Relative coupled-minus-uncoupled error |
| --- | --- | --- | --- |
| $+1$ | zero / zero | $2.24\times10^{-14}$ | exactly zero |
| $+1$ | $(2\times10^{-4}+10^{-5}i)$ / zero | $2.08\times10^{-14}$ | $1.14\times10^{-12}$ |
| $+1$ | $(2\times10^{-4}+10^{-5}i)$ / $(10^{-4}+10^{-5}i)$ | $2.35\times10^{-14}$ | $2.82\times10^{-12}$ |
| $-1$ | zero / zero | $3.99\times10^{-14}$ | exactly zero |
| $-1$ | $(2\times10^{-4}+10^{-5}i)$ / zero | $4.23\times10^{-14}$ | $5.60\times10^{-12}$ |
| $-1$ | $(2\times10^{-4}+10^{-5}i)$ / $(10^{-4}+10^{-5}i)$ | $4.23\times10^{-14}$ | $3.13\times10^{-12}$ |

Relative errors are Euclidean complex Cartesian-vector errors normalized by
the independent reference norm. Changing quadrature order changed the full
field by at most $7.05\times10^{-14}$ relatively. A $10^{-10}$ full-field
comparison threshold was specified as a floating-point consistency guard,
not source uncertainty or a fitted physical tolerance. The weaker relative
precision of the small subtraction follows from cancellation. This probe was
bounded to modest optical depth; it is not an opaque-stack conditioning
survey or an independently published anchor. Scratch code/results are
uncommitted diagnostics, not durable regression evidence.

## Checks, limitations, and verdict

The focused command `uv run pyrite-dev test` on
`tests/montecarlo/spectrum/test_dtr_history.py`,
`tests/montecarlo/spectrum/test_dtr_production.py` and
`tests/dev/test_docs.py` passed: **24 tests**. The three-history coherent
production smoke emitted the documented unresolved negative pair-estimate
warning; that warning does not certify population convergence. Existing
straight-passage, subdivision, zero-harmonic and replay checks are internal
consistency evidence. The CBS/Feranchuk checks in the history module passed
but do not extend this charge-current verdict to `line-tr-phase-reconciliation`.

The final focused documentation tests passed again: **9 tests**. An isolated
Sphinx/MyST HTML build of this page with the repository's dollar/AMS math
extensions passed with warnings as errors: **125** elements had both `math`
and `notranslate` classes, with **zero** literal dollar delimiters in the
article text. The canonical full documentation build also emitted this page
with the same **125** math elements and **zero** raw dollar delimiters. It
reported the expected missing-toctree warning for this new page; navigation
integration and a clean full-build gate remain the task owner's check.

Units, limits and signs pass at the scope stated here. No reference table
values were transcribed, no external-code or CPU/GPU parity artifact was
used, and no reference-data uncertainty was assigned. The numerical cases
are newly executed at base revision `e350ed4fc`; they do not promote an
existing implementation-consistency test to an independent anchor.

This verdict covers the mathematical transported charge-current extension and
its field/power bookkeeping. It does not remove the locally straight TR
baseline approximation, validate sampled transport trajectories, prove recoil
bounds, certify CBS force phases, establish CUDA parity or reproduce a
published dynamical diffraction spectrum. The public author page and module
docstrings still describe this claim as unverified at the inspected revision;
updating that status language and navigation belongs to the task owner.

- **Claim**: `dtr-transported-current` — `src/pyrite/montecarlo/spectrum/dtr_history.py` and `src/pyrite/montecarlo/runner/dtr_emission.py` — Maxwell curl equations, the finite point-current modal Green integral, and signed field accounting derived above.
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`.
- **Re-derivation**: `matches` — longitudinal source contact term, finite modal integral, exterior own fields, field subtraction, history grouping and stored-baseline population convention agree; no divergent term found.
- **Verdict**: `rederived`.
- **Write-up**: `docs/validation/radiation-physics/dtr-transported-current.md`.
- **Suggested ledger change**: owner changes this row from `unverified` to `rederived`, links this derivation and records the six-case error bounds and attribution limits. Retain the existing internal-consistency anchor classification and all scope restrictions. Human sign-off remains pending.

