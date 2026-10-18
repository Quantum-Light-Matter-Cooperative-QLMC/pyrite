# Finite-transfer coherent TDS gain: scoped independent verification

Validation: `cr-axial-zone-handoff-accounting`.

This review covers only `checks/axial_escape_accounting.py::coherent_tds_gain`.
The overall physical entrance-zone handoff claim remains **unverified**.
The verifier derived the following expression from the cited source, signature
and docstring before inspecting the implementation or research derivation.

## Independent derivation

[Castin, Dalibard and Mølmer, arXiv:0805.4002](https://arxiv.org/html/0805.4002),
Eq. (2), gives a gain as a sum of jump sandwiches; Eq. (8) identifies its
trace as a jump rate. Neither equation specifies a crystal boundary.
Use axial length as the evolution variable. Let the supplied finite transfer
polynomial have coefficients $a_q$, reciprocal wavevectors $\mathbf{k}_q$
and integer orders $q$. Coefficients already include source registration
and cell-area normalization. Let $\mathbf{u}$ be centered Gaussian with
covariance $s I$, where $s$ is the supplied displacement variance per axis.
The Gaussian characteristic function follows by completing the square:

$$
D_q=\mathbb{E}\exp(-i\mathbf{k}_q\cdot\mathbf{u})
=\exp(-s|\mathbf{k}_q|^2/2).
$$

In an orthonormal Fourier basis define the translation $T_q|n\rangle=|n+q\rangle$,
without modulo identification. For one independent string of linear density
$\lambda$, the centered jump is

$$
L(\mathbf{u})=\sqrt{\lambda}\sum_q a_q
[\exp(-i\mathbf{k}_q\cdot\mathbf{u})-D_q]T_q.
$$

The covariance and outgoing gain are therefore

$$
K_{qr}=\exp[-s|\mathbf{k}_q-\mathbf{k}_r|^2/2]-D_qD_r,
\qquad
G=\lambda\sum_{q,r}a_qa_r^*K_{qr}T_q\rho T_r^\dagger.
$$

Equivalently, on the unique integer output set of all $n+q$,

$$
G_{bc}=\lambda\sum_{n,m,q,r}
\rho_{nm}a_qa_r^*K_{qr}
\delta_{b,n+q}\delta_{c,m+r}.
$$

A source phase in $a_q$ survives as $a_qa_r^*$; deleting off-diagonal
transfer covariance would generally change the gain. The incident matrix
is not normalized: multiplying it by any nonnegative weight multiplies
the gain by that weight. Independent strings add their separate gains.

## Cheap filters before implementation inspection

Units pass: $s|\mathbf{k}|^2$ and $a_q$ are dimensionless; $G$ has inverse-length
units solely from $\lambda$. The covariance is a Gram matrix of centered
random phases, so the gain preserves positivity and Hermiticity.
For $s=0$ or $\lambda=0$ the gain vanishes. For an incident plane wave of
weight $w$, the diagonal output at $n+q$ is
$w\lambda|a_q|^2[1-\exp(-s|\mathbf{k}_q|^2)]$ for distinct transfers;
off-diagonal output coherences may still remain. The zero-transfer coefficient
has zero centered fluctuation. For small variance,
$K_{qr}=s\mathbf{k}_q\cdot\mathbf{k}_r+O(s^2)$.
The sign of the displacement exponent is immaterial for this centered
isotropic Gaussian covariance; the sign and conjugation of source phases
are material. Integer orders and Cartesian wavevectors must describe the
same reciprocal basis.

## Exact implementation comparison

Inspected the body only after recording the derivation above.
`q2` and `difference2` produce exactly the two squared norms in $K$.
The second exponential uses the sum of squared norms with the same factor
one half; `coefficients[:, None] * coefficients.conj()[None, :]` has the
required conjugation. `rho[:, None, :, None]` indexes incident bra and ket
independently. The unique literal integer sums and `np.add.at` correctly
accumulate interfering paths sharing an output, with no modulo folding.
There is no density normalization or entrywise covariance clipping.
The research note's coherent finite-transfer section gives the same formula,
absorbing $\lambda a_q a_r^*$ into its definition of $K$.
No divergent factor, sign, exponent or index convention was found.

## Numeric evidence (2026-10-18)

An independent 28-by-28 Gauss–Hermite integration formed the displacement
jump matrices directly and averaged their sandwiches, without using the
implementation's covariance or contraction helpers. Inputs were incident
orders $(0,0),(1,-1),(-1,1)$; transfers
$(0,0),(1,-1),(-1,1),(2,0)$; reciprocal-basis rows
$(0.9,0.3),(-0.2,1.2)$ Å$^{-1}$; coefficients
$(0.2i,0.7+0.3i,-0.2+0.4i,0.1-0.5i)$; variance $0.27$ Å$^2$;
and linear density $0.63$ Å$^{-1}$. The positive incident matrix was
$\rho=X^T X^*$ with rows of $X$ equal to
$(1,0.2i,-0.4)$ and $(0.3i,0.8,0.2)$.
Its trace is $1.97$, deliberately not unity.

The eight-output gain matched quadrature to maximum absolute error
$1.40\times10^{-16}$ Å$^{-1}$. Its smallest eigenvalue was
$1.26\times10^{-17}$ Å$^{-1}$ and trace was
$0.6001992751616119$ Å$^{-1}$ (imaginary roundoff below
$2\times10^{-17}$ Å$^{-1}$). Scaling the incident density by $3.7$
scaled the gain with maximum error $5.56\times10^{-17}$ Å$^{-1}$.
Zero variance and zero string density both returned exact zero.

The project runner passed all 12 tests in
`tests/montecarlo/test_channeling_axial_escape_accounting.py`, including its
separate 16-by-16 Gaussian quadrature, coherent interference, plane-wave
rate reduction and zero-limit checks. All eight `tests/dev/test_docs.py`
checks passed.

## Verdict and limitations

**Scoped algebra verdict: rederived. Overall ledger claim: unverified.**
The finite supplied transfer polynomial gives a positive coherent gain with
the declared units and incident weight. Positivity is an exact-arithmetic
statement; tiny-variance covariance cancellation and extreme numerical
ranges were not bounded. Consistency of the integer and Cartesian basis,
and physical normalization of supplied coefficients, remain caller
assumptions rather than properties established by this helper.

This review does not verify transfer-tail completion, equality with production
jump loss or its numerical sink, a physical matching surface, omitted return
flux, a phase-space handoff instrument, energy/frame/native stream mapping,
or source scoring. No physical escape direction distribution is certified.
The supplied finite support must remain fixed during this calculation;
independent strings contribute gains separately. Suggested ledger edit:
retain `Status: unverified`; add this write-up as scoped independent
finite-transfer algebra evidence and retain all outstanding physical checks.
Only a human can grant sign-off.

# Continuum independent-atom window split: scoped verification (2026-10-18)

This section covers only
`checks/axial_escape_accounting.py::continuum_tds_window_split` and the way
`continuum_report` passes production inputs to it. The finite-transfer
section above is unchanged. The overall handoff claim remains **unverified**.
The verifier derived the expressions below from the TDS jump definition in the
`axial_dissipator` module docstring and from Castin, Dalibard and Mølmer,
[arXiv:0805.4002](https://arxiv.org/html/0805.4002), Eq. (2). This was done
before reading the function body or the research-note section "Cutoff-free
continuum window split". Only the signature and docstring were read first.

## Independent derivation

From the module docstring, an atom at $\mathbf R$ with displacement
$\mathbf u\sim\mathcal N(0,a^2 I)$ adds the real phase
$\phi_{\mathbf u}(\mathbf r)=\phi_0(\mathbf r-\mathbf R-\mathbf u)$, with
$\phi_0=-V_{\rm proj}/\hbar v$. Its jumps are
$L_{\mathbf u}=\phi_{\mathbf u}-\langle\phi\rangle$, and their rate measure is
$\lambda\,p(\mathbf u)\,d^2u$. Write
$\phi_0(\mathbf r)=\int\frac{d^2q}{(2\pi)^2}t(q)e^{i\mathbf q\cdot\mathbf r}$.
Here $t$ is real and even, with units of Å$^2$. The Gaussian characteristic
function is $\chi(q)=\mathbb E\,e^{-i\mathbf q\cdot\mathbf u}=e^{-a^2q^2/2}$.
Take one string with no image correlation, so its displacement is independent
of every other string. Incident plane waves are
$\psi_i=e^{i\mathbf p_i\cdot\mathbf r}/\sqrt A$. Under
$\langle\mathbf k|\mathbf k'\rangle=(2\pi)^2\delta$, the jumped state has
Fourier amplitude

$$
\widetilde{L_{\mathbf u}\psi_i}(\mathbf k)=A^{-1/2}\,t(|\mathbf q_i|)\,
e^{-i\mathbf q_i\cdot\mathbf R}\left(e^{-i\mathbf q_i\cdot\mathbf u}-\chi_i\right),
\qquad \mathbf q_i=\mathbf k-\mathbf p_i .
$$

The Castin–Dalibard–Mølmer gain $\lambda\,\mathbb E_{\mathbf u}
L_{\mathbf u}\rho L_{\mathbf u}^\dagger$ has momentum diagonal
$\lambda\,\mathbb E\sum_{ij}\rho_{ij}\widetilde{L\psi_i}\,\widetilde{L\psi_j}^{\,*}$
per $d^2k/(2\pi)^2$. The position phases combine to
$e^{-i\mathbf q_i\cdot\mathbf R}e^{+i\mathbf q_j\cdot\mathbf R}
=e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}$. The displacement average is
$\mathbb E[e^{-i(\mathbf q_i-\mathbf q_j)\cdot\mathbf u}]-\chi_i\chi_j$, and
$\mathbf q_i-\mathbf q_j=\mathbf p_j-\mathbf p_i$. Per $d^2k$, the result is

$$
\boxed{g(\mathbf k)=\frac{\lambda}{A(2\pi)^2}\sum_{ij}\rho_{ij}
e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}\,t_it_j
\left[e^{-a^2|\mathbf p_i-\mathbf p_j|^2/2}-\chi_i\chi_j\right]}
$$

This expression is
$\lambda A^{-1}(2\pi)^{-2}\,\mathbb E_{\mathbf u}\lvert\sum_i c_i X_i\rvert^2$
for any decomposition $\rho=\sum c c^\dagger$. It is therefore pointwise
nonnegative whenever $\rho\succeq0$.

For the total, set
$\mathbf G=\mathbf p_i-\mathbf p_j$ and
$\mathbf k=\mathbf q+(\mathbf p_i+\mathbf p_j)/2$. Then
$|\mathbf q_i|=|\mathbf q-\mathbf G/2|$ and $|\mathbf q_j|=|\mathbf q+\mathbf G/2|$.
The two squared norms in $\chi_i\chi_j$ sum to $2q^2+G^2/2$. Hence

$$
\int d^2k\,g=\frac{\lambda}{A(2\pi)^2}\sum_{ij}\rho_{ij}
e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}D(|\mathbf p_i-\mathbf p_j|),
$$

$$
D(G)=\int d^2q\,t(|\mathbf q-\tfrac{\mathbf G}{2}|)\,t(|\mathbf q+\tfrac{\mathbf G}{2}|)
\left[e^{-a^2G^2/2}-e^{-a^2(q^2+G^2/4)}\right].
$$

**Production diagonal.** The variance of a single non-periodic string is
$C_1(\mathbf r)=\lambda\operatorname{Var}\phi_{\mathbf u}(\mathbf r)
=\frac{\lambda}{(2\pi)^4}\int d^2G\,D(G)e^{i\mathbf G\cdot(\mathbf r-\mathbf R)}$.
Poisson summation over the cell lattice of independent atoms gives harmonics
$\lambda A_{\rm cell}^{-1}e^{-i\mathbf G\cdot\mathbf R}D(G)/(2\pi)^2$. This
equals the production
$n_{s,\mathbf G}T_s(|\mathbf G|)/(2\pi\hbar v)^2$ with
$n_{s,\mathbf G}=\lambda e^{-i\mathbf G\cdot\mathbf R}/A_{\rm cell}$ and
$D=T_s/(\hbar v)^2$, where $t=-v/\hbar v$. Its expectation is
$\operatorname{tr}(\rho C)=\sum_{ij}\rho_{ij}\langle\mathbf p_j|C|\mathbf p_i\rangle$.
That expectation selects the harmonic $\mathbf G=\mathbf p_j-\mathbf p_i$, whose
phase is $e^{-i(\mathbf p_j-\mathbf p_i)\cdot\mathbf R}
=e^{+i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}$. So the production phase
$e^{-i\mathbf G\cdot\mathbf R}$ and the gain phase
$e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}$ are the same convention, and the
two totals are equal. The production full diagonal is the independent-atom
diagonal. It is not the image-correlated variance.

**Image and supercell independence.** Replace the string by its images at
$\mathbf R+\mathbf L$. If every $\mathbf p_i-\mathbf p_j$ is a reciprocal vector
of the image lattice, each image's phase factor is
$e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf L}=1$. A $c_x\times c_y$ supercell
holding $c_xc_y$ independent copies, with plane waves normalized on
$A=c_xc_yA_{\rm cell}$, therefore reproduces the single-cell result. Harmonics
on the supercell lattice but off the cell lattice cancel in the image sum.

**Periodic lattice sum.** Correlated images at supercell period $A_s$ give
$\phi=A_s^{-1}\sum_{\mathbf q\in\Lambda_s^*}t(q)e^{i\mathbf q\cdot(\mathbf r-\mathbf R-\mathbf u)}$.
Its outgoing weight at a lattice momentum $\mathbf k$ is $g(\mathbf k)$
multiplied by $(2\pi)^2/A_s$. The image-correlated gain is therefore
$\sum_{\mathbf k\in\mathbf p+\Lambda_s^*}g(\mathbf k)(2\pi)^2/A_s$. With box
edges at half-integer lattice coordinates, the lattice points are the cell
centres, so this sum is the midpoint rule of the continuum box integral. The
statement also holds for coherent inputs, because the outputs
$\mathbf p_i+\mathbf q$ share one lattice.

## Cheap filters before implementation inspection

| Check | Result |
| --- | --- |
| Units | $t$: Å$^2$. $D$: Å$^2$. $\lambda/(A(2\pi)^2)\,D$: Å$^{-1}$. $g$ per $d^2k$: Å. Pass. |
| $a^2\to0$ or $\lambda=0$ | bracket $1-1=0$ or prefactor zero, so all weights vanish. Pass. |
| Single plane wave | $g=\frac{\lambda}{A(2\pi)^2}t^2(1-e^{-a^2q^2})$ and total $=\frac{\lambda}{A(2\pi)^2}D(0)$, independent of $\mathbf p$. Pass. |
| Small $a^2$ | bracket $\to a^2\mathbf q_i\cdot\mathbf q_j$. Pass. |
| Enclosing box | represented $\to$ total, with the tail $\propto Q^{-2}$ because $t\propto q^{-2}$. Pass. |
| Sign of $t$ | enters only as $t_it_j$, so it is immaterial. The sign of the $\mathbf R$ phase is material for coherences. Pass. |
| Hermiticity | $g$ is real for Hermitian $\rho$. Pass. |

## Implementation comparison

The body was inspected only after the derivation above. `mixed` is
$\rho_{ij}e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}$, built as
`difference = p[:, None] - p[None, :]`. `coherence` is
$e^{-a^2|\mathbf p_i-\mathbf p_j|^2/2}$, and `damped` is $t_i\chi_i$. The two
`einsum` terms give exactly the bracket. `scale` is
$\lambda/(A(2\pi)^2)$. The box integral uses tensor Gauss–Legendre panels
over $d^2k$, and the total is `scale * sum(mixed * D)`. Both agree with the
derivation in every factor, sign, exponent and phase.

In `continuum_report`, `phase` is $-v/\hbar v$ and `diagonal` is
`_tds_transform / hbar_v**2`. That is $D$, and it uses the production
$a^2$ of the source. `area` is the 1×1 grid area, which equals
`profile.area_ang2`, and the box edges are $(j-\tfrac12)\Delta$. The lattice
probe's prefactor is $\lambda/(A\cdot c^2A)$. That is $c^2$ strings, each with
weight $\lambda/A_s^2$, which is consistent with the midpoint statement. The
research-note equations agree with this derivation. The note gives $g$ per
$d^2k/(2\pi)^2$, while the docstring gives it per $d^2k$; this is the same
quantity. The note's Gaussian-test $D$ was rechecked:
$g^2e^{-bG^2/2}[e^{-a^2G^2/2}\pi/(2b)-e^{-a^2G^2/4}\pi/(2b+a^2)]$.
No divergent factor, sign, exponent or convention was found.

## Independent numeric evidence (2026-10-18)

The case is Si [001] at 5 MeV, from `axial_profile` and `axial_dissipator`,
with one string. Its parameters are $a^2=6.1003\times10^{-3}$ Å$^2$,
$\lambda=0.18413$ Å$^{-1}$ and $A=3.68683$ Å$^2$.

- **Production closure.** A 256² dissipator has no sink clamp. The FFT
  harmonics of its `total_rate` gave $\operatorname{tr}(\rho C)$ for four
  inputs: single waves (0,0) and (7,7), the in-phase (0,0)+(1,0) state, and a
  complex rank-two three-wave $\rho$ on orders (0,0), (1,0) and (1,1). The
  function's total agreed in all four cases to $\le1.3\times10^{-11}$ relative.
  For example, $9.265009930\times10^{-5}$ Å$^{-1}$ for the complex state.
- **Own $D$.** An independent polar quadrature over the full $2\pi$, using its
  own nodes and bracket, gave $D(0)$, $D(g)$ and $D(\sqrt2 g)$. Each agreed with
  `_tds_transform`$/(\hbar v)^2$ to $\le1\times10^{-6}$ relative when
  integrated to $Q_{\max}=10^4$ Å$^{-1}$. A first attempt truncated at
  120 Å$^{-1}$ was 0.6% low. That shortfall is the expected
  $Q^{-2}$ tail, not a discrepancy.
- **Large-box closure.** For a box of $\pm150$ Å$^{-1}$, the represented weight
  falls short of the total by $2.4\times10^{-7}$ Å$^{-1}$ for the single wave.
  The analytic tail
  $\frac{\lambda}{A(2\pi)^2}\int_{Q}^\infty2\pi q\,t^2\,dq\approx2.9\times10^{-7}$
  Å$^{-1}$, using $t\approx1.29/q^2$, matches this. In the report the unresolved
  weight falls by $\approx4$ per window doubling.
- **Independent box weight.** A 40×40 Gauss–Hermite average over $\mathbf u$
  of the squared jump amplitude was formed directly, for the complex three-wave
  state in an off-centre 12×12-panel box. It reproduced the represented weight,
  $4.469986849\times10^{-5}$ Å$^{-1}$, to $5\times10^{-16}$ relative. Its
  minimum integrand sample was positive.
- **Supercell independence.** For a coherent complex $\rho$ on cell orders
  (0,0) and (1,0), summing the function over the four image strings of a 2×2
  supercell with $A=4A_{\rm cell}$ reproduced the 1×1 represented, unresolved
  and total weights to roundoff. A coherence at half a cell reciprocal vector
  gave the same total as the mixture, as image cancellation requires.
- **Midpoint rule.** The report's lattice-versus-box rows have relative
  differences $-2.35\times10^{-2}$, $-3.4\times10^{-4}$ and $2.1\times10^{-6}$
  for $c=1,2,4$. The lattice sum converges to the continuum box integral.

The report's 16² `production_grid_loss` values are 0.6%, 0.5% and 1.7% above
the totals. Probing showed that at 16² the sink clamp
`max(full - grid, 0)` is active on 14.8% of points. At 32², where there is no
clamp, the grid mean matches the total to $5\times10^{-10}$ relative. For the
uniform mixture, then, the excess is the clamp and not aliasing. The research
note names both causes. This is a wording nuance outside the function's scope;
it does not affect the closure.

The project runner passed all 17 tests in
`tests/montecarlo/test_channeling_axial_escape_accounting.py`.

Out-of-scope aside: the `_tds_transform` docstring labels its result
[eV² Å⁶]. Its quadrature is $\int d^2q\,v^2$ with $v$ in eV Å$^3$, which gives
eV² Å⁴. The numerics are unaffected, and that docstring belongs to
`cr-lindblad-jump-operators`.

## Verdict and limitations

**Scoped verdict for `continuum_tds_window_split`: rederived. Overall ledger
claim: unverified.**

The pointwise nonnegative independent-atom gain density, its box integral and
its analytic total are correct in units, phase convention, the $1/A$ and
$(2\pi)^2$ normalization, and image-supercell independence. The total closes on
the expectation of the production full TDS diagonal. The periodic lattice sum
is the midpoint rule of the same integrand. Box accuracy depends on panel
resolution relative to the width of $t$; the caller sets that resolution, and it
is not bounded here.

Several things remain outside this review. The box is a declared numerical
window, not a physical retained projector or matching surface. The production
grid kernel uses image-correlated in-grid variance, and its clamp-dependent
partition is not certified by this check. The review also does not cover
omitted return flux, the phase-space handoff instrument, energy, frame or
native stream mapping, or source scoring.

Suggested ledger edit: keep `Status: unverified`, and cite this section as
scoped fresh-context evidence for the continuum split. Only a human can grant
sign-off.

# Energy-projector return-flux study: scoped verification (2026-10-18)

Scope: `checks/axial_matching_study.py::{periodic_tds_rates,vectors_loss,retained_populations,report}`
and the research-note section "Energy-defined retained projector and return
flux". The derivation below was written from the TDS jump definition in the
`axial_dissipator` module docstring, Castin–Dalibard–Mølmer arXiv:0805.4002
Eq. (2), and the secular (Pauli) limit. Only the docstrings and signatures were
read before it. The overall claim stays `unverified`.

## Independent derivation

**Periodic string phase.** One string sits at $\mathbf R$ in a cell of area
$A$, with reciprocal orders $\mathbf G_n=(n_xg_x,n_yg_y)$. Every image carries
the same in-plane displacement $\mathbf u\sim N(0,a^2I)$. Let $\tilde\phi(q)$
[Å$^2$] be the 2D transform of the per-atom phase. Then

$$
\phi_{\mathbf u}(\mathbf r)=\sum_n f_n\,e^{-i\mathbf G_n\cdot\mathbf u}\,e^{i\mathbf G_n\cdot\mathbf r},
\qquad
f_n=\frac{\tilde\phi(G_n)}{A}\,e^{-i\mathbf G_n\cdot\mathbf R}.
$$

Here $f_n$ is dimensionless. The mean is $\langle\phi\rangle_n=f_n\chi_n$ with
$\chi_n=e^{-a^2G_n^2/2}$. The jump $J_{\mathbf u}=\phi_{\mathbf u}-\langle\phi\rangle$
has rate measure $\lambda\,p(\mathbf u)\,d^2u$, where $\lambda$ is in Å$^{-1}$.
In the cell-normalized plane-wave basis, $J_{\mathbf u}$ is multiplicative, so

$$
\langle p+n|J_{\mathbf u}|p\rangle=f_n D_n(\mathbf u),
\qquad
D_n(\mathbf u)=e^{-i\mathbf G_n\cdot\mathbf u}-\chi_n .
$$

The transfer $T_n$ therefore maps order $p$ to $p+n$, and $D_0=0$.

**Eigenstate rates.** Write $|a\rangle=\sum_pc^a_p|p\rangle$ and
$M^{ba}_n=\sum_pc^{b*}_{p+n}c^a_p$. The Lindblad gain of Castin et al. Eq. (2),
averaged over the jump measure, gives

$$
W_{ba}=\lambda\,\mathbb E_{\mathbf u}\Bigl|\sum_nM^{ba}_nf_nD_n\Bigr|^2
=\sum_{n,m}M^{ba}_n\,\mathcal C_{nm}\,M^{ba*}_m,
$$

$$
\mathcal C_{nm}=\lambda f_nf_m^*\bigl[\chi_{n-m}-\chi_n\chi_m\bigr].
$$

$\mathcal C$ is Hermitian and positive semidefinite, because it is a covariance.
Any factorization $\mathcal C=KK^\dagger$, for example $K=U\sqrt{w}$ from the
eigendecomposition, gives
$W_{ba}=\sum_j\bigl|\sum_nM^{ba}_nK_{nj}\bigr|^2$. Dropping eigenvalues
$w_j<\epsilon\,w_{\max}$ changes $W$ by at most
$\epsilon\,w_{\max}\,\mathrm{rank}\,\sum_n|M_n|^2$. Only transfers $n$ in the
difference set of the represented orders enter.

**Full loss.** $\Gamma=\lambda\,\mathbb E_{\mathbf u}J^\dagger J=\lambda\operatorname{Var}_{\mathbf u}\phi(\mathbf r)$
is a multiplicative operator with harmonics

$$
\Gamma_{\mathbf G}=\lambda\sum_nf_nf^*_{n-G}\bigl[\chi_G-\chi_n\chi_{n-G}\bigr],
\qquad \Gamma_{-\mathbf G}=\Gamma_{\mathbf G}^*.
$$

This is the docstring formula with $c_n\equiv f_n$ and $\chi_G=e^{-a^2G^2/2}$.
Then $\langle a|\Gamma|a\rangle=\sum_{p,q}c^{a*}_p\Gamma_{p-q}c^a_q$. Only $G$
in the represented difference set is needed. Completeness gives
$\sum_{b\,\rm all}W_{ba}=\langle a|\Gamma|a\rangle$, so over the represented
basis $\Gamma_a-\sum_bW_{ba}\ge0$ is the weight leaving it. At large $n$ the
summand tends to $|f_n|^2\chi_G$ with $f_n\propto q^{-2}$. The lattice tail
beyond $|n|\le T$ then falls as $T^{-2}$.

**Secular rate equation.** Under $H_\perp$ a coherence $\rho_{ab}$ rotates at
$(E_a-E_b)/\hbar v$ per Å. If every nondegenerate spacing satisfies
$|E_a-E_b|/\hbar v\gg\Gamma_a,\Gamma_b$, coherences between nondegenerate
states decouple, and the populations obey

$$
\frac{dn_b}{dz}=\sum_aW_{ba}n_a-\Gamma_bn_b .
$$

The diagonal $W_{bb}$ appears in both the gain and the loss. Inside a degenerate
multiplet the secular limit keeps the intra-multiplet coherences, so the Pauli
form depends on the basis chosen there. It is exact only if $\rho$ is
proportional to the identity on each multiplet. Here that holds by symmetry: the
$C_{4v}$ string, isotropic $\mathbf u$ and the zero-tilt plane wave (an $A_1$
state) give an invariant $\rho$, and Schur's lemma applies.

**Absorbing projector and flux balance.** For retained $P$ and complement $F$,
take $\dot n_P=(W_{PP}-\operatorname{diag}\Gamma_P)n_P$, $\dot A_F=\mathbf 1^TW_{FP}n_P$
and $\dot A_O=(\Gamma-\mathbf 1^TW)_Pn_P$. The sum is

$$
\frac{d}{dz}\Bigl(\sum_Pn+A_F+A_O\Bigr)
=\sum_{a\in P}n_a\Bigl[\sum_{b\in P}W_{ba}-\Gamma_a+\sum_{b\in F}W_{ba}+\Gamma_a-\sum_bW_{ba}\Bigr]=0 .
$$

Closure is therefore exact for any rates. Each cumulative absorbed weight is a
linear functional of $\int_0^zn_P\,dz'$.

## Cheap filters before implementation inspection

| Filter | Result |
| --- | --- |
| Units | $\tilde\phi/A$ is dimensionless, so $W$ and $\Gamma$ are in Å$^{-1}$ through $\lambda$. With $v$ in eV Å$^3$, $\tilde\phi=-v/\hbar v$ is in Å$^2$. Pass. |
| Zero limits | $a=0$ makes $\chi\equiv1$, so $\mathcal C=0$ and $\Gamma_G=0$. $\lambda=0$ gives zero. Populations freeze. Pass. |
| Positivity | $\mathcal C\succeq0$, so $W\ge0$. Represented loss is at most the total by completeness. Pass. |
| Sign of $\phi$ | Enters quadratically. Immaterial. |
| Sign of the $\mathbf R$ phase | Material for $W$ at an off-centre string with non-symmetric states. Checked numerically below. |
| Flux closure | Algebraically exact (above). Pass. |

## Implementation comparison

The bodies were read after the derivation.

- `periodic_tds_rates`: `c` is $\tilde\phi(|\mathbf q|)/A\,e^{-i\mathbf q\cdot\mathbf R}$
  on the transfer grid $-\text{span}\ldots\text{span}$. `covariance` is
  $\lambda c_nc_m^*[\chi_{n-m}-\chi_n\chi_m]$. `eigh` with relative truncation
  gives `kraus`$=U\sqrt w$, and negative roundoff eigenvalues are dropped.
  `diff` is `orders[p]-orders[q]`, so `block[p, q]` is $\langle p|J|q\rangle$
  with $p=q+n$, consistent with $T_n:q\to q+n$. The flat `index` matches the
  `meshgrid(..., indexing="ij")` layout. The `einsum` is
  $\sum_{pq}c^{b*}_pK_{p-q,j}c^a_q$. This agrees in every factor, sign and phase.
- `vectors_loss`: `ifft2(F·F*)` is $\sum_nf_nf^*_{n-G}$ at index $G$. The
  array offset cancels in a correlation. Zero padding to $2(2T+1)$ prevents
  circular wrap for $|G|\le2T$. `harmonic`$=\lambda(\chi_G\,\text{first}-\text{second})$,
  with `second` the correlation of $f\chi$. `loss[p,q]`$=\Gamma_{p-q}$. This
  agrees.
- `retained_populations`: the generator, $W_{FP}$ column sums, the unresolved
  rate $(\Gamma-\mathbf 1^TW)_P$ and $\int_0^ze^{\mu s}ds$ via `expm1` all
  match. The eigen-decomposition assumes a diagonalizable generator, which is
  generic; the numeric check below agrees with `expm`.
- `report`: $\tilde\phi=-$`_potential`$/\hbar v$, with production $a^2$,
  $\lambda$ and $A$. Cuts sit at gap midpoints. Bound means $E<U_{\max}$, the
  cell maximum, as the docstring states.

No divergent factor, sign, exponent, index or convention was found in the four
functions.

## Independent numeric evidence (2026-10-18)

The scratch script is not committed. Its inputs:

- a rectangular cell, $L_x=2.3$ Å and $L_y=1.7$ Å;
- an off-centre string at $\mathbf R=(0.37,-0.21)$ Å;
- a Gaussian per-atom phase with width $0.25$ Å;
- $a=0.12$ Å and $\lambda=0.05$ Å$^{-1}$;
- 20 orders, with 17 random Hermitian eigenvectors (an incomplete set).

$\phi_{\mathbf u}$ was built in real space from $7\times7$ images. A
$40\times40$ Gauss–Hermite grid sampled $\mathbf u$, and a $96^2$ midpoint grid
did the cell integrals.

- **$W$ by direct displacement quadrature** of
  $|\langle b|J_{\mathbf u}|a\rangle|^2$ agreed to $2.4\times10^{-14}$ relative.
  With $\mathbf R\to-\mathbf R$ the difference was 125%, so the phase sign is
  tested and correct.
- **$\Gamma_a$ by direct real-space variance**,
  $\lambda\int|\psi_a|^2\operatorname{Var}_{\mathbf u}\phi$, agreed to
  $1.9\times10^{-14}$ relative. $\Gamma_a-\sum_bW_{ba}>0$ for every state.
- **Absorbing model against `scipy.linalg.expm`** of the augmented
  $(n_P,A_F,A_O)$ generator: populations agreed to $3\times10^{-17}$ and absorbed
  weights to $10^{-18}$, and the balance closed to $10^{-16}$.

Si [001] at 5 MeV, $\pm10$ orders (441 states), $\hbar v=1964.77$ eV Å:

- **Against production.** The periodic $\Gamma_a$ of the six $E<U_{\max}$
  states and the first state above $U_{\max}$ is 0.956 to 0.998 of
  $\langle a|C(\mathbf r,\mathbf r)|a\rangle$ from a 64² production dissipator.
  The cell-lattice sum is a coarse midpoint rule of the continuum transfer
  integral. The represented fraction $\sum_bW_{ba}/\Gamma_a$ is 0.86 to 0.97.
- **Truncations.** Changing `rank_rtol` from $10^{-14}$ to $10^{-12}$ changes
  $W$ by $1.8\times10^{-14}$ relative; the note says $5\times10^{-14}$, which is
  immaterial. Tail 300→600 changes $\Gamma$ by $1.10\times10^{-4}$. The
  150→300 / 300→600 ratio is 4.01, the expected $T^{-2}$.
- **Secular condition.** Under 176 eV there are 145 states, 10440 pairs and 36
  degenerate pairs, leaving 10404 nondegenerate pairs. Of these, 64 have
  $\Delta E/\hbar v<(\Gamma_a+\Gamma_b)/2$, and 98 have
  $\Delta E/\hbar v<\Gamma_a+\Gamma_b$. The note's count uses the first
  criterion without stating it. Every failing pair lies at $E\ge8.71$ eV, above
  $U_{\max}$, and none straddles any of the four cuts. Gaps at the cuts are
  1.74, 3.79, 3.73 and 8.76 eV. The 8.710/8.743 eV pair has
  $\Delta E/\hbar v=1.68\times10^{-5}$ Å$^{-1}$, against $\Gamma\approx0.9$ to
  $1.1\times10^{-4}$ Å$^{-1}$.
- **Multiplet basis.** A random unitary was applied inside every degenerate
  multiplet. $N_B$ changed by $\le8\times10^{-14}$ relative for cuts 7.84 and
  94.93 eV, and by $9\times10^{-11}$ for $P=R$. The basis caveat is correct in
  general but does not affect these numbers, consistent with the symmetry
  argument.
- **Report reproduction.** `uv run python checks/axial_matching_study.py` took
  21 s. Every $\pm10$ table entry, the grid series for $P=R$, the 27% and
  $4\times10^{-6}$ entrance weights, and $U_{\min,\max}=-77.2$ and $8.19$ eV
  match the output. Flux-balance defects are $\le4\times10^{-15}$. One
  rounding nit: the 175.3 eV, $3\times10^4$ Å entry is 0.020549, printed as
  0.0205.
- **Tests.** The project runner passed all 6 tests in
  `tests/montecarlo/test_channeling_axial_matching_study.py`.

## Characterisation discrepancies

1. **"Saddle" means the cell maximum.** The note says "An absorbing boundary
   at the saddle energy … loses 11%" and calls the 8.71/8.74 eV pair "just above
   the saddle". The ledger Notes repeat this as "saddle-energy absorbing
   boundary" and "≥ one well depth above the saddle". But $U_{\max}=8.19$ eV is
   the cell maximum, at the centre of the square of strings, $(L/2,L/2)$. The
   string-to-string saddle, at the midpoint $(L/2,0)$, is $U_s=4.97$ eV. Two of
   the six "bound" states ($5.75$ and $6.97$ eV) lie above $U_s$, so they are
   not confined to one string. The 7.84 eV cut is a cell-maximum boundary. The
   numbers are right for the defined quantity, $E<U_{\max}$, but the label is
   wrong. Replace "saddle" with "cell maximum $U_{\max}$" in both places, or
   recompute with $U_s$ if a saddle boundary is intended.
2. **The "converges in the grid to 0.3%" wording is not supported as
   stated.** At $E_m\approx U_{\max}+(U_{\max}-U_{\min})$ and $10^4$ Å, the
   deficit for $\pm6,\pm8,\pm10$ orders is 0.63%, 1.57% and 1.97%. It rises
   monotonically, with steps of 0.94 and 0.40 percentage points. $N_B$ itself
   changes by 0.2% from ±8 to ±10. At 175 eV the deficit is 0.07%, 0.54% and
   0.86%. "About 2% to $10^4$ Å" is supported at ±10. Convergence of the deficit
   to 0.3% is not demonstrated; the note should state the last-step change and
   that the sequence is still increasing.

The other conclusions are supported by the numbers:

- the deficit grows with depth at every $E_m$;
- the $P=R$ reference is unconverged beyond a few $10^4$ Å;
- the result needs a depth-budget or architecture decision.

The caveats (secular approximation, TDS only, a single periodic string per cell
at the Γ point, zero-tilt entrance) are stated. Bloch-$\mathbf k$ and supercell
convergence are named as open.

## Verdict and limitations

**Scoped verdict for `periodic_tds_rates`, `vectors_loss`,
`retained_populations` and `report` arithmetic: rederived. Research-note and
ledger interpretation: discrepancy (the "saddle" label, and grid-convergence
wording). Overall ledger claim: unverified.**

This review does not validate:

- a physical matching surface;
- the supercell-periodic (image-correlated) displacement model as a stand-in for
  independent string displacements, beyond the 0.96–1.00 total-loss agreement
  above;
- electronic jumps;
- non-secular coherences among the 64 near-degenerate pairs above $U_{\max}$;
- the Breuer–Petruccione section citation, which was not checked against the
  book.

Suggested ledger edit: keep `Status: unverified`. In Notes, replace "saddle"
with "cell-maximum ($U_{\max}$)" and cite this section as scoped fresh-context
evidence. Only a human can grant sign-off.

# Bloch-block TDS rates: scoped verification (2026-10-18)

Scope: `checks/axial_matching_study.py::{bloch_tds_rates,_kraus_rates,_kraus,bloch_report}`
and the research-note paragraphs "Bloch-$\mathbf k$ sampling" and "Provisional
TDS-only depth budget". The derivation below was written from the TDS jump
definition in the `axial_dissipator` module docstring and Castin–Dalibard–Mølmer
arXiv:0805.4002 Eq. (2). Only the docstrings and signatures were read before it.
The Γ-point functions were reviewed in the previous section. The overall claim
stays `unverified`.

## Independent derivation

**Supercell and jumps.** Take a $c\times c$ supercell of area $S=c^2A$, periodic
with the supercell lattice. Its strings sit at $\mathbf R_j=\mathbf R+\mathbf t_j$,
where $\mathbf t_j$ runs over the $c^2$ cell translations inside it. Each string
has its own displacement $\mathbf u_j\sim N(0,a^2I)$, repeated only in the
supercell images. The dissipator docstring defines one jump family per atom,
$L_{\mathbf R_j,\mathbf u}=\phi_{\mathbf u}-\langle\phi\rangle$, with rate
measure $\lambda\,p(\mathbf u)\,d^2u$. The supercell reciprocal lattice is
$\mathbf g/c$. With supercell-normalized plane waves
$|\mathbf Q\rangle=e^{i\mathbf Q\cdot\mathbf r}/\sqrt S$,

$$
\langle\mathbf Q'|L_{j,\mathbf u}|\mathbf Q\rangle
=\frac{\tilde\phi(\lvert\mathbf q\rvert)}{S}\,
e^{-i\mathbf q\cdot(\mathbf R+\mathbf t_j)}\,D_{\mathbf q}(\mathbf u),
\qquad \mathbf q=\mathbf Q'-\mathbf Q,
$$

with $D_{\mathbf q}=e^{-i\mathbf q\cdot\mathbf u}-\chi(\mathbf q)$ and
$\chi(\mathbf q)=e^{-a^2q^2/2}$.

**Bloch blocks.** A cell-periodic Hamiltonian is block diagonal in
$\boldsymbol\kappa=(i,j)\mathbf g/c$. A state of block $\boldsymbol\kappa$ is
$\psi=\sum_nc_n\,e^{i(\boldsymbol\kappa+\mathbf n\cdot\mathbf g)\cdot\mathbf r}/\sqrt S$,
with $\sum_n\lvert c_n\rvert^2=1$, in the cell orders $\mathbf n$. Between
$\lvert\boldsymbol\kappa a\rangle$ and $\lvert\boldsymbol\kappa' b\rangle$, the
transfer from incoming order $\mathbf n$ to outgoing order $\mathbf m$ is

$$
\mathbf q_{\mathbf m-\mathbf n}=\boldsymbol\kappa'-\boldsymbol\kappa+(\mathbf m-\mathbf n)\cdot\mathbf g .
$$

This is final minus initial momentum. The image phase factorizes:
$e^{-i\mathbf q_k\cdot\mathbf t_j}=e^{-i(\boldsymbol\kappa'-\boldsymbol\kappa)\cdot\mathbf t_j}$,
because $\mathbf g\cdot\mathbf t_j\in2\pi\mathbb Z$. It is the same for every
transfer $k$, so it drops out of $\lvert\cdot\rvert^2$. Each of the $c^2$ images
therefore contributes the same squared amplitude, and

$$
W_{\boldsymbol\kappa'b,\boldsymbol\kappa a}
=c^2\,\lambda\,\mathbb E_{\mathbf u}\Bigl\lvert\sum_kM_k\frac{\tilde\phi_k}{S}e^{-i\mathbf q_k\cdot\mathbf R}D_{\mathbf q_k}\Bigr\rvert^2
=\sum_{k,l}M_k\,\mathcal C_{kl}\,M_l^*,
$$

$$
\mathcal C_{kl}=\lambda f_kf_l^*\bigl[\chi(\mathbf q_k-\mathbf q_l)-\chi(\mathbf q_k)\chi(\mathbf q_l)\bigr],
\qquad
f_k=\frac{\tilde\phi(\lvert\mathbf q_k\rvert)}{cA}\,e^{-i\mathbf q_k\cdot\mathbf R}.
$$

Here $M_k=\sum_{\mathbf m-\mathbf n=k}c^{b*}_{\mathbf m}c^a_{\mathbf n}$. The
factor is $c^2/S^2=1/(cA)^2$, so the effective area is $\boxed{cA}$, not $c^2A$.
For $c=1$ the shift is zero, and this is the Γ-point formula of the previous
section. The common shift phase $e^{-i(\boldsymbol\kappa'-\boldsymbol\kappa)\cdot\mathbf R}$
is also global. The $\mathbf R$ phase that matters is the relative one,
$e^{-i(k-l)\cdot\mathbf g\cdot\mathbf R}$, as at Γ.

**Loss.** Summing over images,
$\Gamma(\mathbf r)=\lambda\sum_j\operatorname{Var}_{\mathbf u}\phi_j(\mathbf r)$.
Its harmonics on the $\mathbf g/c$ lattice carry
$\sum_je^{-i\mathbf G\cdot\mathbf t_j}=c^2\,\delta_{\mathbf G\in\mathbf g\text{-lattice}}$.
Only cell vectors $\mathbf G$ survive, with

$$
\Gamma_{\mathbf G}=\lambda\sum_{\mathbf Q\in\mathbf g/c\text{ lattice}}f_{\mathbf Q}f^*_{\mathbf Q-\mathbf G}\bigl[\chi(\mathbf G)-\chi(\mathbf Q)\chi(\mathbf Q-\mathbf G)\bigr],
$$

using the same $f$ with area $cA$. $\Gamma$ is cell periodic, so it does not mix
blocks. Then
$\langle\boldsymbol\kappa a|\Gamma|\boldsymbol\kappa a\rangle=\sum_{\mathbf n,\mathbf m}c^{a*}_{\mathbf n}\Gamma_{(\mathbf n-\mathbf m)\cdot\mathbf g}c^a_{\mathbf m}$.
A tail of $T$ cell orders is $cT$ orders on the fine lattice. The sum
$\sum_{\mathbf Q}(cA)^{-2}$ approaches the $c$-independent continuum integral,
because it has $c^2$ more terms per unit area.

**Block-diagonal state.** The image sum also gives
$\sum_jL_j\,|\boldsymbol\kappa a\rangle\langle\boldsymbol\kappa a|\,L_j^\dagger\propto\delta_{\boldsymbol\kappa'\boldsymbol\kappa''}$
in the outgoing blocks. A $\boldsymbol\kappa$-diagonal density matrix therefore
stays $\boldsymbol\kappa$-diagonal. Inter-block coherences are exactly zero in
the ensemble, so the Pauli model only has to treat degeneracies within one
block. Completeness over all blocks gives
$\sum_{\boldsymbol\kappa'b}W_{\boldsymbol\kappa'b,\boldsymbol\kappa a}=\Gamma_{\boldsymbol\kappa a}$.

## Cheap filters before implementation inspection

| Filter | Result |
| --- | --- |
| Units | $\tilde\phi/(cA)$ is dimensionless, and $W,\Gamma$ are in Å$^{-1}$ through $\lambda$. Pass. |
| $c=1$ limit | Shift zero, area $A$, lattice $\mathbf g$: the Γ-point formula exactly. Pass. |
| Continuum scaling | Single-block rates scale as $c^{-2}$, with $c^2$ blocks of final states. $\Gamma$ is a Riemann sum with $c$-independent limit. Pass. |
| Zero limits | $a=0$ or $\lambda=0$ gives zero. Pass. |
| Shift sign | $\mathbf q=\boldsymbol\kappa'-\boldsymbol\kappa+\ldots$ is final minus initial. It is material for $c\ge2$. Tested numerically below. |

## Implementation comparison

The bodies were read after the derivation.

- `bloch_tds_rates`: `area = cells * area_ang2` is $cA$. The shift is
  `(kappa_out - kappa_in) * g / cells`, which is $\boldsymbol\kappa'-\boldsymbol\kappa$.
  The loss maps orders to fine orders `orders * cells + kappa` on `g / cells`,
  with area $cA$ and tail `tail_orders * cells`. `vectors_loss` evaluates
  harmonics at every fine difference, but `loss[p, q]` reads only differences
  that are multiples of `cells`, which are the cell vectors. The fine FFT has
  size $2(2cT+1)$, which avoids circular wrap for the represented span. Global
  offsets follow block order. This agrees.
- `_kraus_rates`: `low = out.min - in.max` and `high = out.max - in.min` bound
  $\mathbf m-\mathbf n$ over all represented pairs. `diff = out - in - low`,
  flattened with stride `high[1] - low[1] + 1`, matches the
  `meshgrid(..., indexing="ij")` layout in `_kraus`. The `einsum`
  `"pb,pqj,qa->jba"` is $\sum_{pq}c^{b*}_pK_{p-q,j}c^a_q$. This agrees.
- `_kraus`: $\mathbf q=$`shift + grid * g`, and
  $f=\tilde\phi(\lvert\mathbf q\rvert)/(cA)\,e^{-i\mathbf q\cdot\mathbf R}$.
  The covariance is $\lambda f_kf_l^*[\chi(\mathbf q_k-\mathbf q_l)-\chi(\mathbf q_k)\chi(\mathbf q_l)]$.
  Truncated `eigh` gives $K=U\sqrt w$. This agrees.
- **Cache key.** The key is (`shift` rounded to $10^{-12}$ Å$^{-1}$, `low`,
  `high`). The other inputs to `_kraus` (`g`, $\mathbf R$, $a^2$, $\lambda$,
  area, phase, `rank_rtol`) are fixed within one `bloch_tds_rates` call, and the
  cache is local to that call, so the key is sufficient. Distinct shifts differ
  by at least $\lvert\mathbf g\rvert/c\gg10^{-12}$. Passing one external
  `cache` to `_kraus_rates` across different parameters would be unsafe. The
  function is private, and no caller does so.
- **$c=1$ reduction.** `periodic_tds_rates` calls `_kraus_rates` with shift
  `(0.0, 0.0)` and `vectors_loss` on the same arguments. `bloch_tds_rates` at
  `cells=1` makes identical calls, so the result is bitwise equal; the test uses
  `assert_array_equal`.
- `bloch_report`: blocks come from `axial_states` at
  $\boldsymbol\kappa=(i,j)\mathbf g/c$. Its basis is $\boldsymbol\kappa+\mathbf n\cdot\mathbf g$
  with $\lvert n_{x,y}\rvert\le M$, the convention assumed above. Block 0 is
  $\boldsymbol\kappa=0$, and the entrance is the zero-tilt plane wave there.
  "Bound" is $E<U_{\max}$ over all blocks. Cuts sit at gap midpoints of the
  pooled spectrum. A cut between degenerate states of different blocks is
  harmless, because there are no inter-block coherences.

No divergent factor, sign, area, index offset or cache convention was found in
the four functions.

## Independent numeric evidence (2026-10-18)

**Real-space supercell.** The scratch script is not committed and does not call
`periodic_tds_rates`, `_kraus` or `vectors_loss` for the reference. Its inputs:

- a cell of $2.3\times1.7$ Å, and a string at $\mathbf R=(0.37,-0.21)$ Å;
- a Gaussian per-atom phase of width $0.25$ Å;
- $a=0.12$ Å and $\lambda=0.05$ Å$^{-1}$;
- cell orders $\pm2$, with 8 random orthonormal vectors per block, for
  $c=1,2,3$.

Each of the $c^2$ strings got its own displacement. $\phi_{j,\mathbf u}$ was
built in real space on the $c\times c$ supercell, using $5\times5$ supercell
images and a $24c\times24c$ midpoint grid. Each $\mathbf u_j$ was averaged with
$20\times20$ Gauss–Hermite nodes, with $\langle\phi\rangle$ taken from the same
quadrature. Then $\sum_j\lambda\,\mathbb E\lvert\langle b|L_j|a\rangle\rvert^2$
and $\sum_j\lambda\int\lvert\psi_a\rvert^2\operatorname{Var}\phi_j$ were formed
directly.

| $c$ | states | max rel. diff. $W$ | max rel. diff. $\Gamma$ | $\mathbf R\to-\mathbf R$ | shift sign flipped |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | $5.1\times10^{-15}$ | $2.8\times10^{-15}$ | 150% | (no shift) |
| 2 | 32 | $2.5\times10^{-15}$ | $3.1\times10^{-15}$ | 98% | 53% |
| 3 | 72 | $4.8\times10^{-15}$ | $2.9\times10^{-15}$ | 105% | 62% |

The shift sign, the $\mathbf R$ phase and the $cA$ area are each tested and
correct. An area of $c^2A$ would scale $W$ by $1/c^2$, and the agreement rules
that out. $\Gamma_a-\sum_bW_{ba}\ge1.9\times10^{-4}$ for every state.

**Tests.** The project runner passed all 8 tests in
`tests/montecarlo/test_channeling_axial_matching_study.py`. The new $2\times2$
test sums `periodic_tds_rates` over four image strings on the fine lattice. That
is a different route, but it shares `_kraus`. The real-space check above is
independent of it.

**Multiplet basis.** For Si [001], 5 MeV, $\pm6$, $c=2$, a random unitary was
applied inside every degenerate multiplet of every block. The reference exposure
changed by $\le1.7\times10^{-9}$ relative, and the matched exposure at
$E_m=94.97$ eV by $\le5\times10^{-11}$.

**Report reproduction.** `uv run python checks/axial_matching_study.py bloch`
took 8 min 20 s here, under concurrent load (the docstring says ~4 min). Every
entry of the note's Bloch table matches the output after rounding:

- $\pm6$, $c=1$: 0.133, 0.750, 2.50, 3.59%;
- $\pm6$, $c=2$: 0.171, 0.951, 3.09, 4.27%;
- $\pm6$, $c=3$ at 93.62 eV: 0.184, 1.015, 3.33, 4.58%;
- $\pm8$, $c=1$: 0.334, 1.80, 5.80, 8.40%;
- $\pm8$, $c=2$: 0.389, 2.11, 6.79, 9.59%.

The reference exposures 5416, 5684 and 5768 Å match as well. "c=3 adds about
0.3 points" matches: it is 0.31.

**Depth-budget arithmetic.** `report()` gives these $\pm10$, $c=1$ deficits:

| $E_m$ (eV) | $10^4$ Å | $3\times10^4$ Å |
| ---: | ---: | ---: |
| 94.93 | 0.420% | 2.29% |
| 175.27 | 0.178% | 1.09% |

Scaling by 1.2 and interpolating log-log between $10^4$ and $3\times10^4$ Å
puts the 1% crossing at 1.56 µm and 2.55 µm. Linear interpolation gives 1.44
and 2.43 µm. With the $c=3$ factor of about 1.35, the crossings are 1.44 and
2.37 µm. "About 1–2 µm" and "about 2–3 µm" follow from either factor.

## Characterisation discrepancies

1. **"The Bloch dimension converges by $c\approx3$" is overstated.** At
   $\pm6$ and $3\times10^5$ Å, the $c=1\to2$ step is +0.68 points and the
   $c=2\to3$ step is +0.31 points, and the reference exposure steps are 268
   and 84 Å. At $10^5$ Å the deficit steps are +0.59 and +0.24 points. The increment shrinks
   (ratio about 0.3–0.5), but convergence is shown only by the trend, and only
   at $\pm6$ orders. The $c=3$ row also uses $E_m=93.6$ eV rather than 95.0 eV.
   At this steepness, $d\ln(\text{deficit})/dE_m\approx-0.026$ eV$^{-1}$ at
   $3\times10^4$ Å, so the lower $E_m$ inflates that row by about 4%. Suggested
   wording: "the $c=2\to3$ increment is about half the $c=1\to2$ increment at
   ±6 orders; residual Bloch sensitivity is a few tenths of a point at
   $3\times10^5$ Å".
2. **The "~1.2 Bloch factor" is the $c=2$ factor, not the near-converged
   one.** The $c=2/c=1$ ratios at about 95 eV are 1.19–1.29 at $\pm6$ and
   1.14–1.17 at $\pm8$. The $c=3/c=1$ ratios at $\pm6$ are 1.27–1.38. The note
   then calls $c\approx3$ converged but scales by ~1.2. With ~1.35 the budgets
   move by under 0.2 µm and stay inside the stated ranges, so the conclusion
   survives. The text should name the factor's $c$, or use the $c=3$ value. At
   175 eV the measured $c=2$ factor is 0.98–1.11, with $E_m$ differing by up to
   3 eV between grids, so 1.2 is conservative there.
3. **"Because the deficits are lower bounds, these budgets are upper bounds"
   is a trend, not a bound.** The deficit rises monotonically with orders
   ($\pm6,\pm8,\pm10$: 0.13, 0.33, 0.42% at $10^4$ Å and 95 eV) and with $c$.
   Nothing proves that it keeps rising. The statement should be conditional:
   the budgets are upper bounds "if the observed monotone increase with orders
   and $c$ persists". The $\pm8\to\pm10$ step at $3\times10^4$ Å is still +27%
   relative.

## Minor implementation observation (not a rate error)

`bloch_report` samples $\boldsymbol\kappa\in[0,\mathbf g)$ unfolded, so the
order window $\lvert n\rvert\le M$ is centred on $\boldsymbol\kappa$, not on
the first Brillouin zone. At $c=3$, $\boldsymbol\kappa=(2,0)\mathbf g/3$ is the
time-reversal partner of $(1,0)\mathbf g/3$, but its truncated spectrum differs:
by 0.016 eV in the lowest 8 levels and by up to 7.5 eV below 180 eV.
$(\pm1,0)\mathbf g/3$ agree to $6\times10^{-14}$ eV. This is a valid truncated
basis at that $\boldsymbol\kappa$, and it belongs to the plane-wave grid error
the note already names as dominant. Still, it breaks the
$\boldsymbol\kappa\leftrightarrow-\boldsymbol\kappa$ symmetry of the $c=3$
sample, and it probably contributes to the shifted 93.6 eV cut. Folding
$\boldsymbol\kappa$ into $(-\mathbf g/2,\mathbf g/2]$ would make the truncation
symmetric. This is a recommendation; the code was not changed.

## Verdict and limitations

**Scoped verdict for `bloch_tds_rates`, `_kraus_rates`, `_kraus` and
`bloch_report` arithmetic: rederived. Research-note wording: discrepancy
(items 1–3 above; the numbers themselves reproduce, and the depth-budget ranges
survive). Overall ledger claim: unverified.**

This review does not validate:

- physical convergence in $c$ or in the plane-wave grid;
- a physical matching surface;
- electronic jumps;
- the Pauli treatment of the near-degenerate pairs above $U_{\max}$;
- non-zero entrance tilt.

Suggested ledger edit: keep `Status: unverified`. In Notes, cite this section
as the scoped fresh-context review of the Bloch-block functions (`rederived`),
and record the research-note wording corrections. Only a human can grant
sign-off.

# MCWF return-flux comparison: scoped verification (2026-10-18)

Scope: the `jump_hook` argument of `AxialMCWF.run`
(`src/pyrite/montecarlo/channeling/axial_mcwf.py`), and `GridProjector`,
`projective_hook`, `_job` and `main` in `checks/axial_mcwf_return_flux.py`.
Fresh context; the derivation below was written from the research note and
the ledger rows before the diff bodies were compared.

## Independent derivation

**Grid Hamiltonian.** The split-step propagates
$\psi=e^{i\mathbf k_t\cdot\mathbf r}\phi$ with
$\exp[-i\,\Delta z\,(T+U-iW)/\hbar v]$ and
$T=\frac{\hbar^2}{2\gamma m}\lvert\mathbf k_t+\mathbf k_{\mathbf m}\rvert^2$,
the relativistic transverse mass $\gamma m$. Pointwise multiplication by $U$ on
an $N$-point periodic grid is the circular convolution
$(U\phi)_{\mathbf m}=\sum_{\mathbf m'}\hat U_{(\mathbf m-\mathbf m')\bmod N}\phi_{\mathbf m'}$.
If $U$ has the crystal period and the grid spans $c$ cells with $c\mid N$, the
sampled $U$ is periodic with $N/c$ points, so $\hat U_{\mathbf n}=0$ unless
$\mathbf n\equiv0\pmod c$. Hence $H=T+U$ couples only orders of equal residue
$\mathbf m\bmod c$ (the $c^2$ Bloch momenta $\boldsymbol\kappa=\mathbf r\,\mathbf g/c$),
and diagonalising each block gives $P(E_m)=\sum_{E_\alpha<E_m}\lvert\alpha\rangle\langle\alpha\rvert$
exactly for the represented grid. FFT index $i$ and signed order $m$ agree
modulo $c$ because $c\mid N$.

**Projective unravelling.** For $\rho=P\rho P$ the retained generator
(`eq-axial-retained-generator` in the research note) has no-jump operator
$PH_\perp P-\tfrac{i}{2}\hbar v\,\Gamma$ with $\Gamma=P(\sum_aJ_a^\dagger J_a)P$, and
gain $\sum_aPJ_a\rho J_a^\dagger P$. Measuring $P$ after a jump on
$\lvert\chi\rangle=J_a\lvert\psi\rangle/\lVert J_a\psi\rVert$ keeps
$P\chi/\lVert P\chi\rVert$ with probability $\lVert P\chi\rVert^2$; averaged
over jumps this reproduces the gain term exactly. The MCWF no-jump evolution
uses the full $W$, not $PWP$, so for a state in $P$ it generates an $F$
amplitude of first order in $\epsilon=\lvert\langle f\rvert W\lvert p\rangle\rvert/\lvert E_f-E_p\rvert$
(adiabatic dressing plus an oscillating part). The post-jump outcome
probability therefore differs at $O(\epsilon)$ through the cross term
$2\,\mathrm{Re}\langle PJ_aP\psi\vert PJ_aF\psi\rangle$, and the retained
weight at $O(\epsilon^2)$. For a pure entry state the deterministic entry
projection times $p_0=\lVert P\phi_0\rVert^2$ is exact: the measured ensemble
is $p_0\cdot\{\text{trajectories from }P\phi_0/\sqrt{p_0}\}+(1-p_0)\cdot0$.

**Nesting.** With $E_m\ge U_{\max}$ inside a gap at the entry tilt,
$P(U_{\max})\le P(E_m)$, so the model and reference bound weights coincide at
$z=0$ and the deficit starts at zero.

## Cheap filters before implementation inspection

Units: $\hbar^2/2m$ in eV Å$^2$ divided by $\gamma$, $k$ in Å$^{-1}$, so $T$ is
in eV; the tilt passed to the hook is $\mathbf k_t$ in Å$^{-1}$
(`momentum * tilt_rad`). Limits: cut above the spectrum gives the identity
hook; cut below it absorbs at entry. Sign: absorption can only lower the
ensemble bound weight. All pass.

## Implementation comparison

- `_half_kinetic` uses `HBAR2_2M_EV_ANG2 / self.gamma * (kx**2 + ky**2)` with
  `kx = grid.kx + tilt`; `_setup` passes the same scale and `grid.kx`. Match.
- `GridProjector` builds residue blocks from `fftfreq` orders and the coupling
  `u[(i - j) % N]`. Match with the circular convolution above.
- `run` calls the hook once per electron at entry (before the threshold draw)
  and after each normalised TDS/electronic jump, with the kicked tilt after an
  electronic jump. It never touches the Philox streams, so an identity hook
  only renormalises an already normalised state. Absorption sets
  `live[b] = False` and `lagged[b] = 0`, mirroring the sink branch; absorbed
  electrons contribute zero to `observables` and are excluded from `surviving`.
- `projective_hook` streams are `SeedSequence((seed, 0x548, electron))`,
  independent of the MCWF keys. Entry is projected deterministically and
  `_job` scales by `p0` from the same zero tilt that `run` uses by default.

## Independent numeric evidence (2026-10-18)

Default grid ($2\times2$ cells, $64^2$, TDS-only absorptive part): Γ-point gaps
at the three cuts 7.97, 94.93, 178.07 eV are 1.49, 3.73, 2.26 eV, and
$W$ ranges $5\times10^{-4}$ to 2.2 eV (mean 0.063 eV). Over straddling pairs
in the Γ eigenbasis, $\max\epsilon=0.016$, 0.006, 0.009 and
$\max_p\sum_f\epsilon^2=3.5\times10^{-4}$, $7\times10^{-5}$, $9\times10^{-5}$:
at zero tilt the leakage is off-resonant. At three random tilts with
$\lvert\mathbf k_t\rvert\approx0.1$ to 0.3 Å$^{-1}$, the nearest level to the
same cuts lies 0.01 to 1.0 eV away, comparable to $W$; there the cut is not in
a gap and $\epsilon$ is not small. One $64^2$ block set costs about 3 s and
67 MB of eigenvectors. Focused tests pass (3 + 3), and `--smoke` runs in about
7 s with zero entry deficit.

## Discrepancies and recommendations

1. **Caveat incomplete for electronic jumps.** Electronic kicks give each
   electron a generic tilt. The cuts sit in gaps only at Γ, so for runs with
   electronic jumps the coherent $P\leftrightarrow F$ leakage is not
   off-resonant in general, and $P(E_m)$ at a generic tilt can split
   near-degenerate levels.
2. **Cache memory (`GridProjector.spectrum`, check line 72).** Electronic kicks
   produce a new continuous tilt per jump, so caching up to 257 tilts holds
   about 17 GB per worker at the default grid. After clearing, cached tilts
   are evicted and get recomputed at every observation depth. This is a cost
   and memory bug for the lab-box default, not a physics error.
3. **Estimator bias.** The deficit is a mean of per-chunk ratios
   $1-X/X_{\rm ref}$, which is biased at $O(\mathrm{CV}^2)$ per chunk of 256
   electrons. Use a ratio of pooled means, with a jackknife or delta-method
   SEM, or state the bias. `deficit.mean` together with `nanstd` is
   inconsistent but harmless here.
4. **Wording.** The `AxialMCWFResult` docstrings for `surviving` ("not removed
   by a sink jump") and `jumps` omit `absorbed`. The research note says the
   `--mechanisms tds` run "cross-checks" the secular table, but the default
   grid ($\pm16$ orders per cell, $c=2$) matches no row of that table, and
   MCWF keeps coherences. It is a comparison only at matched orders and $c$.
5. **Test independence.** The block test applies $H$ by FFT kinetic plus
   pointwise $U$, which is a genuine independent route. Not tested: the solver's
   kinetic scale (the test uses $\gamma=10$), hook-plus-electronic-jump
   ordering, Born-rule frequencies, stream disjointness, and the $p_0$
   scaling.

## Verdict and limitations

**Scoped verdict: `jump_hook` mechanics and `GridProjector` algebra
rederived. The research-note caveat is a discrepancy (item 1); items 2 and 3
are implementation findings. Overall ledger claim: unverified.** Electronic
loss magnitudes were not checked (SBETHE data were unavailable). Suggested
ledger edit: keep `Status: unverified`. Cite this section as the scoped
review, and record items 1, 3 and 4 in the note. Only a human can grant
sign-off.

### Follow-up applied (2026-10-18)

The review findings were addressed in `checks/axial_mcwf_return_flux.py`.
`GridProjector.gap_cut` places the cut at each tilt in the widest level gap
within `--gap-window-eV` of the nominal matching energy, and each row reports
the smallest and median gaps used, so $\epsilon$ can be judged after the run.
The per-tilt cache is now least-recently-used with 160 entries. The default
grid is $32^2$ ($2\times2$ cells, about 4 MB per entry). The deficit is the
ratio of pooled means, with a delete-one-chunk jackknife SEM (`None` for one
chunk, which removes the NaN that previously broke JSON output). New tests
cover the solver's kinetic scale, the Born-rule keep frequency, the gap
choice, and the pooled estimator. `AxialMCWFResult` docstrings now name
`absorbed`. The hook combined with electronic jumps is still untested here
because SBETHE data is not installed.
