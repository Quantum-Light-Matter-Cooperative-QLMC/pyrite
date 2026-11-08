# `cr-axial-fiber-regression`: companion two-time source of the fibred absorbing axial model

**Status:** see the [ledger row](../ledger-channeling.md#cr-axial-fiber-regression). Task-owner derivation (2026-11-07, #992); the fresh-context verification is recorded below it.

## Problem

The on-axis photon source of `cr-axial-qrt-source-normalization` needs the
two-time current correlation

$$
G_{ab}(z_i,z_j)=\operatorname{Tr}\!\left[J_a\,\Lambda_{z_i-z_j}\!\left(\rho(z_j)J_b^\dagger\right)\right],
\qquad z_i\ge z_j ,
$$

where $\Lambda_\tau$ is the depth-$\tau$ map of the dissipative model and
$J_a$ ($a=x,y$) the positive-frequency (lowering) velocity/c current. The
dense `regression_correlation` builds $\Lambda$ on one finite Hilbert space.
The physical axial dissipator of `cr-axial-fiber-unraveling` does not fit
there: an electronic kick $S_ie^{i\mathbf q\cdot\mathbf r}$ with uniform
azimuth and a continuum TDS jump $J_{Q,u}$ move the Bloch carrier
$\kappa\to\kappa'$ over a continuum. Its density stays block diagonal in
$\kappa$, but the reachable set of carriers is uncountable.

## Model

In fibre $\kappa$ the retained subspace $P(\kappa)$ is spanned by the
orthonormal columns $V_\kappa$ of eigenvectors of $H(\kappa)$ below the cut.
The absorbing fibred map is the Dyson expansion of

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}\left(H_\mathrm{eff}\rho-\rho H_\mathrm{eff}^\dagger\right)
+\sum_r\int d\mu_r\,P A_r\,\rho\,A_r^\dagger P ,
\qquad H_\mathrm{eff}=V^\dagger\!\left(H-\tfrac{i\hbar v}{2}\Gamma\right)V ,
$$

with $\Gamma$ the total loss (lattice or string TDS plus kicks), $A_r$ the
jump operators with their continuous labels $\mu_r$ (kick azimuth; string
$(Q,u)$; lattice Kraus column) and $P$ the retained projector of the output
carrier. Outputs outside $P(\kappa')$ (outgoing, unresolved, tail) leave the
model. Every term maps a $\kappa$-block operator to a $\kappa'$-block
operator with **the same** operator acting on the ket and on the bra side.

For a rank-one operator $X=|u\rangle\langle w|$ in one fibre,

$$
\Lambda_\tau(X)=\sum_{\text{records }R}K_R\,|u\rangle\langle w|\,K_R^\dagger ,
\qquad K_R=U(\tau-t_n)\,PA_{r_n}\,U(t_n-t_{n-1})\cdots PA_{r_1}\,U(t_1),
$$

with $U(s)=e^{-isH_\mathrm{eff}/\hbar v}$ in the current fibre, and the sum
standing for the integral over jump depths, channels and continuous labels.

## Estimator

The fibred unraveling samples a record of a normalized primary $u$ with
probability density $p_R=\lVert K_R u\rVert^2$ (the MCWF record law of
Castin, Dalibard & Mølmer, arXiv:0805.4002, Eqs. (4), (8), as already
verified for this engine). Records that end outside the retained subspace
have no $K_R$ term. Hence

$$
\operatorname{Tr}[J_a\Lambda_\tau(|u\rangle\langle w|)]
=\sum_R\langle K_Rw|J_a|K_Ru\rangle
=\mathbb E_{R\sim p}\!\left[\frac{\langle K_Rw|J_a|K_Ru\rangle}{\lVert K_Ru\rVert^2}\right].
$$

The summand is linear in $K_Ru$, so it vanishes on every record with
$p_R=0$: sampling by the primary's own law is unbiased for any $w$, with no
absolute-continuity condition on $w$. Define the companion
$\hat w_R=K_Rw/\lVert K_Ru\rVert$ and the normalized primary
$\hat u_R=K_Ru/\lVert K_Ru\rVert$; the estimate is
$\langle\hat w_R|J_a|\hat u_R\rangle$.

The unraveling factorizes $\lVert K_Ru\rVert$ step by step: after a no-jump
segment the primary is divided by $\lVert U(s)c\rVert$; after a retained jump
by $\lVert PA\chi\rVert$, with $\chi$ the normalized pre-jump state. The
companion receives the identical operator and the identical divisor at each
step, so its product of divisors equals $\lVert K_Ru\rVert$.

With the incident density $\rho(z_j)=W\,\mathbb E[\mathbb 1_\mathrm{alive}
|\hat u(z_j)\rangle\langle\hat u(z_j)|]$ (the `observe` contract of the
fibre claim) and $\rho J_b^\dagger=W\,\mathbb E[\,|\hat u\rangle\langle
J_b\hat u|\,]$, insert $w=J_b\hat u(z_j)$ at every recorded depth and use
the Markov property of the primary's future draws:

$$
G_{ab}(z_i,z_j)=W\,\mathbb E\!\left[\mathbb 1_{\mathrm{alive}}(z_i)\,
\langle\hat w_{jb}(z_i)|J_a(\kappa_{z_i})|\hat u(z_i)\rangle\right].
$$

$J_a(\kappa)$ is evaluated in the fibre the record occupies at $z_i$. The
equal-depth term is $W\langle J_b\hat u|J_a\hat u\rangle$. The reverse pair
follows from $G(z_j,z_i)=G(z_i,z_j)^\dagger$ in the component indices.

### Implementation algebra

`_Companions` stores companions in the right-eigenvector coordinates
$y$ of $H_\mathrm{eff}=R\,\mathrm{diag}(\lambda)R^{-1}$, in the frame where
the primary's unnormalized state is $R\,e^{-i(z-z_0)\lambda/\hbar v}a$,
$a=R^{-1}c$. At a recorded depth it multiplies $y$ by the step factor and
evaluates $\langle w|J_au\rangle=y^\dagger R^\dagger(J_au)/n$, $n$ the
primary norm, without forming $w$. A new companion is
$y=R^{-1}(nJ_b\hat u)$. At a retained jump it forms the window-order
companion $V R\,y/n$, applies the drawn operator (the batched box map
returned by `_jump`), projects with $V_{\kappa'}^\dagger$ on the window
orders and divides by the primary's $\lVert c_\mathrm{new}\rVert$, then
returns to eigen-coordinates of the new fibre. The hooks draw no random
numbers, so the primary events are bitwise those of `unravel_fiber_outgoing`.

### Currents and photon law

In fibre $\kappa$ with window orders $\mathbf n$,
$J_a=\hbar c\,V^\dagger\mathrm{diag}(\kappa_a+n_ag_a)V/(\gamma mc^2)$,
which is $v_a/c$ when the kinetic term is $\hbar^2k^2/(2\gamma m)$. Only
elements $[f,i]$ with $E_i-E_f$ above the Bohr tolerance are kept. An
optional `boundary_eV` keeps final states strictly below it, the axial block
of `cr-axial-source-partition`.

Per primary, the spectrum is
$\alpha E\sum_a\sum_{ij}v_{ia}\hat G_{ij,aa}v_{ja}^*/(4\pi^2(\hbar v)^2)$
with $v_{ia}=w_iA_i(E)e^{iE(1-\beta)z_i/\hbar v}$, trapezoid weights $w_i$
and amplitude attenuation $A=e^{-\mu L/2}$ of `cr-axial-source-partition`.
Hermitian symmetry gives
$2\,\mathrm{Re}\,v\,\mathrm{tril}'(\hat G_{aa})\,v^\dagger$, with the
diagonal halved. The mean of independent primaries is unbiased; their
spread is the standard error. Single-primary values are signed and are not
clipped.

## Assumptions and scope

- The absorbing no-return fibred model of `cr-axial-fiber-unraveling` and its
  retained subspaces define the source. Emission after a record leaves
  $P(\kappa)$, and its interference with earlier emission, is not
  represented; native transport must own it, or a bound must be given.
- On-axis observation only: no transverse photon recoil, so $J_a$ keeps the
  carrier.
- Constant longitudinal velocity and depth as the evolution coordinate.
- Gains that keep emission coherence through a jump are part of the map. A
  pure-damping model, which removes them, is a different (incorrect) model.
- Material completeness, grid/window/cut/ensemble convergence and agreement
  with measured spectra are outside this algebraic claim.

## Limiting cases

- No jumps: each primary is deterministic and the estimate is the exact
  regression for every primary.
- Ground-state primary: $J_b\hat u=0$, so no companion and zero emission.
- One fibre and positive gains only: the dense one-sector
  `regression_correlation` value.
- The photon law of a positive dense correlation reproduces
  `on_axis_photon_spectrum`.

## Checks (anchor module)

`tests/montecarlo/quantum/test_channeling_axial_fiber_regression.py`:

- jump-free model equals the dense regression to 1e-14 and has zero spread;
- `signed_on_axis_spectrum` equals `on_axis_photon_spectrum` to 1e-10;
- two-carrier commensurate model ($q=g_x/2$, azimuths $0,\pi$) against an
  independent block GKSL oracle, with kicks alone and with a lattice Kraus
  column: correlation $z$-scores have RMS below 1.3 and maximum below 5,
  spectrum $|z|<4.5$; primary events equal `unravel_fiber_outgoing`;
- removing every gain (pure damping) deviates by more than 10 standard
  errors, so the estimator resolves surviving jump coherence;
- batched kick, lattice and string maps equal the drawn single outputs.

## Fresh-context verification (2026-11-07)

Verifier: fresh context (Claude Opus 5.5) that did not write the implementation.
Commit under test: `a3c144c34` (rebased onto `main` `65623db78`; content
unchanged from the pre-rebase review). Scratch oracles live under the
worktree's ignored `scratch/verify992/` and are not part of the anchor.

### Independent derivation

For each carrier $\kappa$ the model is the absorbing generator on the retained
coefficients,

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}\left(H_\mathrm{eff}\rho-\rho H_\mathrm{eff}^\dagger\right)
+\sum_\ell \int d\mu_\ell\;G_{\ell,\mu}\,\rho\,G_{\ell,\mu}^\dagger ,
\qquad G_{\ell,\mu}=V_{\kappa'}^\dagger\,[A_{\ell,\mu}]_\mathrm{window}V_\kappa ,
$$

with $H_\mathrm{eff}=\mathrm{diag}(E_\kappa)-\tfrac{i}{2}\hbar v\,V_\kappa^\dagger\Gamma V_\kappa$
and $\Gamma=\Gamma_\mathrm{lat}+\sum_i S_i^\dagger S_i+\sum_s\Gamma_s$ the
*full* loss (all outputs). The jump labels are: the lattice Kraus column $j$;
for a kick, the channel $i$ and the azimuth $\theta$ (operator
$N_\theta^{-1/2}S_ie^{i\mathbf q_\theta\cdot\mathbf r}$ for a discrete set, or
density $1/2\pi$); for a string, the folded output carrier and displacement
$(\kappa',\mathbf u)$ with the coherent harmonic sum $A_{\kappa',\mathbf u}$.
Since every gain is of the form $G\rho G^\dagger$ with one operator on both
sides, the Dyson series gives, for any rank-one block operator,

$$
\Lambda_\tau(|u\rangle\langle w|)=\int dR\;K_R|u\rangle\langle w|K_R^\dagger ,
\qquad
\operatorname{Tr}[J_a\Lambda_\tau(|u\rangle\langle w|)]
=\int dR\,\langle K_Rw|J_a|K_Ru\rangle .
$$

**Sampled law.** From normalized $c$ the engine draws a threshold $r$ and the
root $t$ of $\lVert U(t)c\rVert^2=r$, so the depth density is
$-\partial_t\lVert U(t)c\rVert^2=\langle U c|V^\dagger\Gamma V|Uc\rangle$
(per Å). With $\chi=VU(t)c/\lVert U(t)c\rVert$ the group is drawn with
probability $\langle\chi|\Gamma_g|\chi\rangle/\langle\chi|\Gamma|\chi\rangle$,
then: lattice column $\lVert BL_j\chi\rVert^2/\langle\chi|\Gamma_\mathrm{lat}|\chi\rangle$
(remainder unresolved); kick azimuth uniform, which is exact because
$\lVert S_ie^{i\mathbf q\cdot\mathbf r}\chi\rVert^2=\lVert S_i\chi\rVert^2$; string
output momentum with density $\lvert(A_{\kappa',\mathbf u}\chi)_o\rvert^2$,
so the in-box decision has probability $\lVert BA\chi\rVert^2/\lVert A\chi\rVert^2$
and the operator kept, $BA_{\kappa',\mathbf u}$, does not depend on the landing
order $o$. Finally `_split` keeps the retained part with probability
$\lVert P BA\chi\rVert^2/(\lVert PBA\chi\rVert^2+\lVert(B-P)A\chi\rVert^2+\text{outside})$.
The products telescope: the density of a record that stays retained is
$\lVert G_{\ell,\mu}U(t)c\rVert^2\,dt\,d\mu$, and over a full record
$p_R=\lVert K_Ru\rVert^2$. Outgoing, unresolved and tail outcomes terminate
the record, which is the absorbing model.

**Estimator.** With $f_R=\langle K_Rw|J_a|K_Ru\rangle/\lVert K_Ru\rVert^2$,
$\mathbb E_{p}[f_R]=\int_{p_R>0}dR\,\langle K_Rw|J_a|K_Ru\rangle$, and the
integrand is linear in $K_Ru$, so the excluded set $p_R=0$ contributes
nothing. The identity holds for every $w$; there is no likelihood ratio and so
no absolute-continuity condition. Inserting $w=J_b\hat u(z_j)$, using
$\rho(z_j)=W\,\mathbb E[\mathbb 1\,|\hat u\rangle\langle\hat u|]$ and the Markov
property of the future draws, gives
$G_{ab}(z_i,z_j)=W\,\mathbb E[\mathbb 1(z_i)\langle\hat w_{jb}(z_i)|J_a|\hat u(z_i)\rangle]$,
with $\hat w=K w/\lVert K\hat u\rVert$. This is
$\operatorname{Tr}[J_a\Lambda(\rho J_b^\dagger)]=\langle J_b^\dagger(z_j)J_a(z_i)\rangle$,
the right insertion of `RegressionCorrelation`. The left insertion
$\operatorname{Tr}[J_a\Lambda(J_b^\dagger\rho)]$ is a different quantity. The
reverse pair is $G_{ab}(z_j,z_i)=\overline{G_{ba}(z_i,z_j)}$.

**Norm bookkeeping.** The primary divisors are $\lVert U(t_1)c_0\rVert$,
$\lVert P A\chi_1\rVert$, $\lVert U(t_2-t_1)c_1\rVert,\dots$, whose product is
$\lVert K_Ru\rVert$. A companion that gets the same operators and the same
divisors therefore equals $K_Rw/\lVert K_Ru\rVert$ exactly.

**Currents and law.** $v_a/c=\hbar c(\kappa_a+n_ag_a)/(\gamma mc^2)$ for kinetic
energy $\hbar^2k^2/2\gamma m$. The positive-frequency part in the retained
eigenbasis is $[J_a]_{fi}$ with $E_i-E_f>0$ (Heisenberg $e^{-i(E_i-E_f)\tau/\hbar}$).
From the classical on-axis amplitude
$e^2\omega^2/(4\pi^2c)\lvert\int\boldsymbol\beta_\perp e^{i\omega(t-z/c)}dt\rvert^2$
with $dt=dz/v$, the number spectrum is
$dN/dE\,d\Omega=\alpha E\,(4\pi^2(\hbar v)^2)^{-1}\sum_a\sum_{ij}v_{ia}G_{ij,aa}\bar v_{ja}$,
where $v_{ia}=w_iA_ie^{iE(1-\beta)z_i/\hbar v}$. Here
$1-\beta=1/(\gamma^2(1+\beta))$ and $A_i=e^{-\mu L_i/2}$ is the amplitude
attenuation. The resonance sits at $E=(E_i-E_f)/(1-\beta)$ (the Doppler sign
is correct).

### Source-to-code comparison

| Item | Derivation | Code | Result |
| --- | --- | --- | --- |
| frame | unnormalized primary $R\,e^{-i\lambda(z-z_0)/\hbar v}a$, $a=R^{-1}c$ | `segment`: `u = right @ (exp(rate*(z-start)) * a)`; `_Fiber.evolve` is the same exponential | match |
| step factor | $y\mapsto e^{-i\lambda\,\delta z/\hbar v}y$ | `self.y * np.exp(rate * (z - here))[:, None]`, closed to `stop` on non-final segments | match |
| contribution | $\langle\hat w\vert J_a\hat u\rangle=y^\dagger R^\dagger(J_a\hat u)/n$ | `self.y.conj().T @ (right.conj().T @ ju.T) / norm` | match |
| insertion | $y=R^{-1}(nJ_b\hat u)$ | `inverse @ (norm * ju[live].T)` | match |
| jump | $y'=R'^{-1}V'^\dagger[\mathcal A(VRy/n)]_\mathrm{window}/\lVert c_\mathrm{new}\rVert$, with $n=\lVert\text{state}\rVert$ at the root | `jump(before, np.linalg.norm(state), operators[0], fiber, c_new)` | match |
| equal time | $\langle J_b\hat u\vert J_a\hat u\rangle$ | `(ju.conj() @ ju.T).T` (indices $[a,b]$) | match |
| scatter | `values[i, j_k, a, b_k]` | `values[i, labels[:,0], :, labels[:,1]]`: advanced indices split by a slice put the companion axis first, (ncomp, 2), equal to the RHS; no duplicate $(j,b)$ | match |
| completion | $G_{ab}(z_i,z_j)=\overline{G_{ba}(z_j,z_i)}$ for $i<j$ | `swapaxes(mean,0,1)[upper].conj().swapaxes(-1,-2)` | match |
| currents | $\hbar c\,V^\dagger\mathrm{diag}(k_a)V/(\gamma mc^2)$, $[f,i]$ with $E_i-E_f>\epsilon$, optional $E_f<$ boundary | `fiber_lowering_currents` | match |
| law | $2\,\mathrm{Re}\,v\,\mathrm{tril}'(G)\,v^\dagger$ with the diagonal halved, prefactor $\alpha E/(4\pi^2(\hbar v)^2)$ | `signed_on_axis_spectrum` (phase `E/(gamma**2*(1+beta))/hbar_v`) | match |
| SEM | $\sqrt{\widehat{\mathrm{Var}}/N}$ | spectrum `std(ddof=1)/sqrt(N)`; correlation `sqrt((E[x^2]-m^2)/(N-1))`, which is the same | match |

**Hooks.** `operators` only appends closures. They are the batched lattice
column (`lattice_column_outputs`), the kick with the drawn `shift`
(`kick_outputs`) and the string `apply` with the drawn column kernel. They
capture already-drawn values and make no `rng` call, and `_Companions` holds
no generator. The no-carry path differs from `HEAD~1` only by guarded
appends.

### Numerical checks (verifier-owned, independent of the anchor)

1. **Model A, own dense block GKSL oracle** (`scratch/verify992/oracle.py`;
   `expm` of the $\le 42^2$ superoperator; my own explicit convolution and
   relabel matrices, currents and photon double sum). The model has a
   rectangular cell $g=(1.0,1.6)$ Å$^{-1}$, an off-centre carrier
   $(0.13,-0.21)$, a complex off-centre potential and the physical kinetic
   term at 5 MeV. The window is $M=(2,1)$, the box $B=(3,2)>M$ and the
   retained dimension 7. There are two lattice Kraus columns ($D=(5,3)$) plus
   a $5\times10^{-6}$ Å$^{-1}$ unresolved remainder. Two kick channels
   ($q_0=\lvert(g_x/3,g_y/2)\rvert$ and $2q_0$) use four non-axis azimuths and
   reach 6 commensurate carriers with zone-crossing shifts. Absorption
   attenuation is on. The table gives z-scores against the oracle:

   | run | primaries | off-diagonal $G$ RMS / max $\lvert z\rvert$ (624) | diagonal (72) | spectrum RMS / max | gain term resolved |
   | --- | --- | --- | --- | --- | --- |
   | kicks + lattice, seed 7 | 60k | 0.94 / 2.92 | 1.15 / 2.45 | 1.25 / 1.45 | 94 SE |
   | kicks only, seed 7 | 60k | 1.45 / 3.66 | 1.11 / 2.67 | 2.07 / 2.45 | 109 SE |
   | kicks only, seed 8 | 240k | 1.12 / 2.98 | 1.41 / 2.42 | 1.24 / 1.41 | 219 SE |

   The 60k kicks-only excess did not grow at 4× the sample (the 24 spectrum
   points and 624 correlation entries are strongly correlated), so it is
   fluctuation, not bias. The retained weight agrees ($\max\lvert z\rvert\le1.9$).
   The left-insertion alternative is rejected at $>3\times10^3$ SE and the
   pure-damping (no-gain) model at 94–219 SE. Primary events are identical to
   `unravel_fiber_outgoing` in every run.
2. **Pathwise $K_R$ reconstruction with continuum strings**
   (`scratch/verify992/pathwise.py`). The model has a lattice column, a
   continuous-azimuth kick (`kick_azimuths=None`), one `TdsString` and a tail
   cap. The engine is wrapped only to log the record (depths, drawn
   $(Q,\mathbf u)$, carriers, choices). $K_R$ is rebuilt in the window basis
   from my own `expm` propagators $V e^{-iH_\mathrm{eff}s/\hbar v}V^\dagger$
   and explicit convolutions. The string column is `StringChannel.column`, the
   operator definition of `cr-axial-continuum-tds-jumps`; the kick shift is
   recovered from $\lvert\kappa'-\kappa+\mathbf s\cdot\mathbf g\rvert=q$. Then
   $W\langle K w|J_aK\hat u\rangle/\lVert K\hat u\rVert^2$ is evaluated per
   primary. Over 300 primaries (65 retained string, 71 kick and 10 lattice
   jumps; 38 tails, 126 unresolved, 130 outgoing) the companion values agree to
   $1.7\times10^{-11}$ relative.
3. **Absolute-continuity stress (model B)** (`scratch/verify992/oracle_b.py`,
   `oracle_b2.py`). The model is $x$-mirror symmetric with an $x$-even
   primary and an $x$-odd kick column along $\pm y$, which reaches a carrier
   whose retained space holds only $x$-even states (1, then 2 states with a
   nonzero $J_y$). There $\lVert P'A\hat u\rVert=9\times10^{-17}$ while
   $\lVert P'AJ_x\hat u\rVert/\lVert J_x\hat u\rVert=3.4\times10^{-3}$, about
   half of $\lVert A\hat u\rVert$. Against the dense oracle at 60k primaries:
   off-diagonal RMS 0.82 (max 2.80), spectrum max $\lvert z\rvert$ 0.73, gain
   term resolved at 108 SE. The estimate is correct even though the primary
   never samples the channel that only $J_b\hat u$ populates; in the exact map
   that channel's term $(P'A\hat u)(P'AJ_b\hat u)^\dagger$ is zero.
4. **Draw invariance.** With the pathwise model at 3000 primaries (1114 kicks,
   3244 TDS jumps, 381 tails), the events of `fiber_on_axis_source` and of the
   current `unravel_fiber_outgoing` equal those of the `HEAD~1` engine
   (`axial_fiber_unraveling.py`/`axial_tds_continuum.py` loaded from `git show`)
   field by field (`scratch/verify992/bitwise.py`).
5. **Photon law** (`scratch/verify992/law.py`). On a non-uniform depth grid
   with attenuation and a garbage upper triangle,
   `signed_on_axis_spectrum` and `on_axis_photon_spectrum` both equal my
   direct double sum to $1.4\times10^{-15}$ relative.
6. **Anchor module** and the fibre-unraveling module: 41 passed
   (`pyrite-dev test`, 31.5 s).

### Scope assessment

The stated scope is accurate and does not overreach. The claim covers the
absorbing no-return fibred model and its retained subspaces only. Emission
after a record leaves $P(\kappa)$, and its interference with earlier
emission, is excluded and stated as excluded. Observation is on axis
($J_a$ keeps the carrier). Unbiasedness is relative to the engine's own
sampled law: the root-found jump depth (`brentq`, `xtol=depth_tolerance_ang`)
is shared by primary and companion, so it adds no estimator-specific error
beyond `cr-axial-fiber-unraveling`. The ledger row's
$\operatorname{Tr}[J_a\Lambda(\rho J_b^\dagger)]$ matches the
`RegressionCorrelation` convention.

### Minor findings (no discrepancy)

- The anchor's dense comparison uses $B=M$, axis-only kicks and no strings.
  The string and box paths are anchored only by the batched-map equality
  test. Checks 1–2 cover $B>M$, non-axis and continuous-azimuth kicks, and
  strings with tails. A $B>M$ case in the anchor would pin the
  `window_in_box` path.
- A recorded depth exactly at a jump depth is assigned the post-jump state
  (half-open segments, the same convention as `_observe_segment`). This is a
  measure-zero choice.
- `_admit`'s working-memory estimate omits the companion store
  ($\le 2n_z$ columns of the retained dimension), which is negligible.
- Nothing can check at run time that the kinetic term is $\hbar^2k^2/(2\gamma m)$,
  because the `spectrum` provider is opaque. This remains a stated assumption.

### Verdict

Re-derivation `matches`: no factor, sign, insertion or bookkeeping divergence.
Verdict `rederived`. The verifier-owned oracle passes and the anchor module is
green locally, so the evidence supports the owner advancing the row to
`anchored`. Human sign-off remains in #277.
