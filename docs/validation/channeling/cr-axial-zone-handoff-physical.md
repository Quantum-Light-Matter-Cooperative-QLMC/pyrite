# `cr-axial-zone-handoff-physical`: physical axial outgoing instrument

**Status:** `anchored` (#883 scope); fresh-context verdict `rederived` 2026-11-01, findings resolved below.
Code: `src/pyrite/montecarlo/channeling/axial_outgoing_launch.py::{sample_physical_handoff,zone_energy_and_clock,handoff_balance,handoff_stream,AxialMatchingDeclaration,_Mapper}`
and the shared kinematics
`axial_zone_handoff.py::{crystal_to_lab_frame,momentum_inv_ang,on_shell_direction}`.
Builds on `cr-axial-fiber-unraveling` (absorbing fibre model, outgoing box and
Weyl measurement), `cr-axial-continuum-tds-jumps`/`cr-axial-tds-tail` (string
tails) and `cr-axial-phase-space-instrument` (finite Weyl POVM).

**Scope (#883).** This claim covers the map from the absorbing fibre model to a
correlated launch record: depth, lab position, kinetic energy, both lab
direction components, clock, incident ID and native stream key, with
retained, physical outgoing and unresolved weights kept apart. The depth
budget, the bound on omitted return flux and convergence of the matching
surface belong to #884. The photon source belongs to #885 and native
`LaunchState` conversion and admission to #886. None of them is claimed here.

## Model and fates

The fibre model (Castin, Dalibard & Mølmer, arXiv:0805.4002, Eqs. (4), (8))
evolves a retained amplitude in $P(\kappa)$, the spectral subspace below the
matching cut $E_c$ of the carrier-$\kappa$ Hamiltonian on the order window.
Every jump output is split into three parts: retained $P(\kappa')$ (the history
continues), represented outgoing $B-P(\kappa')$ on the box $\mathcal B$, and
the rest outside the box. A continuum-string output outside the box with
$|K|\le K_c$ is a `tail` record.

**Entrance.** The incident state is the plane wave $\psi_0$ at carrier
$\kappa_0$ (an order-window vector). Its complementary projectors at the
surface are $P(\kappa_0)$ and $B-P(\kappa_0)$; $\psi_0$ lies in the window, so
nothing falls outside the box. The entry must have unit norm and $\kappa_0$ must
lie in the half-open zone $[-g/2,g/2)$; anything else is refused. With $c=P\psi_0$ and $w=\lVert c\rVert^2$,

$$
\Pr(\text{enter})=w,\qquad
\Pr(\text{outgoing at }z=0)=\lVert f\rVert^2=1-w,\qquad f=B\psi_0-c .
$$

One uniform draw on the incident's `handoff_stream` (domain `0x6230883`,
version 1) selects the branch; the surface branch Weyl-measures $f$ on the same
generator (position, then momentum). Entered incidents use `fiber_stream` and
`phase_space_stream` unchanged. The entry draw and the fibre draws are
independent, so the composite law is the mixture of the two branches.

**Physical vs numerical.** Box-outgoing (including the surface complement) and
tail events are labelled `physical_outgoing` **only** under a declared matching
cut that meets the research rule. The target must satisfy
$E_t\ge U_{\max}+(U_{\max}-U_{\min})$, and every cut the unraveling selects
must lie within $|E_c-E_t|\le\delta$. The fibre engine evaluates the cut
through a wrapper that refuses otherwise. A tail is above the cut when its
kinetic floor exceeds it:

$$
\frac{(\hbar c\,k_{\min})^2}{2(E+mc^2)}+U_{\min}\ge E_t+\delta,
\qquad k_{\min}=\min_i (B_i+\tfrac12)g_i ,
$$

because any out-of-box $K$ has $|K_i|\ge(B_i+\tfrac12)g_i$ on some axis, and
$E_t+\delta$ is the highest cut the wrapper admits. This bounds the energy
expectation ($H\ge T+U_{\min}$); within the model a tail is in any case exactly
orthogonal to $P$, which lives in the window. $U_{\max}$ and $U_{\min}$ are
declared and are not checked against the spectrum.
Otherwise the call refuses. Out-of-box lattice and kick remainders, and string
outputs beyond $K_c$, are `unresolved` and never launched. Above-barrier
occupation, wrapping and numerical sinks are therefore never relabelled.

**Balance.** Each incident has exactly one fate, so the unit-weight fractions
$R(z)+F(z)+U(z)=1$ at every depth. The expectations are

$$
F(z)=(1-w)+\operatorname{tr}Q_F(z)+T(z),\qquad U(z)=U_O(z),
\qquad R(z)=\operatorname{tr}\rho(z),
$$

with $Q_F$, $U_O$ and $\rho$ as defined for the absorbing model at initial
density $cc^\dagger$ (`cr-axial-absorbing-jump-instrument`), and $T$ the tail
weight.

## Kinematic map

Let $L_i=2\pi/g_i$ and $N_i=2B_i+1$. The Weyl outcome $(a,b)$ on the box
grid, with the window peak at grid index 0, gives

$$
x_a=a\,L/N\in[0,L),\qquad m=\begin{cases}b,&b\le B\\ b-N,&b>B\end{cases},\qquad
\mathbf k_\perp=\kappa+m\,\mathbf g .
$$

The sign of $m$ follows from the layout: orders $n$ are placed at FFT index
$n\bmod N$ and the state is `ifft2(·, ortho)`. The forward FFT of the
windowed state therefore peaks at index $n\bmod N$. A tail record supplies
$\mathbf k_\perp=\kappa+\mathbf Q$ directly, at in-cell position
$\mathbf s+\mathbf u$ (struck string plus its sampled displacement).

**Lattice image.** The fibre state is cell-periodic in density, so the cell
index is not in the reduced state. For a beam much wider than a cell, the
incident's classical impact point $\mathbf r_s$ (crystal frame, entrance origin
on a lattice point) fixes the cell: $\mathbf X=\lfloor\mathbf r_s/L\rfloor L+x$.
Every window/box state at a folded carrier has $|\mathbf k_\perp|<k_b=
|(\mathbf B+1)\mathbf g|$. With $\sin\theta_b=k_b/p_1$ at the launch energy,
the group-velocity drift before the event is at most $z\tan\theta_b$, reported
as `lateral_bound_ang`. This is a semiclassical bound on the cell
prescription, since the reduced state carries no cell index. A box whose reach
exceeds $p_1$ is refused.

**Direction.** The plane-wave order is the kinetic transverse momentum. Native
transport has no continuum potential, so this is the momentum it continues
with. The neglected potential energy is $\lesssim U_{\max}-U_{\min}\ll E_c$ at a
rule-satisfying cut. With $p=\sqrt{E(E+2mc^2)}/\hbar c$ at the launch energy,

$$
\hat v_{\rm crystal}=\big(k_x/p,\ k_y/p,\ \sqrt{1-k_\perp^2/p^2}\big),\qquad
\hat v_{\rm lab}=R\,\hat v_{\rm crystal},
$$

and the call refuses if $k_\perp\ge p\sin\theta_{\max}$ (no clipping). $R$ must
be orthonormal and right-handed.

**Energy and clock (declared CSDA).** The zone propagates at the entrance
energy, so it admits only $Sz_{\max}/E_0\le\epsilon$ (default $10^{-3}$) and
refuses before any work otherwise. The launch energy is $E_1=E_0-Sz$. Straggling,
and any correlation between kick count and energy loss, are omitted. With
$\beta=pc/(E+mc^2)$ and $dE=-S\,dz$,

$$
\Delta t=\int_0^z\frac{dz'}{\beta}=\frac{p_0c-p_1c}{S}
=z\,\frac{E_0+E_1+2mc^2}{p_0c+p_1c},
$$

which uses $p_0^2-p_1^2=(E_0-E_1)(E_0+E_1+2mc^2)$, has no cancellation, and
gives $z/\beta_0$ at $S=0$. Then $t=t_{\rm surface}+\Delta t$ (Å, $c=1$). The
omitted transverse path delay is at most $z(\sec\theta_b-1)/\beta_1$, with
$\beta_1\le\beta$ along the zone. It is reported as `clock_bound_ang`.

**Lab position.** $\mathbf r_{\rm lab}=\mathbf r_0+R\,(X,Y,z)$.

## Limiting cases

- **Zero scattering.** With no channels and an entry in $P$, every incident is retained and nothing is launched.
- **Free particle.** With zero potential and identity eigenvectors, a single
  kick of $\mathbf q=\pm g\hat x$ and a uniform window (exact momentum) give
  $\mathbf k_\perp=\kappa_0\pm\mathbf q$ exactly: the classical kick added to
  the incident tilt. Depths follow $1-e^{-|s|^2z}$, the no-jump norm.
- **Unstopped clock.** $S=0$ gives $z/\beta_0$.
- **Determinism.** Given the fibre event, the launch is a function, so the
  joint law of (depth, carrier, order, window position) is the fibre model's
  and no marginal is resampled.

## Regression anchor

`tests/montecarlo/quantum/test_channeling_axial_outgoing_launch.py`:
balance and surface-complement Weyl law against the independent dense box
oracle; depth-split momentum law (chi-square) and exact event-to-launch
identity; frame/energy/clock against an independent `quad` of $1/\beta$; free
$\pm x$ kick kinematics and depth law; zero scattering; the tail launch at
$\mathbf s+\mathbf u$ with exact momentum; incident-order independence and
stream disjointness; refusals (cut outside the window, barrier-adjacent target,
small tail box, paraxial budget, energy-loss budget, duplicate ID, clock order,
stream key, left-handed frame). Four injected faults are caught: unsigned FFT
order, inverted entry branch, unstopped clock and missing lattice image.

## Fresh-context verification

**2026-11-01, fresh context (Claude Opus 5.5), verdict `rederived`.** Source
commit `b99ad1749`. Items 1–6 below were derived from the ledger row, this
page's statement and the upstream contracts before reading the bodies of
`axial_outgoing_launch.py` and `axial_zone_handoff.py`. The code was then
compared term by term. No implementation, test or ledger row was changed.

**Filters.** Units pass: eV, Å and Å$^{-1}$ throughout,
$\hbar c=1973.269804$ eV Å, $mc^2=510998.95$ eV, clock in Å at $c=1$.
Limits pass: $S=0$ gives $z/\beta_0$, zero channels retain every entry, a
uniform window gives exact momentum and a delta window exact position.
Signs and conventions pass: signed FFT order, a right-handed frame, and a
positive square root for the forward direction.

### Independent re-derivation

1. **Entrance split.** $\psi_0$ lies in the window and the box contains the window, so
   $B\psi_0=\psi_0$. $P=P(\kappa_0)$ is an orthogonal projector on the window,
   so $\lVert(B-P)\psi_0\rVert^2=\lVert\psi_0\rVert^2-\lVert P\psi_0\rVert^2=1-w$
   for unit $\psi_0$. The surface instrument $\{P,\,B-P\}$ followed by the
   fibre unraveling from $c/\sqrt w$, or by the Weyl POVM on $f/\sqrt{1-w}$, gives
   $R=wR_1$, $F=(1-w)+wF_1$ and $U=wU_1$, where $R_1+F_1+U_1=1$ are the fractions of the
   normalized fibre model. This is the page's expression at initial density
   $cc^\dagger$ ($\operatorname{tr}=w$) by linearity. Matches. The code uses
   one independent uniform draw against $w=\lVert V^\dagger\psi_0\rVert^2$. The
   fibre renormalizes $c$ itself, so the conditional fibre law is unchanged.
2. **FFT order and position.** With $c_n$ at index $n\bmod N$,
   $\psi_j=N^{-1/2}\sum_kc_ke^{2\pi ijk/N}=N^{-1/2}\sum_nc_ne^{in g x_j}$ with
   $x_j=jL/N$ and $L=2\pi/g$. Index $a$ is therefore the physical point $aL/N$ of the
   periodic factor, and a forward `fft2` returns $c_n$ at $b=n\bmod N$, so
   $m=b$ for $b\le B$ and $m=b-N$ for $b>B$. With the window peak at index 0, a
   window rolled by $a$ is centred at $x_a$. Matches `_Mapper.box_kinematics`,
   the entry layout (identical to the fibre's `_measure`) and
   `sample_phase_space` (`np.roll(g, position)`).
3. **Direction and frame.** For conserved transverse momentum
   $\hbar\mathbf k_\perp$ and $p(E_1)$,
   $\hat v=(k_x/p,\,k_y/p,\,\sqrt{1-k_\perp^2/p^2})$, and
   $\hat v_{\rm lab}=R\hat v$ with $R^{\mathsf T}R=I$ and $\det R=+1$. The
   code refuses when $k_\perp^2/p^2\ge\sin^2\theta_{\max}$. Matches.
4. **CSDA clock.** $dt=dz/\beta=-(E+mc^2)\,dE/(S\,pc)=-d(pc)/S$, so
   $\Delta t=(p_0c-p_1c)/S=z(E_0+E_1+2mc^2)/(p_0c+p_1c)$, using
   $p_0^2-p_1^2=(E_0-E_1)(E_0+E_1+2mc^2)$ and $E_0-E_1=Sz$. Matches
   `zone_energy_and_clock`.
5. **Tail floor.** `string_jump` folds the absolute $\mathbf K=\kappa+\mathbf Q$.
   Out of box means $\lvert n_i\rvert\ge B_i+1$ for the folded order on some axis,
   so $\lvert K_i\rvert\ge(B_i+\tfrac12)g_i$. This holds for any $\kappa$ because the fold is
   absolute. Then $\lvert\mathbf K\rvert\ge k_{\min}$, and the operator inequality
   $H\ge T+U_{\min}$ gives
   $\langle H\rangle\ge(\hbar ck_{\min})^2/(2(E+mc^2))+U_{\min}$. Matches
   `_check_tail_reach`. See L2 for the comparison value.
6. **Bounds.** Before the event, the history is retained in the window. With a folded carrier,
   $\lvert k_i\rvert\le(M_i+\tfrac12)g_i\le(B_i+\tfrac12)g_i$. The exact lateral
   drift rate is $\tan\theta$ with $\sin\theta=k/p_1$. The exact path delay rate
   is $(\sec\theta-1)/\beta$, not $\theta^2/2$. The stated quantities
   $z\theta_b$ and $z\theta_b^2/2$, with $\theta_b=\lvert(\mathbf B+1)\mathbf g\rvert/p_0$,
   are bounds only through the margin of $(B+1)$ over $(M+\tfrac12)$. See L1 and own check D.
7. **Honesty of the label.** The label is `physical_outgoing` only for `outgoing`/`tail` events
   under a validated `AxialMatchingDeclaration`. Every cut that
   `_Engine.fiber` selects goes through `_checked_cut`, including the entry
   carrier. Lattice and kick remainders outside the box and string outputs beyond
   `tail_q_max_inv_ang` stay `unresolved` and are never launched. Window
   states between $U_{\max}$ and $E_c$ (above-barrier occupation) stay
   retained. The fibre model has no clamped numerical sink. The page and
   docstrings keep the depth budget, return flux and matching convergence (#884) out
   of scope. The labelling is stated honestly, with L3 as its caveat.

### Anchor tests

`uv run pyrite-dev test tests/montecarlo/quantum/test_channeling_axial_outgoing_launch.py`:
17 passed (11.0 s).

### Own checks

Script: `scratch/v883_independent.py` (ignored). It uses a free-particle model
with $\mathbf g=(1.1,0.9)$ Å$^{-1}$, a $3\times3$ window, box $B=4$, carrier
$\kappa_0=(0.21,-0.13)$ Å$^{-1}$, a cut at 0.8 eV and a random complex unit
entry. The retained orders are $(0,0)$ and $(0,1)$, with $w=0.27822$.

- **A. Entry branch and momentum law** (uniform window, 20,000 incidents).
  The surface-outgoing fraction is 0.7186 against $1-w=0.7218$ ($z=-1.02$). All 14,371
  launches sit exactly on $\kappa_0+m\mathbf g$ for a non-retained order. The
  momentum law against $\lvert\psi_{0,m}\rvert^2$ gives $\chi^2=4.30$ for 6 dof
  ($p=0.64$). `handoff_balance` gives $R+F+U=1$ at depths 0 and 5 Å.
- **B. Position law and lattice image** (delta window, impact in cell
  $(3,-2)$, 20,000 incidents). The oracle is a direct sum
  $\lvert\sum_nf_ne^{i n\mathbf g\cdot\mathbf x_a}\rvert^2$ with no FFT. No launch
  lies off the $aL/N$ grid or outside the impact cell, and $\chi^2=76.6$ for 80 dof
  ($p=0.59$). The mirrored-coordinate control gives $\chi^2=7.8\times10^4$, so
  the position check can detect the sign convention.
- **C. Clock.** `zone_energy_and_clock` against `quad` of $1/\beta(E_0-Sz)$
  for $E_0\in\{0.2,1,5,50\}$ MeV and loss fractions $\{0,10^{-4},10^{-3}\}$ at
  $z=2\times10^4$ Å: largest relative error $3.8\times10^{-16}$.
- **D. Bound tightness.** The ratio exact/stated is computed at the worst
  window momentum $\lvert(\mathbf M+\tfrac12)\mathbf g\rvert$, energy $E_1=0.999E_0$. Lateral:
  0.90 at $B=M=4$, 0.98–0.99 at $B=M=24$ and 0.26 at $B=24$, $M=6$, so it always holds.
  Clock: 0.82 at 5 MeV with $B=M=4$, 0.97 at 5 MeV with $B=M=24$, **1.02** at 1 MeV with $B=M=24$, **1.17** at
  0.2 MeV with $B=M=4$ and **1.40** at 0.2 MeV with $B=M=24$, but 0.07–0.10 for $B=24$, $M=6$.
- **E. Entry normalization probe.** With `entry_orders` $=2\psi_0$ the call
  is accepted, `entered_weight` $=4w=1.11$ and no incident takes the surface branch
  (normalized law: 72%).

### Findings

- **M1 (medium, input validation).** `sample_physical_handoff` does not
  check $\lVert\psi_0\rVert=1$. The branch probability `entered_weight` is
  $\lVert P\psi_0\rVert^2$ unnormalized, while the fibre renormalizes, so a
  non-unit entry silently gives a wrong surface law (check E). The derivation
  holds for the declared unit plane wave. Suggested fix (task owner): refuse
  `entry_orders` whose norm differs from 1, or normalize it and document the change.
- **L1 (low, bound wording).** `clock_bound_ang` $=z\theta_b^2/2$ omits
  $1/\beta$ and the $\sec\theta-1-\theta^2/2$ excess. It holds only through the
  $(B+1)$ versus $(M+\tfrac12)$ margin, and fails for $B=M$ at low energy or
  large $B$ (check D: up to 1.40). Suggest $z(\sec\theta_b-1)/\beta_1$ with
  $\sin\theta_b=\lvert(\mathbf B+1)\mathbf g\rvert/p_1$, or calling it an
  estimate. `lateral_bound_ang` $=z\theta_b$ holds in every tested case (strictly the
  drift is $z\tan\theta$). Its bound is an Ehrenfest/group-velocity bound on a
  semiclassical prescription, since the reduced state carries no cell index.
- **L2 (low, comparison value).** `_check_tail_reach` compares the floor with
  `target_cut_eV`, but `_checked_cut` admits cuts up to
  `target_cut_eV + window_eV`. "Above every selected cut" needs
  floor $\ge E_t+\delta$. This has no effect at realistic box sizes, where the floor is far above.
- **L3 (low, declared inputs).** `potential_max_eV`/`potential_min_eV` are
  declared, not checked against the spectrum's potential. The initial
  `carrier_inv_ang` is not folded or refused outside $[-g/2,g/2)$, while $\theta_b$
  assumes a folded carrier. Kick and string carriers are folded.
- **I1 (info).** The tail floor is a bound on the expectation value ($H\ge T+U_{\min}$), not on the spectral
  support. Within the model a tail is exactly orthogonal to $P$ because $P$
  lives in the window. The tail position $\mathbf s+\mathbf u$ is a semiclassical
  localization, with error of order the string phase range, and no reported bound.

**Resolution (task owner, 2026-11-01).** M1: non-unit `entry_orders` now
refuse. L1: `lateral_bound_ang` $=z\tan\theta_b$ and `clock_bound_ang`
$=z(\sec\theta_b-1)/\beta_1$ with $\sin\theta_b=k_b/p_1$, and a box reach
$\ge p_1$ refuses. L2: the tail floor is compared with $E_t+\delta$. L3: a
carrier outside $[-g/2,g/2)$ refuses; the potential extrema stay declared, as
stated. I1: stated in the derivation. Regressions:
`test_non_unit_entry_and_unfolded_carrier_refuse` and
`test_reported_bounds_are_the_exact_box_reach_geometry` (21 anchor tests pass).

**Suggested ledger change (task owner applies; never `signed-off`).** Set the status to
`rederived`. Advance it to `anchored` (the anchor tests pass) once M1 is addressed or explicitly
accepted, and L1/L2 are fixed or restated in Claim/Checks. Add a Notes entry linking
this section.
