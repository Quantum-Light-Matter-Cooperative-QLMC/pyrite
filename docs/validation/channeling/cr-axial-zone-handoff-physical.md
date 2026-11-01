# `cr-axial-zone-handoff-physical`: physical axial outgoing instrument

**Status:** task-owner derivation (2026-11-01), awaiting fresh-context verification.
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
nothing falls outside the box. With $c=P\psi_0$ and $w=\lVert c\rVert^2$,

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
\frac{(\hbar c\,k_{\min})^2}{2(E+mc^2)}+U_{\min}\ge E_t,
\qquad k_{\min}=\min_i (B_i+\tfrac12)g_i ,
$$

because any out-of-box $K$ has $|K_i|\ge(B_i+\tfrac12)g_i$ on some axis.
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
The omitted transverse drift before the event is at most
$z\,\theta_b$, where $\theta_b=|(\mathbf B+1)\mathbf g|/p$ bounds the transverse
velocity of every window/box state. Each launch reports this bound as
`lateral_bound_ang`.

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
transverse path excess $z\langle\theta^2\rangle/2\le z\theta_b^2/2$ is
reported as `clock_bound_ang`.

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

Pending.
