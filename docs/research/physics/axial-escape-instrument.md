# Axial entrance-zone escape: operator accounting

The maintainer selected the entrance-zone architecture for
[#548](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/548)
on 2026-10-17. This note derives the accounting that an escape model must
satisfy and gives a numerical counterexample to identifying the current grid
sink with outgoing population. It does **not** select or validate a physical
matching threshold, rechanneling approximation, or native launch distribution.
The accounting claim `cr-axial-zone-handoff-accounting` is `rederived`; the physical handoff, `cr-axial-zone-handoff-physical`, remains `unverified`.

## Three subspaces, not two loss labels

Let $R$ project onto the numerically represented transverse space. Choose a
physical retained projector $P\le R$; it can contain bound states **and** slow
above-barrier states. Let $F=R-P$ be represented candidate outgoing states and
$O=I-R$ the unresolved space. These orthogonal projectors sum to the identity.
A Fourier grid edge defines $R$, not $P$. Neither a potential saddle nor a
change in bound population alone establishes irreversible escape.

Start with the Lindblad jump gain and loss terms in
{cite:t}`castin2008`, Eq. (2), with rates expressed per unit axial depth.
Write jumps as $J_a$ with units Å$^{-1/2}$, absorbing their quadrature weights
into the operators. The existing TDS label $a$ includes string, thermal
displacement and its probability measure. For $\rho=P\rho P$, define

```{math}
:label: eq-axial-escape-effects

A_a=PJ_aP,\qquad
\Gamma_F=\sum_a PJ_a^\dagger FJ_aP,\qquad
\Gamma_O=\sum_a PJ_a^\dagger OJ_aP,\qquad
\Gamma=\sum_a PJ_a^\dagger J_aP.
```

Inserting $P+F+O=I$ gives the positive operator identity

```{math}
:label: eq-axial-escape-closure

\sum_a A_a^\dagger A_a+\Gamma_F+\Gamma_O=\Gamma.
```

Each effect has units Å$^{-1}$ and is positive semidefinite. In particular,
$\Gamma_O$ is a projected **operator** product, not an arbitrary pointwise
difference of two diagonal rates.

For a fixed $P$ commuting with the transverse Hamiltonian $H_\perp$, an
absorbing, no-return model on the retained subspace is

```{math}
:label: eq-axial-retained-generator

\frac{d\rho}{dz}=
-\frac{i}{\hbar v}[PH_\perp P,\rho]
+\sum_a A_a\rho A_a^\dagger-\frac12\{\Gamma,\rho\},
\qquad
-\frac{d\operatorname{tr}\rho}{dz}
=\operatorname{tr}(\rho\Gamma_F)+\operatorname{tr}(\rho\Gamma_O).
```

The second equality follows by tracing the first and using
{eq}`eq-axial-escape-closure`; it is exact for this specified absorbing model.
This is not a proof that discarding return flux and retained/outgoing
coherences approximates the full crystal dynamics. A changing stopping energy
changes an energy-defined $P$; basis motion and additional projection flux
then require separate accounting. If $[H_\perp,P]\ne0$, coherent leakage is
also present and the stated closure is insufficient.

Integrating the traced equation gives retained weight plus cumulative $F$
weight plus unresolved $O$ weight equal to the initial retained weight.
Entrance populations already outside $P$ must be recorded separately, with a
declared preparation/matching map; silently renormalizing $P\rho_{in}P$ loses
incident weight. Retained above-barrier population is reported separately from
bound occupation. Native handoff counts must not be described as all physical
dechanneling events if slow unbound population remains quantum mechanical.

## A total loss effect does not determine an outgoing instrument

The represented outgoing gain operator per Å is

```{math}
:label: eq-axial-outgoing-gain

\sigma_F=\sum_a FJ_aP\rho PJ_a^\dagger F,
\qquad \operatorname{tr}\sigma_F=\operatorname{tr}(\rho\Gamma_F).
```

To sample a positive outgoing measure, specify a positive-operator-valued
measure $M(d\xi)$ on $F$, normalized to $F$. Then
$d\Phi(z,\xi)=\operatorname{tr}[M(d\xi)\sigma_F(z)]\,dz$ is nonnegative and
integrates to the outgoing weight. The label $\xi$ must resolve whatever
native launch data the matching model needs. This is an instrument choice,
not information recoverable from $\Gamma_F$ alone.

A minimal counterexample is a one-dimensional retained state $|b\rangle$ and
two orthogonal outgoing directions $|+\rangle$, $|-\rangle$.
$J_+=\sqrt{\gamma}|+\rangle\langle b|$ and
$J_-=\sqrt{\gamma}|-\rangle\langle b|$ have the same loss effect
$\gamma|b\rangle\langle b|$, yet launch in opposite directions. Here each
operator defines a different alternative model, not two simultaneous jumps.
No diagonal-loss accessor can distinguish them. The same problem applies to
the current unresolved TDS sink.

A momentum marginal can be positive without supplying a joint sharp position
and momentum law. General coherent states cannot be sampled by multiplying
their independent position and momentum marginals or treating a possibly
negative Wigner function as a probability density. A localized phase-space
measurement or a justified semiclassical matching construction must declare
its resolution and preserve relevant position/direction correlations. The
two transverse momentum components must remain joint throughout sampling.

## Why the current TDS sink is not a Fourier escape projector

The diagnostic TDS jump multiplies the wavefunction by a phase fluctuation.
Band-limiting that phase and applying the product on a periodic grid is not
the same operation as applying the full jump and projecting its **outgoing
state** back into the represented space. The latter depends on incoming
momentum as well as momentum transfer. Grid products can wrap unresolved
outgoing modes back into the grid.

For a normalized incident plane wave of integer order $\mathbf p$ in a
supercell of area $A$, write the phase fluctuation coefficients for a source
at $\mathbf r_s$ as
$s_{s,\mathbf n}e^{-i\mathbf q_{\mathbf n}\cdot\mathbf r_s}
(e^{-i\mathbf q_{\mathbf n}\cdot\mathbf u}-e^{-a_s^2|\mathbf q_{\mathbf n}|^2/2})$.
Here $\mathbf q_{\mathbf n}$ is the Cartesian wave vector corresponding to
integer transfer order $\mathbf n$, $s$ includes the $1/A$ normalization,
$a_s^2$ is the isotropic
displacement variance in Å² and $\lambda_s$ is the string density in Å$^{-1}$.
The Gaussian characteristic function gives the transfer rate

```{math}
:label: eq-axial-plane-wave-transfer

w_{s,\mathbf n}=\lambda_s|s_{s,\mathbf n}|^2
\left(1-e^{-a_s^2|\mathbf q_{\mathbf n}|^2}\right),\qquad
r_O(\mathbf p)=\sum_{s,\mathbf n}
w_{s,\mathbf n}\,\mathbf1_{\{\mathbf p+\mathbf n\notin\mathcal R\}}.
```

Here $\mathcal R$ is the set of represented integer momentum orders; the
indicator adds only dimensionless integer orders. All transfers are explicitly
cutoff-limited; this is not the full
infinite-transfer escape rate. Distinct outgoing plane waves are orthogonal,
so rates add. General incident superpositions have interfering paths to the
same outgoing momentum and cannot use this classical convolution.

Two incident plane waves have identical uniform spatial density, so the
current expectation of `tds_sink_rate` is identical for both. Their $r_O$
can differ because the indicator depends on $\mathbf p$. This is a direct
test of whether a position-diagonal sink can stand in for an outgoing-state
projector, without choosing any physical dechanneling threshold.

`checks/axial_escape_accounting.py` evaluates this example for Si [001],
5 MeV, 293.15 K, one periodic cell, a 16² state window and 32² phase-transfer
grid. It compares origin/edge/corner incident orders without modulo wrapping,
and checks that the finite transfer-rate sum equals the mean finite-band TDS
variance. These tiny grids demonstrate the distinction only; they are not
converged material predictions. No transport or stochastic histories are run.

The bounded check gives the following rates (Å$^{-1}$):

| Incident order | Current diagonal sink | Escape from the 16² state window, 32² transfer cutoff |
| --- | ---: | ---: |
| (0, 0) | 9.808715e-6 | 5.131780e-6 |
| (7, 0) | 9.808715e-6 | 2.787127e-5 |
| (0, 7) | 9.808715e-6 | 2.787127e-5 |
| (7, 7) | 9.808715e-6 | 4.153174e-5 |

The common finite-transfer total is 6.078322e-5 Å$^{-1}$; its difference
from the mean finite-band TDS variance is 5.42e-20 Å$^{-1}$. The present
diagonal sink uses its existing continuum/full-diagonal construction, whereas
the comparison uses a finite periodic phase polynomial. Their totals need not
coincide. The relevant counterexample is the incident-momentum dependence:
no fixed position-diagonal effect reproduces both escape rates of these
equal-density inputs. The six fast analytic tests cover corner escape counted
once, no modulo wrapping, an enclosing window, zero jumps and invalid rates.
This is implementation-context evidence, not a fresh-context verdict or a
validated physical escape law.

## Coherent finite-transfer gain

The plane-wave convolution above extends to a density operator only if the
off-diagonal transfer covariance is retained. For one string, absorb its
position phase into $c_n=s_n e^{-i q_n\cdot r_s}$ and let $T_n|p\rangle
=|p+n\rangle$ on the **unwrapped** integer-order lattice. The finite jump is
$J(u)=\sqrt{\lambda}\sum_n c_n(e^{-iq_n\cdot u}-\chi_n)T_n$, where
$\chi_n=e^{-a^2|q_n|^2/2}$. Gaussian averaging gives

```{math}
:label: eq-axial-coherent-transfer-covariance

K_{nm}=\lambda c_n c_m^*
\left[e^{-a^2|q_n-q_m|^2/2}
-e^{-a^2(|q_n|^2+|q_m|^2)/2}\right],\qquad
\mathcal G(\rho)=\sum_{n,m}K_{nm}T_n\rho T_m^\dagger.
```

$K$ is positive semidefinite because it is the covariance of the random
coefficient vector. Factoring $K=BB^\dagger$ therefore yields finite Kraus
jumps $L_j=\sum_n B_{nj}T_n$ and $\mathcal G(\rho)=\sum_j L_j\rho
L_j^\dagger\ge0$. No entrywise clipping of $K$ is part of this derivation.
The loss effect of this same finite model is
$\Gamma=\sum_{n,m}K_{nm}T_m^\dagger T_n$; its expectation is
$\operatorname{tr}\mathcal G(\rho)$. It need not equal the production
dissipator's continuum diagonal plus numerical sink.

For a finite incident support, enumerate the union of literal sums $p+n$.
This union contains the entire gain of the declared finite transfer model;
no FFT periodic reduction is applied. A represented output projector $R$
gives positive gains $R\mathcal G(\rho)R$ and
$(I-R)\mathcal G(\rho)(I-R)$. Their traces sum to the full trace even
though cross-block coherences can exist. This is numerical window accounting,
not the selection of the physical $P$ in the first section. The unresolved
weight here excludes transfers omitted from the finite polynomial; it is
not a bound on the infinite-transfer tail.

`coherent_tds_gain` in `checks/axial_escape_accounting.py` evaluates this map
for a supplied incident density and one independent string. Sum maps for
independent strings; do not add their amplitudes coherently. Coefficients
must include source registration and cell-area normalization; Cartesian $q$
and integer orders must refer to the same reciprocal basis. The implementation
keeps the incident trace rather than normalizing it. $K$, the gain and its
trace have units Å$^{-1}$; zero displacement or zero string density gives
zero gain. A single incident plane wave recovers the earlier transfer rates.
Both Cartesian momentum components remain joint in the outgoing labels.
The helper allocates a dense outgoing density and a four-index contraction;
it is for bounded research probes, not production grid sizes.

For a concrete interference test choose incident and transfer orders
$(-1,0),(1,0)$, $q_n=n$ Å$^{-1}$, $c_n=1$, $a^2=0.5$ Å² and
$\lambda=1$ Å$^{-1}$. The equal mixtures of the incident modes and their
in-phase/antiphase superpositions have the same incident momentum
probabilities. Their gain at outgoing order $(0,0)$ is respectively
$v$, $v+c$, $v-c$, where $v=1-e^{-0.5}$ and
$c=e^{-1}-e^{-0.5}<0$, in Å$^{-1}$. A diagonal transfer-rate convolution
misses this interference. This synthetic example tests the operator algebra;
it is not a crystal prediction.

Focused tests compare the map with direct two-dimensional Gaussian
displacement quadrature for a complex mixed incident state, check positivity
and represented/unresolved trace closure, and exercise this interference
example, the plane-wave reduction, and zero-rate limits. The
[fresh-context verification](../../validation/channeling/cr-axial-zone-handoff-accounting.md)
rederives this finite-transfer algebra and compares an independent Gaussian
quadrature with the implementation. It does not certify physical matching:
`cr-axial-zone-handoff-physical` remains `unverified`.

## Cutoff-free continuum window split

The finite polynomial above omits every transfer outside its support, and its
supercell phase repeats each displacement in all periodic images. Both are
removed by treating each atom as independently displaced. Let incident plane
waves $e^{i\mathbf p_i\cdot\mathbf r}/\sqrt A$ be normalized on the supercell
area $A$ with positive matrix $\rho_{ij}$, and let
$t(q)=-v(q)/\hbar v$ be the real, even transform of one atom's phase (Å²),
with $v$ the projected Mott–Bethe potential of `cr-axial-realspace-dissipator`.
For a string at $\mathbf R$, the Fourier amplitude of the jump acting on the
state is
$\sqrt{\lambda/A}\sum_i c_i t(|\mathbf q_i|)e^{-i\mathbf q_i\cdot\mathbf R}
(e^{-i\mathbf q_i\cdot\mathbf u}-\chi_i)$ with $\mathbf q_i=\mathbf k-\mathbf p_i$
and $\chi_i=e^{-a^2q_i^2/2}$. Gaussian averaging gives the outgoing momentum
density

```{math}
:label: eq-axial-continuum-gain-density

g(\mathbf k)=\frac{\lambda}{A}\sum_{ij}\rho_{ij}
e^{i(\mathbf p_i-\mathbf p_j)\cdot\mathbf R}\,t_it_j
\left[e^{-a^2|\mathbf p_i-\mathbf p_j|^2/2}-\chi_i\chi_j\right]\ge0,
```

per $d^2k/(2\pi)^2$. It is an average of squared moduli, so it is pointwise
nonnegative for any coherent input. Images in other supercells give the same
density because $\mathbf p_i-\mathbf p_j$ is a supercell reciprocal vector.
Substituting $\mathbf q=\mathbf k-(\mathbf p_i+\mathbf p_j)/2$ turns the total
integral into

```{math}
:label: eq-axial-continuum-gain-total

\int\frac{d^2k}{(2\pi)^2}g=\frac{\lambda}{A(2\pi)^2}\sum_{ij}\rho_{ij}
e^{i\mathbf G_{ij}\cdot\mathbf R}D(|\mathbf G_{ij}|),\qquad
\mathbf G_{ij}=\mathbf p_i-\mathbf p_j,
```

where $D(G)=\int d^2q\,t(|\mathbf q-\mathbf G/2|)t(|\mathbf q+\mathbf G/2|)
[e^{-a^2G^2/2}-e^{-a^2(q^2+G^2/4)}]$ is `_tds_transform` divided by
$(\hbar v)^2$. The right side is $\operatorname{tr}(\rho\,C)$ for the
production full diagonal $C(\mathbf r,\mathbf r)$, whose harmonic at
$\mathbf G$ is $\lambda A^{-1}e^{-i\mathbf G\cdot\mathbf R}D(G)/(2\pi)^2$.
The gain of the full jump therefore closes exactly on the production no-jump
loss effect, with no transfer cutoff. The weight inside a declared momentum box
$\mathcal B$ is the integral of {eq}`eq-axial-continuum-gain-density` over
$\mathcal B$, and the unresolved weight is the analytic total minus that
integral. For the grid window of orders $[-n/2,n/2)$ the box edges are
$(j-\tfrac12)\Delta$ with $\Delta=2\pi/L$. This box is a numerical convention
for a non-periodic outgoing packet; its $\pm\Delta/2$ edge ambiguity vanishes
as the supercell grows.

The supercell-periodic phase of the finite polynomial evaluates the same
integrand at lattice transfers $\mathbf q_n$ with weight $(2\pi)^2/A$: it is the
midpoint rule of the continuum box integral, which is why its total earlier
differed from the continuum diagonal.

`continuum_tds_window_split` in `checks/axial_escape_accounting.py` evaluates
the box integral by tensor Gauss–Legendre panels and the total from $D$.
Its tests use a Gaussian $t(q)=g e^{-bq^2}$, for which
$D(G)=g^2e^{-bG^2/2}[e^{-a^2G^2/2}\pi/(2b)-e^{-a^2G^2/4}\pi/(2b+a^2)]$.
A large box closes on this total to $10^{-12}$ relative for a complex
three-wave mixed state with an off-origin string. An independent
Gauss–Hermite displacement average of the squared jump amplitude reproduces
the box weight. Plane waves are translation covariant, and zero displacement or
zero string density gives zero weight.

The bounded Si [001], 5 MeV, 293.15 K example (one 1.92 Å cell, single string)
gives a total of 6.428955e-5 Å$^{-1}$ for every plane wave. The unresolved
weight (Å$^{-1}$) is:

| Window | (0, 0) | (7, 0) | (7, 7) |
| ---: | ---: | ---: | ---: |
| 16² | 7.200e-6 | 3.000e-5 | 4.367e-5 |
| 32² | 1.927e-6 | 2.929e-6 | 3.861e-6 |
| 64² | 4.924e-7 | 5.421e-7 | 5.909e-7 |
| 128² | 1.232e-7 | 1.262e-7 | 1.291e-7 |

The origin-wave ratios per doubling are 3.74, 3.91 and 4.00, consistent with
the $Q^{-2}$ tail of unscreened single nuclear scattering, $t(q)\propto q^{-2}$.
This large-angle weight leaves any finite window and is physically outgoing,
but it says nothing about where slow states should be matched. Gauss–Legendre
orders 4, 6 and 10 change the (7, 7) 16² represented weight by 4.0e-9 and
3.8e-11 Å$^{-1}$. For equal superpositions of orders (0, 0) and (1, 0), the
mixture, in-phase and antiphase totals are 6.429e-5, 1.204e-4 and
8.161e-6 Å$^{-1}$, because the coherent state concentrates on or away from
the string. The production grid samples of $C(\mathbf r,\mathbf r)$ give
6.465e-5, 1.210e-4 and 8.297e-6 Å$^{-1}$: 0.6%, 0.5% and 1.7% higher. The
excess comes from the zero clamp on the production sink, active on 14.8% of
the 16² grid points of this 1×1-cell example, not from the closure; at 32², where the clamp is
inactive for this example (it stays active on the 2×2-cell MCWF grid), the mixture's grid mean matches the total to 5e-10. Supercell lattice sums over
orders $|n_{x,y}|\le15c$ approach the matching continuum box integral as
6.078e-5 versus 6.225e-5, 6.2288e-5 versus 6.2309e-5, and 6.22787e-5 versus
6.22785e-5 Å$^{-1}$ for $c=1,2,4$. The $c=1$ lattice sum reproduces the
finite-transfer total of the earlier table.

This removes the omitted-transfer caveat from the numerical window split and
ties the outgoing gain to the production TDS loss effect. It does not choose
the physical retained projector $P$, bound return flux, supply an outgoing
phase-space instrument, or treat electronic jumps; the box remains numerical
accounting. A [fresh-context review](../../validation/channeling/cr-axial-zone-handoff-accounting.md)
rederives this split; `cr-axial-zone-handoff-physical` remains `unverified`.

## Energy-defined retained projector and return flux

Take $P(E_m)=\sum_{E_\alpha\le E_m}|\alpha\rangle\langle\alpha|$, the spectral
projector of the represented transverse Hamiltonian. It commutes with
$H_\perp$, so {eq}`eq-axial-retained-generator` and its trace balance apply
without coherent leakage. Choosing $E_m$ inside a spectral gap keeps every
degenerate multiplet whole. Stopping loss is not modelled; at fixed energy
the projector does not move.

In the eigenbasis, the TDS jumps of one supercell-periodic string give the
rates $W_{\beta\alpha}=\lambda\,\mathbb E_u|\langle\beta|J_u|\alpha\rangle|^2$
for represented $\beta$. The loss of state $\alpha$ is
$\Gamma_\alpha=\langle\alpha|\Gamma|\alpha\rangle$, and
$\Gamma_\alpha-\sum_\beta W_{\beta\alpha}$ leaves the represented space.
Keeping only populations gives the secular (Pauli) rate equation
(Breuer and Petruccione, *The Theory of Open Quantum Systems*, Sec. 3.3.1):

```{math}
:label: eq-axial-retained-pauli

\frac{dn_\alpha}{dz}=\sum_{\beta\in P}W_{\alpha\beta}n_\beta-\Gamma_\alpha n_\alpha,
\qquad \alpha\in P.
```

Every jump that leaves $P$ is absorbed, either into represented $F$ states
or out of the represented space. The retained population plus the two
cumulative absorbed weights equals the initial retained weight. The
reference is $P=R$, which drops only the unresolved weight. The
below-barrier population $N_B(z)=\sum_{E_\alpha<U_{\max}}n_\alpha$ then
measures what the absorbing boundary discards as return flux. The secular
limit needs level spacings $\Delta E/\hbar v$ much larger than the rates.
It drops coherences, and within degenerate multiplets its rates depend on the
basis chosen inside the multiplet. In the study below, 64 of the 10404
nondegenerate pairs under 175 eV have $\Delta E/\hbar v$ below their mean
width $(\Gamma_\alpha+\Gamma_\beta)/2$. All lie above $U_{\max}$, and none
straddles a matching cut. The 8.71/8.74 eV pair is one of them, with a spacing of
$1.7\times10^{-5}$ Å$^{-1}$ against widths near $10^{-4}$ Å$^{-1}$.

`checks/axial_matching_study.py` builds $W$ from a Kraus factorization of
the finite transfer covariance. That factorization is exact for transfers
between represented orders, up to a relative eigenvalue truncation of
$10^{-14}$. It builds $\Gamma_\alpha$ from variance harmonics summed by FFT
over $|n|\le300$ orders. Focused tests check three things: $W$ matches the
coherent gain {eq}`eq-axial-coherent-transfer-covariance` projected on each
basis vector to $10^{-12}$; $\Gamma_\alpha$ matches a direct double lattice
sum; and the two-state solution and flux balance close exactly.

**Bounded study.** Si [001], 5 MeV, Γ point of one cell, TDS only:
$U_{\min}=-77.2$ eV, cell maximum $U_{\max}=8.19$ eV (at the cell centre) and
string-to-string saddle $U_s=4.97$ eV. Six states lie below $U_{\max}$; two of
them (5.75 and 6.97 eV) are above $U_s$. "Bound" below means $E<U_{\max}$. A
plane wave at zero tilt has 27% of its entrance weight above $U_{\max}$ and
$4\times10^{-6}$ above 49.6 eV. The table gives $N_B$ for $\pm10$ orders (441 states),
with the relative deficit from the $P=R$ reference in parentheses. Matching
energies sit in gaps nearest $U_{\max}$ and $U_{\max}+\{0.5,1,2\}(U_{\max}-U_{\min})$.

| $E_m$ (eV) | $z=10^3$ Å | $10^4$ Å | $3\times10^4$ Å | $10^5$ Å |
| ---: | ---: | ---: | ---: | ---: |
| 7.8 | 0.5882 (0.6%) | 0.1409 (11%) | 0.0083 (65%) | 1.1e-4 (98%) |
| 49.6 | 0.5915 (0.06%) | 0.1518 (4.3%) | 0.0144 (39%) | 6.7e-4 (85%) |
| 94.9 | 0.5918 (0.02%) | 0.1554 (2.0%) | 0.0180 (23%) | 1.3e-3 (70%) |
| 175.3 | 0.5918 (0.01%) | 0.1572 (0.9%) | 0.0205 (12%) | 2.3e-3 (50%) |
| $P=R$ | 0.5919 | 0.1586 | 0.0234 | 4.5e-3 |

For $E_m=7.8$ eV the deficit also contains the 27% entrance weight that is
handed off at $z=0$. Most of it would not have reached bound states, so the
deficit at $10^3$ Å is only 0.6%.

The $P=R$ reference itself depends on the grid. For $\pm6$, $\pm8$ and
$\pm10$ orders, $N_B$ is 0.1622, 0.1582 and 0.1586 at $10^4$ Å; 0.0203,
0.0222 and 0.0234 at $3\times10^4$ Å; and 0.0022, 0.0034 and 0.0045 at
$10^5$ Å. Weight leaving the represented space is not physically lost either:
part of it would return. Changing the Kraus truncation from $10^{-14}$ to
$10^{-12}$ changes $W$ by less than $10^{-13}$ relative. Moving the variance tail
from 300 to 600 orders changes $\Gamma_\alpha$ by $1.1\times10^{-4}$, with the
expected $Q^{-2}$ ratio.

A channeling-radiation yield weights each state's emission rate by its
depth-integrated population. The exposure
$X_B(z)=\int_0^z N_B\,dz'$ (Å) is the state-blind proxy for that weight.
At $\pm10$ orders the reference gives $X_B$ = 654, 3544, 4791, 5469 and 5794 Å
at $z=10^3$, $10^4$, $3\times10^4$, $10^5$ and $3\times10^5$ Å. The relative
deficit of the absorbing model is:

| $E_m$ (eV) | $10^4$ Å | $3\times10^4$ Å | $10^5$ Å | $3\times10^5$ Å |
| ---: | ---: | ---: | ---: | ---: |
| 7.8 | 3.2% | 9.7% | 19% | 24% |
| 49.6 | 1.0% | 4.4% | 12% | 16% |
| 94.9 | 0.4% | 2.3% | 7.7% | 12% |
| 175.3 | 0.2% | 1.1% | 4.5% | 7.9% |

At $3\times10^5$ Å the reference exposure for $\pm6$, $\pm8$ and $\pm10$ orders is
5416, 5577 and 5794 Å, still rising, so the long-depth deficits are lower
bounds. Below $10^4$ Å the exposure is set by the entrance population and the
handoff costs under 1% for $E_m\gtrsim50$ eV.

**Bloch-$\mathbf k$ sampling.** The transverse Hamiltonian is block diagonal
in the Bloch momentum $\boldsymbol\kappa$, but TDS jumps move it:
a jump from $\boldsymbol\kappa$ to $\boldsymbol\kappa'$ transfers
$\mathbf q=\boldsymbol\kappa'-\boldsymbol\kappa+\mathbf n\cdot\mathbf g$.
Sampling $\boldsymbol\kappa=(i,j)\mathbf g/c$ is the $c\times c$ supercell with
one independent string per image. Each image contributes the same squared
amplitude, so the rates are the one-cell expression with area $cA$, and the
loss harmonics sum on the $\mathbf g/c$ lattice. `bloch_tds_rates`
implements this, and a focused test checks it against an explicitly embedded
$2\times2$ supercell with four image strings. `bloch_report` folds
$\boldsymbol\kappa$ into $(-\mathbf g/2,\mathbf g/2]$, so time-reversal partners
share a symmetric truncated window. For the plane-wave entrance at
$\boldsymbol\kappa=0$, the exposure deficit is:

| Orders, $c$ | $E_m$ (eV) | $10^4$ Å | $3\times10^4$ Å | $10^5$ Å | $3\times10^5$ Å |
| --- | ---: | ---: | ---: | ---: | ---: |
| ±6, 1 | 95.0 | 0.1% | 0.8% | 2.5% | 3.6% |
| ±6, 2 | 95.0 | 0.2% | 1.0% | 3.1% | 4.3% |
| ±6, 3 | 93.4 | 0.2% | 1.0% | 3.4% | 4.6% |
| ±8, 1 | 94.9 | 0.3% | 1.8% | 5.8% | 8.4% |
| ±8, 2 | 94.9 | 0.4% | 2.1% | 6.8% | 9.6% |

Going from $c=1$ to $c=2$ raises the deficit by roughly a fifth, and $c=3$
adds about 0.3 points at $3\times10^5$ Å. The reference exposure at
$3\times10^5$ Å for ±6 orders rises 5416, 5684 and 5772 Å for $c=1,2,3$.
The steps shrink (268 Å, then 88 Å), which suggests convergence near
$c\approx3$. That is shown only by this trend, and only at ±6 orders. The
$c=3$ row's lower $E_m$ also inflates it by a few percent. The plane-wave grid
remains the dominant unconverged dimension.

**Provisional TDS-only depth budget.** For a 1% exposure-deficit tolerance,
take the ±10, $c=1$ values and scale them by the Bloch factor. That factor
is 1.2–1.3 at $c=2$ and 1.3–1.4 at $c=3$ near 95 eV, and 1.0–1.1 near
175 eV. The result is about 1.4–1.6 µm at $E_m\approx95$ eV and about
2.4–2.6 µm at $E_m\approx175$ eV. The deficits have so far risen
monotonically with grid size and $c$; the ±8 to ±10 step at 3 µm is still
+27%. If that trend continues, these budgets are upper bounds, but that is a
trend, not a proof. Electronic jumps can move them in either direction.

**What this establishes.** An absorbing boundary at the barrier, just below
$U_{\max}$, is not a physical retained projector. Over $10^4$ Å it loses 11% of
the bound population to omitted return flux. At $\pm10$ orders,
$E_m\gtrsim U_{\max}+(U_{\max}-U_{\min})$ keeps that loss near 2% to $10^4$ Å.
That figure is not grid-converged. At the 94.9 eV cut and $10^4$ Å the deficit
for $\pm6$, $\pm8$ and $\pm10$ orders is 0.63%, 1.57% and 1.97%; at 175.3 eV
it is 0.07%, 0.54% and 0.86%. Both are still rising, so treat them as lower
bounds. Beyond a few $10^4$ Å the below-barrier tail is set by
rechanneling, and no tested $E_m$ or grid converges it: the deficit grows
with depth for every matching energy. A fixed-energy handoff to native
transport is therefore defensible only up to a depth budget where the
return-flux deficit stays below a declared tolerance. A long-depth handoff
needs either a quantum zone extending through the depth of interest or a
native model of rechanneling. Choosing between those changes scope and is a
maintainer decision. The numbers depend on the secular approximation,
TDS-only scattering, the Γ point of one cell, and a zero-tilt plane-wave
entrance. Electronic jumps, Bloch-$\mathbf k$ and supercell convergence, and a
full coherent comparison remain open. These are research diagnostics, not
dechanneling predictions.

## MCWF return-flux comparison with electronic jumps

The secular model cannot carry electronic jumps without Bloch-$\mathbf k$
resolution finer than its dense rate tables allow, so
`checks/axial_mcwf_return_flux.py` measures the same deficit with the
production MCWF unravelling (`cr-axial-realspace-dissipator`), TDS and
electronic jumps together. The reference ensemble runs unchanged. Each
absorbing ensemble measures the grid spectral projector $P(E_m)$ at the
electron's carrier tilt at entry and after every jump, through the
`AxialMCWF.run(jump_hook=...)` hook: outcome $P$ continues with the projected
state, outcome $F$ is handed off. The no-jump loss operator does not commute
with $P$, so this differs from the absorbing generator by the coherent
$P\leftrightarrow F$ leakage of the no-jump evolution, of relative size
$\epsilon=|W_{fp}|/\Delta E$ across the cut, and it enters jump outcomes at
first order through cross terms. At the $\Gamma$-point gaps of the
$2\times2$, $64^2$ grid $\max\epsilon\approx0.016$. Electronic kicks give each
electron a generic tilt, at which a fixed cut need not sit in a gap (nearest
levels 0.01 to 1 eV away against $|W|$ up to 2.2 eV). The script therefore
places the cut at each tilt in the widest level gap within
`--gap-window-eV` (default 2 eV) of the nominal matching energy, and
reports the smallest and median gaps used; $\epsilon$ is bounded only where
those gaps are wide against $W$. Entry is projected
deterministically and the ensemble scaled by the entrance weight $p_0$,
which removes entry sampling noise. The bound observable is the weight below
$U_{\max}$; the cuts are the spectral gaps nearest $U_{\max}$,
$U_{\max}+(U_{\max}-U_{\min})$ and $U_{\max}+2(U_{\max}-U_{\min})$. Chunks
share seeds across models and measurement streams are disjoint from the jump
streams, so deficits are paired and the SEM is taken across chunks. The
deficit is the ratio of pooled means per depth, $1-\langle X\rangle/\langle
X_{\rm ref}\rangle$, with a delete-one-chunk jackknife SEM. The default
(Si [001], 5 MeV, $2\times2$ cells, $32^2$, 3 µm, 32×128 electrons) is a
lab-box job; `--smoke` is a seconds-long TDS-only plumbing check. A
`--mechanisms tds` run is comparable with the secular table above only at
matched plane-wave orders and Bloch sampling; the default ($\pm8$ orders per
cell, $c=2$) matches no row there.

**Electronic-jump pilot (2026-10-21; research only).** The first lab-box run
of the 2 × 2 cell, 16-point-per-cell grid stopped because the gap selector
required a *gap midpoint* within 2 eV of the nominal cut. A cut may lie in a
wide level-free interval even when its midpoint is farther away. The
diagnostic now selects the largest symmetric level-free interval centred
within the allowed cut window and reports its width; outside the represented
spectrum the retained projector is trivially empty or full. This repairs the
measurement diagnostic, not the physical matching model.

A pilot with only the high candidate cut used `--cut-eV 178 --workers 4
--chunks 4 --electrons-per-chunk 16 --depth-max-ang 10000
--depth-step-ang 1000 --step-ang 10` (SLURM job 1640, both TDS and electronic
jumps, seed chunks 0–3). The zero-tilt gap cut was 179.532 eV. The reference
64-electron ensemble counted 46 TDS jumps, 207 electronic jumps and 24
numerical sinks; the measured model absorbed 12 electrons. At 10,000 Å the
pooled bound-exposure deficit was $-0.0060\pm0.0151$ (delete-one-chunk
jackknife standard error). The minimum reported centred level-free width was
0.593 eV, only 0.296 eV clearance on either side; the median chunk median
width was 1.413 eV. The earlier non-Hermitian cross-cut estimate reaches
2.2 eV after some electronic kicks, so a small coherent-leakage error is not
established by this pilot. The uncertainty also exceeds the provisional 1%
exposure criterion. There is no validated depth budget or admitted case from
these numbers. Reproduction requires cut/clearance distributions, coherent
projected evolution, more histories, and independent grid and supercell
convergence before a physical matching threshold can be selected.

## Observing the cross-cut no-jump loss

`checks/axial_mcwf_return_flux.py --loss-coupling` optionally records the
absorptive operator at each **entry or post-jump projection event**. This is
research instrumentation under `cr-axial-zone-handoff-accounting`; the physical
claim `cr-axial-zone-handoff-physical` remains `unverified`. The MCWF no-jump convention follows
[Castin, Dalibard and Mølmer, Eq. (4)](https://arxiv.org/abs/0805.4002):

$$H_{\rm eff}=H-iW,\qquad W=\frac{\hbar v_z}{2}\sum_m C_m^\dagger C_m.$$

Here the jump rates are per propagation length, so $W$ is the production
`dissipator.absorptive_eV`, in eV. For the selected spectral cut $E_c$ at
carrier tilt $\mathbf q$, define $P=1_{H(\mathbf q)<E_c}$ and $F=1-P$.
The diagnostic measures

$$c_W=\|FWP\|_2,\qquad
\Delta=\min\operatorname{spec}(H|_F)-\max\operatorname{spec}(H|_P),\qquad
\eta=c_W/\Delta.$$

Both $c_W$ and $\Delta$ have eV units; $\eta$ is dimensionless. $\Delta$ is
the actual separation between the represented sectors, **not** twice the
clearance from the selected cut to the nearest level. For a cell-periodic
$W$, the Fourier matrix is a direct sum of Bloch blocks. The full cross norm
is the maximum block singular value; the sector separation uses the entire
represented spectrum, including levels in different blocks. Nonperiodic loss
arrays are refused rather than silently discarding their inter-block terms.
Constant $W$ gives zero cross norm up to roundoff. Empty/full sectors have
zero cross norm and no defined separation or ratio (JSON `null`).

For a state initially in $P$, the exact finite-grid no-jump generator gives
$F\partial_z\phi=-FWP\phi/(\hbar v_z)$ at that instant because $FHP=0$.
Thus $c_W/(\hbar v_z)$ measures the largest instantaneous outgoing-amplitude
derivative per Å for a unit retained vector. Neither this derivative nor
$\eta$ is a finite-depth probability, return-flux or photon-source error
bound. Repeated jumps, long-time evolution, discretization error and
unrepresented transfers still require separate treatment.

The JSON `matching_diagnostics_by_chunk` retains fixed-size histograms and
extrema for selected cut, nearest-level clearance, $c_W$, $\Delta$ and
$\eta$. Each hook call contributes once, including deterministic entrance
projection; the distribution is weighted by observed hook events, not depth,
incident weight or unique tilt. Bin counts cover values below the first edge,
half-open interior intervals, and values at/above the last edge; undefined
quantities increment `missing`. These samples do not bound unvisited tilts or
tilts of already removed electrons. Eigenvectors and loss diagnostics use
bounded LRU caches; enabling observation consumes no random draws and changes
no propagation or acceptance rule. Dense singular-value calculations add
cost, so use the flag explicitly in lab-box studies.

Independent dense real-space operator regressions, constant-loss/trivial-sector
limits and exact enabled/disabled RNG and worker-result comparisons are in
`tests/montecarlo/test_channeling_axial_mcwf_return_flux.py`. They check the
represented-grid algebra, not physical eligibility; fresh-context review and
the coherent projected-evolution comparison remain required.

**Bounded lab-box pilot (SLURM 1641, 2026-10-21).** The
[raw result and code identity](../../validation/check-records/axial-zone/loss-coupling-pilot-1641.json)
record Si [001], 5 MeV, $2\times2$ cells, $32^2$ points, both TDS and
electronic jumps, 10,000 Å depth and 10 Å steps, with two seed chunks of eight
incident electrons. The nominal 178 eV target selects 179.532 eV at entry.
Across 78 entry/post-jump projection events, nearest-level clearance reached
0.320 eV, the global sector separation reached 0.640 eV, and $c_W$ ranged
from 0.797 to 0.883 eV. The largest observed $c_W/\Delta$ was **1.257**.
The ratio uses the global separation, so it is conservative when the nearest
levels belong to uncoupled Bloch blocks; it does not demonstrate large actual
leakage. It also does not certify a uniformly small perturbative scale.

The bound-exposure deficit at 10,000 Å was $0.001224\pm0.001575$
(pooled ratio, delete-one-chunk jackknife SEM). Two chunks cannot establish an
admission confidence bound, and a small measured exposure deficit does not
bound the omitted coherent correction. This pilot supplies no eligible case
or depth budget. It motivates the coherent projected-evolution comparison
before increasing histories for a depth calibration. Reproduce on the lab box
with the script's `--loss-coupling --cut-eV 178 --workers 2 --chunks 2
--electrons-per-chunk 8 --depth-max-ang 10000 --depth-step-ang 1000
--step-ang 10` options. Source/table digests and exact script SHA-256 are in
the linked record; no peak-RSS claim is made because SLURM accounting is
disabled on this host.

(axial-return-flux-paired-study)=
### Larger paired study (SLURM 1642–1647)

The [study record](../../validation/check-records/axial-zone/study-1642-1647.json)
contains settings, paired-error method, resource records, source/table digests
and identifies the six losslessly compressed raw results linked below. Source checkpoint
`82e4d1f3` retains each chunk's reference and model exposures so comparisons
can delete the **same seed chunk** from both arms. For each arm the deficit
is $D=1-\sum_i X_i/\sum_i X_{\mathrm{ref},i}$. For an arm difference, the
jackknife recomputes both pooled ratios after deleting seed $i$ and subtracts
them before estimating its standard error. Independent chunks, not individual
hook events, are the statistical units.

All cases use Si [001], 5 MeV, TDS plus electronic jumps and a nominal 178 eV
target, except the additional low-cut baseline. The control is $2\times2$
cells at 16 points/cell, 10 Å steps and 10,000 Å depth. Each numerical-control
arm has eight chunks of 32 incident electrons (256 total), with shared seeds
0–7. Only the named control changes. The precision arm instead uses 16 chunks
of 128 (2,048 total). It overlaps some control streams and is reported
separately, never pooled as independent additional data.

| Arm | Change from control | Deficit at 1 µm (%) | Paired change from control (percentage points) |
| --- | --- | ---: | ---: |
| [Control](../../validation/check-records/axial-zone/study-1642-baseline.json.gz) | none | 0.677 ± 0.759 | — |
| [Step](../../validation/check-records/axial-zone/study-1643-step.json.gz) | 5 Å propagation step | 0.099 ± 0.733 | −0.578 ± 0.314 |
| [Grid](../../validation/check-records/axial-zone/study-1644-grid.json.gz) | 24 points/cell, $48^2$ | 0.894 ± 0.718 | +0.217 ± 0.714 |
| [Supercell](../../validation/check-records/axial-zone/study-1645-supercell.json.gz) | $3\times3$ cells, 16 points/cell, $48^2$ | 0.305 ± 0.978 | −0.372 ± 0.709 |
| [Depth](../../validation/check-records/axial-zone/study-1646-depth.json.gz) | propagate to 2 µm, compare at 1 µm | 0.677 ± 0.759 | zero to roundoff |
| [Precision](../../validation/check-records/axial-zone/study-1647-precision.json.gz) | 2,048 incident electrons | 0.427 ± 0.205 | not a matched comparison |

Errors are **jackknife SEMs**, not admission confidence bounds. At 2 µm the
depth arm gives $1.992\%\pm1.255\%$. The low-cut control (nominal 95 eV)
gives $0.873\%\pm1.365\%$ at 1 µm. The actual zero-tilt selected cuts and
their event distributions are retained in the raw results: the adaptive
matching rule changes its selected energy when the represented spectrum
changes, so the grid/supercell comparison includes that discretization effect.

The precision arm reduces sampling uncertainty, but does not certify the
provisional 1% error criterion. Step refinement shifts the estimate by about
0.58 percentage points, and the paired grid/supercell errors remain too wide
to establish the needed convergence. The largest observed conservative
$c_W/\Delta$ is 2.101 in the precision arm and 2.926 in the supercell arm.
These remain diagnostic scales, not measured leakage probabilities. No
physical outgoing instrument, calibrated depth budget or axial run admission
is established. A coherent projected-evolution comparison is still required
before interpreting the exposure differences as physical handoff errors.

All six jobs completed with exit code zero within three minutes, each under
a ten-minute/four-CPU allocation (12 GB memory, 24 GB for the refined grid).
GNU time's reported MaxRSS is retained, but coverage of forkserver workers
was not established, so it is not an allocation peak. Some Python 3.14 jobs
emitted semaphore-cleanup warnings after writing their results; the warnings
did not change the successful scheduler exit codes. These host timings are
provenance, not performance comparisons between different numerical cases.

### Coherently projected no-jump reference

`checks/axial_mcwf_return_flux.py --projected-no-jump` adds a CPU research
comparison using `montecarlo/channeling/axial_projection.py`. This is not a
physical escape model; `cr-axial-zone-handoff-physical` stays `unverified`.
The source is Castin, Dalibard and Mølmer, Eqs. (2), (4), (8), with the
represented-grid generator compressed before exponentiation:

$$H_P=P H P,\qquad W_P=P W P,\qquad
U_P(\Delta z)=\exp\!\left[-\frac{i\Delta z}{\hbar v_z}(H_P-iW_P)\right].$$

The exponential acts **within** the retained subspace. It is not
$P\exp[-i(H-iW)\Delta z/(\hbar v_z)]P$, and it does not add repeated
finite-step measurement losses. In the spectral basis of $H(\mathbf q)$,
$H_P$ is diagonal; the supplied cell-periodic $W$ is compressed in each
Fourier/Bloch block. Dense matrix exponentials remove kinetic/potential
Strang splitting from this reference, up to floating-point matrix-function
error. Jump times remain resolved to one propagation step.

The exact no-jump norm obeys

$$-\partial_z\|\phi\|^2=\frac{2}{\hbar v_z}
\langle\phi|W_P|\phi\rangle.$$

Here $W$ is in eV and $\hbar v_z$ in eV Å, so the hazard is per Å. The full
native jump selection and jump operators are evaluated on the normalized
retained state. The same stochastic post-jump $P/F$ measurement supplies
retained channels $P J_m P$; sink outcomes remain separately recorded as
numerically unresolved. Electronic jumps update the carrier and therefore
the selected projector. Entry and post-jump states must already be normalized
inside that subspace; otherwise the reference refuses instead of silently
losing unaccounted weight. The pure entry's projected norm is applied externally
as in the existing study. General mixtures need their own component weights.

The five arms are:

1. Native Strang reference with no handoff.
2. Native Strang propagation with entry/post-jump handoff measurements.
3. Full-space matrix-exponential reference with no handoff.
4. Full-space matrix propagation with the same handoff measurements.
5. Retained-space matrix propagation with those measurements.

The difference between arms 4 and 5 isolates the compression effect from
Strang splitting at a fixed jump-depth step. Arms 1–4 expose the independent
splitting effect. Deficits of arms 4 and 5 share arm 3's pooled reference
exposure; their difference and paired delete-one-chunk SEM are stored under
`spectral_comparison`. Per-chunk exposures are retained for all arms. More
than one jump per step is not resolved by either implementation, so step
convergence is still required even with matrix-exponential propagation.

For $W=0$ the retained evolution is unitary. Constant scalar $W$ attenuates
amplitude by $\exp[-W\Delta z/(\hbar v_z)]$. Empty/full cuts recover the
zero/full subspace. If $[P,W]=0$, no-jump evolution starting in $P$ stays in
$P$ in the full reference as well. Independent real-space DOP853 integration,
analytic constant-loss/semigroup checks, seeded native sink waiting times and
carrier-update regressions are in
`tests/montecarlo/test_channeling_axial_projection.py`. These are numerical
implementation checks; fresh-context source-to-code verification remains open.

### Projected comparison results (SLURM 1648–1650)

Source checkpoint `4384e473`; all three jobs completed with exit zero. The
16-electron pilot (1648) was followed by matched 256-electron runs at 10 Å
(1649) and 5 Å (1650), each with eight seed chunks of 32 histories. Both use
2×2 cells, 16 points per cell, TDS plus electronic jumps, 1 µm depth and the
selected 179.532 eV cut. The pilot overlaps the main streams and is not pooled.

At 1 µm, deficits are percentages of each propagator's no-handoff reference
bound-state exposure. Errors are paired delete-one-chunk jackknife SEMs:

| Propagation / measurement | 10 Å step | 5 Å step |
|---|---:|---:|
| Native Strang with handoff | 0.677 ± 0.759% | 0.099 ± 0.733% |
| Full matrix with handoff | 0.398 ± 0.862% | 0.301 ± 0.699% |
| Retained matrix with handoff | 0.409 ± 0.863% | 0.277 ± 0.710% |
| Retained minus full-matrix handoff deficit | +0.0108 ± 0.0100 pp | −0.0243 ± 0.0271 pp |

The last row uses a common full-matrix reference and correlated chunks; its
error is not obtained by adding the marginal errors. Halving the step changes
the retained-matrix deficit by −0.132 ± 0.437 percentage points (paired SEM).
The native counterpart changes by −0.578 ± 0.314 points. These samples show no
resolved large compression effect for this case; they neither establish a
step-converged error bound nor justify a physical handoff. More histories can
narrow sampling error, but cannot supply the missing physical outgoing
instrument, source closure, or independent verification. Bound-state exposure
remains a proxy rather than a photon-source error measurement.

The summary retains per-arm source identities, settings, environment, batch
scripts and resource records; the raw results retain every chunk exposure.
Reported GNU time RSS does not establish forkserver-worker/allocation peaks.
Main/half-step scheduler runtimes were 91/90 s on four allocated CPUs with
12 GB requested memory. Evidence:
{download}`summary <../../validation/check-records/axial-zone/projected-1648-1650.json>`,
{download}`pilot <../../validation/check-records/axial-zone/projected-1648-pilot.json.gz>`,
{download}`10 Å <../../validation/check-records/axial-zone/projected-1649-main.json.gz>`,
{download}`5 Å <../../validation/check-records/axial-zone/projected-1650-halfstep.json.gz>`.

### Matching-energy, grid and supercell scan (SLURM 1651–1656)

The same five-arm comparison was repeated at several matching energies and
two numerical refinements, without source changes (script and spectral
kernel hashes as for 1648–1650). All six jobs (three 16-electron pilots,
then 1654–1656) completed with exit zero. Settings: Si [001], 5 MeV, TDS
plus electronic jumps, 10 Å steps, 1,000 Å recording interval, gap window
±2 eV.

- **Cut scan (1654):** $2\times2$ cells, 16 points/cell, nominal targets
  8.19 (≈ $U_{\max}$), 50, 95 and 178 eV, 2 µm depth, 32 chunks × 32
  electrons (1,024). All cuts share one reference per propagator, so
  cut-to-cut differences are paired. Its seeds 0–7 reproduce the 1649
  retained-matrix exposures at 1 µm exactly (zero difference).
- **Grid (1655):** 24 points/cell ($48^2$), 95 and 178 eV, 1 µm, 256 electrons.
- **Supercell (1656):** $3\times3$ cells at 16 points/cell, 95 and 178 eV,
  1 µm, 256 electrons.

Retained-matrix (projected) deficits, jackknife SEM, with zero-tilt selected
cuts:

| Arm | Selected cut (eV) | 1 µm (%) | 2 µm (%) |
|---|---:|---:|---:|
| Cut scan | 7.95 | 11.39 ± 2.16 | 19.39 ± 2.27 |
| Cut scan | 49.11 | 0.29 ± 0.76 | 2.23 ± 0.94 |
| Cut scan | 94.96 | 0.22 ± 0.48 | 1.41 ± 0.66 |
| Cut scan | 179.53 | 0.31 ± 0.33 | 0.93 ± 0.51 |
| Grid $48^2$ | 94.93 / 178.04 | 1.19 ± 1.54 / 0.56 ± 0.77 | — |
| Supercell $3\times3$ | 95.50 / 177.64 | 1.03 ± 1.64 / 0.42 ± 0.67 | — |

Paired cut differences (lower minus higher cut, percentage points): 7.95 vs
49.11 eV is **+11.11 ± 1.85** at 1 µm and +17.17 ± 2.05 at 2 µm. 49.11 vs
94.96 eV is +0.06 ± 0.59 and +0.82 ± 0.84; 94.96 vs 179.53 eV is −0.09 ± 0.50
and +0.48 ± 0.68. Retained-minus-full-matrix handoff effects stay below
0.16 points for every cut and arm, none resolved beyond 1.3 SEM. At 178 eV
the grid and supercell arms differ from the 1649 control by +0.15 ± 1.00
and +0.01 ± 0.69 points (same seeds, but streams diverge, so weakly paired).
Native Strang and full-matrix hook rows agree with the retained-matrix rows
within their errors; all rows are retained in the raw results.

Electronic kicks put the cut at generic tilts. Across hook events, the
smallest level-free width around a selected cut was 0.36–0.64 eV on
$2\times2$ cells, 0.42–0.50 eV on the $48^2$ grid and 0.27 eV on
$3\times3$ cells. The median per-chunk median width fell from about 1.0–1.4 eV
to 0.65–0.74 eV on $3\times3$ cells. Larger supercells fold more levels into
each window, so clearance, not only cut energy, must be tracked when they grow.

**Reading.** A barrier-adjacent cut is excluded: it discards about a tenth
of the bound exposure within 1 µm, consistent in sign with the TDS-only
secular table above (3.2% at $10^4$ Å, 7.8 eV) but three to four times
larger with electronic jumps on this $2\times2$ grid. For cuts from about $U_{\max}+(U_{\max}-U_{\min})/2$ upward, the
1 µm differences are unresolved at the 0.5–0.6 point level, and grid and
supercell changes are unresolved at the 0.7–1.0 point level. At 2 µm the
estimates order as expected (lower cut, larger deficit) but the pairwise
differences are below 1.3 SEM. These are SEMs from 256–1,024 histories,
not confidence bounds. They do not establish a depth budget, a physical
outgoing instrument or a photon-source error. Bound-state exposure remains
the proxy.

Evidence: {download}`summary <../../validation/check-records/axial-zone/matching-1651-1656.json>`
(settings, batch script, provenance, paired analysis and pilot digests),
{download}`cut scan <../../validation/check-records/axial-zone/matching-1654-cuts.json.gz>`,
{download}`grid <../../validation/check-records/axial-zone/matching-1655-grid.json.gz>`,
{download}`supercell <../../validation/check-records/axial-zone/matching-1656-supercell.json.gz>`.
Scheduler runtimes were 6 min 40 s (16 CPUs), 13 min 17 s (8 CPUs) and
2 min 6 s (12 CPUs).

## Material TDS jump maps for the absorbing instrument

`checks/axial_outgoing_maps.py` turns the Kraus factorization of
{eq}`eq-axial-coherent-transfer-covariance` into the inputs of
`axial_jump_instrument.absorbing_axial_model` for a complete Bloch eigenbasis
on a rectangular order window $R$ and an energy cut $P$ (so $[H,P]=0$):
$A_j=PL_jP$, $B_j=(R-P)L_jP$ mapped to the window's discrete position basis,
and $E_O=P(\Gamma-\sum_jL_j^\dagger RL_j)P$. Here $\Gamma$ is the lattice-sum
loss of the same periodic finite-transfer model, so
$E_O=\sum_jPL_j^\dagger(1-R)L_jP\ge0$ up to the order tail and Kraus rank
truncation; the builder reports its smallest eigenvalue and refuses a
materially negative one. These are differential maps from the material
covariance, not a relabelled grid sink.

Focused tests check the operator closure
$\sum A^\dagger A+\sum B^\dagger B+E_O=P\Gamma P$, agreement of the diagonal
with the Pauli rate matrix of `periodic_tds_rates`, and, as an independent
off-diagonal oracle, equality of $\sum_jB_j\rho B_j^\dagger$ with the
literal-sum `coherent_tds_gain` of a mixed retained state projected on
$R-P$, plus zero-rate limits and refusals.

Bounded example (Si [001], 5 MeV, one cell, Γ point, TDS only, orders
$|n|\le4$, cut 29.39 eV, 24 retained states, 183 Kraus jumps, entrance
weight 0.99999; 34 s, 0.5 GB RSS):

| Depth (Å) | Retained | Represented outgoing | Unresolved |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.8716 | 0.0214 | 0.1071 |
| 10,000 | 0.3952 | 0.1057 | 0.4991 |
| 30,000 | 0.0945 | 0.1637 | 0.7418 |

Closure defects are below $1.5\times10^{-14}$ and the smallest unresolved
eigenvalue is $+1.2\times10^{-11}$ Å$^{-1}$. The unresolved column is
dominated by transfers leaving the $9\times9$ window. It is window
accounting, not a physical escape fraction. A sampled event returns a
correlated depth/position/momentum index on that window.

**Scaling limit and trajectory unraveling.** The dense instrument costs
$(2n^2)^2$. The provisional ~95 eV rule retains 77 Γ-point states in one cell
(about 2 GiB per dense matrix, more for supercells and Bloch sampling), so the
dense form is a small-$n$ oracle only.
`axial_jump_unraveling.unravel_absorbing_outgoing` samples the same law with
terminal events: between jumps $\psi$ evolves under $H-i\hbar v\Gamma/2$; the
jump depth solves $\lVert\psi(z)\rVert^2=u$ by root finding (not step
quantized); the channel is drawn with weights $\lVert A_a\psi\rVert^2$,
$\lVert B_b\psi\rVert^2$, $\langle\psi|E_O|\psi\rangle$. Retained jumps
continue; an outgoing jump measures $B_b\psi$ with the Weyl instrument and
ends; the unresolved channel ends. Incident-weighted event fractions estimate
$\operatorname{tr}Q_F(z)$ and $U_O(z)$. Propagation uses one
eigendecomposition of the non-Hermitian $H_\mathrm{eff}$, with an `expm`
fallback when its eigenvectors are ill conditioned. Each incident has its own
versioned jump stream plus the existing measurement stream.

Focused tests compare 3,000 incidents of a mixed three-state model, with
about four coherent recycling jumps per incident, against the dense oracle
tallies (4.5σ binomial) and the joint Weyl outcome law ($\chi^2$). They also
check the scalar exponential depth law (KS), zero jumps, processing-order
independence, equality of the fallback path and refusals. Injected faults are
detected: a wrong no-jump loss factor fails the tally and depth-law tests,
and discarding recycling phases shifts the outgoing tally by about 8σ.

Material run, 20,000 incidents, seed 0
([record](../../validation/check-records/axial-zone/unravel-local-20k.json);
local CPU, 3 min 20 s, 0.68 GB RSS). At the 29.39 eV cut (orders $|n|\le4$,
24 states) every unraveled tally agrees with the dense oracle within 1.3σ.
At the provisional 94.97 eV cut (orders $|n|\le6$, 77 states, 175 s), the
values are below, with binomial standard errors:

| Depth (Å) | Retained | Represented outgoing | Unresolved |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.904 ± 0.002 | 0.021 ± 0.001 | 0.075 ± 0.002 |
| 10,000 | 0.490 ± 0.004 | 0.121 ± 0.002 | 0.389 ± 0.003 |
| 30,000 | 0.177 ± 0.003 | 0.205 ± 0.003 | 0.618 ± 0.003 |

"Retained" counts every state below the cut, not bound exposure. Unresolved
weight is mostly TDS transfer out of the 13×13 order window. These numbers
are therefore window accounting and are not comparable with the MCWF
bound-exposure deficits above. The model omits electronic jumps, Bloch
sampling and supercells, continuum transfers and return flux. It is not a
dechanneling fraction or depth budget.

**Electronic kicks on carrier fibres.** An electronic jump
$\sqrt{c_q\rho_q(\mathbf r)}\,e^{i\mathbf q\cdot\mathbf r}$ moves the Bloch
carrier by $\mathbf q$. The density matrix stays block diagonal in the
carrier, so a classical folded carrier label is exact.
`axial_fiber_unraveling.unravel_fiber_outgoing`
([derivation and review](../../validation/channeling/cr-axial-fiber-unraveling.md))
keeps one retained state per history on fibre $P(\boldsymbol\kappa)$. The
algorithm, step by step:

- Between jumps the state evolves under that fibre's no-jump generator.
- At each jump it draws a TDS Kraus column, the lattice unresolved remainder
  or a kick node.
- A kick then draws a uniform azimuth.
- The kicked carrier folds into $[-g/2,g/2)$ and the orders are relabelled.
- The output is split into retained (continue), represented outgoing (Weyl
  measurement, end) or outside the window (unresolved, end).

At the Γ point with TDS only, the fibre engine's rate operators equal the
`tds_outgoing_maps` operators above to about $10^{-15}$.

Local study (Si [001], 5 MeV, orders $|n|\le6$, ~95 eV cut, entry at Γ;
[record](../../validation/check-records/axial-zone/fiber-local-4k.json)).
The electronic model uses 680 kick nodes on a $32^2$ density grid, with
harmonics $|n|\le4$ dropping at most $1.3\times10^{-3}$ of $\int\rho_q$. The
electronic run took 13.5 min for 4,000 incidents.

| Depth (Å) | TDS outgoing | TDS unresolved | TDS + electronic outgoing | TDS + electronic unresolved |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.030 ± 0.003 | 0.078 ± 0.004 | 0.024 ± 0.002 | 0.074 ± 0.004 |
| 10,000 | 0.133 ± 0.005 | 0.384 ± 0.008 | 0.148 ± 0.006 | 0.371 ± 0.008 |
| 30,000 | 0.207 ± 0.006 | 0.617 ± 0.008 | 0.259 ± 0.007 | 0.555 ± 0.008 |

The table uses seed 0 with 4,000 incidents for each column. A repeat of the
TDS-only run (seed 1, 8,000 incidents) gives outgoing 0.024, 0.130 and 0.211.
That agrees with the 20k Γ-point unraveling within 2σ at every depth. At
1e3 Å the seed-0 value sits 3.3σ high, which is a fluctuation.

Each electronic history makes about 5.6 kicks and 1.6 TDS jumps, yet
retained weight is unchanged within errors (0.186 vs 0.175 at 3e4 Å). Kicks
mostly recycle inside the retained subspace across fibres. Their net effect
in this window is to move about 6% of the weight at 3e4 Å from unresolved to
represented outgoing. This is still window accounting for one string per
cell with an absorbing no-return model: no supercells, continuum transfers,
return flux or matching-energy convergence.

**Outgoing order box.** Most of the unresolved weight above is not
unrepresentable. It is TDS output that lands just outside the 13×13 retained
window. `FiberChannels.outgoing_order_max` = B now represents and measures
jump outputs on a box $[-B,B]^2 \supseteq R$. Retained weight is still
$P(\boldsymbol\kappa')$, outgoing is box $-P$, and only weight outside the box
is unresolved. The box is exact when the Kraus columns span $D\ge B+M$. Then
every box order receives every transfer, and the remainder is a sum of PSD
terms over outputs outside the box
([derivation and re-verification](../../validation/channeling/cr-axial-fiber-unraveling.md#implementation-change-outgoing-order-box-2026-10-31-awaiting-re-verification)).
In the Γ fibre the TDS remainder falls as $B^{-2}$, the $Q^{-2}$ single-nuclear
tail: 2.85e-5, 5.9e-6, 2.5e-6 and 1.4e-6 Å$^{-1}$ at B = 6, 12, 18 and 24.

Repeating the study at B = 18 (D = 24, 872 Kraus columns, balanced Gaussian
Weyl window on the 37×37 box; 4,000 incidents, seed 0;
[record](../../validation/check-records/axial-zone/fiber-box18-local-4k.json))
gives:

| Depth (Å) | TDS outgoing | TDS unresolved | TDS + electronic outgoing | TDS + electronic unresolved |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.090 ± 0.005 | 0.009 ± 0.001 | 0.091 ± 0.005 | 0.008 ± 0.001 |
| 10,000 | 0.481 ± 0.008 | 0.036 ± 0.003 | 0.484 ± 0.008 | 0.037 ± 0.003 |
| 30,000 | 0.775 ± 0.007 | 0.056 ± 0.004 | 0.762 ± 0.007 | 0.052 ± 0.004 |

What changes and what doesn't:

- **Retained weight is unchanged** from the B = 6 runs (0.168 and 0.185 at
  3e4 Å, against 0.175 and 0.186). It should be: the box changes only how
  terminal weight is labelled.
- **Unresolved weight falls** from 0.62 to 0.06 at 3e4 Å.
- **Most outgoing events are measured outside the retained window**: 72%
  (TDS) and 66% (TDS + electronic) have order $|n|>6$.
- **Electronic kicks no longer move weight between the terminal labels at B = 18.** The earlier
  difference between the two runs was electronic kicks recycling weight that
  TDS would otherwise have sent just outside the window, and the box now
  represents that weight directly.

The remaining unresolved weight is the TDS tail beyond 18 orders
(transverse energy ≳ 1.2 keV). It can be bounded by its $B^{-2}$ law, or
sampled as single large-angle nuclear scattering.

The model still has one string per cell. All strings vibrate together, so
TDS transfers are confined to the reciprocal lattice and TDS never changes
the carrier. Independent strings give
$\sum_R L_R\rho L_R^\dagger=\int_{\rm BZ}d^2k\,J_k\rho J_k^\dagger$ with
$J_k=e^{i\mathbf k\cdot\mathbf r}\varphi_k(\mathbf r)$, where $\varphi_k$ is
cell periodic with harmonics $\hat f(\mathbf k+\mathbf G)$. That is the same
carrier-moving structure as the electronic kicks, but with a column that
depends on the full vector $\mathbf k$. Tabulating $\mathbf k$ on an
$N\times N$ grid is equivalent to an $N\times N$ TDS supercell.

**Independent-string (continuum) TDS.** `axial_tds_continuum` implements
that family. Each Kraus label is the pair (carrier shift $\Delta$, struck-string
displacement $u$), because the thermal average over $u$ already is the
factorization. Each TDS jump draws (output momentum, $u$) by exact rejection
from the single-atom incoherent bound, at 0.1–2.5 ms per draw. The jump moves
the carrier, and it is unresolved exactly when the sampled momentum leaves
the box. No eigendecomposition, transfer truncation or $D\ge B+M$ condition
is needed
([derivation](../../validation/channeling/cr-axial-continuum-tds-jumps.md)).
Any number of strings per cell is supported. For Si [001] there is one
string per cell, so this removes the image correlation completely and a
supercell adds nothing. Si [111] has 2 strings per cell and [110] has 4.

Si [001], B = 18, 4,000 incidents
([record](../../validation/check-records/axial-zone/fiber-continuum-001-local-4k.json)).
The TDS-only run takes 92 s, about 10× faster than the lattice model at the
same box. At 3e4 Å:

| Model | Retained | Outgoing | Unresolved |
| --- | ---: | ---: | ---: |
| TDS, one-cell lattice | 0.168 ± 0.006 | 0.775 ± 0.007 | 0.056 ± 0.004 |
| TDS, independent strings | 0.186 ± 0.006 | 0.762 ± 0.007 | 0.051 ± 0.003 |
| TDS + electronic, lattice | 0.185 ± 0.006 | 0.762 ± 0.007 | 0.052 ± 0.004 |
| TDS + electronic, strings | 0.192 ± 0.006 | 0.758 ± 0.007 | 0.050 ± 0.003 |

The one-cell model's 1.7% lower mean TDS rate and its carrier-conserving
transfers shift these window tallies by only 1–2σ.

**Tail beyond the box (#881).** The unresolved string weight above is the
sampled single-nuclear tail: its momentum is already drawn, so it is now
recorded as a `"tail"` event (absolute $K$, string displacement, depth)
within a paraxial cap $|K|\le0.1k$
([derivation](../../validation/channeling/cr-axial-tds-tail.md)). Draws are
unchanged. At 3e4 Å the tail is 0.049 (TDS) and 0.045 (TDS + electronic) and
unresolved falls to 0.0025 and 0.0045
([record](../../validation/check-records/axial-zone/fiber-tail-001-local-4k.json)).
The TDS-only residual is the tail beyond the cap, about 6% of the
out-of-box weight as the large-$Q$ law predicts for the square box (12
expected, 10 observed). In the electronic run, half of the 18 unresolved
events are unclassified large-$q$ kick remainders. A tail is a model record, not a physical escape.

Si [111] has two strings per cell (cut target 106.1 eV by the provisional
rule, 199 retained states at Γ, square window M = 12, box B = 30;
[record](../../validation/check-records/axial-zone/fiber-continuum-111-local.json)).
At 3e4 Å, TDS alone (1,000 incidents) gives retained 0.261 ± 0.014, outgoing
0.668 ± 0.015 and unresolved 0.071 ± 0.008. TDS + electronic (500 incidents)
gives retained 0.226 ± 0.019, outgoing 0.736 ± 0.020 and unresolved
0.038 ± 0.009. Caveats for [111]:

- The electronic harmonics drop up to 1.8% of $\int\rho_q$ on this cell.
- The electronic run peaked at 8.05 GB, at the local memory cap.
- The square order window makes the $x$ reach the binding box edge.

**Reach-matched rectangular window (#880).** On a rectangular cell the
window and box extents can be chosen per axis, so that $|g_i|M_i$ is about
the same on both axes (`matched_order_extents`). For Si [111]
($g=(1.636, 2.834)$ Å⁻¹) the square $M=12$, $B=30$ becomes $M=(12,7)$,
$B=(30,18)$: 375 instead of 625 window orders and 2,257 instead of 3,721 box
orders. The Γ-point count below the cut stays 195
([record](../../validation/check-records/axial-zone/fiber-continuum-111-rect-local.json)).
At 3e4 Å:

| Si [111] | Window | Retained | Outgoing | Unresolved | Wall | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TDS (1,000) | square | 0.261 ± 0.014 | 0.668 ± 0.015 | 0.071 ± 0.008 | 8:56 | 2.66 GB |
| TDS (1,000) | rectangular | 0.253 ± 0.014 | 0.670 ± 0.015 | 0.077 ± 0.008 | 2:31 | 0.37 GB |
| TDS + el. (500) | square | 0.226 ± 0.019 | 0.736 ± 0.020 | 0.038 ± 0.009 | 17:11 | 8.05 GB |
| TDS + el. (500) | rectangular | 0.210 ± 0.018 | 0.732 ± 0.020 | 0.058 ± 0.010 | 5:43 | 0.37 GB |

All tallies at the three depths agree within $|z|\le1.9$ (the largest is the
electronic unresolved fraction). The rectangular run takes about a third of
the wall time. Most of the old peak memory was cyclic garbage, not live
state. SciPy's `brentq` wrapper kept each waiting-time root's closure in a
reference cycle, and that closure held a fibre eigenbasis. Electronic runs
grew by about 18 MB per incident until a full GC pass. With the closure
holding only a cleared list, the same runs peak at 0.37 GB, with bitwise
identical results (rectangular rows above; the square rows predate the
fix).

Si [110] has four strings per cell, $g=(1.636, 1.157)$ Å⁻¹, and a cut target
of 139.25 eV (same provisional rule). Its Γ count below the cut is 535 for
the square $M=12$, whose $y$ reach is only 13.9 Å⁻¹. It converges to 655
once both axes reach about 20 Å⁻¹, for example $M=(13,17)$ with 945 orders;
a square window needs $M=17$ (1,225 orders).

Si [110] window convergence (box $B=(30,42)$, remote CPU array 3158;
[record](../../validation/check-records/axial-zone/fiber-continuum-110-remote-3158.json)).
At 3e4 Å:

| Si [110] | Window (orders) | Retained | Outgoing | Unresolved | TDS jumps |
| --- | --- | ---: | ---: | ---: | ---: |
| TDS (1,000) | (13,17) (945) | 0.303 ± 0.015 | 0.620 ± 0.015 | 0.077 ± 0.008 | 1.84 |
| TDS (1,000) | (15,21) (1,333) | 0.314 ± 0.015 | 0.597 ± 0.016 | 0.089 ± 0.009 | 1.93 |
| TDS (1,000) | (18,25) (1,887) | 0.303 ± 0.015 | 0.602 ± 0.015 | 0.095 ± 0.009 | 2.08 |
| TDS + el. (1,000) | (13,17) (945) | 0.282 ± 0.014 | 0.633 ± 0.015 | 0.085 ± 0.009 | 1.73 |
| TDS + el. (500) | (15,21) (1,333) | 0.266 ± 0.020 | 0.626 ± 0.022 | 0.108 ± 0.014 | 1.84 |

The (15,21) and (18,25) TDS windows agree on every tally to $|z|\le0.54$.
(13,17) sits 1–2σ high in outgoing at 1e3–1e4 Å, so (15,21) is the
converged choice at this precision. The mean TDS jump count still rises from
(15,21) to (18,25), by 2.3σ, while the terminal tallies stay the same; that
is open. Electronic kicks change the [110] tallies by at most about 1σ. The
fraction unresolved at the $B=(30,42)$ box is about 0.09–0.11, above [111]'s
0.06–0.08; box and tail convergence belong to #881/#882. Each task peaked
below 1.6 GB.

## Paired numerical convergence and return deficit of the fibre model (#882)

`checks/axial_outgoing_convergence.py` runs one-change arms of the fibre
unraveling (Si [001], 5 MeV, continuum TDS + electronic kicks, seed 0) around
the control M = 6, nominal 95 eV cut, box B = 18 and 200 Å search step. All
arms share incident IDs, so every difference is paired per incident.
`unravel_fiber_outgoing(observe=...)` reports each surviving incident's
normalized retained amplitude at 61 checkpoints (0–3 µm), and the study
tallies the below-$U_{\max}$ population $N_B$ and its exposure $X_B$
([derivation](../../validation/channeling/cr-axial-fiber-return-deficit.md)).
The return deficit compares the absorbing control with the $P=R$ reference
(cut removed, same window and box), as in the secular study above. It
includes return through jumps plus two sign-indefinite terms: the coherent
no-jump coupling across the cut and jump interference of $P$–$Q$
coherences recycled into $P$. It is accounting at a fixed window, not a physical
escape fraction.

**Results (SLURM 3050;
[record](../../validation/check-records/axial-zone/outgoing-convergence-3050.json)).**
16,000 incidents per arm (4,000 for the step arms, paired against the same
control IDs). Paired differences from the control, mean ± SEM. Relative
exposure is $1-X_B(\text{arm})/X_B(\text{control})$, so a positive value
means the arm has less bound exposure:

| Arm | ΔR at 1 µm | ΔR at 3 µm | Rel. $X_B$ at 1 µm | Rel. $X_B$ at 3 µm |
| --- | ---: | ---: | ---: | ---: |
| step 100 Å / 400 Å | 0 (identical) | 0 | 0 | 0 |
| B = 24 | −0.0005 ± 0.0008 | −0.0005 ± 0.0006 | +0.01% ± 0.03% | +0.04% ± 0.05% |
| B = 12 | +0.0026 ± 0.0013 | +0.0009 ± 0.0011 | −0.09% ± 0.05% | −0.13% ± 0.09% |
| M = 5 | −0.0022 ± 0.0031 | +0.0024 ± 0.0033 | −2.88% ± 0.18% | −3.54% ± 0.31% |
| M = 7 | +0.0047 ± 0.0027 | −0.0039 ± 0.0031 | +0.78% ± 0.13% | +0.91% ± 0.23% |
| M = 8 | +0.0070 ± 0.0029 | −0.0064 ± 0.0031 | +0.94% ± 0.14% | +1.22% ± 0.25% |
| target 50 eV | −0.110 ± 0.003 | −0.091 ± 0.003 | +0.40% ± 0.15% | +1.77% ± 0.26% |
| target 140 eV (136.8) | +0.075 ± 0.003 | +0.071 ± 0.003 | −0.28% ± 0.10% | −1.12% ± 0.20% |
| target 180 eV | +0.109 ± 0.003 | +0.114 ± 0.004 | −0.26% ± 0.11% | −1.40% ± 0.21% |

Control tallies at 1 and 3 µm: retained 0.476 ± 0.004 and 0.194 ± 0.003,
outgoing 0.489 and 0.755, unresolved 0.035 and 0.050; $N_B$ = 0.1503 and
0.0247; $X_B$ = 3502 ± 17 and 4729 ± 30 Å.

- **Search step.** It does not enter. The no-jump norm is monotone, so the
  step only changes where the root bracket starts; events are identical.
- **Box.** B = 18 is converged: against B = 24 the exposure changes by
  ≤0.04% ± 0.05%; only the terminal labels move (unresolved 0.050 → 0.027
  at 3 µm). B = 12 changes exposure by ≤0.13% but raises unresolved to
  0.117.
- **Window M.** Exposure falls by 0.9% (M = 7) and 1.2% (M = 8) at 3 µm, and
  0.8–0.9% at 1 µm; M = 5 is 2.9–3.5% high. Convergence in M at the 1% level
  holds to about 1 µm, not to 3 µm.
- **Cut target.** Retained weight counts everything below the cut, so it
  moves with the cut by construction and is not a convergence observable.
  Bound exposure gains 1.1% (137 eV) and 1.4% (180 eV) at 3 µm, and
  0.3% at 1 µm. At 180 eV it equals the $P=R$ reference within errors
  (4795 vs 4794 Å).

Return deficit, $1-X_B(\text{absorbing})/X_B(P=R)$ at the provisional 95 eV
cut (paired, mean ± SEM):

| Window | 0.1 µm | 1 µm | 3 µm |
| --- | ---: | ---: | ---: |
| M = 6 | 0.01% ± 0.03% | 0.20% ± 0.11% | 1.37% ± 0.20% |
| M = 8 | 0.05% ± 0.04% | 0.50% ± 0.13% | 2.51% ± 0.22% |

The deficit in the below-barrier population itself, $1-N_B/N_B^{\rm ref}$,
is 1.2% ± 0.4% (M = 6) and 1.8% ± 0.4% (M = 8) at 1 µm, and 10% ± 2% and
19% ± 2% at 3 µm. The $P=R$ reference exposure is 0.45% ± 0.04%,
0.64% ± 0.14% and 0.06% ± 0.23% lower at M = 8 than at M = 6 at 0.1, 1 and
3 µm (paired).

**What this establishes (numerical accounting only).** With 95% confidence
(±2 SEM), for Si [001] 5 MeV in this model:

- step and box (B ≥ 18) are converged to well below 0.1% of bound exposure;
- to 1 µm, window (M = 6 → 8) and cut (95 → 180 eV) each change bound
  exposure by at most about 1.2%, and the return deficit is below 0.8%
  (M ≤ 8);
- to 3 µm, the return deficit is 1.0–3.0% and grows with M, as in the
  secular study, so it is not converged in the window; the cut target and
  M each move exposure by about 1–1.8%.

The 95 eV rule therefore keeps return-deficit and cut errors near the 1%
level to about 1 µm. Beyond that the deficit grows with depth and window,
consistent with the depth-budget conclusion above. The first 1,500 incidents
of the same array, recorded earlier as a preliminary result, overstated the
M = 6 deficit (2.6% ± 0.6% at 3 µm); that partial record is superseded.

## Matching prescription still to establish

**Provisional numerical matching rule (research, from 1651–1656).** At each
entry and post-jump tilt, cut in the widest level-free interval within
±2 eV of a nominal target no lower than
$U_{\max}+(U_{\max}-U_{\min})$ (about 95 eV for Si [001] at 5 MeV), and
record the selected cut and clearance. Refuse barrier-adjacent cuts. This
rule fixes only the numerical retained/outgoing split for further study. It
does not supply the physical outgoing state, a depth budget or admission.

Retain slow free states in the quantum zone. Investigate native matching only
where the outgoing transverse motion is sufficiently kinetic dominated and
coherences/return to channeling have a bounded effect. A high-energy matching
surface must converge as it moves while the represented grid and differential
transfer range are enlarged independently. A saddle-energy classification is
an observable to report, not automatically that surface. No numerical cutoff
or planar calibration is selected here.

The outgoing instrument must supply a joint depth/position/momentum/energy
record for each incident ID. Map the direction in the existing axial-to-lab
frame, enforce the declared kinetic-energy shell, and preserve the launch
clock and native stream key. If a sampled transverse momentum exceeds the
allowed shell or paraxial regime, refuse the approximation; clipping its angle
to a grid edge does not repair it. Potential-energy reference and stopping
loss must be explicit before interpreting transverse eigenenergy as a vacuum
angle.

The energy-defined projector study above makes the matching condition
depth-dependent. Next physical evidence:
- a decision on the long-depth architecture. The options are an extended
  quantum zone, or a re-entry hybrid: native transport between fresh quantum
  entrance zones whenever multiple scattering returns the electron inside the
  matching cone. Classical bound orbits are not adequate for six bound
  states;
- Bloch-$\mathbf k$/supercell convergence of the return-flux deficit, which
  must come before electronic jumps. Their transfers are mostly smaller than
  the 3.27 Å$^{-1}$ reciprocal spacing and move the Bloch
  momentum off Γ, so a Γ-only basis cannot hold them;
- a coherent (non-secular) comparison on a small basis;
- a native phase-space launch law.

Fresh-context review and a physical anchor remain necessary before the claim
can reach `anchored`.
