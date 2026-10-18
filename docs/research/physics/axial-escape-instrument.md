# Axial entrance-zone escape: operator accounting

The maintainer selected the entrance-zone architecture for
[#548](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/548)
on 2026-10-17. This note derives the accounting that an escape model must
satisfy and gives a numerical counterexample to identifying the current grid
sink with outgoing population. It does **not** select or validate a physical
matching threshold, rechanneling approximation, or native launch distribution.
The claim `cr-axial-zone-handoff-accounting` remains `unverified`.

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
`cr-axial-zone-handoff-accounting` remains `unverified`.

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
the 16² grid points, not from the closure; at 32², where the clamp is
inactive, the mixture's grid mean matches the total to 5e-10. Supercell lattice sums over
orders $|n_{x,y}|\le15c$ approach the matching continuum box integral as
6.078e-5 versus 6.225e-5, 6.2288e-5 versus 6.2309e-5, and 6.22787e-5 versus
6.22785e-5 Å$^{-1}$ for $c=1,2,4$. The $c=1$ lattice sum reproduces the
finite-transfer total of the earlier table.

This removes the omitted-transfer caveat from the numerical window split and
ties the outgoing gain to the production TDS loss effect. It does not choose
the physical retained projector $P$, bound return flux, supply an outgoing
phase-space instrument, or treat electronic jumps; the box remains numerical
accounting. A [fresh-context review](../../validation/channeling/cr-axial-zone-handoff-accounting.md)
rederives this split; `cr-axial-zone-handoff-accounting` remains `unverified`.

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
cell, $c=2$) matches no row there. Results are pending.

## Matching prescription still to establish

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
