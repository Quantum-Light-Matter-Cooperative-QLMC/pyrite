# `cr-axial-fiber-return-deficit`: retained-state observation and paired return deficit

**Status:** task-owner derivation (2026-11-01, #882), awaiting fresh-context
verification.
Code: the observation hook of
`src/pyrite/montecarlo/channeling/axial_fiber_unraveling.py`
(`unravel_fiber_outgoing(observe=..., checkpoints_ang=...)`, `_observe_segment`)
and the estimators of `checks/axial_outgoing_convergence.py`
(`bound_weight`, `observables`, `paired`, `return_deficit`, `analyze`, `run_arm`).
The underlying unraveling is `cr-axial-fiber-unraveling`.

## Why

The fibre unraveling recorded only terminal events. Retained weight at depth
$z$ follows from them, but the return-flux question of #882 needs *which*
retained states are populated: the below-barrier population and its exposure,
compared between the absorbing model and a reference that keeps the whole
order window. This matches the observable of the secular $P=R$ study in
[the escape research note](../../research/physics/axial-escape-instrument.md#energy-defined-retained-projector-and-return-flux).

## Derivation

**Model.** In fibre $\kappa$ the retained density obeys
(Castin, Dalibard & Mølmer, arXiv:0805.4002, Eqs. (2), (4))

$$
\frac{d\rho}{dz}=-\frac{i}{\hbar v}\left(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger\right)
+\sum_j A_j\rho A_j^\dagger,\qquad
H_{\rm eff}=P(H-\tfrac{i\hbar v}{2}\Gamma)P,
$$

with $A_j$ the retained parts $P(\kappa')J_jP(\kappa)$ of the jumps (lattice,
string or kick, after carrier relabelling) and all other outputs absorbed.

**Waiting-time unraveling** (same source, Eq. (8)). From a normalized state
$\psi_k$ at the last jump depth $z_k$, $\tilde\psi(z)=e^{-iH_{\rm eff}(z-z_k)/\hbar v}\psi_k$;
the next jump depth $s$ has survival $\Pr(s>z)=\|\tilde\psi(z)\|^2$ and rate
density $-\tfrac{d}{dz}\|\tilde\psi\|^2=\langle\tilde\psi|\Gamma|\tilde\psi\rangle$;
the branch $A_j$ (continue with $A_j\tilde\psi/\|A_j\tilde\psi\|$) has
probability $\|A_j\tilde\psi\|^2/\langle\tilde\psi|\Gamma|\tilde\psi\rangle$,
every other branch terminates.

**Claim 1 (observation).** Let $a(z)$ be 1 when the trajectory has not
terminated by $z$, and $\hat\psi(z)=\tilde\psi(z)/\|\tilde\psi(z)\|$ on the
current segment. Then

$$
\mathbb E\!\left[w\,a(z)\,\hat\psi(z)\hat\psi(z)^\dagger\right]=\rho(z),
$$

with $w$ the entrance weight ($\rho(0)=w\,\psi_0\psi_0^\dagger$).
*Proof.* Condition on $\psi_k$ at $z_k$. No jump in $(z_k,z]$ has probability
$\|\tilde\psi(z)\|^2$, so that event contributes
$\|\tilde\psi\|^2\hat\psi\hat\psi^\dagger=\tilde\psi\tilde\psi^\dagger$: the
no-jump propagation of $\psi_k\psi_k^\dagger$. A recycling jump at
$s\in(z_k,z]$ into branch $j$ has density
$\langle\tilde\psi(s)|\Gamma|\tilde\psi(s)\rangle\cdot\|A_j\tilde\psi(s)\|^2/\langle\tilde\psi(s)|\Gamma|\tilde\psi(s)\rangle\,ds$
and restarts from $A_j\tilde\psi(s)/\|A_j\tilde\psi(s)\|$, so by induction on the
number of jumps its contribution is the no-jump propagation from $s$ to $z$ of
$A_j\tilde\psi(s)\tilde\psi(s)^\dagger A_j^\dagger\,ds$. Summing over jump
numbers is the Dyson series of the model in the no-jump/jump split, i.e.
$\rho(z)/w$. Terminating branches contribute nothing. Because the model is
block diagonal in the carrier, the ensemble average also sums over carriers.

The hook evaluates $\hat\psi$ at fixed checkpoints by the same deterministic
$e^{-iH_{\rm eff}\Delta z/\hbar v}$ the waiting search uses, so it draws no
random numbers; the trajectory law is unchanged. A checkpoint equal to a jump
depth is assigned to the post-jump segment (the event tallies count a
terminal event at depth $d$ as terminated at $z\ge d$; both conventions
agree). The amplitude is reported in the eigenbasis of $H(\kappa)$ with its
energies.

**Claim 2 (observables).** With $c$ the eigenbasis amplitude of $\hat\psi$:
retained $R(z)=w\,a$, bound $N_B(z)=w\,a\sum_{E_\alpha<U_{\max}}|c_\alpha|^2$,
exposure $X_B(z)=\int_0^zN_B\,dz'$. By Claim 1 their means are
$\operatorname{Tr}\rho$, $\operatorname{Tr}\Pi_B\rho$ and $\int_0^z\operatorname{Tr}\Pi_B\rho$,
with $\Pi_B$ the below-$U_{\max}$ spectral projector of $H(\kappa)$ in each
fibre. $X_B$ uses the trapezoid rule on the checkpoint grid (61 points, 500 Å),
error $O(\Delta z^2\,\max|N_B''|)$ per unit depth; it is a grid quantity and is
reported with its grid.

**Claim 3 (paired errors).** Each incident has its own jump and measurement
streams keyed by (seed, incident ID), so per-incident observables are i.i.d.
across incidents within an arm. Two arms run with the same seed and IDs give
per-incident pairs $(x_i,y_i)$; $d_i=x_i-y_i$ are i.i.d. and
$\bar d\pm s_d/\sqrt N$ is the paired mean and SEM, valid whatever the
correlation between arms. Relative deficits $1-\bar x/\bar y$ use the
delta-method SEM of $r_i=(x_i-\hat r y_i)/\bar y$ with $\hat r=\bar x/\bar y$.

**Return deficit (definition, not a claim of physical escape).** The
reference arm uses $P=R$: the energy cut is removed, so every window state is
retained; outputs outside the window still terminate in both arms (same
outgoing box). The deficit $D_B=N_B^{\rm ref}-N_B^{\rm abs}$ (and the exposure
deficit) is the bound weight the absorbing cut loses at a fixed window. It
contains (i) return: weight that left the cut through a jump and later
re-entered below $U_{\max}$; (ii) the coherent no-jump coupling
$P\Gamma Q$ across the cut, which the absorbing generator compresses away;
and (iii) jump interference $\sum_j PJ_j(\rho_{PQ}+\rho_{QP})J_j^\dagger P$,
by which reference-arm $P$–$Q$ coherence is carried into $P$ by a recycled
jump. (ii) and (iii) have no fixed sign, so $D_B$ is not a nonnegative
return flux. It is the
same quantity as the secular study's deficit, now non-secular, with
continuum strings, electronic kicks and moving carriers.

## Assumptions

- Absorbing no-return fibred model and its jump families
  (`cr-axial-fiber-unraveling`, `cr-axial-continuum-tds-jumps`).
- The reference itself is window dependent (it drops everything that leaves
  the order window); compare `ref` and `ref-m8`.
- Bound means $E<U_{\max}$ of $H(\kappa)$ (cell maximum of the axial potential).

## Limiting cases

- Cut above every window level: absorbing arm = reference arm incident by
  incident; $D_B\equiv0$ with zero paired SEM.
- No channels: $a\equiv1$, $\hat\psi$ is the unitary evolution of the entrance;
  $N_B$ constant.
- Exactly proportional arms: relative deficit equals the factor with zero
  paired SEM (tested).

## Checks (task owner)

- `tests/montecarlo/test_channeling_axial_fiber_unraveling.py::test_observation_draws_no_random_numbers_and_tracks_survival`:
  events identical with and without observation; checkpoints reported exactly
  for surviving incidents; normalized amplitudes; retained energies.
- `...::test_observed_states_reproduce_the_dense_retained_density`: ensemble
  populations vs the independent dense `absorbing_axial_model` with lattice and
  uniform-azimuth kicks (3,000 incidents, 4.5σ).
- `tests/montecarlo/test_channeling_axial_outgoing_convergence.py`: bound
  weight, terminal tallies, trapezoid exposure, paired SEM, proportional limit,
  seed/ID and chunk-gap refusals, observed-vs-event survival refusal.

## Verification request

Fresh context: re-derive Claim 1 (including the checkpoint-at-jump
convention), Claim 3 and the delta-method SEM; confirm the hook cannot change
the trajectory law; confirm the reference arm is the stated $P=R$ model
(cut replaced by $+\infty$, same box) and that the deficit's coherent part is
correctly described as sign-indefinite.

## Fresh-context verification (2026-11-01)

Independent verifier (fresh context; derivation written from the ledger row
and this page's statement before the implementation body was read).
Verdict: **`anchored`** (re-derivation matches; anchors green). One minor,
non-blocking descriptive finding on the deficit decomposition (below).

### Independent derivation

**Claim 1.** Let $\psi_k$ be the normalized state at the start $z_k$ of a
segment, $\tilde\psi(z)=e^{-iH_{\rm eff}(z-z_k)/\hbar v}\psi_k$ and
$\Gamma_P=P\Gamma P$ with $\Gamma$ the *total* loss (every output, recycling
or terminating). Then $\frac{d}{dz}\|\tilde\psi\|^2=-\langle\tilde\psi|\Gamma_P|\tilde\psi\rangle$,
so $\|\tilde\psi(z)\|^2$ is the survival of the waiting law. Define
$\sigma(z)=\mathbb E[a(z)\,\hat\psi\hat\psi^\dagger\mid\psi_k]$. Splitting on
the first jump after $z_k$:

$$
\sigma(z)=\tilde\psi(z)\tilde\psi(z)^\dagger
+\int_{z_k}^{z}\sum_j\|A_j\tilde\psi(s)\|^2\,
\mathcal T_{z\leftarrow s}\!\left[\frac{A_j\tilde\psi(s)\tilde\psi(s)^\dagger A_j^\dagger}{\|A_j\tilde\psi(s)\|^2}\right]ds,
$$

where the jump density $\langle\tilde\psi|\Gamma_P|\tilde\psi\rangle$ cancels
against the branch denominator, terminating branches carry $a=0$, and
$\mathcal T_{z\leftarrow s}$ is the same expectation map restarted at $s$
(a linear map on the restart density). This is the Duhamel/Dyson identity
of $\dot\rho=-\tfrac{i}{\hbar v}(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger)+\sum_jA_j\rho A_j^\dagger$
split into no-jump propagation plus recycled jumps, so $\sigma$ equals the
propagated $\psi_k\psi_k^\dagger$ and $w\,\sigma$ from $z=0$ is $\rho(z)$.
The entrance is $w=\|P\psi_{\rm in}\|^2$, $\psi_0=P\psi_{\rm in}/\sqrt w$, so
$\rho(0)=P\psi_{\rm in}\psi_{\rm in}^\dagger P$. Carrier block diagonality
lets the average sum over fibres. A checkpoint exactly at a jump depth has
probability zero (continuous jump depths), so the convention does not affect
expectations; assigning it to the post-jump segment makes a terminal event
at $d$ dead at $z=d$, which matches the tallies' `depth <= z` /
`depth > z` split.

**Claim 3.** With per-incident streams keyed by (seed, ID), the pairs
$(x_i,y_i)$ are i.i.d. across $i$, hence so are $d_i=x_i-y_i$:
$\bar d\pm s_d/\sqrt N$ ($s_d$ with ddof 1), whatever the within-pair
correlation. For $\hat r=\bar x/\bar y$, linearizing about the means,
$\hat r-r\approx(\bar x-r\bar y)/\mu_y$, so

$$
\operatorname{Var}\hat r\approx\frac{\operatorname{Var}x-2r\operatorname{Cov}(x,y)+r^2\operatorname{Var}y}{N\mu_y^2},
$$

estimated by the sample variance of $r_i=(x_i-\hat r y_i)/\bar y$ over $N$.
$1-\hat r$ has the same SEM. Exactly proportional arms give $r_i\equiv0$.

**Deficit structure.** Write $Q=R-P$ and $\Gamma$ on $R$. Projecting the
reference ($P=R$) equation onto $PP$ (using $[H(\kappa),P]=0$) and subtracting
the absorbing equation, $\delta=\rho^{\rm ref}_{PP}-\rho^{\rm abs}$ obeys
$\dot\delta=\mathcal L_{\rm abs}\delta+s_{\rm ret}+s_{\rm jx}+s_{\rm coh}$ with

$$
s_{\rm ret}=\sum_jPJ_j\rho_{QQ}J_j^\dagger P\succeq0,\quad
s_{\rm jx}=\sum_jPJ_j(\rho_{PQ}+\rho_{QP})J_j^\dagger P,\quad
s_{\rm coh}=-\tfrac12\left(P\Gamma Q\rho_{QP}+\rho_{PQ}Q\Gamma P\right),
$$

($\rho=\rho^{\rm ref}$) and $\delta(0)=P\psi_{\rm in}\psi_{\rm in}^\dagger P-P\psi_{\rm in}\psi_{\rm in}^\dagger P=0$.
$\mathcal L_{\rm abs}$ is CP, so the return part of $D_B={\rm Tr}\,\Pi_B\delta$
is nonnegative; $s_{\rm coh}$ and $s_{\rm jx}$ have no fixed sign. Example for
$s_{\rm coh}$ (no recycling, $H\to0$, $\Pi_B=P=|1\rangle\langle1|$,
$\Gamma=\gamma\begin{pmatrix}1&1\\1&1\end{pmatrix}$): the bright entrance
$(|1\rangle+|2\rangle)/\sqrt2$ gives $D_B=\tfrac12(e^{-2\gamma z}-e^{-\gamma z})<0$,
the dark one $(|1\rangle-|2\rangle)/\sqrt2$ gives $\tfrac12(1-e^{-\gamma z})>0$.
So $D_B$ is fixed-window accounting, not a return flux. In the study
$\Pi_B\subset P$: the Si [001] cut (92.9–96.9 eV over sampled carriers, M = 6
and 8) lies far above $U_{\max}=8.19$ eV, so $D_B(0)=0$.

### Code comparison

- `_observe_segment` is called once per waiting segment with the segment's
  start `c`, `start`, `fiber` and `kappa` (before `_jump` updates them), and
  reports `fiber.evolve(c, z - start)` normalized for `start <= z < stop`, plus
  `z == max_depth` on the final retained segment. `evolve` is the same
  deterministic eigen-propagator the waiting search uses; the hook draws no
  random numbers and touches no engine state (the fibre cache is not
  queried). Next segment starts at the jump depth, so a checkpoint at a
  recycling-jump depth is post-jump; at a terminal depth it is unobserved.
- `c` is in the orthonormal retained eigenbasis `v` of $H(\kappa)$, so
  $\|c\|=\|\hat\psi\|$ and `fiber.energies` = `energies[keep]` are its
  eigenvalues; `bound_weight` sums $|c_\alpha|^2$ for $E_\alpha<U_{\max}$.
- `run_arm` `ref`: `dataclasses.replace(channels, cut=lambda e: np.inf)`;
  `cut` is read only in `_Engine.fiber` (`keep = energies < cut`), so
  $P(\kappa)=R$ in every fibre, with the same window, box, channels, step,
  seed and IDs; `_split` then sends box − R to outgoing and outside-box to
  unresolved, as in the absorbing arm.
- `observables`: $N_B=w\cdot$`bound` (zero when dead), trapezoid exposure from
  0, survival from the hook; outgoing/unresolved use `depth <= z`.
  `paired`: per-incident differences, ddof 1; relative entry
  `resid = (xa - ratio xb)/mean(xb)`, ddof 1, $/\sqrt N$ — the delta-method
  formula above. `return_deficit` negates `paired(base, ref)` so $D=$ ref −
  base, and the relative entry is $1-X^{\rm abs}/X^{\rm ref}$.

### Numerical checks (own scripts, not the tests' helpers)

| Check | Model | Result |
|---|---|---|
| Hook ensemble vs own dense GKSL (`expm` of the vectorized generator; own convolution Kraus operators $C_j(p-p')$, own loss $\sum L_j^\dagger L_j$ + extra PSD) | lattice-only fibre, M = 1, D = 2, J = 2, random non-eigen entrance, random spectrum, cut 3.0, $U_{\max}=1.7$, $\hbar v=1.3$; abs and ref arms; 100k incidents | every real and imaginary element of $\rho(z)$ at $z=0,0.7,3,10$: worst $\lvert z\rvert=2.43$ over all elements of both arms; ${\rm Tr}\,\rho$ and $N_B$ within $1.1\sigma$ |
| Paired $D_B$ vs dense | same | $z=0.7,3,10$: $+0.00692\pm0.00052$ / $+0.00702$; $+0.01953\pm0.00061$ / $+0.01976$; $+0.00630\pm0.00025$ / $+0.00621$; $D_B(0)=0$ exactly; paired SEM 1.4–2.4× below unpaired |
| Bitwise event identity with/without `observe` | Si [001] 5 MeV, M = 3, B = 4, lattice columns + continuum strings + electronic kicks, abs and ref, 120 incidents, 41 checkpoints | observed and unobserved event lists equal for both arms (TDS/kick jumps 145/355 abs, 154/405 ref); observed survival equals terminal events at every checkpoint; $\lvert\lVert c\rVert^2-1\rvert\le7\times10^{-16}$; energies below cut |
| Terminal-depth checkpoint | dense model, 29 terminal incidents | checkpoint at `depth_ang` never observed; at `nextafter(depth, 0)` always observed |
| `paired`/`return_deficit` vs own SEMs | synthetic correlated paired arms, 4,000 incidents | means and SEMs agree to roundoff (proportional end point: script 0, own $2\times10^{-9}$) |
| Delta-method coverage | 300 replicas of $N=1000$ vs $4\times10^5$-sample truth | $z$ mean 0.04, sd 1.05, 92% within 1.96 |
| Deficit decomposition | 8 random dense 7-level models | $D_B=$ ret + jx + coh to $10^{-16}$; $s_{\rm jx}$ contribution negative in 4 of 8 (e.g. ret $+0.0037$, jx $-0.0090$, coh $+0.0064$) |
| Owner anchors | `test_channeling_axial_fiber_unraveling.py`, `test_channeling_axial_outgoing_convergence.py` | 25 passed |

### Findings

- **Minor (descriptive, non-blocking).** The return-deficit paragraph and the
  ledger claim name two parts, return through a jump and the coherent
  no-jump $P\Gamma Q$ coupling. There is a third, sign-indefinite part: the jump
  interference $s_{\rm jx}=\sum_jPJ_j(\rho_{PQ}+\rho_{QP})J_j^\dagger P$, which
  carries reference-arm $P$–$Q$ coherence into $P$ through a recycled jump.
  It is not "weight that left the cut and re-entered", and in random models it
  is as large as the return term. The conclusion stands (sign-indefinite
  accounting, not a return flux); suggest the task owner add this term to the
  decomposition text here, in the ledger claim and in the
  `checks/axial_outgoing_convergence.py` docstring ("the absorbing arm drops
  both").
- **Minor.** Checkpoints below 0 or above `max_depth_ang` are silently
  skipped, and duplicate checkpoints are reported twice; `run_arm` is
  unaffected (it builds the grid and its index map).
