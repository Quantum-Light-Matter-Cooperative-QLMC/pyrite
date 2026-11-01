# Terminal-event unraveling of the absorbing axial jump model

## Independent derivation

Claim: `cr-axial-absorbing-jump-unraveling`. This fresh verification context
completed this section from the ledger row, the cited source and the anchored
dense oracle `cr-axial-absorbing-jump-instrument` before reading the
implementation body. The source is
[Castin, Dalibard and Mølmer, arXiv:0805.4002](https://arxiv.org/abs/0805.4002),
Eqs. (4) (waiting-time law) and (8) (jump selection). The claim is finite
sampling algebra on a supplied model; it establishes no physical escape,
matching surface or admission.

### Dense law

Use the notation of the dense oracle: Hermitian $H$ in eV, retained jumps
$A_a$ and outgoing jumps $B_b$ in Å$^{-1/2}$, a positive unresolved effect
$E_O$ in Å$^{-1}$, $h=\hbar v$ in eV Å, and an initial positive density
$\rho_0$ of incident weight $t_0=\operatorname{tr}\rho_0\le1$. With

$$
\Gamma=\sum_a A_a^\dagger A_a+\sum_b B_b^\dagger B_b+E_O,
$$

the retained density obeys

$$
\frac{d\rho}{dz}=-\frac{i}{h}[H,\rho]+\sum_a A_a\rho A_a^\dagger
-\tfrac12\{\Gamma,\rho\},
$$

with $\sigma_F=\sum_b B_b\rho B_b^\dagger$, $Q_F(z)=\int_0^z\sigma_F\,ds$ and
$U_O(z)=\int_0^z\operatorname{tr}(E_O\rho)\,ds$.

### No-jump propagator

Define $H_{\rm eff}=H-\tfrac{i}{2}h\Gamma$ (eV). Then

$$
-\frac{i}{h}\left(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger\right)
=-\frac{i}{h}[H,\rho]-\tfrac12\{\Gamma,\rho\},
$$

so the dense equation is
$\dot\rho=-\tfrac{i}{h}(H_{\rm eff}\rho-\rho H_{\rm eff}^\dagger)+\sum_aA_a\rho A_a^\dagger$.
The no-jump amplitude is $\tilde\psi(z)=e^{-iH_{\rm eff}z/h}\psi$. The sign is
fixed: to first order in $z$, $e^{-iH_{\rm eff}z/h}\approx1-iHz/h-\Gamma z/2$,
which damps. The factor $\tfrac12$ appears once on $\Gamma$ in $H_{\rm eff}$.

### Monotone norm

$$
\frac{d}{dz}\lVert\tilde\psi\rVert^2
=\frac{i}{h}\langle\tilde\psi|H_{\rm eff}^\dagger-H_{\rm eff}|\tilde\psi\rangle
=-\langle\tilde\psi|\Gamma|\tilde\psi\rangle\le0,
$$

because $\Gamma\ge0$. The norm is nonincreasing, so for uniform
$u\in(0,1)$ the first step whose end norm is $\le u$ brackets a unique crossing
(up to plateaus where $\Gamma\tilde\psi=0$, which cannot hold the norm at $u$
from above). Bracketing plus a root solve is therefore valid.

### Terminal-event expectation

Start from a normalized $\psi$ at depth $z_0$. The first-event density is
$p(s)=-\frac{d}{ds}\lVert\tilde\psi(s)\rVert^2=\langle\tilde\psi|\Gamma|\tilde\psi\rangle$,
which is sampled by solving $\lVert\tilde\psi(s)\rVert^2=u$ (Eq. 4). The
channel probabilities given an event at $s$ are
$\lVert A_a\tilde\psi\rVert^2$, $\lVert B_b\tilde\psi\rVert^2$ and
$\langle\tilde\psi|E_O|\tilde\psi\rangle$ divided by
$\langle\tilde\psi|\Gamma|\tilde\psi\rangle$ (Eq. 8); these sum to one by the
definition of $\Gamma$, and normalizing $\tilde\psi$ first leaves the ratios
unchanged. Multiplying density by probability, the event rate into channel
$B_b$ at $s$ with post-event unnormalized state $B_b\tilde\psi$ is
$B_b|\tilde\psi\rangle\langle\tilde\psi|B_b^\dagger$, and similarly for
$A_a$ and $E_O$. Recycling on $A_a$ restarts the same process from
$A_a\tilde\psi/\lVert A_a\tilde\psi\rVert$ with a fresh $u$; trajectory weight
stays $t_0$.

Let $\bar\rho(z)$ be the expectation of $|\psi\rangle\langle\psi|$ over
trajectories that have not terminated by $z$. Summing the Dyson series over
recycling histories,

$$
\bar\rho(z)=\tilde\rho(z)+\int_0^z ds\,
\mathcal S(z-s)\Big[\sum_aA_a\bar\rho(s)A_a^\dagger\Big],
\qquad \mathcal S(z)X=e^{-iH_{\rm eff}z/h}Xe^{iH_{\rm eff}^\dagger z/h},
$$

which is the integral form of the dense equation. Hence $\bar\rho=\rho$,
$t_0\,P(\text{retained at }z)=\operatorname{tr}\rho(z)$, and

$$
t_0\,E[\mathbf 1\{B_b\text{ terminal by }z\}]=\int_0^z
\operatorname{tr}B_b\rho B_b^\dagger\,ds,\qquad
t_0\,E[\mathbf 1\{E_O\text{ by }z\}]=U_O(z).
$$

Summing over $b$ gives $\operatorname{tr}Q_F(z)$.

### Mixed entry

Write $\rho_0=\sum_k\lambda_k|k\rangle\langle k|$ by eigendecomposition. Draw
$k$ with probability $\lambda_k/t_0$ and attach weight $t_0$. By linearity of
the dense equation the trajectory average reproduces $\rho_0$ exactly,
including its off-diagonal coherences in any other basis; a basis-diagonal
draw would not.

### Joint Weyl law

Given a $B_b$ event, the Weyl instrument applied to $\phi=B_b\tilde\psi$ gives
outcome $(a,b')$ with probability
$\lvert\langle g_{ab'}|\phi\rangle\rvert^2/(M\lVert\phi\rVert^2)$, with
$g_{ab'}(n)=g(n-a)e^{2\pi i b'\cdot n/N}$ and $M=n_xn_y$
(`cr-axial-phase-space-instrument`). Combining with the event rate,

$$
t_0\,P(\text{outgoing by }Z,\ a,b')=\frac1M\int_0^Z
\langle g_{ab'}|\sigma_F(z)|g_{ab'}\rangle\,dz
=\frac1M\langle g_{ab'}|Q_F(Z)|g_{ab'}\rangle.
$$

The C-order reshape of $B_b\psi$ must match the oracle's C-order outgoing
index.

### Eigen propagator

If $H_{\rm eff}=R\,\mathrm{diag}(\lambda)R^{-1}$, then
$e^{-iH_{\rm eff}z/h}=R\,e^{-i\lambda z/h}R^{-1}$, with
$\operatorname{Im}\lambda\le0$ by the numerical-range bound from $\Gamma\ge0$.
$H_{\rm eff}$ is non-normal whenever $[H,\Gamma]\ne0$, and it can be defective
at exceptional points. The relative error of the eigen form scales with
$\kappa(R)\epsilon$. A dense `expm` fallback is required when $\kappa(R)$ is
large or $R$ is non-finite.

## Filters

- **Units:** $h\Gamma$ is eV, $z/h$ is eV$^{-1}$, so the exponent is
  dimensionless; rates $\lVert A\psi\rVert^2$ are Å$^{-1}$. Pass.
- **Limits:** zero jumps give $\Gamma=0$, unit norm and only retained events.
  For scalar $\Gamma=\gamma$ the waiting law is $e^{-\gamma z}$. Pure-recycling
  models conserve trace. Pass.
- **Signs and conventions:** $-i$ in the propagator, $+\tfrac{i}{2}$ moved
  into $-\tfrac{i}{2}h\Gamma$, forward-FFT Weyl sign and C-order are
  consistent with the anchored oracle and measurement. Pass.

## Source-to-code comparison

Implementation: `src/pyrite/montecarlo/channeling/axial_jump_unraveling.py`.

- `_prepare` forms `loss` as $E_O+\sum A^\dagger A+\sum B^\dagger B$ and
  `h_eff = h - 0.5j * hbar_v * loss` (line 136). This matches
  $H-\tfrac{i}{2}h\Gamma$ in eV with $h$ in eV Å.
- `_Model.evolve` applies `expm(-1j * depth / hbar_v * h_eff)` or
  `right @ (exp(-1j*depth/hbar_v*lam) * (inverse @ psi))`. Both match the
  derived propagator. `_eigen` falls back when $R$ is non-finite or
  $\kappa(R)>10^8$ (line 80).
- `_waiting_depth` compares the absolute squared norm of the evolved
  normalized segment state with $u$ = `rng.random()` (line 178), then calls
  `brentq` inside the first bracketing step. This matches Eq. 4 and the
  monotonicity argument.
- `_trajectory` normalizes the state, builds rates
  `[‖A_a ψ‖², ‖B_b ψ‖², ⟨ψ|E_O|ψ⟩]` and chooses by `rates / total`. A
  retained index (`channel < len(retained)`, line 200) recycles
  `A_a ψ / ‖A_a ψ‖`, keeping complex phases. The last index terminates as
  `unresolved`; the middle indices measure the C-order reshaped `B_b ψ`. With
  no supplied jumps, the single zero matrix in each stack has rate zero and is
  never chosen, so the indexing remains consistent.
- The mixed entry uses `eigh(rho)`, positive eigenvalues as probabilities, and
  carries `weight` $=\sum\lambda_k^+$ unchanged on every event.
- Streams: jump draws use `SeedSequence((seed, 0x6230001, 1, id))`. The single
  measurement uses `phase_space_stream` (domain `0x6205053`), created once per
  incident. The streams are independent and do not depend on processing order.
  `sample_phase_space` normalizes $\phi$ internally.
- The `retained` record has `depth_ang = max_depth_ang` and is excluded from
  the cumulative tallies. `event_tallies` sets retained to weight minus both
  cumulative tallies, which equals $\operatorname{tr}\rho(z)$ for
  $z\le$ `max_depth_ang`.

The derivation and the implementation match.

### Minor finding (no physics divergence)

`event_tallies` (line 292) does not refuse depths beyond the trajectories'
`max_depth_ang`. For such $z$, trajectories still retained at `max_depth_ang`
are counted as retained, not resolved, so the outgoing and unresolved tallies
are biased low against the dense oracle. The scenario is caller misuse, but
the function cannot detect it, because events do not carry `max_depth_ang`
separately from retained depths.

The ledger `Code` field lists `unraveling_stream` and `_trajectory`, but
neither docstring carries `Validation: cr-axial-absorbing-jump-unraveling`.
At `4dbc7ffc0`, `tests/dev/test_validation_ledger.py::test_ledger_python_owners_have_matching_markers`
fails for these two symbols. This is a traceability finding, not a physics
divergence; the verifier did not repair the code.

## Independent numerical evidence (2026-10-31)

A scratch reference written by this context (not using implementation
helpers) integrated $[\rho,\int\rho]$ with DOP853 (`rtol=1e-11`). It formed
$\operatorname{tr}Q_F$, $U_O$ and $\operatorname{tr}\rho$, and built the Weyl
weights from explicit $g_{ab}$ vectors. The test model had $n=4$, a $3\times2$
outgoing grid, $h=2000$ eV Å, random Hermitian $H$ ($\sim$ 30 eV scale), two
complex retained jumps (one upper-triangular), two outgoing jumps, and a
positive $E_O$. The resulting $\lVert[H_{\rm eff},H_{\rm eff}^\dagger]\rVert\approx4\times10^3$,
so the model is strongly non-normal. The entry was a rank-2 coherent density
with trace 0.9. Dense closure held to $2\times10^{-16}$, and $\sum q$ equaled
$\operatorname{tr}Q_F(Z)$ to $10^{-16}$.

- Three seeds (40k, 100k and 100k incidents, $Z=150$ Å, mean 0.9 recycles,
  maximum 10) matched the retained, outgoing and unresolved tallies at five
  depths. The 100k runs fell within $|z|\le2.0$. The 40k run's correlated
  cumulative $\approx2\sigma$ offset did not reproduce, and reversed sign in
  seed 22.
- Joint $(a,b')$ outcome $\chi^2$ with 35 degrees of freedom gave
  $p=0.86,\,0.55,\,0.99$.
- At an exceptional point $H_{\rm eff}=[[-i\gamma,\gamma/2],[\gamma/2,0]]$,
  `_eigen` returned `None` and `evolve` equaled `expm` exactly. On the main
  model ($\kappa(R)=2.65$), the eigen form agreed with `expm` to
  $2\times10^{-16}$, and the no-jump norm was monotone over 300 Å.
- `pyrite-dev test tests/montecarlo/quantum/test_channeling_axial_jump_unraveling.py`:
  11 passed.

Verdict: `rederived`. The regression file satisfies the anchor requirement for
`anchored`. Human sign-off remains in #277.
