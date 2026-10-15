# `cr-mcwf-unravelling` — quantum-jump unravelling of the channeling Lindblad equation

Verifier: fresh context. Ledger: `docs/validation/ledger-channeling.md`. Code: `src/pyrite/montecarlo/channeling/mcwf.py::{MCWFSolver,MCWFSolver.run,mcwf_stream_key}`. It shares its operators with `cr-density-matrix-split-operator` ([write-up](cr-density-matrix-split-operator.md)) and its kernel with `cr-lindblad-jump-operators`. Sources: Dalibard, Castin & Mølmer, PRL 68, 580 (1992); Mølmer, Castin & Dalibard, JOSA B 10, 524 (1993); Carmichael (1993).

Independence: the derivation and the jump-operator, ensemble and statistics checks below were set up from the ledger row and the docstrings. The draw-order replica (check 3) and the shared-state test (finding 3) were written after reading the body.

## Claim

Each electron carries a pure state $\psi$ in one Bloch block. $\psi$ evolves unnormalized under the density solver's half-step propagators, and a jump occurs when $\lVert\psi\rVert^2$ falls below a uniform threshold $r$. The jump is placed at the step midpoint. The jump operators come from $K_r=\sum_s\lambda_su_su_s^\dagger$, with $L_{rs}=\sqrt{\lambda_s}\sum_hu_s[h]e^{i(\kappa_r+hg)x}$ mapping block $b\to b+r$. Channel $(r,s)$ is chosen with probability $\lVert PL_{rs}\hat\psi\rVert^2/\langle\hat\psi\rvert C(x,x)\lvert\hat\psi\rangle$. The remainder is overflow. The collapse keeps the electron with probability $\lVert P_S\phi\rVert^2/\lVert\phi\rVert^2$; otherwise it dechannels into $n\in F$ with weight $\lvert\langle n\vert\phi\rangle\rvert^2$. The survivor mean of $\lvert\hat\psi\rangle\langle\hat\psi\rvert$ equals $\rho_S$.

## Independent derivation

**Jump operators.** $K_r$ is Hermitian and PSD (validated upstream), so $K_r=\sum_s\lambda_su_su_s^\dagger$ with $\lambda_s\ge0$. Then

$$
C(x,x')=\sum_{r,i,j}K_r[i,j]e^{ik_ix}e^{-ik_jx'}=\sum_{r,s}L_{rs}(x)L_{rs}^*(x'),\qquad L_{rs}(x)=\sqrt{\lambda_s}\sum_iu_s[i]e^{i(\kappa_r+h_ig)x}.
$$

On a block-$b$ state $\psi=\sum_m\psi[m]\lvert\kappa_b+h_mg\rangle$, multiplication by $e^{i(\kappa_r+hg)x}$ is a unitary shift to $\kappa_b+\kappa_r+(h+h_m)g$. That momentum lies in block $(b+r)\bmod n$ at order $o=h_m+h+c$, where the carry is $c=\lfloor(b+r)/n\rfloor\in\{0,1\}$. Hence

$$
(PL_{rs}\psi)[o]=\sqrt{\lambda_s}\sum_hu_s[h]\,\psi[o-h-c],\qquad \lvert o\rvert\le H .
$$

Since $\sum_{r,s}PL_{rs}\psi\psi^\dagger L_{rs}^\dagger P=P J(\lvert\psi\rangle\langle\psi\rvert)P$, the kept channels sum to the in-band jump trace, and the overflow is $\langle\psi\rvert D\lvert\psi\rangle-\sum\lVert PL\psi\rVert^2\ge0$.

**No-jump evolution and waiting time.** $\partial_z\lVert\tilde\psi\rVert^2=-\langle\tilde\psi\rvert P_SDP_S\lvert\tilde\psi\rangle=-\langle\tilde\psi\rvert D\lvert\tilde\psi\rangle$ for $\tilde\psi\in S$. This equals the full loss used to normalize the channel probabilities, so the overflow residual is a proper probability. With $r\sim U(0,1)$ drawn when the norm is 1, the conditional probability of a jump in a step that starts at norm $\nu_0$ is $(\nu_0-\nu_1)/\nu_0$, which is exact. The no-jump branch weight $\lVert\tilde\psi\rVert^2\,\hat\psi\hat\psi^\dagger=\tilde\psi\tilde\psi^\dagger$ reproduces $e^{A}\rho$.

**Midpoint placement.** $\nu_0-\nu_1=\int_0^{dz}\langle D\rangle\,dz'=dz\,\langle\tilde\psi_{1/2}\rvert D\lvert\tilde\psi_{1/2}\rangle+O(dz^3)$ by the midpoint rule. The jump branch therefore contributes $U_{1/2}\bigl[dz\sum_kL_k\rho_{1/2}L_k^\dagger\bigr]U_{1/2}^\dagger+O(dz^3)$, the density solver's first-order jump term. The collapse gives $\mathbb E[\text{survivor}]=\sum_k\lVert\phi_k\rVert^2\cdot\frac{\lVert P_S\phi_k\rVert^2}{\lVert\phi_k\rVert^2}\frac{P_S\phi_k\phi_k^\dagger P_S}{\lVert P_S\phi_k\rVert^2}=P_SJP_S$. The free flux is $\operatorname{tr}P_FJ$, distributed over $n\in F$ as ${\rm diag}(P_FJP_F)$, as in the density solver.

**One jump per step.** A jump found in a step is placed at the midpoint. A second threshold crossing in the second half-step is detected in the next step, so it is deferred rather than dropped. For a constant rate $c$ with $a=c\,dz$, the inter-jump interval is $dz\max(1,\lceil\tau/dz-\tfrac12\rceil)$ with $\tau\sim{\rm Exp}(c)$, which gives

$$
\frac{\langle\Delta\rangle}{1/c}=a\Bigl(1+\frac{e^{-3a/2}}{1-e^{-a}}\Bigr)=1+0.458\,a^2+O(a^3).
$$

The bias is second order, consistent with the density solver's $O(dz^2)$.

## Implementation diff

| Item | Code | Verdict |
| --- | --- | --- |
| Channels | `eigh` per residue; drops $\lambda<10^{-12}\lambda_{\max}$ and stores $\sqrt\lambda$, $u$ (mcwf.py:149–152) | matches. The dropped weight becomes overflow. |
| $PL_{rs}\psi$ | `lag = o − h − carry + H`, `carry = (b+r)//n`, target `(b+r) % n`, amplitudes times $\sqrt\lambda$ (mcwf.py:170–176) | matches the convolution above |
| Channel probability | $\lVert PL_{rs}\hat\psi\rVert^2$, overflow `max(loss − Σ, 0)` (mcwf.py:320–328) | matches, with $\sqrt\lambda$ counted once |
| Waiting time | `middle = U½ψ`, `end = U½ middle`; jump if `‖end‖² < r`; jump from normalized `middle`; survivor gets `U½`; new $r$ (mcwf.py:272–293) | matches |
| Collapse | `rng·total < stay`, survivor `P_S φ/√stay`; else $n\in F$ by cumulative weight (mcwf.py:179–191) | matches |
| Entry | samples $l$ from `entry_lattice`, then collapses; out of band counts as entry flux with a padding draw (mcwf.py:193–205) | matches the density `entry_state` |
| Survivor mean | $\sum_{\rm alive}\hat\psi\hat\psi^\dagger/N_{\rm launched}$ per block | matches $\rho_S$ normalization |
| RNG | Philox keyed by SplitMix64 of `(seed ^ salt) + φ·(i+1)`; per-electron order: entry pick, entry collapse, threshold, then per jump a channel, collapse(es) and threshold | matches the docstring |

## Independent numerical checks

Si(110), 17 MeV, 293 K, TDS plus electronic kernel unless stated.

**1. Jump-operator completeness.** For a random normalized state in each block, $\sum_{r,s}(PL_{rs}\psi)(PL_{rs}\psi)^\dagger$ from `_jump_candidates` was placed in its target blocks. I compared it with my dense $J(\lvert\psi\rangle\langle\psi\rvert)$ from the cr-density write-up's reference. The maximum relative error is $7.7\times10^{-16}$ (real kernel, $n=3$, $H=5$) and $6.9\times10^{-16}$ (random rank-2 $K_r$ in all residues, $n=4$, $H=4$, every carry path). `_loss` equals my $\langle\psi\rvert D\lvert\psi\rangle$ to $10^{-19}$.

**2. Ensemble against an independent reference (not the density solver).** The reference is my dense projected-model Liouvillian, with free and overflow fluxes integrated as augmented rows and exponentiated exactly. Setup: $n=3$, $H=5$, margin 5 eV, $dz=50$ Å, 1 µm, records every 0.2 µm. Survival, free+entry and overflow are binomial; band populations use the code's SEM.

| run | max $\lvert z\rvert$, survival | max $\lvert z\rvert$, free+entry | max $\lvert z\rvert$, overflow | max $\lvert z\rvert$, bands |
| --- | --- | --- | --- | --- |
| $l_0=0$, 4000 e⁻ | 1.31 | 1.04 | 1.51 | 2.20 |
| $l_0=2$, 4000 e⁻ | 2.34 | 2.23 | 0.91 | 1.98 |
| $l_0=2$, 16000 e⁻, seed 31 | 1.62 (at $z=0$) | 1.62 | 0.54 | — |
| $l_0=0$, 16000 e⁻, seed 32 | 1.51 (at $z=0$) | 1.51 | 1.51 | — |

The 4000-electron $l_0=2$ excursion is a fluctuation: at 16000 electrons the same quantity is at $+0.51\sigma$. Final trace at 16000 electrons: 0.6626 against 0.6607 ($\sigma=0.0037$), and 0.6595 against 0.6569.

**3. Waiting-time rule and draw order.** I used a pure-overflow uniform loss: `momentum_kernel = 0` and $C(x,x)=c=10^{-4}$ Å⁻¹, margin $10^7$ eV, $dz=100$ Å, 2 µm, 3000 electrons, seed 99. I regenerated each electron's Philox stream from `mcwf_stream_key` alone and took the third draw as the threshold $r$. The predicted dechanneling depth $(\lceil-\ln r/(c\,dz)\rceil-\tfrac12)dz$ matches the code's depth for all 3000 electrons, with 0 mismatches. Survival is 0.1340 against $e^{-cz}=0.13534$ ($z=-0.21$). The density solver's overflow is 0.864661 against $1-e^{-cz}=0.864665$.

**4. Jump statistics and one-jump-per-step.** Identity jump $C=c=10^{-3}$ Å⁻¹ (kept), 2000 electrons, 2 µm, so $cz=20$:

| $a=c\,dz$ | jumps per electron | deferral law $20/(\langle\Delta\rangle c)$ |
| --- | --- | --- |
| 0.1 | 19.85 | 19.91 |
| 0.5 | 18.07 | 18.18 |
| 1.0 | 14.56 | 14.78 |

The residual 0.3–1.5% is the first and last interval, which the stationary law ignores. Jumps are deferred, not lost, and the rate bias is $O(a^2)$. For the physical kernel at $dz=50$ Å, $a\approx0.02$, giving a bias of about $2\times10^{-4}$.

**5. Stream keys.** 200 seeds × $10^4$ electrons give $2\times10^6$ distinct keys. `mix` is a bijection on 64 bits, so keys collide only if $(s\oplus{\rm salt})-(s'\oplus{\rm salt})\equiv\varphi\,(i'-i)\pmod{2^{64}}$, which is not met in this range. Batch independence is pinned by `test_runs_are_reproducible_and_batch_independent`.

## Findings

1. **Wording (ledger row and module docstring).** The row defines $L_{rs}$ with $\sqrt{\lambda_s}$ and then states the channel probability as "$\lambda_s\lVert PL_{rs}\hat\psi\rVert^2/\ldots$", which counts $\lambda_s$ twice. The code (mcwf.py:176, 320) is correct: $\lVert PL_{rs}\hat\psi\rVert^2$ with $\sqrt{\lambda_s}$ applied once. Fix the text to $\lVert PL_{rs}\hat\psi\rVert^2/\langle\hat\psi\rvert C(x,x)\lvert\hat\psi\rangle$, or to $\lambda_s\lVert Pe_{rs}\hat\psi\rVert^2$ with unit-weight $e_{rs}$.
2. **Wording.** "At most one jump per step; the neglected double-jump probability is $O((\text{rate}\,dz)^2)$" is imprecise. A second crossing is deferred to the next step's midpoint, not neglected. The ensemble effect is a relative rate bias of $0.458(\text{rate}\,dz)^2$ (check 4), still second order.
3. **Robustness (not the physics).** `MCWFSolver.__init__` resynchronizes the shared `DensityMatrixSolver` to the entry basis (mcwf.py:142–143), but `run` does not (mcwf.py:227–230). A later `DensityMatrixSolver.propagate(..., stopping_eV_per_ang>0)` leaves `solver.basis` at the degraded $\gamma$ and `_scale` $\ne1$. A subsequent `run` then mixes that basis and kernel scale with the entry $\hbar v_0$. Failing case: `m = MCWFSolver(s)`, `s.propagate(2e4, 200, stopping_eV_per_ang=1.0)` (basis $\gamma$ 34.2294 against $\gamma_0=34.2682$), then `m.run(200, 2000, 100, seed=1)`. Its `final_density` differs from a fresh solver's by up to 0.445. Move the resynchronization into `run`, or document that the solver must not be propagated with energy loss after construction.
4. **Scope (already in Notes).** MCWF has no energy loss. Compare it only with density runs at `stopping_eV_per_ang=0`.

## Verdict

`rederived`. The unravelling reproduces the projected Lindblad dynamics:

- The jump operators sum exactly to $J$.
- The ensemble agrees with an independent dense integration within statistics at 16000 electrons.
- The waiting-time rule and RNG order reproduce exactly.

Findings 1–2 are text fixes. Finding 3 is a state-sharing hazard outside the documented call pattern; it is worth fixing, but it does not affect the claim.

## Resolution (lead, after the verdict)

The wording findings are fixed in the module docstring and the ledger row. The channel probability now reads $\lVert PL_{rs}\hat\psi\rVert^2/\langle\hat\psi|C(x,x)|\hat\psi\rangle$. A second crossing in one step is described as deferred, with the $0.46(\text{rate}\cdot dz)^2$ bias. The robustness finding is fixed: `MCWFSolver.run` now resets a shared solver's basis and kernel scale to the entry energy, and `test_run_resyncs_a_slowed_shared_solver` pins it. The fix was not re-verified in a fresh context.
