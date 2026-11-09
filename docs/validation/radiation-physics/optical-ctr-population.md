# `optical-ctr-population`

## Claim and source

- Claim: for independent identically distributed electrons the optical TR power per physical electron is $\mathrm E|\mathbf S|^2+(N-1)|F|^2|\mathrm E\mathbf S|^2$ with $N=Q/e$ from the bunch charge alone, estimated from $M$ histories with positive importance weights by $\sum w|\mathbf S|^2/W$ and $(|\sum w\mathbf S|^2-\sum w^2|\mathbf S|^2)/(W^2-W_2)$. $|F|^2=1$ for histories carrying their own offsets; the Gaussian or resolved empirical characteristic function at $\boldsymbol\kappa=(\omega,-\omega n_x,-\omega n_y)/\hbar c$ applies only to offset-free histories. Correlated sources are accepted only as an explicit realization; unsupported contracts are refused.
- Code: `montecarlo/spectrum/optical_transition.py::{OpticalBunchSource, optical_transition_radiation, gaussian_coherent_band_eV}`; `montecarlo/spectrum/transition_radiation_crossings.py::history_field_sums`; `montecarlo/spectrum/optical_transition_validity.py::radiated_energy_fraction`.
- Sources: squared field-sum expansion (Shkvarunets and Fiorito, arXiv:0712.2848, Eq. 1; standard expansion, not re-read); self-normalized importance sampling.
- Units: $\omega=E/\hbar c$ in Å$^{-1}$; times in Å of light travel, lengths in Å; output photons eV$^{-1}$ sr$^{-1}$ per physical electron.
- Frozen revision: commit `3caced48` (fresh-context verifier, 2026-10-22).

## Independent derivation

### Population law

With $\mathbf T=\sum_{j=1}^N\mathbf S_j$ and $\mathbf S_j$ i.i.d.,

$$
\mathrm E|\mathbf T|^2=N\,\mathrm E|\mathbf S|^2+N(N-1)\,|\mathrm E\mathbf S|^2 ,
$$

so per physical electron $I=\mathrm E|\mathbf S|^2+(N-1)|\mathrm E\mathbf S|^2$. Transport through a laterally infinite planar stack is translation invariant, so an entrance offset $\boldsymbol\xi=(\delta t,\delta x,\delta y)$ multiplies every crossing by $e^{i\boldsymbol\kappa\cdot\boldsymbol\xi}$, $\boldsymbol\kappa=(\omega,-\omega n_x,-\omega n_y)/\hbar c$. For offsets independent of the offset-free field $\mathbf S^0$: $\mathrm E|\mathbf S|^2=\mathrm E|\mathbf S^0|^2$ and $\mathrm E\mathbf S=\chi\,\mathrm E\mathbf S^0$, $\chi=\mathrm E\,e^{i\boldsymbol\kappa\cdot\boldsymbol\xi}$, so $|F|^2=|\chi|^2$. For $\boldsymbol\xi\sim\mathcal N(0,\Sigma)$, $|F|^2=e^{-\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa}$. If the histories already contain their offsets, an analytic factor would count them twice.

### Estimators

Draw $\mathbf S_e\sim q$ with $w_e=c\,p/q$. Then $\sum w|\mathbf S|^2/\sum w\to\mathrm E_p|\mathbf S|^2$ and

$$
\frac{|\sum_ew_e\mathbf S_e|^2-\sum_ew_e^2|\mathbf S_e|^2}{W^2-W_2}
=\frac{\sum_{e\ne f}w_ew_f\,\mathbf S_e\cdot\mathbf S_f^\ast}{\sum_{e\ne f}w_ew_f}
\;\to\;\frac{|\mathrm E_q[w\mathbf S]|^2}{(\mathrm E_qw)^2}=|\mathrm E_p\mathbf S|^2 .
$$

Both are ratios of U-statistics: consistent, invariant to a common scale $c$, with $O(1/M_{\rm eff})$ bias for non-constant weights; for unit weights exactly $G/M$ and $(P-G)/(M(M-1))$, which are unbiased. The weight enters the self term linearly, never squared. Scaling the crossing charges of history $e$ by $c_e$ makes the history sums return $\sum c_e^2|\mathbf S_e|^2$ and $\sum c_e\mathbf S_e$; $c_e=w_e$ and $c_e=\sqrt{w_e}$ give exactly the three moments.

### Empirical law

For $K$ i.i.d. draws, $\mathrm E|\sum_e e^{i\phi_e}|^2=K+K(K-1)|\chi|^2$, so $(|\sum e^{i\phi}|^2-K)/(K^2-K)$ is unbiased (weighted analogue: the same ratio as above). For a stated density on a grid, $|\sum w e^{i\phi}|^2/W^2$ is its transform.

### Coherent window

At normal incidence with a time-only law, $|F_t|^2=e^{-(E\sigma_t/\hbar)^2}$. By Cauchy–Schwarz $|\mathrm E\mathbf S|^2\le\mathrm E|\mathbf S|^2$, so the pair term is below $\epsilon$ times the self term above $E_w=(\hbar/\sigma_t)\sqrt{\ln((N-1)/\epsilon)}$. For $\sigma_t=200$ fs, $\hbar/\sigma_t=3.291$ meV; at 1 pC and $\epsilon=10^{-3}$, $E_w=15.63$ meV; at 0.2 eV $(E\sigma_t/\hbar)^2=3693$.

## Cheap filters

| Check | Result |
|---|---|
| Units | $\boldsymbol\kappa$ [Å$^{-1}$] times $\boldsymbol\xi$ [Å] is dimensionless; $\Sigma$ in Å$^2$; $\hbar=0.65821$ eV fs from $\hbar c/c$. Pass. |
| Limits | $N=1$ gives the self term only; $F\to0$ gives $I_1$; identical in-phase histories give $N I_1$; $M$-invariance at fixed charge. Pass. |
| Conventions | $\boldsymbol\kappa$ sign follows the amplitude phase; $|F|^2$ is sign-blind, the sampled law is not, and the code's phase is the derived one. Pass. |

## Comparison with the implementation

- `_weighted_moments`: unit weights reuse one `history_field_sums` call; weighted runs scale crossing charges by $w_e$ and $\sqrt{w_e}$. The chunked path multiplies charges into amplitudes and the CUDA path passes them to `history_sums`. Matches.
- `optical_transition_radiation`: self, pair and total $=$ self $+\max(N-1,0)|F|^2$ pair as derived; the realization law gives $|\sum\mathbf S|^2/M$ and refuses weights and $|N-M|>0.5$; $W^2-W_2\le0$ with $N>1$ is refused; signed and unclipped. Matches.
- `form_factor_squared`: `exp(-einsum(kappa, cov, kappa))` with `kappa = (ω, -ω n_x, -ω n_y)`; `samples` and `profile` transforms as derived.
- `_require_offset_free`: refuses analytic or empirical laws when entrance time or transverse spreads exceed $10^{-6}$ rad of phase; a common offset is harmless.
- `gaussian_coherent_band_eV`: `hbar_ev_fs = HBARC_EV_ANG / C_ANG_PER_FS` as derived; 0 when $(N-1)/\epsilon\le1$.

### Independent numerical checks

1. Brute force: 7 random tilted thin-film histories, absorbing film, random weights $w\in[0.2,3]$, $N=10^3$; explicit $\sum_{e\ne f}$ double loop and $\sum w|\mathbf S|^2/W$ against `optical_transition_radiation`: self, pair and total agree to $3\times10^{-16}$; unit weights against $G/M$, $(P-G)/(M(M-1))$: $4\times10^{-16}$.
2. Translation phase: a shift by $(37,-55,21)$ Å multiplies the summed amplitude by $e^{i\boldsymbol\kappa\cdot\boldsymbol\xi}$ to $2\times10^{-16}$ for both faces; the opposite sign misses by 0.08–0.10.
3. Correlated Gaussian: code $|F|^2=0.999747$ against a $4\times10^5$-draw Monte Carlo 0.999748.
4. Window numbers: $E_{1/e}=3.2911$ meV, $E_w=15.6298$ meV.

## Adjudication

No divergent factor, sign, exponent, unit or convention. Verdict: **rederived**. Anchoring is to closed-form and limiting-case references only ($N=1$, $F=0$, $F=1$, $M$-invariance, analytic Gaussian against sampled offsets, weight scale, 200 fs window). There is no published or measured CTR anchor.

### Findings and owner disposition

1. Medium, guard scope: `radiated_energy_fraction` cannot detect coherent energy below a band whose lower edge lies above the Gaussian window (estimate for 30 keV on gold at 200 fs: about $2\times10^{-4}$ of the kinetic energy at 1 pC, $2\times10^{-3}$ at 10 pC). Applied as documentation: the budget is stated as in-band only in the module, physics page and ledger. Superseded (#615 follow-up, post-verification): `out_of_band_coherent_energy_eV` adds a perfect-conductor envelope with analytic Gaussian tails to the budget (1.8e-4 for the case above at 1 pC, consistent with this estimate); it awaits fresh-context review.
2. Low, wording: weighted estimators are consistent, not unbiased. Applied (test renamed, ledger Checks reworded).
3. Low, resolution criterion: stated as sampling interval (grid spacing), not RMS blur. Applied to the source docstring and physics page.
4. Low, scope: `gaussian_coherent_band_eV` is a time-only, normal-incidence window; a tilted front observed obliquely can widen it. Applied to the physics page.
5. Low, scope: the formation measure bounds the self power only. Applied.

### Caveats

- The `sampled` law with histories that carry no offsets is a point bunch ($|F|^2=1$) by contract; this is not detectable.
- Bunch evolution (space charge, wakefields) is outside the passive-source model.

## Fresh-context review of the #615 follow-up (2026-11-08)

Scope: commits `e02620930` and `b0860fbfa`. `optical_transition_validity.py::{pec_crossing_energy_per_eV, out_of_band_coherent_energy_eV, coherent_formation_error_bound}` and the caller `runner/transition_emission.py::_optical_validity`. The ledger rows and docstrings were read first. The derivation below was written before the implementation bodies were read.

### Independent derivation: out-of-band envelope

**Single crossing, perfect conductor.** A charge $e$ meets a perfect-conductor plane. Its vacuum field is the field of the charge plus the field of an image charge $-e$ at the mirrored position, with velocity $\boldsymbol\beta'=\boldsymbol\beta-2(\boldsymbol\beta\cdot\hat{\mathbf z})\hat{\mathbf z}$. Both stop abruptly at the surface (or start there, for an exit). The Liénard–Wiechert spectral amplitude of a stopping or starting charge is $\propto\mathbf n\times(\mathbf n\times\boldsymbol\beta)/(1-\mathbf n\cdot\boldsymbol\beta)$. In Gaussian units $d^2I/d\omega\,d\Omega=(e^2/4\pi^2c)\,\lvert\sum_j q_j\,\mathbf n\times\boldsymbol\beta_j/(1-\mathbf n\cdot\boldsymbol\beta_j)\rvert^2$. Dividing by $\hbar$ gives the energy per unit photon energy:

$$
\frac{dU}{dE\,d\Omega}=\frac{\alpha}{4\pi^2}
\left\lvert\mathbf n\times\Bigl(\frac{\boldsymbol\beta}{1-\mathbf n\cdot\boldsymbol\beta}
-\frac{\boldsymbol\beta'}{1-\mathbf n\cdot\boldsymbol\beta'}\Bigr)\right\rvert^2 ,
$$

This is dimensionless (eV per eV per sr), and the image enters with a minus sign. At normal incidence $\boldsymbol\beta'=-\boldsymbol\beta$, so the bracket is $2\boldsymbol\beta/(1-\beta^2\cos^2\theta)$. Then

$$
\frac{dU}{dE\,d\Omega}=\frac{\alpha\beta^2\sin^2\theta}{\pi^2(1-\beta^2\cos^2\theta)^2},
\qquad
\int_{2\pi}\frac{dU}{dE\,d\Omega}d\Omega=\frac{\alpha}{\pi}\Bigl[\frac{1+\beta^2}{2\beta}\ln\frac{1+\beta}{1-\beta}-1\Bigr].
$$

Dividing by $E$ gives the Ginzburg–Frank conductor yield $\alpha\beta^2\sin^2\theta/(\pi^2E(1-\beta^2\cos^2\theta)^2)$ photons eV$^{-1}$ sr$^{-1}$. The result does not depend on $E$.

**Chain to the pair term.** The per-electron pair yield is $(N-1)|F|^2|\langle\mathbf S\rangle|^2$.

1. Jensen's inequality gives $|\langle\mathbf S\rangle|^2\le\langle|\mathbf S|^2\rangle$.
2. Cauchy–Schwarz over the $n_h$ crossings, at a fixed direction $\mathbf n$, gives $|\mathbf S_h(\mathbf n)|^2\le n_h\sum_c|\mathbf A_c(\mathbf n)|^2$.
3. Integrating over all directions and replacing each $\int|\mathbf A_c|^2d\Omega$ by its envelope $U'_c$ gives $\int|\mathbf S_h|^2\le n_h\sum_cU'_c$.
4. The Gaussian law has $\boldsymbol\kappa=(\omega,-\omega n_x,-\omega n_y)/\hbar c$. With no $t$–$x$ or $t$–$y$ covariance and a PSD transverse block, $\boldsymbol\kappa^{\mathsf T}\Sigma\boldsymbol\kappa\ge(E\sigma_t/\hbar c)^2$, where $\sigma_t$ is in Å of light travel. Therefore $|F|^2\le e^{-(E\sigma_t/\hbar c)^2}$.
5. With $s=\hbar c/\sigma_t$, $\int_0^{E_{\rm lo}}e^{-(E/s)^2}dE+\int_{E_{\rm hi}}^\infty e^{-(E/s)^2}dE=s\frac{\sqrt\pi}{2}[\operatorname{erf}(E_{\rm lo}/s)+\operatorname{erfc}(E_{\rm hi}/s)]$.
6. $\langle n_h\sum_cU'_c\rangle$ is estimated self-normalized, $B=\sum_hw_hn_h\sum_cU'_c/\sum_hw_h$. This matches the self-term estimator. It is consistent, not a deterministic bound.

The result is $(N-1)B\,s(\sqrt\pi/2)[\operatorname{erf}+\operatorname{erfc}]$ in eV.

### Independent derivation: formation propagation

Let $\mathbf S'_h=\mathbf S_h+\boldsymbol\delta_h$ with $\lvert\boldsymbol\delta_h\rvert\le b_h$ in $\mathbb C^3$. Then $\bigl\lvert|\mathbf S'_h|^2-|\mathbf S_h|^2\bigr\rvert=\lvert2\,\mathrm{Re}(\mathbf S_h\cdot\boldsymbol\delta_h^\ast)+|\boldsymbol\delta_h|^2\rvert\le2b_h|\mathbf S_h|+b_h^2$. Write $\mathbf C=\sum w\mathbf S$. Then $\lvert\delta\mathbf C\rvert\le D=\sum wb$ and $\bigl\lvert|\mathbf C'|^2-|\mathbf C|^2\bigr\rvert\le2D|\mathbf C|+D^2$. By the triangle inequality:

$$
\Bigl\lvert\delta\sum w|\mathbf S|^2\Bigr\rvert\le\sum w(2b|\mathbf S|+b^2),\qquad
\Bigl\lvert\delta\bigl(|\mathbf C|^2-\sum w^2|\mathbf S|^2\bigr)\Bigr\rvert\le2D|\mathbf C|+D^2+\sum w^2(2b|\mathbf S|+b^2).
$$

The total self$/W+(N-1)|F|^2\,$pair$/(W^2-W_2)$ is linear in the two numerators, so its absolute bound is $dg/W+(N-1)|F|^2\,d_{\rm pair}/(W^2-W_2)$. The realization total $|\mathbf C|^2/M$ with unit weights has bound $(2D|\mathbf C|+D^2)/M$. Histories without crossings have $\mathbf S=0$ and $b=0$. They still count in $W$ and $W_2$.

### Cheap filters

| Check | Result |
|---|---|
| Units | $U'$ is dimensionless (eV/eV); $s=\hbar c/\sigma_t$ in eV with $\sigma_t$ in Å; the result is in eV per physical electron. Pass. |
| Limits | Normal incidence gives the Ginzburg–Frank conductor form. $N=1$ gives zero. $\sigma_t\to\infty$ gives zero tails. Straight infinite runs give a zero formation bound. Pass. |
| Signs | The image term enters with a minus sign. A backscatter exit through face 0 radiates into the same vacuum hemisphere as the entrance. The bound is invariant under $\boldsymbol\beta\to-\boldsymbol\beta$. Pass. |

### Comparison with the implementation

- `pec_crossing_energy_per_eV`: $\beta$ comes from the crossing's kinetic energy. The image is `beta * [1, 1, -1]`, the bracket uses the minus sign, and the prefactor is `ALPHA_FS/(4 pi^2)`. Face 0 uses the back hemisphere, the last face the front one, and inner faces the larger of the two. This matches the derivation.
- `out_of_band_coherent_energy_eV`: $\sigma=\sqrt{\Sigma_{tt}}$ [Å], `scale = HBARC_EV_ANG / sigma`, and `tails` as in step 5. $B$ is the weighted mean of `n_cross * summed`. Correlated laws, i.e. nonzero `cov[0, 1:]`, are refused. `OpticalBunchSource` enforces symmetry and PSD, so step 4 holds for every accepted covariance. This matches the derivation.
- `form_factor_squared`: `exp(-kappa^T cov kappa)` with $\boldsymbol\kappa$ as above. The inequality in step 4 holds.
- `coherent_formation_error_bound` / `_history_sums`: `np.add.at` on the transposed views `summed.transpose(1, 0, 2)` and `bound.T` writes into the owning arrays. `_history_relative_error` now passes compact `inverse` ids. The weights are `ones(count)` or `history_weights`. `pair_den <= 0` gives zero, consistent with the estimator's refusal. This matches the derivation term by term.
- `_optical_validity`: refuses on `max(total)` and records `formation_self_bound`. The out-of-band energy is added to the budget, and laws without an envelope are recorded as `out_of_band_unbounded`. This is consistent.

### Independent numerical checks

Scripts are in the session scratchpad; none uses the code under review except where noted.

1. The hemisphere closed form, $(\alpha/\pi)[\dots]$, matches direct quadrature of the general image formula to $10^{-6}$: $3.48614\times10^{-4}$ at 30 keV and $6.19817\times10^{-3}$ at $\beta=0.95$.
2. Oblique image formula against the finite-ε planar solver (`interface_transition_radiation_amplitude`, verified independently under `optical-planar-tr`). At $\varepsilon=(-1+i)10^7$ the integrated ratio is 0.994–0.998 for tilts 0–80° and $\beta=0.328,0.95$. The image formula and its sign are confirmed at oblique incidence.
3. Algebra of the formation bound: 20 000 random trials (2–7 histories, random weights, random and adversarially aligned $\boldsymbol\delta$, empty histories). Largest change/bound is 0.9994 for the self term (tight) and 0.78 for the pair term. No violation.
4. Repository tests: `test_optical_transition_validity.py`, `test_optical_finite_convergence.py` and `test_coherent_formation_bound_propagates_to_pair_and_total` pass (13 passed).

### Is the PEC yield an upper bound of the finite-ε yield?

The integrated ratio $\int U_\varepsilon\,d\Omega/\int U_{\rm PEC}\,d\Omega$ is what step 3 needs. The normal-incidence values come from an independent implementation of the Ginzburg–Tsytovich single-interface formula. The oblique values come from the planar solver.

| Medium, face | $\beta$ | tilt | ratio |
|---|---|---|---|
| Au-like $\varepsilon=-2000+400i$ (sub-band IR), entrance/exit | 0.328–0.999 | 0–80° | 0.86–0.99 |
| Au $\varepsilon=-40+3i$ (1.4 eV) | 0.328–0.95 | 0–80° | 0.69–0.88 |
| $\varepsilon=-1.5+5i$ (Au 2.4 eV-like), exit | 0.95 | 80° | **1.41** |
| $\varepsilon=-0.5+0.1i$ (near plasma edge), exit/entrance | 0.95 | 80° | **1.62 / 1.52** |
| Si $\varepsilon=11.7$ (below 1.1 eV), exit to vacuum | 0.328 | 0° | **1.25** |
| $\varepsilon=9.8$, exit | 0.328 | 0° | **275** (refracted Cherenkov pole, $\varepsilon-\sin^2\theta=1/\beta^2$) |
| glass $\varepsilon=2.25$, exit | 0.95 | 0° | **2.2** |
| Si $\varepsilon=12.9$, exit | 0.95 | 60° / 80° | **42 / 389** |
| any dielectric, entrance (backward) | all | all | < 0.5 |

Pointwise in angle, real metals exceed PEC by up to $10^2$ at oblique incidence. That does not matter for step 3, which integrates first. The integrated envelope does hold for gold below its plasma edge, which covers the documented 30 keV-on-gold case. It fails in two situations:

- **Exit faces into vacuum from a dielectric whose $\beta^2\,\mathrm{Re}\,\varepsilon\gtrsim1$.** The semi-infinite formula there contains medium Cherenkov radiation refracted into vacuum. It is unbounded at the pole, and for a finite layer it grows with thickness. This happens at 30 keV for Si below the gap, and for any glass at MeV energies. Fresnel $|r|\le1$ holds in all these cases, so the stated assumption is not the operative condition, and "dielectrics" is a wrong example.
- **Grazing incidence on media near their plasma edge** (ratio up to about 1.6). For gold this lies above a visible band, where the Gaussian factor is negligible ($e^{-3700}$ at 0.2 eV for 200 fs). It is relevant only for short bunches or bands below the plasma edge of a low-$|\varepsilon|$ medium.

Separately, a fixed 24×24 quadrature under-integrates the PEC hemisphere once $\gamma\gtrsim20$: 24×24 / 800×1600 is 0.73 at $\beta=0.99999$ normal incidence and 0.60 at $\beta=0.999$, 85° tilt. It is exact to $10^{-5}$ at 30 keV and to $2\times10^{-3}$ at $\beta=0.95$. An envelope that undershoots is not an envelope.

### Verdict

- `pec_crossing_energy_per_eV` formula, `out_of_band_coherent_energy_eV` arithmetic and unit conversions, and the $|F|^2$ inequality: **match**.
- `coherent_formation_error_bound` and its use in `_optical_validity`: **match**, verdict **rederived**. The factor-2 premise for $b_h$ stays a heuristic, as documented.
- The validity statement of the out-of-band envelope: **discrepancy (medium, scope)**. The first divergent item is the assumption "$|r|\le1$ out of band (metals below their plasma edge, dielectrics)". The PEC hemisphere energy is not an upper bound of the single-interface yield for (a) exit faces from media with $\beta^2\mathrm{Re}\,\varepsilon\gtrsim1$ (Cherenkov), by factors 1.25 to unbounded, or (b) grazing incidence on low-$|\varepsilon|$ media (up to about 1.6). In addition, the fixed 24×24 quadrature undershoots by up to 40% for $\gamma\gtrsim20$. Neither affects the documented 30 keV gold case.

### Suggested owner actions (not applied)

1. Restate the assumption as: no out-of-band Cherenkov in traversed media ($\beta^2\mathrm{Re}\,\varepsilon<1$ on exit faces), and media not near their plasma edge at grazing incidence. Drop "dielectrics" as an example. Alternatively, record `out_of_band_unbounded` when a traversed non-vacuum layer has an exit crossing with $\beta^2\varepsilon_{\rm static}\ge1$.
2. Use the closed form at normal incidence, and scale `n_cos`/`n_phi` with $\gamma$ (or refuse $\gamma>10$), so that the envelope does not undershoot.
3. State that $B$ is a sample estimate of $\langle n_h\sum_cU'_c\rangle$ (consistent, with Monte Carlo error), not a deterministic bound.
4. Inner faces: the larger PEC hemisphere ignores thin-film transmission into the opposite hemisphere and any multilayer enhancement. It is a heuristic and should be labelled as one.
5. Optional: add a randomized perturbation test of the formation propagation, like check 3 above.

Suggested ledger change for `optical-ctr-population`: keep `rederived` for the population law and estimators. Record in Notes that the out-of-band envelope formula and the formation propagation were fresh-context verified on 2026-11-08 at `b0860fbfa`, and that the envelope's validity statement is a scope discrepancy (Cherenkov exit faces, grazing low-$|\varepsilon|$, $\gamma\gtrsim20$ quadrature) pending owner action. For `optical-planar-tr`: keep `anchored` and record `coherent_formation_error_bound` as rederived. Never `signed-off`.

### Owner disposition (2026-11-08)

1. Envelope scope: applied. In-band `beta^2 Re eps >= 1` on any layer (proxy for the out-of-band values) records `out_of_band_unbounded`; grazing low-|ε| media (≤1.6×) documented; inner-face hemisphere rule labelled heuristic.
2. Quadrature: refused above γ = 5 (24×24 rule within 1 % at 1.2 rad tilt there); the closed form was not substituted.
3. B labelled a weighted sample estimate.
4. Optional perturbation-inside-bound regression: not added.
