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

## Published comparisons (fresh-context review, 2026-11-10)

Scope: the two published-calculation anchors of `optical-ctr-population` at `f844eeff8`, `tests/montecarlo/spectrum/test_optical_ctr_casalbuoni_anchor.py` and `tests/montecarlo/spectrum/test_optical_ctr_schroeder_anchor.py`. The population law itself was rederived above and is not repeated. The verifier did not write the implementation or the tests. All figure readings, quadratures and scorer drivers below are the verifier's own (session scratch scripts); none uses the tests' helper functions, except the mutation runs, which call the test functions against in-memory mutated copies of the production modules.

### A. Casalbuoni, Schmidt and Schmüser, TESLA Report 2005-15

#### Independent derivation (SI)

Source Eq. (8) is the one-sided ($\omega>0$) far-field energy of one electron entering an infinite perfect conductor,

$$
\frac{d^2U_{\rm GF}}{d\omega\,d\Omega}
=\frac{e^2}{4\pi^3\varepsilon_0 c}\,
\frac{\beta^2\sin^2\theta}{(1-\beta^2\cos^2\theta)^2},
\qquad
\frac{e^2}{4\pi^3\varepsilon_0 c}=\frac{\alpha\hbar}{\pi^2}.
$$

For $N$ electrons with independent transverse positions drawn from a uniform disc of radius $r_b$ and no time spread, the field sum gives $N+N(N-1)\lvert F_\perp\rvert^2$ times the single-electron energy, with

$$
F_\perp(\theta,\omega)=\frac{1}{\pi r_b^2}\int_{\rho<r_b}e^{-i\mathbf{k}_\perp\cdot\boldsymbol\rho}\,d^2\rho
=\frac{2J_1(x)}{x},\qquad x=\frac{\omega r_b\sin\theta}{c}.
$$

This is the first term of source Eq. (37); the screen term $T$ vanishes for an infinite screen. The source writes $N^2$ where the population law has $N+N(N-1)\lvert F\rvert^2$; at $N=Q/e=6.2415\times10^{9}$ the difference is below $2\times10^{-10}$ wherever $\lvert F\rvert^2\gtrsim1$ and cannot be resolved by any figure. The energy inside the cone per frequency interval $\Delta f$ is, with $d\omega=2\pi\,df$,

$$
\boxed{\;
\frac{\Delta U}{\Delta f}
=\frac{Q^2}{4\pi^3\varepsilon_0 c}\;2\pi
\int_0^{\theta_{\rm det}}
\frac{\beta^2\sin^2\theta}{(1-\beta^2\cos^2\theta)^2}
\left[\frac{2J_1(x)}{x}\right]^2 2\pi\sin\theta\,d\theta\;}
$$

Note $N^2e^2=Q^2$: the elementary charge cancels. Point-bunch check with $\gamma=1000$ in the small-angle limit: the cone integral is $\pi[\ln(1+\gamma^2\theta_{\rm det}^2)-1+(1+\gamma^2\theta_{\rm det}^2)^{-1}]$, i.e. 25.794 at 100 mrad and 11.388 at 10 mrad. With $Q^2/(4\pi^3\varepsilon_0c)=3.0376\times10^{-18}$ J s/sr and $\Delta f=1$ GHz this gives 0.4923 and 0.2174 µJ/GHz (exact-angle quadrature: 0.49239 and 0.21735).

Units chain from the scorer. `bunch_yield` is $N\,\times$ photons/eV/sr per physical electron. For photon energy $E=hf$:

$$
\frac{\Delta U}{\Delta f}\,[\mathrm{J}]
=\underbrace{N\,y}_{\rm photons/eV/sr}\;
\underbrace{d\Omega}_{\rm sr}\;
\underbrace{E}_{\rm eV/photon}\;
\underbrace{e}_{\rm J/eV}\;
\underbrace{h}_{\rm eV\,s}\;
\underbrace{\Delta f}_{\rm Hz}.
$$

The test's `photons_per_eV * energy * _E_C * (_H_EV_S * 1e9) * 1e6` is exactly this in µJ. Its reference `_eq37_uJ_per_GHz` uses `population**2 * ALPHA_FS * hbar_J_s / np.pi**2 * cone * 2 * np.pi * 1e9 * 1e6`, identical to the boxed expression. Factor, exponents, $2\pi$ and $N^2$ all match. The scorer's own Ginzburg–Frank normalization is $y=\alpha\beta^2\sin^2\theta/[\pi^2E(1-\beta^2\cos^2\theta)^2]$, consistent with $\alpha\hbar/\pi^2$ above.

Fig. 15 (left): a Gaussian charge density with RMS $\sigma$ has $F_{\rm long}=e^{-\omega^2\sigma^2/2}$ by source Eq. (35), so $\lvert F_{\rm long}\rvert^2=e^{-\omega^2\sigma^2}$.

Forward stand-in: for a perfect conductor the exit-face field of a charge leaving the metal is the image problem mirrored in $z$, so the forward pattern equals the report's backward pattern. Numerically the production conductor-exit crossing (`eps = -1e12 + 1e9j`) reproduces the boxed expression to $\le4\times10^{-4}$. The backward code path (`n_z < 0`, face flip) is therefore not exercised by this anchor.

#### Independent figure reading

Page 22 was converted with `pdftocairo -svg`; curves were taken from the stroked vector paths and calibrated on the major ticks (1 and 10 THz; ordinate 0 and 0.5, 0 and 0.2, 0 and 1.0). The frame edges then fall at 0.5003 and 30.01 THz. Path coordinates are quantized at one plot unit: 0.0003 µJ/GHz (left) or 0.00015 (right), i.e. 0.06 % of the plateau and 0.4 % at the lowest ordinate, and 0.18 % in $f$.

Assignment, confirmed by matching each colour to the equation at 12 THz (residual $\le2\times10^{-3}$ with neighbouring radii $\ge20$ % away in the left panel; residual $\le4\times10^{-4}$ with neighbours $\ge0.8$ % away in the right panel): blue 50 µm, red 150 µm, green 350 µm, black 1000 µm; left panel $\theta_{\rm det}=100$ mrad (plateau 0.4950 vs equation 0.4924), right panel 10 mrad (0.2186 vs 0.2174).

Columns: verifier figure reading, JSON, verifier quadrature, production scorer with the `empirical` law (verifier's own equal-width ring profile with area weights, not the test's sunflower lattice). All in µJ/GHz.

| $\theta_{\rm det}$, $r_b$ | $f$ [THz] | figure | JSON | equation | scorer | fig/eq $-1$ | scorer/eq $-1$ |
|---|---|---|---|---|---|---|---|
| 0.1, 50 µm | 0.6 | 0.49499 | 0.49501 | 0.49233 | 0.49233 | +0.0054 | 0.0000 |
| 0.1, 50 µm | 28 | 0.40846 | 0.40844 | 0.40599 | 0.40617 | +0.0061 | +0.0004 |
| 0.1, 350 µm | 0.6 | 0.49055 | 0.49056 | 0.48952 | 0.48953 | +0.0021 | 0.0000 |
| 0.1, 350 µm | 8 | 0.32642 | 0.32639 | 0.32450 | 0.32451 | +0.0059 | 0.0000 |
| 0.1, 1000 µm | 0.6 | 0.46315 | 0.46316 | 0.47058 | 0.47058 | −0.0158 | 0.0000 |
| 0.1, 1000 µm | 0.8 | 0.45189 | 0.45199 | 0.45589 | 0.45589 | −0.0088 | 0.0000 |
| 0.1, 1000 µm | 1.0 | 0.43763 | 0.43755 | 0.43946 | 0.43946 | −0.0042 | 0.0000 |
| 0.1, 1000 µm | 1.2 | 0.42230 | 0.42223 | 0.42250 | 0.42250 | −0.0005 | 0.0000 |
| 0.1, 1000 µm | 1.5 | 0.39939 | 0.39930 | 0.39812 | 0.39812 | +0.0032 | 0.0000 |
| 0.1, 1000 µm | 2.0 | 0.36625 | 0.36624 | 0.36401 | 0.36401 | +0.0062 | 0.0000 |
| 0.1, 1000 µm | 15 | 0.13473 | 0.13462 | 0.13390 | 0.13390 | +0.0062 | 0.0000 |
| 0.1, 1000 µm | 28 | 0.07574 | 0.07587 | 0.07518 | 0.07519 | +0.0073 | 0.0000 |
| 0.01, 50 µm | 0.6 | 0.21857 | 0.21858 | 0.21735 | 0.21735 | +0.0056 | 0.0000 |
| 0.01, 350 µm | 15 | 0.20280 | 0.20287 | 0.20175 | 0.20180 | +0.0052 | +0.0002 |
| 0.01, 1000 µm | 8 | 0.18501 | 0.18502 | 0.18402 | 0.18411 | +0.0054 | +0.0005 |
| 0.01, 1000 µm | 28 | 0.07535 | 0.07540 | 0.07497 | 0.07510 | +0.0052 | +0.0018 |

Per-curve summary over 12 frequencies from 0.6 to 28 THz:

| curve | fig/eq $-1$: min, median, max | max JSON/fig $-1$ | max scorer/eq $-1$ |
|---|---|---|---|
| 0.1, 50 µm | +0.0052, +0.0055, +0.0061 | 0.0003 | 0.0004 |
| 0.1, 150 µm | +0.0046, +0.0049, +0.0060 | 0.0011 | 0.0003 |
| 0.1, 350 µm | +0.0021, +0.0033, +0.0071 | 0.0004 | 0.0001 |
| 0.1, 1000 µm | −0.0158, +0.0046, +0.0073 | 0.0017 | 0.0000 |
| 0.01, 50 µm | +0.0051, +0.0056, +0.0057 | 0.0001 | 0.0000 |
| 0.01, 150 µm | +0.0050, +0.0052, +0.0057 | 0.0004 | 0.0002 |
| 0.01, 350 µm | +0.0050, +0.0052, +0.0059 | 0.0005 | 0.0007 |
| 0.01, 1000 µm | +0.0043, +0.0048, +0.0072 | 0.0010 | 0.0018 |

Readings.

- JSON data: confirmed against the verifier's reading to $\le0.17$ % (the path quantization at low ordinates).
- Agreement within 1 %: confirmed for scorer vs figure everywhere except the stated corner (largest 0.73 %).
- Constant +0.55 %: confirmed as an approximate statement. Curve medians are +0.46 to +0.56 %; individual points range from +0.2 to +0.7 % outside the corner. It is not exactly constant. Since $N^2e^2=Q^2$, only $\varepsilon_0$, $c$ and $\pi$ enter; rounded $\varepsilon_0$ and $c$ change the result by 0.02 %. One candidate, $N$ from $e=1.6\times10^{-19}$ C with a CODATA $e^2$, gives +0.27 %; this is unverified. Cause not identified, as the test says.
- Corner: the 1000 µm, 100 mrad curve is below the equation by 1.58 % at 0.6 THz and 1.8 % at the lowest JSON point (0.547 THz), crossing zero at 1.2 THz. "Up to 1.9 %" is confirmed within reading precision. Two refinements: measured against the +0.55 % offset the shortfall persists to about 2 THz (+0.32 % at 1.5 THz), and the 350 µm, 100 mrad curve shows the same effect in miniature (+0.21 % at 0.6 THz, +0.27 % at 0.8 THz). The shortfall grows with $r_b$ and exists only in the 100 mrad panel. The equation with $T=0$ does not produce it; the cause is not identified.
- Fig. 15 (left): the curve equals $e^{-\omega^2\sigma^2}$ with $\sigma=50$ fs to 0.0012 at ten frequencies from 0.6 to 8 THz (e.g. 0.4126 vs 0.4114 at 3 THz). The field-RMS reading $e^{-\omega^2\sigma^2/2}$ gives 0.641 and a 50 fs FWHM gives 0.852 at 3 THz; both are excluded. The scorer's `gaussian` form factor and its energy ratio equal $e^{-\omega^2\sigma^2}$ to four digits. An additional check outside the test: 20 000 iid arrival times carried by the histories (`sampled` law) give the same ratio within 0.006.

#### Sampled law and the lattice

Six sets of 4000 iid uniform-disc crossing positions through the `sampled` law ($\lvert F\rvert^2=1$ applied) against the test's 1000-point sunflower lattice:

| case | $f$ [THz] | equation | iid mean ± s.e.m. | iid/eq $-1$ | lattice/eq $-1$ | lattice/fig $-1$ |
|---|---|---|---|---|---|---|
| 0.1, 350 µm | 0.6 | 0.48952 | 0.48952 ± 0.00001 | 0.0000 | 0.0000 | −0.0021 |
| 0.1, 350 µm | 8 | 0.32450 | 0.32432 ± 0.00015 | −0.0005 | −0.0005 | −0.0064 |
| 0.01, 1000 µm | 10 | 0.16922 | 0.16910 ± 0.00009 | −0.0007 | −0.0003 | −0.0053 |
| 0.01, 1000 µm | 18 | 0.11474 | 0.11459 ± 0.00013 | −0.0013 | −0.0009 | −0.0083 |
| 0.01, 1000 µm | 28 | 0.07497 | 0.07484 ± 0.00010 | −0.0016 | −0.0019 | −0.0070 |

Genuine iid draws reproduce the equation within their statistical error, so the estimator is tested as an estimator, and the lattice is a legitimate low-variance stand-in. A deterministic lattice has no pair fluctuation, so the distinct-pair formula is biased by about $-(1-\lvert F\rvert^2)/(M\lvert F\rvert^2)$, which is the observed −0.1 to −0.2 % at $M=1000$. The suppressed `CoherentSamplingWarning` concerns 4 of 960 and 16 of 1152 bins with pair values no lower than $-2\times10^{-6}$ of the maximum, at the disc form-factor zeros; the suppression is legitimate, and the equation assertion would catch a real loss. The margin of the 1 % figure tolerance is thin at the high-frequency end of the 10 mrad, 1000 µm case (0.83 % at 18 THz), because the figure offset and the lattice bias add.

#### Mutations

In-memory mutated copies of `optical_transition.py` and `transition_radiation.py` (source files untouched); test functions called directly. "C" columns are the Casalbuoni tests (8 profile cases, 2 sampled cases, Gaussian), "S" the Schroeder tests.

| mutation | C profile | C sampled | C Gaussian | S collimated | S divergent |
|---|---|---|---|---|---|
| none | pass | pass | pass | pass | pass |
| pair term scaled by 1 instead of $N-1$ ($N$ vs $N^2$) | **8/8 fail** | **2/2 fail** | **fail** | pass | pass |
| $N=2Q/e$ | **8/8 fail** | **2/2 fail** | pass | pass | pass |
| $N-1\to N$ | 8/8 fail (internal identity only) | pass | pass | pass | pass |
| drop $-\sum w^2\lvert S\rvert^2$ | **8/8 fail** | pass | pass | pass | pass |
| denominator $W^2$ instead of $W^2-W_2$ | **8/8 fail** | pass | pass | pass | pass |
| pair moment replaced by the incoherent one | pass | **2/2 fail** | pass | **fail** | **fail** |
| prefactor $16\pi^4\to8\pi^4$ | **8/8 fail** | **2/2 fail** | pass | **fail** | **fail** |
| sign of transverse $\boldsymbol\kappa$ in `form_factor_squared` | pass | pass | pass | pass | pass |
| $\sin\theta$ dropped from transverse $\boldsymbol\kappa$ | **8/8 fail** | pass | pass | pass | pass |
| empirical $\lvert F\rvert$ instead of $\lvert F\rvert^2$ | **8/8 fail** | pass | pass | pass | pass |
| Gaussian $\sigma\to\sigma/\sqrt2$ | pass | pass | **fail** | pass | pass |
| Gaussian $\sigma$ read as FWHM | pass | pass | **fail** | pass | pass |
| `sampled` law applies $\lvert F\rvert^2=0.5$ | pass | **2/2 fail** | pass | pass | pass |
| crossing transverse phase removed | pass | **2/2 fail** | pass | pass | pass |
| crossing transverse phase sign flipped | pass | pass | pass | pass | pass |
| crossing time phase sign flipped | pass | pass | pass | pass | pass |

What the anchors pin: $N$ from the charge and the $N^2$ scale; the unbiased pair estimator (subtraction and denominator, through the two identical histories of the profile test, where a biased form is wrong by a factor 2); the coherent pair moment from crossing positions; the absolute $E n_z^2/(16\pi^4)$ normalization; the transverse wavevector magnitude $\omega\sin\theta/c$; $\lvert F\rvert^2$ rather than $\lvert F\rvert$; the Gaussian exponent and the RMS meaning of $\sigma$.

What they do not pin: the sign of any phase (the disc, the Gaussian and the Schroeder momentum law are all even, so $\lvert F\rvert^2$ is sign-blind); $N-1$ versus $N$ (not resolvable at 1 nC; caught only by the test's own bookkeeping identity); the self term (below $10^{-9}$ of the signal); importance weights; t–x correlation. At $M=1000$ and above, the sampled and Schroeder tests alone do not distinguish the unbiased estimator from the biased one. These are covered, if at all, by the closed-form tests in `test_transition_radiation_optical.py`, not by a published value.

#### Exclusions

The stated exclusions are accurate: solid curves (12.5 mm screen) and Fig. 15 right (near field, finite screen) need a finite radiator, and nothing here is a measurement. Not stated but also outside the comparison: the backward vacuum-to-metal geometry itself (forward stand-in), a real metal permittivity, oblique incidence, and all phase signs.

### B. Schroeder, Esarey, van Tilborg and Leemans, PRE 69, 016501 (2004)

#### Source equations

Read from the PDF pages (016501-5 to -7). Eqs. (17) and (18), with $u=\gamma\beta$ and the electron direction $(\sin\psi\cos\varphi,\sin\psi\sin\varphi,\cos\psi)$:

$$
\mathcal{E}_\parallel=\frac{u\cos\psi\,[u\sin\psi\cos\varphi-(1+u^2)^{1/2}\sin\theta]}
{[(1+u^2)^{1/2}-u\sin\psi\cos\varphi\sin\theta]^2-u^2\cos^2\psi\cos^2\theta},
\qquad
\mathcal{E}_\perp=\frac{u^2\cos\psi\sin\psi\sin\varphi\cos\theta}
{[(1+u^2)^{1/2}-u\sin\psi\cos\varphi\sin\theta]^2-u^2\cos^2\psi\cos^2\theta}.
$$

Eq. (22):

$$
\frac{d^2W}{d\omega\,d\Omega}=\frac{e^2N}{\pi^2c}\left\{\int d^3p\,(\mathcal{E}_\parallel^2+\mathcal{E}_\perp^2)\,g
+(N-1)\left[\Big\lvert\int d^3p\,g\,\mathcal{E}_\parallel F\Big\rvert^2+\Big\lvert\int d^3p\,g\,\mathcal{E}_\perp F\Big\rvert^2\right]\right\}.
$$

Eq. (36), collimated: $(e^2/\pi^2c)\,N(N-1)\sin^2\theta\,\lvert\int du\,g_\parallel F\,u(1+u^2)^{1/2}/(1+u^2\sin^2\theta)\rvert^2$. It follows from Eq. (17) at $\psi=0$ because $(1+u^2)-u^2\cos^2\theta=1+u^2\sin^2\theta$. Eq. (29): $g_\parallel=u_t^{-1}e^{-u/u_t}$, with $u_t=10$ stated in the text for the 5 MeV case. Sect. III C: $g_\perp\propto\exp(-\sin^2\psi/\sigma_\perp^2)\sin\psi\cos\psi\,d\psi\,d\varphi$, so $\sin^2\psi$ is exponential with mean $\sigma_\perp^2$ and the azimuth is uniform. The test docstring transcribes all of these correctly, and its sampling of $u$ and $\sin\psi$ follows these laws.

Normalization. One electron at $\psi=0$ gives $(e^2/\pi^2c)\,\beta^2\sin^2\theta/(1-\beta^2\cos^2\theta)^2$, the Gaussian-unit Ginzburg–Frank energy per $d\omega\,d\Omega$. The scorer returns $y=\alpha\,\lvert\mathcal{E}\rvert^2/(\pi^2E)$ photons/eV/sr, and $d^2W/d\omega\,d\Omega=\hbar E\,y$, so

$$
\frac{\pi^2c}{e^2}\,\frac{d^2W}{d\omega\,d\Omega}=\frac{\pi^2E\,y}{\alpha}.
$$

The test's `np.pi**2 * _ENERGY / ALPHA_FS` applied to `pair_yield` is therefore the paper's $\lvert\langle\mathcal{E}_\parallel\rangle\rvert^2+\lvert\langle\mathcal{E}_\perp\rangle\rvert^2$. Confirmed.

#### Figure reading

Fig. 3 is an embedded 861 × 587 one-bit image (300 ppi). The verifier digitized it independently (ticks: $\theta=0$ to 0.5 at 146 px per 0.1 rad, ordinate 0 to 4 at 110.5 px per unit; curve centroids by column, and by row on the steep rise). Reading precision is about 1 % and worse on the rise and where dashes are absent. The JSON data agree with this digitization to within 1 % at most angles (2–3 % on the dashed rise below 0.02 rad). The JSON dashed curve ends at 0.345 rad and the solid one at 0.261 rad; the figure continues to 0.52 rad, where both curves coincide.

Collimated (dashed) curve. Columns: Eq. (36) by quadrature, its square root, verifier figure reading, JSON, and the scorer's normalized pair yield from 200 000 iid histories (verifier's driver).

| $\theta$ [rad] | Eq. (36) | $\sqrt{\rm Eq.\,(36)}$ | figure | JSON | fig/$\sqrt{\ }-1$ | fig/Eq. (36) | scorer/Eq. (36) $-1$ |
|---|---|---|---|---|---|---|---|
| 0.02 | 8.839 | 2.973 | 2.936 | 3.001 | −0.013 | 0.33 | −0.014 |
| 0.05 | 16.463 | 4.058 | 4.043 | 4.054 | −0.004 | 0.25 | −0.008 |
| 0.07 | 16.481 | 4.060 | 4.045 | 4.046 | −0.004 | 0.25 | −0.006 |
| 0.10 | 14.557 | 3.815 | 3.769 | 3.799 | −0.012 | 0.26 | −0.005 |
| 0.15 | 11.015 | 3.319 | 3.339 | 3.317 | +0.006 | 0.30 | −0.003 |
| 0.20 | 8.389 | 2.896 | 2.901 | 2.886 | +0.001 | 0.35 | −0.003 |

The plotted ordinate is the square root of the quantity named in the caption: a log–log fit of the verifier's digitization against Eq. (36) gives exponent 0.505 and scale 0.983 (1.01 and 0.98 in the test's square-root terms; the docstring's 0.992 and 0.997 are consistent within digitization error). Read as the energy itself, the figure would be a factor 3 to 4 too low. Supporting evidence: the same $g$ gives, for the incoherent Fig. 2 as an energy, a peak of 38.65 at 0.035 rad, 8.3 at 0.25 rad and 25.1 on axis for an RMS divergence of 0.02, which match that figure by eye (about 38.5, 8.5 and 25); and Fig. 3 read as an energy would peak at 4.2 where Eq. (36) gives 16.7. The reading remains an inference that contradicts the caption's literal label.

Divergent (solid) curve, $\langle\psi^2\rangle^{1/2}=0.05$. Reference: the verifier's quadrature of Eqs. (17)/(18) over $g_\parallel g_\perp$ (Gauss–Laguerre in $u$, 4000-point rule in $\sin^2\psi$, 96 azimuths; $\langle\mathcal{E}_\perp\rangle=0$ by symmetry).

| $\theta$ [rad] | $\lvert\langle\mathcal{E}\rangle\rvert^2$ | $\sqrt{\ }$ | figure | JSON | fig/$\sqrt{\ }-1$ | scorer/quadrature $-1$ |
|---|---|---|---|---|---|---|
| 0.015 | 1.003 | 1.002 | 1.205 | 1.212 | +0.20 | −0.014 |
| 0.02 | 1.727 | 1.314 | 1.567 | 1.576 | +0.19 | −0.012 |
| 0.03 | 3.555 | 1.885 | 2.132 | 2.164 | +0.13 | −0.011 |
| 0.04 | 5.605 | 2.367 | 2.568 | 2.598 | +0.085 | −0.010 |
| 0.05 | 7.566 | 2.751 | 2.901 | 2.912 | +0.054 | −0.007 |
| 0.06 | 9.218 | 3.036 | 3.133 | 3.151 | +0.032 | −0.004 |
| 0.08 | 11.253 | 3.355 | 3.373 | 3.379 | +0.006 | −0.002 |
| 0.10 | 11.788 | 3.433 | 3.421 | 3.421 | −0.004 | −0.002 |
| 0.15 | 10.222 | 3.197 | 3.190 | 3.181 | −0.002 | −0.001 |
| 0.20 | 8.101 | 2.846 | 2.846 | 2.838 | 0.000 | 0.000 |
| 0.25 | 6.434 | 2.537 | 2.539 | 2.533 | +0.001 | 0.000 |

Readings.

- Above 0.08 rad the figure, the paper's equations and the scorer agree within 0.6 %. The divergence still lowers the curve there by 16 % at 0.08 rad, 10 % at 0.10 rad and 4 % at 0.15 rad relative to the collimated one, so the retained range does test the direction average against a 1.5 % tolerance.
- Below 0.08 rad the figure lies above the paper's own Eqs. (17)/(18) by 3 % (0.06 rad) rising to 19–20 % (0.015–0.02 rad). The docstring's "up to 18 %" is confirmed within digitization error. The scorer follows the equations (within the 1.4 % noise of the iid driver), not the figure. No single $\sigma_\perp$ repairs it: 0.045 still leaves +10 % at 0.02 rad and −3 % at 0.08 rad, and $\sigma_\perp=0.05/\sqrt2$ is 6–12 % low throughout. The exclusion is an honest figure/equation difference of the source, of unknown cause. It removes the range where the divergence matters most, which should stay visible in the scope wording.
- The scorer's normalized self yield also equals the incoherent quadrature $\langle\mathcal{E}_\parallel^2+\mathcal{E}_\perp^2\rangle$ within 0.2 % (divergent) and 2 % (collimated, heavy-tailed iid noise).

Mutations: see the table in part A. The Schroeder tests read `pair_yield` directly, so they pin the vector pair moment $\lvert\langle\mathbf{S}\rangle\rvert^2$ over an energy and direction spread, the oblique conductor amplitude and the absolute per-pair normalization. They do not involve $N$, $N(N-1)$, the charge, the self-term subtraction (invisible at $M\ge12\,800$) or any form factor.

### Verdict

- **Claim**: `optical-ctr-population` — `montecarlo/spectrum/optical_transition.py::{OpticalBunchSource, optical_transition_radiation}`, `montecarlo/spectrum/transition_radiation_crossings.py::history_field_sums` — published comparisons: Casalbuoni et al., TESLA 2005-15, Eqs. (8), (35), (37), Figs. 14–15; Schroeder et al., PRE 69, 016501, Eqs. (17), (18), (22), (36), Fig. 3.
- **Filters**: units pass (photons/eV/sr per electron → µJ/GHz chain; $\pi^2E/\alpha$ normalization); limits pass (point-bunch plateau 0.4924 and 0.2174 µJ/GHz; $\psi=0$ reduction of Eq. (17) to Eq. (36); $\lvert F\rvert^2\to1$ at low frequency); signs/conventions pass for what is tested ($\sigma$ is the RMS of the charge density; Fig. 3 ordinate is a square root). Phase signs are not testable by either anchor.
- **Re-derivation**: `matches`. Verifier quadratures, production scorer and both figures agree within the stated tolerances; both source figure/equation differences are reproduced independently of the tests' helpers.
- **Verdict**: `rederived` stands; the two tests are valid published-value anchors and support `anchored` with the scope below. No discrepancy in the implementation. Wording findings are listed for the owner.
- **Write-up**: this section.
- **Suggested ledger change** (owner applies; never `signed-off`): status `anchored`, with Notes scope: "Published calculations, perfect conductor, far field, forward conductor-exit crossing. Casalbuoni Figs. 14–15 fix $N=Q/e$ and the absolute $N^2$ energy scale, the unbiased distinct-pair estimator, the transverse-disc `empirical` and position-`sampled` form factors and the Gaussian RMS time form factor. Schroeder Fig. 3 (square-root reading, inferred) fixes the vector pair moment over energy and direction spread at $F=1$ for $\theta\ge0.08$ rad and the collimated curve; it does not involve $N$. Not pinned by any published value: phase signs, $N-1$ vs $N$, the self term, importance weights, correlated or tilted laws, real permittivities, backward geometry, finite screens, near field, measurements."

Wording to correct (owner):

1. Test docstring, ledger Notes and physics page: "a constant 0.55 % above Eq. (37)". The offset is +0.46 to +0.56 % in the curve medians and +0.2 to +0.7 % point by point. Suggested: "about 0.5 % above (0.2–0.7 %)".
2. Same places: the low-frequency shortfall is described as confined to the 1000 µm, 100 mrad curve below 1.2 THz. It is 1.6 % at 0.6 THz and 1.8–1.9 % at the lowest plotted point there, and it is also visible at 350 µm, 100 mrad (+0.2 % instead of +0.5 %). Suggested addition: "a smaller shortfall of the same kind appears at 350 µm; it grows with $r_b$".
3. Ledger Notes: "2e-3 to 4e-3 of an independent Eq. (37) quadrature". Those are the test's assertion tolerances against its own in-file quadrature. Measured agreement with the verifier's quadrature is $\le4\times10^{-4}$ for resolved profiles and $\le2\times10^{-3}$ for the sampled lattice.
4. Ledger Notes scope "pair estimator with transverse phases": add "magnitude only; phase signs are not tested (even laws)".
5. Physics page, "Both are calculations, not measurements: they fix the absolute $N^2$ scale, the transverse and Gaussian form factors and the population average": the Schroeder comparison does not involve $N$ or a form factor. Suggested: "Together they fix …; the absolute $N^2$ scale and the form factors come from the first, the momentum-spread pair average from the second."
6. Ledger Notes, Schroeder: "exceeds the paper's own Eqs. (17)/(18) by up to 18 %": the verifier reads 19–20 % at 0.015–0.02 rad; "up to about 20 %" is safer. Add that the excluded range is where the divergence effect is largest, and that the JSON curves stop at 0.345 rad (dashed) and 0.261 rad (solid).

Commands (all exit 0): `uv run pyrite-dev test tests/montecarlo/spectrum/test_optical_ctr_casalbuoni_anchor.py tests/montecarlo/spectrum/test_optical_ctr_schroeder_anchor.py` (15 passed); scratch scripts `scratch/v615/{extract,casal_check,mutate,schro_fig,schro_check}.py` via `uv run python`.

### Owner disposition (2026-11-10)

All six wording corrections are applied to the test docstrings, the ledger Notes and the physics page. The ledger row moves to `anchored` with the scope above; the unpinned items (phase signs, $N-1$ versus $N$, self term, importance weights, correlated or tilted laws, real permittivities, backward geometry, finite screens, near field, measurements) are listed in its Notes. No mutation-killing test for phase signs was added: no published value in these sources depends on them.
