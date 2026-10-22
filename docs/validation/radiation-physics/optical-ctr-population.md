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

1. Medium, guard scope: `radiated_energy_fraction` cannot detect coherent energy below a band whose lower edge lies above the Gaussian window (estimate for 30 keV on gold at 200 fs: about $2\times10^{-4}$ of the kinetic energy at 1 pC, $2\times10^{-3}$ at 10 pC). Applied as documentation: the budget is stated as in-band only in the module, physics page and ledger.
2. Low, wording: weighted estimators are consistent, not unbiased. Applied (test renamed, ledger Checks reworded).
3. Low, resolution criterion: stated as sampling interval (grid spacing), not RMS blur. Applied to the source docstring and physics page.
4. Low, scope: `gaussian_coherent_band_eV` is a time-only, normal-incidence window; a tilted front observed obliquely can widen it. Applied to the physics page.
5. Low, scope: the formation measure bounds the self power only. Applied.

### Caveats

- The `sampled` law with histories that carry no offsets is a point bunch ($|F|^2=1$) by contract; this is not detectable.
- Bunch evolution (space charge, wakefields) is outside the passive-source model.
